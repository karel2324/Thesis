/* ===============================================================
   SQL1_utils.sql - MIMIC-IV Utility Tables

   Analogue to AUMCdb SQL1_utils.sql

   Creates derived tables for:
   - Visit times
   - Demographics
   - Death
   - Var map (itemid -> variable name)
   - Weight (first + effective)
   - Urine output rates
   - GCS map
   =============================================================== */


/* ===============================================================
   1) ICU VISIT TIMES
   Note: Cast all datetime columns to TIMESTAMP for consistency
   =============================================================== */
CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.visit_times` AS
SELECT
  ie.subject_id,
  ie.hadm_id,
  ie.stay_id,
  TIMESTAMP(ie.intime)  AS icu_intime,
  TIMESTAMP(ie.outtime) AS icu_outtime,
  TIMESTAMP(adm.admittime) AS hospital_admittime,
  TIMESTAMP(adm.dischtime) AS hospital_dischtime
FROM `physionet-data.mimiciv_3_1_icu.icustays` ie
JOIN `physionet-data.mimiciv_3_1_hosp.admissions` adm
  ON ie.hadm_id = adm.hadm_id
WHERE ie.subject_id IS NOT NULL
  AND ie.stay_id IS NOT NULL
;


/* ===============================================================
   2) DEMOGRAPHICS (person-level)
   =============================================================== */
CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.demographics` AS
SELECT
  subject_id,
  gender,
  anchor_age,
  anchor_year,
  anchor_year_group
FROM `physionet-data.mimiciv_3_1_hosp.patients`
WHERE subject_id IS NOT NULL
;


/* ===============================================================
   3) DEATH (person-level)
   Note: Cast DATE to TIMESTAMP for consistency
   =============================================================== */
CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.death` AS
SELECT
  subject_id,
  TIMESTAMP(dod) AS death_dt
FROM `physionet-data.mimiciv_3_1_hosp.patients`
WHERE subject_id IS NOT NULL
  AND dod IS NOT NULL
;


/* ===============================================================
   4) ICU ADMISSION AGE (stay-level)
   =============================================================== */
CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.icu_admission_age` AS
SELECT
  ie.subject_id,
  ie.hadm_id,
  ie.stay_id,
  pat.anchor_age
    + DATE_DIFF(DATE(ie.intime), DATE(CONCAT(pat.anchor_year, '-01-01')), YEAR)
    AS age_at_icu_admission
FROM `physionet-data.mimiciv_3_1_icu.icustays` ie
JOIN `physionet-data.mimiciv_3_1_hosp.patients` pat
  ON ie.subject_id = pat.subject_id
;


/* ===============================================================
   5) VAR_MAP (itemid -> var + source + ranges + lookback)
   Uses unified cfg_params with mimic_ids
   Note: source is determined by itemid range (chart vs lab)
   =============================================================== */
CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.var_map` AS
WITH cfg AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.cfg_params` LIMIT 1
)
SELECT
  v.var,
  -- Determine source based on itemid (lab items < 100000, chart items > 200000)
  CASE
    WHEN itemid < 100000 THEN 'lab'
    ELSE 'chart'
  END AS source,
  v.lb_h,
  -- Bounds in MIMIC units (for validation before conversion)
  -- Special handling for temperature: C to F = C * 9/5 + 32
  CASE
    WHEN v.var = 'temperature' THEN v.min_clin * 9/5 + 32  -- Celsius to Fahrenheit
    WHEN v.conv_factor IS NOT NULL AND v.conv_factor != 1.0 THEN v.min_clin / v.conv_factor
    ELSE v.min_clin
  END AS min_val_clin,
  CASE
    WHEN v.var = 'temperature' THEN v.max_clin * 9/5 + 32  -- Celsius to Fahrenheit
    WHEN v.conv_factor IS NOT NULL AND v.conv_factor != 1.0 THEN v.max_clin / v.conv_factor
    ELSE v.max_clin
  END AS max_val_clin,
  CASE
    WHEN v.var = 'temperature' THEN v.min_stat * 9/5 + 32  -- Celsius to Fahrenheit
    WHEN v.conv_factor IS NOT NULL AND v.conv_factor != 1.0 THEN v.min_stat / v.conv_factor
    ELSE v.min_stat
  END AS min_val_stat,
  CASE
    WHEN v.var = 'temperature' THEN v.max_stat * 9/5 + 32  -- Celsius to Fahrenheit
    WHEN v.conv_factor IS NOT NULL AND v.conv_factor != 1.0 THEN v.max_stat / v.conv_factor
    ELSE v.max_stat
  END AS max_val_stat,
  -- Bounds in SI units (AUMCdb standard, for output after conversion)
  v.min_clin AS min_val_clin_si,
  v.max_clin AS max_val_clin_si,
  v.min_stat AS min_val_stat_si,
  v.max_stat AS max_val_stat_si,
  -- Conversion factor and formula
  v.conv_factor,
  v.conv_formula,
  itemid
FROM cfg
CROSS JOIN UNNEST(cfg.var_defs) AS v
CROSS JOIN UNNEST(v.mimic_ids) AS itemid
;


/* ===============================================================
   6) GCS MAP (itemid -> component)
   Uses unified cfg_params with gcs_itemids_mimic struct
   =============================================================== */
CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.gcs_map` AS
WITH cfg AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.cfg_params` LIMIT 1
)
SELECT cfg.gcs_itemids_mimic.eye_itemid AS itemid, 'eye' AS component FROM cfg
UNION ALL
SELECT cfg.gcs_itemids_mimic.verbal_itemid, 'verbal' FROM cfg
UNION ALL
SELECT cfg.gcs_itemids_mimic.motor_itemid, 'motor' FROM cfg
;


/* ===============================================================
   7) WEIGHT: first measurement per ICU stay
   Uses unified cfg_params with weight_ids.mimic_ids
   =============================================================== */
CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.weight_first_stay` AS
WITH cfg AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.cfg_params` LIMIT 1
),

weight_ranked AS (
  SELECT
    ie.subject_id,
    ie.stay_id,
    TIMESTAMP(ce.charttime) AS charttime,
    ce.valuenum AS weight_kg,
    ROW_NUMBER() OVER (
      PARTITION BY ie.stay_id
      ORDER BY ce.charttime
    ) AS rn
  FROM `physionet-data.mimiciv_3_1_icu.icustays` ie
  JOIN `physionet-data.mimiciv_3_1_icu.chartevents` ce
    ON ce.stay_id = ie.stay_id
   AND TIMESTAMP(ce.charttime) BETWEEN
       TIMESTAMP_SUB(TIMESTAMP(ie.intime), INTERVAL 24 HOUR)
       AND TIMESTAMP_ADD(TIMESTAMP(ie.intime), INTERVAL 24 HOUR)
  CROSS JOIN cfg
  WHERE ce.itemid IN UNNEST(cfg.weight_ids.mimic_ids)
    AND ce.valuenum IS NOT NULL
    AND ce.valuenum BETWEEN 20 AND 300  -- plausible weight in kg
)

SELECT
  subject_id,
  stay_id,
  weight_kg AS weight_kg_first,
  charttime AS weight_ts_first
FROM weight_ranked
WHERE rn = 1
;


/* ===============================================================
   8) WEIGHT: effective per stay (measured -> gender mean -> global mean)
   =============================================================== */
CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.weight_effective_stay` AS
WITH dem AS (
  SELECT subject_id, gender
  FROM `windy-forge-475207-e3.${DATASET}.demographics`
),

w_first AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.weight_first_stay`
),

-- Gender mean (only where weight was measured)
w_by_gender AS (
  SELECT
    d.gender,
    AVG(w.weight_kg_first) AS avg_weight_kg_gender
  FROM w_first w
  JOIN dem d USING (subject_id)
  WHERE w.weight_kg_first IS NOT NULL
  GROUP BY d.gender
),

-- Global mean
w_global AS (
  SELECT AVG(avg_weight_kg_gender) AS avg_weight_kg_global
  FROM w_by_gender
),

stays AS (
  SELECT subject_id, stay_id
  FROM `physionet-data.mimiciv_3_1_icu.icustays`
)

SELECT
  s.subject_id,
  s.stay_id,
  wf.weight_kg_first,
  wf.weight_ts_first,
  COALESCE(
    wf.weight_kg_first,
    wbg.avg_weight_kg_gender,
    wg.avg_weight_kg_global
  ) AS weight_used_kg,
  CASE
    WHEN wf.weight_kg_first IS NOT NULL THEN 'measured_first'
    WHEN wbg.avg_weight_kg_gender IS NOT NULL THEN 'gender_mean'
    ELSE 'global_mean'
  END AS weight_source
FROM stays s
LEFT JOIN w_first wf ON wf.stay_id = s.stay_id
LEFT JOIN dem d ON d.subject_id = s.subject_id
LEFT JOIN w_by_gender wbg ON wbg.gender = d.gender
CROSS JOIN w_global wg
;


/* ===============================================================
   9) URINE OUTPUT RATES
      Uses outputevents with cfg.uo_ids.mimic_ids
      Calculates rate_ml_per_h and rate_ml_per_kg_per_h
   =============================================================== */
CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.uo_rates` AS
WITH cfg AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.cfg_params` LIMIT 1
),

-- Raw urine output events
uo_raw AS (
  SELECT
    oe.subject_id,
    oe.hadm_id,
    oe.stay_id,
    TIMESTAMP(oe.charttime) AS ts,
    SAFE_CAST(oe.value AS FLOAT64) AS urine_ml
  FROM `physionet-data.mimiciv_3_1_icu.outputevents` oe
  CROSS JOIN cfg
  WHERE oe.stay_id IS NOT NULL
    AND oe.charttime IS NOT NULL
    AND oe.value IS NOT NULL
    AND oe.value > 0
    AND oe.itemid IN UNNEST(cfg.uo_ids.mimic_ids)
),

-- Build intervals using LAG (clamped to max 24h back)
uo_intervals AS (
  SELECT
    subject_id,
    hadm_id,
    stay_id,
    GREATEST(
      COALESCE(
        LAG(ts) OVER (PARTITION BY stay_id ORDER BY ts),
        TIMESTAMP_SUB(ts, INTERVAL 1 HOUR)
      ),
      TIMESTAMP_SUB(ts, INTERVAL 24 HOUR)
    ) AS t_start,
    ts AS t_end,
    urine_ml
  FROM uo_raw
),

-- Effective weight per stay
w AS (
  SELECT stay_id, weight_used_kg
  FROM `windy-forge-475207-e3.${DATASET}.weight_effective_stay`
)

SELECT
  u.subject_id,
  u.hadm_id,
  u.stay_id,
  u.t_start,
  u.t_end,
  u.urine_ml,

  -- ml / h
  SAFE_DIVIDE(
    u.urine_ml,
    TIMESTAMP_DIFF(u.t_end, u.t_start, MINUTE) / 60.0
  ) AS rate_ml_per_h,

  -- ml / kg / h
  SAFE_DIVIDE(
    u.urine_ml,
    (TIMESTAMP_DIFF(u.t_end, u.t_start, MINUTE) / 60.0) * w.weight_used_kg
  ) AS rate_ml_per_kg_per_h,

  w.weight_used_kg
FROM uo_intervals u
LEFT JOIN w ON w.stay_id = u.stay_id
WHERE u.t_end > u.t_start
;
