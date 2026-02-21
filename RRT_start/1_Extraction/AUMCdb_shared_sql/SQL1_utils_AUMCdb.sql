
-- ===============================================================
-- SQL1_utils_AUMCdb.sql - ${COHORT_NAME}
-- Dataset: ${DATASET}
-- ===============================================================

/* ===============================================================
   1) VISIT TIMES  (visit-level constant)
   =============================================================== */
CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.visit_times` AS
SELECT
  vo.person_id,
  vo.visit_occurrence_id,
  COALESCE(vo.visit_start_datetime, TIMESTAMP(vo.visit_start_date)) AS admit_dt,
  COALESCE(vo.visit_end_datetime,   TIMESTAMP(vo.visit_end_date))   AS discharge_dt
FROM `amsterdamumcdb.version1_5_0.visit_occurrence` vo
WHERE vo.person_id IS NOT NULL
  AND vo.visit_occurrence_id IS NOT NULL
;

/* ===============================================================
   2) DEATH (person-level constant)
   =============================================================== */
CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.death` AS
SELECT
  person_id,
  COALESCE(death_datetime, TIMESTAMP(death_date)) AS death_dt
FROM `amsterdamumcdb.version1_5_0.death`
WHERE person_id IS NOT NULL
;

/* ===============================================================
   3) DEMOGRAPHICS (person-level constant)
   =============================================================== */
CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.demographics` AS
SELECT
  p.person_id,
  CASE p.gender_concept_id
    WHEN 8507 THEN 'M'
    WHEN 8532 THEN 'F'
    ELSE NULL
  END AS gender,
  DATE(
    IFNULL(p.year_of_birth, 1900),
    IFNULL(p.month_of_birth, 7),
    IFNULL(p.day_of_birth, 1)
  ) AS birth_date
FROM `amsterdamumcdb.version1_5_0.person` p
WHERE p.person_id IS NOT NULL
;

/* ===============================================================
   4) VAR_MAP (cid -> var + ranges + lookback)
   Uses unified cfg_params with aumcdb_ids
   =============================================================== */
CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.var_map` AS
WITH cfg AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.cfg_params` LIMIT 1
)
SELECT
  v.var,
  v.lb_h,
  v.min_clin AS min_val_clin,
  v.max_clin AS max_val_clin,
  v.min_stat AS min_val_stat,
  v.max_stat AS max_val_stat,
  cid AS measurement_concept_id
FROM cfg,
UNNEST(cfg.var_defs) v,
UNNEST(v.aumcdb_ids) AS cid
;

/* ===============================================================
   5) GCS MAP (value_as_concept_id -> component + score)
   Uses unified cfg_params with gcs_map_aumcdb
   =============================================================== */
CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.gcs_map` AS
WITH cfg AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.cfg_params` LIMIT 1
)
SELECT
  gm.value_as_concept_id,
  gm.component,
  gm.score
FROM cfg, UNNEST(cfg.gcs_map_aumcdb) gm
;

/* ===============================================================
   8) WEIGHT: first measurement per person (op basis van meas_state var='weight')
   Uses unified cfg_params with weight_ids.aumcdb_ids
   =============================================================== */
CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.weight_first_person` AS
WITH
cfg AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.cfg_params` LIMIT 1
),
weight_cids AS (
  SELECT cid
  FROM cfg
  CROSS JOIN UNNEST(cfg.weight_ids.aumcdb_ids) cid
)
SELECT
  person_id,
  ANY_VALUE(val) AS weight_kg_first,
  ANY_VALUE(ts)  AS weight_ts_first
FROM (
  SELECT
    m.person_id,
    COALESCE(m.measurement_datetime, TIMESTAMP(m.measurement_date)) AS ts,
    SAFE_CAST(m.value_as_number AS FLOAT64) AS val,
    ROW_NUMBER() OVER (
      PARTITION BY m.person_id
      ORDER BY COALESCE(m.measurement_datetime, TIMESTAMP(m.measurement_date))
    ) AS rn
  FROM `amsterdamumcdb.version1_5_0.measurement` m
  WHERE m.person_id IS NOT NULL
    AND m.value_as_number IS NOT NULL
    AND ((m.provider_id BETWEEN 0 AND 99) OR m.provider_id IS NULL)
    AND m.measurement_concept_id IN (SELECT cid FROM weight_cids)
)
WHERE rn = 1
GROUP BY person_id
;

/* ===============================================================
   9) WEIGHT: effective per person (measured first -> gender mean -> global mean)
      output ook "weight_source" zodat je later kunt zien waar het vandaan komt
   =============================================================== */
CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.weight_effective_person` AS
WITH dem AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.demographics`
),
w_first AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.weight_first_person`
),
w_by_gender AS (
  SELECT
    d.gender,
    AVG(w.weight_kg_first) AS avg_weight_kg_gender
  FROM w_first w
  JOIN dem d USING (person_id)
  GROUP BY d.gender
),
w_global AS (
  SELECT AVG(avg_weight_kg_gender) AS avg_weight_kg_global
  FROM w_by_gender
)
SELECT
  d.person_id,
  w_first.weight_kg_first,
  w_first.weight_ts_first,
  COALESCE(
    w_first.weight_kg_first,
    wbg.avg_weight_kg_gender,
    wg.avg_weight_kg_global
  ) AS weight_used_kg,
  CASE
    WHEN w_first.weight_kg_first IS NOT NULL THEN 'measured_first'
    WHEN wbg.avg_weight_kg_gender IS NOT NULL THEN 'gender_mean'
    ELSE 'global_mean'
  END AS weight_source
FROM dem d
LEFT JOIN w_first ON w_first.person_id = d.person_id
LEFT JOIN w_by_gender wbg ON wbg.gender = d.gender
CROSS JOIN w_global wg
;


/* ===============================================================
   10) URINE OUTPUT RATES (uo_rates)
      - gebruikt cfg.uo_ids.aumcdb_ids
      - bouwt intervals via LAG (clamp naar max 24h terug)
      - rekent rate_ml_per_h en rate_ml_per_kg_per_h met weight_used_kg (person-level)
   =============================================================== */
CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.uo_rates` AS
WITH
cfg AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.cfg_params` LIMIT 1
),

uo_raw AS (
  SELECT DISTINCT
    m.person_id,
    m.visit_occurrence_id,
    COALESCE(m.measurement_datetime, TIMESTAMP(m.measurement_date)) AS ts,
    SAFE_CAST(m.value_as_number AS FLOAT64) AS urine_ml
  FROM `amsterdamumcdb.version1_5_0.measurement` m
  CROSS JOIN cfg
  WHERE m.person_id IS NOT NULL
    AND m.value_as_number IS NOT NULL
    AND ((m.provider_id BETWEEN 0 AND 99) OR m.provider_id IS NULL)
    AND m.measurement_concept_id IN UNNEST(cfg.uo_ids.aumcdb_ids)
),

uo_intervals AS (
  SELECT
    person_id,
    visit_occurrence_id,
    GREATEST(
      COALESCE(LAG(ts) OVER (
        PARTITION BY person_id, visit_occurrence_id ORDER BY ts),
        TIMESTAMP_SUB(ts, INTERVAL 1 HOUR)),
      TIMESTAMP_SUB(ts, INTERVAL 24 HOUR)
    ) AS t_start,
    ts AS t_end,
    urine_ml
  FROM uo_raw
),

w AS (
  SELECT person_id, weight_used_kg
  FROM `windy-forge-475207-e3.${DATASET}.weight_effective_person`
)
SELECT
  u.person_id,
  u.visit_occurrence_id,
  u.t_start,
  u.t_end,
  u.urine_ml,

  -- ml/h
  SAFE_DIVIDE(
    u.urine_ml,
    (TIMESTAMP_DIFF(u.t_end, u.t_start, MINUTE) / 60.0)
  ) AS rate_ml_per_h,

  -- ml/kg/h
  SAFE_DIVIDE(
    u.urine_ml,
    NULLIF(
      (TIMESTAMP_DIFF(u.t_end, u.t_start, MINUTE) / 60.0) * w.weight_used_kg,
      0
    )
  ) AS rate_ml_per_kg_per_h,

  w.weight_used_kg
FROM uo_intervals u
LEFT JOIN w
  ON w.person_id = u.person_id
WHERE u.t_end > u.t_start
;


