/* ===============================================================
   SQL8_renal_trends.sql - MIMIC-IV Renal Trends + KDIGO Stage

   Analogue to AUMCdb SQL8_Renal_trends.sql

   Calculates:
   - creat_min_48h, urea_min_48h, creat_min_7d
   - creat_abs_inc_48h, creat_rel_inc_48h, creat_rel_inc_7d
   - avg_uo_6h, avg_uo_12h, avg_uo_24h
   - kdigo_stage (0-3)
   =============================================================== */

CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.grid_renal_trends` AS
WITH
/* ===============================================================
   1) GRID (single source of truth)
   =============================================================== */
cfg AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.cfg_params` LIMIT 1
),

grid AS (
  SELECT
    subject_id,
    hadm_id,
    stay_id,
    grid_ts
  FROM `windy-forge-475207-e3.${DATASET}.cohort_grid`
),

dem AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.demographics`
),

/* ===============================================================
   1b) AGE INFO
   =============================================================== */
age_info AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.icu_admission_age`
),

/* ===============================================================
   1c) CKD IDENTIFICATION (ICD codes)
   =============================================================== */
ckd_stays AS (
  SELECT DISTINCT ie.stay_id
  FROM `physionet-data.mimiciv_3_1_icu.icustays` ie
  JOIN `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` dx
    ON ie.hadm_id = dx.hadm_id
  WHERE STARTS_WITH(dx.icd_code, '585')    -- ICD-9 CKD
     OR STARTS_WITH(dx.icd_code, 'N18')    -- ICD-10 CKD
),

/* ===============================================================
   2) LABS (creat & urea, raw)
   =============================================================== */
labs AS (
  SELECT
    stay_id,
    ts,
    var,
    value_as_number AS val
  FROM `windy-forge-475207-e3.${DATASET}.cohort_measurements_window_stat_valid`
  WHERE var IN ('creat', 'urea')
),

/* ===============================================================
   3) MINIMUM WITHIN 48h AND ALL-TIME BEFORE GRID
   =============================================================== */
min_windows AS (
  SELECT
    g.subject_id,
    g.hadm_id,
    g.stay_id,
    g.grid_ts,

    -- 48h minima
    MIN(IF(l.var = 'creat' AND l.ts > TIMESTAMP_SUB(g.grid_ts, INTERVAL 48 HOUR), l.val, NULL)) AS creat_min_48h,
    MIN(IF(l.var = 'urea'  AND l.ts > TIMESTAMP_SUB(g.grid_ts, INTERVAL 48 HOUR), l.val, NULL)) AS urea_min_48h,

    -- All-time minimum creatinine (for KDIGO baseline, >= min_clin)
    MIN(IF(l.var = 'creat'
           AND l.val >= (SELECT v.min_clin FROM cfg, UNNEST(cfg.var_defs) v WHERE v.var = 'creat'),
           l.val, NULL)) AS creat_min_alltime

  FROM grid g
  LEFT JOIN labs l
    ON l.stay_id = g.stay_id
   AND l.ts <= g.grid_ts

  GROUP BY g.subject_id, g.hadm_id, g.stay_id, g.grid_ts
),

/* ===============================================================
   3b) BASELINE CREATININE (CKD-aware, MDRD back-calc from eGFR=75)
       Note: measurements are already in µmol/L, so MDRD * 88.4.
   =============================================================== */
creat_baseline AS (
  SELECT
    mw.subject_id,
    mw.hadm_id,
    mw.stay_id,
    mw.grid_ts,
    COALESCE(
      mw.creat_min_alltime,
      CASE
        -- CKD (ICD discharge diagnosis): no baseline assumption
        WHEN ckd.stay_id IS NOT NULL THEN NULL
        -- MDRD back-calculation from eGFR=75 (µmol/L)
        ELSE POWER(
          (175.0 / 75.0)
          * POWER(GREATEST(COALESCE(ai.age_at_icu_admission, 65.0), 18.0), -0.203)
          * IF(COALESCE(d.gender, 'M') = 'F', 0.742, 1.0),
          1.0 / 1.154
        ) * 88.4
      END
    ) AS creat_min_baseline
  FROM min_windows mw
  LEFT JOIN ckd_stays ckd ON ckd.stay_id = mw.stay_id
  LEFT JOIN dem d ON d.subject_id = mw.subject_id
  LEFT JOIN age_info ai ON ai.stay_id = mw.stay_id
),

/* ===============================================================
   4) LAST VALUE AT GRID
   =============================================================== */
last_vals AS (
  SELECT
    stay_id,
    grid_ts,

    MAX(IF(var = 'creat', val_last, NULL)) AS creat_last,
    MAX(IF(var = 'urea',  val_last, NULL)) AS urea_last

  FROM `windy-forge-475207-e3.${DATASET}.grid_last_var_long`
  WHERE var IN ('creat', 'urea')
  GROUP BY stay_id, grid_ts
),

/* ===============================================================
   5) URINE OUTPUT RATES (from uo_rates util)
   =============================================================== */
uo_rates AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.uo_rates`
),

/* ===============================================================
   6) URINE OUTPUT VOLUMES PER GRID (6h, 12h, 24h)
   =============================================================== */
uo_vol_at_grid AS (
  SELECT
    g.subject_id,
    g.hadm_id,
    g.stay_id,
    g.grid_ts,

    -- Volume in 6h window (ml/kg)
    SUM(
      GREATEST(0.0,
        TIMESTAMP_DIFF(
          LEAST(g.grid_ts, r.t_end),
          GREATEST(TIMESTAMP_SUB(g.grid_ts, INTERVAL 6 HOUR), r.t_start),
          MINUTE
        ) / 60.0
      ) * r.rate_ml_per_kg_per_h
    ) AS vol_6h_mlkg,

    -- Volume in 12h window (ml/kg)
    SUM(
      GREATEST(0.0,
        TIMESTAMP_DIFF(
          LEAST(g.grid_ts, r.t_end),
          GREATEST(TIMESTAMP_SUB(g.grid_ts, INTERVAL 12 HOUR), r.t_start),
          MINUTE
        ) / 60.0
      ) * r.rate_ml_per_kg_per_h
    ) AS vol_12h_mlkg,

    -- Volume in 24h window (ml/kg)
    SUM(
      GREATEST(0.0,
        TIMESTAMP_DIFF(
          LEAST(g.grid_ts, r.t_end),
          GREATEST(TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR), r.t_start),
          MINUTE
        ) / 60.0
      ) * r.rate_ml_per_kg_per_h
    ) AS vol_24h_mlkg,

    -- Coverage: total documented hours in each window (MIMIC-style)
    SUM(GREATEST(0.0,
      TIMESTAMP_DIFF(LEAST(g.grid_ts, r.t_end), GREATEST(TIMESTAMP_SUB(g.grid_ts, INTERVAL 6 HOUR), r.t_start), MINUTE) / 60.0
    )) AS uo_tm_6h,
    SUM(GREATEST(0.0,
      TIMESTAMP_DIFF(LEAST(g.grid_ts, r.t_end), GREATEST(TIMESTAMP_SUB(g.grid_ts, INTERVAL 12 HOUR), r.t_start), MINUTE) / 60.0
    )) AS uo_tm_12h,
    SUM(GREATEST(0.0,
      TIMESTAMP_DIFF(LEAST(g.grid_ts, r.t_end), GREATEST(TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR), r.t_start), MINUTE) / 60.0
    )) AS uo_tm_24h

  FROM grid g
  JOIN uo_rates r
    ON r.stay_id = g.stay_id
   AND r.t_end > TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR)
   AND r.t_start < g.grid_ts
  GROUP BY g.subject_id, g.hadm_id, g.stay_id, g.grid_ts
),

/* ===============================================================
   7) URINE OUTPUT AVERAGES (only if full window coverage)
   =============================================================== */
uo_avg AS (
  SELECT
    subject_id,
    hadm_id,
    stay_id,
    grid_ts,

    CASE WHEN uo_tm_6h  >= 6.0  THEN SAFE_DIVIDE(vol_6h_mlkg,  6.0)  END AS avg_uo_6h,
    CASE WHEN uo_tm_12h >= 12.0 THEN SAFE_DIVIDE(vol_12h_mlkg, 12.0) END AS avg_uo_12h,
    CASE WHEN uo_tm_24h >= 24.0 THEN SAFE_DIVIDE(vol_24h_mlkg, 24.0) END AS avg_uo_24h

  FROM uo_vol_at_grid
)

/* ===============================================================
   FINAL
   =============================================================== */
SELECT
  g.subject_id,
  g.hadm_id,
  g.stay_id,
  g.grid_ts,

  -- absolute levels
  lv.creat_last,
  lv.urea_last,

  -- minima
  mw.creat_min_48h,
  mw.urea_min_48h,
  cb.creat_min_baseline AS creat_min_7d,

  -- absolute increase (48h)
  (lv.creat_last - mw.creat_min_48h) AS creat_abs_inc_48h,
  (lv.urea_last  - mw.urea_min_48h)  AS urea_abs_inc_48h,


  -- relative increase (48h-min based)
  SAFE_DIVIDE(lv.creat_last, mw.creat_min_48h) AS creat_rel_inc_48h,
  SAFE_DIVIDE(lv.urea_last,  mw.urea_min_48h)  AS urea_rel_inc_48h,

  -- relative increase (7d baseline)
  SAFE_DIVIDE(lv.creat_last, cb.creat_min_baseline) AS creat_rel_inc_7d,

  -- urine output averages (ml/kg/h)
  uo.avg_uo_6h,
  uo.avg_uo_12h,
  uo.avg_uo_24h,

  -- KDIGO stage (0 = no AKI, 1-3 = stage)
  CASE
    -- Stage 3 (highest priority)
    WHEN (lv.creat_last >= 353.6 AND (lv.creat_last - mw.creat_min_48h) >= 26.5)
      OR SAFE_DIVIDE(lv.creat_last, cb.creat_min_baseline) >= 3.0
      OR uo.avg_uo_24h < 0.3
      OR uo.avg_uo_12h = 0.0
    THEN 3

    -- Stage 2
    WHEN SAFE_DIVIDE(lv.creat_last, cb.creat_min_baseline) >= 2.0
      OR uo.avg_uo_12h < 0.5
    THEN 2

    -- Stage 1
    WHEN (lv.creat_last - mw.creat_min_48h) >= 26.5
      OR SAFE_DIVIDE(lv.creat_last, cb.creat_min_baseline) >= 1.5
      OR uo.avg_uo_6h < 0.5
    THEN 1

    ELSE 0
  END AS kdigo_stage

FROM grid g
LEFT JOIN last_vals lv
  USING (stay_id, grid_ts)
LEFT JOIN min_windows mw
  USING (subject_id, hadm_id, stay_id, grid_ts)
LEFT JOIN creat_baseline cb
  USING (subject_id, hadm_id, stay_id, grid_ts)
LEFT JOIN uo_avg uo
  USING (subject_id, hadm_id, stay_id, grid_ts)
ORDER BY subject_id, stay_id, grid_ts;
