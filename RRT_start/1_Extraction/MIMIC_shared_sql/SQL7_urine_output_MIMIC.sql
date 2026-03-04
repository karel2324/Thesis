/* ===============================================================
   SQL8_urine_output.sql - MIMIC-IV Urine Output at Grid

   Analogue to AUMCdb SQL8_Urine_output.sql

   Integrates UO rates to grid timesteps:
   - uo_6h_mlkgh, uo_12h_mlkgh, uo_24h_mlkgh, uo_24h_ml
   Coverage check: only report if documented hours >= window (MIMIC-style)
   =============================================================== */

CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.grid_urine_output` AS
WITH
/* ===============================================================
   1) GRID
   =============================================================== */
grid AS (
  SELECT
    subject_id,
    hadm_id,
    stay_id,
    grid_ts
  FROM `windy-forge-475207-e3.${DATASET}.cohort_grid`
),

/* ===============================================================
   2) UO RATES (from SQL1_utils)
   =============================================================== */
uo_rates AS (
  SELECT
    stay_id,
    t_start,
    t_end,
    rate_ml_per_kg_per_h,
    rate_ml_per_h
  FROM `windy-forge-475207-e3.${DATASET}.uo_rates`
),

/* ===============================================================
   3) RAW VOLUMES + COVERAGE at each grid point
   =============================================================== */
uo_vol_at_grid AS (
  SELECT
    g.subject_id,
    g.hadm_id,
    g.stay_id,
    g.grid_ts,

    -- Raw volumes (ml/kg)
    SUM(
      GREATEST(0.0,
        TIMESTAMP_DIFF(LEAST(g.grid_ts, r.t_end), GREATEST(TIMESTAMP_SUB(g.grid_ts, INTERVAL 6 HOUR), r.t_start), MINUTE) / 60.0
      ) * r.rate_ml_per_kg_per_h
    ) AS vol_6h_mlkg,

    SUM(
      GREATEST(0.0,
        TIMESTAMP_DIFF(LEAST(g.grid_ts, r.t_end), GREATEST(TIMESTAMP_SUB(g.grid_ts, INTERVAL 12 HOUR), r.t_start), MINUTE) / 60.0
      ) * r.rate_ml_per_kg_per_h
    ) AS vol_12h_mlkg,

    SUM(
      GREATEST(0.0,
        TIMESTAMP_DIFF(LEAST(g.grid_ts, r.t_end), GREATEST(TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR), r.t_start), MINUTE) / 60.0
      ) * r.rate_ml_per_kg_per_h
    ) AS vol_24h_mlkg,

    -- Raw volume (absolute ml, 24h)
    SUM(
      GREATEST(0.0,
        TIMESTAMP_DIFF(LEAST(g.grid_ts, r.t_end), GREATEST(TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR), r.t_start), MINUTE) / 60.0
      ) * r.rate_ml_per_h
    ) AS vol_24h_ml,

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
  LEFT JOIN uo_rates r
    ON r.stay_id = g.stay_id
   AND r.t_end   > TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR)
   AND r.t_start < g.grid_ts

  GROUP BY
    g.subject_id,
    g.hadm_id,
    g.stay_id,
    g.grid_ts
)

/* ===============================================================
   4) FINAL: apply coverage check (NULL if insufficient coverage)
   =============================================================== */
SELECT
  subject_id,
  hadm_id,
  stay_id,
  grid_ts,
  CASE WHEN uo_tm_6h  >= 6.0  THEN SAFE_DIVIDE(vol_6h_mlkg,  6.0)  END AS uo_6h_mlkgh,
  CASE WHEN uo_tm_12h >= 12.0 THEN SAFE_DIVIDE(vol_12h_mlkg, 12.0) END AS uo_12h_mlkgh,
  CASE WHEN uo_tm_24h >= 24.0 THEN SAFE_DIVIDE(vol_24h_mlkg, 24.0) END AS uo_24h_mlkgh,
  CASE WHEN uo_tm_24h >= 24.0 THEN vol_24h_ml END AS uo_24h_ml
FROM uo_vol_at_grid
ORDER BY subject_id, stay_id, grid_ts;
