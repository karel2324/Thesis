/* ===============================================================
   SQL6_last_variables.sql - MIMIC-IV Last Variables per Grid Point

   Analogue to AUMCdb SQL6_Last_variables.sql

   Finds the last measurement within lookback window per grid point
   =============================================================== */

CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.grid_last_var_long` AS
WITH
/* ===============================================================
   1) VAR MAP (deduplicated, exclude gcs)
   =============================================================== */
var_map AS (
  SELECT DISTINCT
    var,
    lb_h
  FROM `windy-forge-475207-e3.${DATASET}.var_map`
  WHERE var != 'gcs'
),

/* ===============================================================
   2) GRID (single source of truth)
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
   3) CANDIDATE MEASUREMENTS WITHIN LOOKBACK
   =============================================================== */
candidates AS (
  SELECT
    g.subject_id,
    g.hadm_id,
    g.stay_id,
    g.grid_ts,

    m.var,
    m.value_as_number AS val,
    m.ts
  FROM grid g
  JOIN `windy-forge-475207-e3.${DATASET}.cohort_measurements_window_stat_valid` m
    ON m.stay_id = g.stay_id
   AND m.ts <= g.grid_ts
  JOIN var_map vm
    ON vm.var = m.var
  WHERE m.source_type = 'state_var'
    AND m.ts > TIMESTAMP_SUB(g.grid_ts, INTERVAL vm.lb_h HOUR)
),

/* ===============================================================
   4) LAST MEASUREMENT PER VAR
   =============================================================== */
last_var AS (
  SELECT
    subject_id,
    hadm_id,
    stay_id,
    grid_ts,
    var,

    val AS val_last,
    ts  AS val_ts,

    -- age since measurement (hours)
    TIMESTAMP_DIFF(grid_ts, ts, MINUTE) / 60.0 AS val_age_hours,

    FORMAT('%s_last', var) AS feature_name
  FROM (
    SELECT
      *,
      ROW_NUMBER() OVER (
        PARTITION BY stay_id, grid_ts, var
        ORDER BY ts DESC
      ) AS rn
    FROM candidates
  )
  WHERE rn = 1
)

SELECT *
FROM last_var
ORDER BY subject_id, stay_id, grid_ts, var;
