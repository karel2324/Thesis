/* ===============================================================
   SQL14_sofa_coag.sql - MIMIC-IV SOFA Coagulation Component

   Analogue to AUMCdb SQL14_Sofa_coag.sql

   Uses platelet count
   =============================================================== */

CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.sofa_coag_24h` AS
WITH
/* ===============================================================
   0) CONFIG
   =============================================================== */
cfg AS (
  SELECT *
  FROM `windy-forge-475207-e3.${DATASET}.cfg_params`
  LIMIT 1
),

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
   2) PLATELET EVENTS -> SOFA COAG SCORE
      MIMIC platelets are in K/uL (same as 10^3/uL)
   =============================================================== */
sofa_coag_events AS (
  SELECT
    stay_id,
    ts,

    CASE
      WHEN value_as_number >= 150 THEN 0
      WHEN value_as_number >= 100 THEN 1
      WHEN value_as_number >= 50  THEN 2
      WHEN value_as_number >= 20  THEN 3
      WHEN value_as_number <  20  THEN 4
    END AS sofa_coag
  FROM `windy-forge-475207-e3.${DATASET}.cohort_measurements_window_stat_valid`
  WHERE var = 'platelets'
    AND value_as_number IS NOT NULL
),

/* ===============================================================
   3a) WORST SOFA COAG - CURRENT WINDOW (grid_ts - 24h, grid_ts]
   =============================================================== */
sofa_coag_24h_current AS (
  SELECT
    g.stay_id,
    g.grid_ts,
    MAX(e.sofa_coag) AS sofa_coag_24h_current
  FROM grid g
  LEFT JOIN sofa_coag_events e
    ON e.stay_id = g.stay_id
   AND e.ts > TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR)
   AND e.ts <= g.grid_ts
  GROUP BY g.stay_id, g.grid_ts
),

/* ===============================================================
   3b) WORST SOFA COAG - FORWARD WINDOW (grid_ts + step - 24h, grid_ts + step]
   =============================================================== */
sofa_coag_24h_forward AS (
  SELECT
    g.stay_id,
    g.grid_ts,
    MAX(e.sofa_coag) AS sofa_coag_24h_forward
  FROM grid g
  CROSS JOIN cfg
  LEFT JOIN sofa_coag_events e
    ON e.stay_id = g.stay_id
   AND e.ts > TIMESTAMP_SUB(
                TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR),
                INTERVAL 24 HOUR)
   AND e.ts <= TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR)
  GROUP BY g.stay_id, g.grid_ts
)

/* ===============================================================
   FINAL
   =============================================================== */
SELECT
  g.subject_id,
  g.hadm_id,
  g.stay_id,
  g.grid_ts,
  c.sofa_coag_24h_current,
  f.sofa_coag_24h_forward
FROM grid g
LEFT JOIN sofa_coag_24h_current c USING (stay_id, grid_ts)
LEFT JOIN sofa_coag_24h_forward f USING (stay_id, grid_ts)
ORDER BY subject_id, stay_id, grid_ts;
