/* ===============================================================
   SQL14_sofa_liver.sql - MIMIC-IV SOFA Liver Component

   Analogue to AUMCdb SQL14_Sofa_liver.sql

   Uses bilirubin (converted to μmol/L in SQL5)
   SOFA thresholds for bilirubin in μmol/L (SI units):
   - 0: <20
   - 1: 20-32
   - 2: 33-101
   - 3: 102-203
   - 4: >=204
   =============================================================== */

CREATE OR REPLACE TABLE `windy-forge-475207-e3.derived_mimic.sofa_liver_24h` AS
WITH
/* ===============================================================
   0) CONFIG
   =============================================================== */
cfg AS (
  SELECT *
  FROM `windy-forge-475207-e3.derived_mimic.cfg_params`
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
  FROM `windy-forge-475207-e3.derived_mimic.cohort_grid`
),

/* ===============================================================
   2) BILIRUBIN EVENTS -> SOFA LIVER SCORE
      Values are in μmol/L (converted from mg/dL in SQL5)
   =============================================================== */
sofa_liver_events AS (
  SELECT
    stay_id,
    ts,

    CASE
      WHEN value_as_number <  20  THEN 0
      WHEN value_as_number <  33  THEN 1
      WHEN value_as_number < 102  THEN 2
      WHEN value_as_number < 204  THEN 3
      WHEN value_as_number >= 204 THEN 4
      ELSE NULL
    END AS sofa_liver
  FROM `windy-forge-475207-e3.derived_mimic.cohort_measurements_window_stat_valid`
  WHERE var = 'bilirubin'
    AND value_as_number IS NOT NULL
),

/* ===============================================================
   3a) WORST SOFA LIVER - CURRENT WINDOW (grid_ts - 24h, grid_ts]
   =============================================================== */
sofa_liver_24h_current AS (
  SELECT
    g.stay_id,
    g.grid_ts,
    MAX(e.sofa_liver) AS sofa_liver_24h_current
  FROM grid g
  LEFT JOIN sofa_liver_events e
    ON e.stay_id = g.stay_id
   AND e.ts > TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR)
   AND e.ts <= g.grid_ts
  GROUP BY g.stay_id, g.grid_ts
),

/* ===============================================================
   3b) WORST SOFA LIVER - FORWARD WINDOW (grid_ts + step - 24h, grid_ts + step]
   =============================================================== */
sofa_liver_24h_forward AS (
  SELECT
    g.stay_id,
    g.grid_ts,
    MAX(e.sofa_liver) AS sofa_liver_24h_forward
  FROM grid g
  CROSS JOIN cfg
  LEFT JOIN sofa_liver_events e
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
  c.sofa_liver_24h_current,
  f.sofa_liver_24h_forward
FROM grid g
LEFT JOIN sofa_liver_24h_current c USING (stay_id, grid_ts)
LEFT JOIN sofa_liver_24h_forward f USING (stay_id, grid_ts)
ORDER BY subject_id, stay_id, grid_ts;
