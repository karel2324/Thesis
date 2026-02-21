/* ===============================================================
   SQL14_sofa_renal.sql - MIMIC-IV SOFA Renal Component

   Analogue to AUMCdb SQL14_Sofa_renal.sql

   Uses creatinine (converted to μmol/L in SQL5) + urine output
   SOFA thresholds for creatinine in μmol/L (SI units):
   - 0: <110
   - 1: 110-170
   - 2: 171-299
   - 3: 300-440
   - 4: >=440

   Original mg/dL thresholds × 88.4 = μmol/L

   SOFA thresholds for urine output (24h):
   - 3: <500 mL
   - 4: <200 mL
   =============================================================== */

CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.sofa_renal_24h` AS
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
   2) CREATININE EVENTS -> SOFA RENAL (CREAT)
      Values are in μmol/L (converted from mg/dL in SQL5)
   =============================================================== */
sofa_renal_creat_events AS (
  SELECT
    stay_id,
    ts,

    CASE
      WHEN value_as_number < 110 THEN 0
      WHEN value_as_number < 171 THEN 1
      WHEN value_as_number < 300 THEN 2
      WHEN value_as_number < 440 THEN 3
      WHEN value_as_number >= 440 THEN 4
    END AS sofa_renal_creat
  FROM `windy-forge-475207-e3.${DATASET}.cohort_measurements_window_stat_valid`
  WHERE var = 'creat'
    AND value_as_number IS NOT NULL
),

/* ===============================================================
   3a) WORST CREATININE SOFA - CURRENT WINDOW (grid_ts - 24h, grid_ts]
   =============================================================== */
sofa_renal_creat_24h_current AS (
  SELECT
    g.stay_id,
    g.grid_ts,
    MAX(e.sofa_renal_creat) AS sofa_renal_creat_24h_current
  FROM grid g
  LEFT JOIN sofa_renal_creat_events e
    ON e.stay_id = g.stay_id
   AND e.ts > TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR)
   AND e.ts <= g.grid_ts
  GROUP BY g.stay_id, g.grid_ts
),

/* ===============================================================
   3b) WORST CREATININE SOFA - FORWARD WINDOW (grid_ts + step - 24h, grid_ts + step]
   =============================================================== */
sofa_renal_creat_24h_forward AS (
  SELECT
    g.stay_id,
    g.grid_ts,
    MAX(e.sofa_renal_creat) AS sofa_renal_creat_24h_forward
  FROM grid g
  CROSS JOIN cfg
  LEFT JOIN sofa_renal_creat_events e
    ON e.stay_id = g.stay_id
   AND e.ts > TIMESTAMP_SUB(
                TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR),
                INTERVAL 24 HOUR)
   AND e.ts <= TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR)
  GROUP BY g.stay_id, g.grid_ts
),

/* ===============================================================
   4) UO RATES (from utils)
   =============================================================== */
uo_rates AS (
  SELECT
    stay_id,
    t_start,
    t_end,
    rate_ml_per_h
  FROM `windy-forge-475207-e3.${DATASET}.uo_rates`
),

/* ===============================================================
   5a) UO INTEGRATION - CURRENT WINDOW: 24h BEFORE grid_ts
   =============================================================== */
uo_24h_current AS (
  SELECT
    g.stay_id,
    g.grid_ts,

    SUM(
      GREATEST(
        0.0,
        TIMESTAMP_DIFF(
          LEAST(g.grid_ts, r.t_end),
          GREATEST(TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR), r.t_start),
          MINUTE
        ) / 60.0
      ) * r.rate_ml_per_h
    ) AS uo_24h_ml_current

  FROM grid g
  LEFT JOIN uo_rates r
    ON r.stay_id = g.stay_id
   AND r.t_end > TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR)
   AND r.t_start < g.grid_ts

  GROUP BY g.stay_id, g.grid_ts
),

/* ===============================================================
   5b) UO INTEGRATION - FORWARD WINDOW: 24h BEFORE (grid_ts + step)
   =============================================================== */
uo_24h_forward AS (
  SELECT
    g.stay_id,
    g.grid_ts,

    SUM(
      GREATEST(
        0.0,
        TIMESTAMP_DIFF(
          LEAST(
            TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR),
            r.t_end
          ),
          GREATEST(
            TIMESTAMP_SUB(
              TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR),
              INTERVAL 24 HOUR
            ),
            r.t_start
          ),
          MINUTE
        ) / 60.0
      ) * r.rate_ml_per_h
    ) AS uo_24h_ml_forward

  FROM grid g
  CROSS JOIN cfg
  LEFT JOIN uo_rates r
    ON r.stay_id = g.stay_id
   AND r.t_end > TIMESTAMP_SUB(
                   TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR),
                   INTERVAL 24 HOUR)
   AND r.t_start < TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR)

  GROUP BY g.stay_id, g.grid_ts
),

/* ===============================================================
   6a) UO -> SOFA RENAL - CURRENT
   =============================================================== */
sofa_renal_uo_current AS (
  SELECT
    stay_id,
    grid_ts,
    CASE
      WHEN uo_24h_ml_current < 200 THEN 4
      WHEN uo_24h_ml_current < 500 THEN 3
      ELSE 0
    END AS sofa_renal_uo_current
  FROM uo_24h_current
),

/* ===============================================================
   6b) UO -> SOFA RENAL - FORWARD
   =============================================================== */
sofa_renal_uo_forward AS (
  SELECT
    stay_id,
    grid_ts,
    CASE
      WHEN uo_24h_ml_forward < 200 THEN 4
      WHEN uo_24h_ml_forward < 500 THEN 3
      ELSE 0
    END AS sofa_renal_uo_forward
  FROM uo_24h_forward
)

/* ===============================================================
   FINAL: COMBINE CREAT + UO FOR BOTH WINDOWS
   =============================================================== */
SELECT
  g.subject_id,
  g.hadm_id,
  g.stay_id,
  g.grid_ts,

  GREATEST(
    IFNULL(cc.sofa_renal_creat_24h_current, 0),
    IFNULL(uc.sofa_renal_uo_current, 0)
  ) AS sofa_renal_24h_current,

  GREATEST(
    IFNULL(cf.sofa_renal_creat_24h_forward, 0),
    IFNULL(uf.sofa_renal_uo_forward, 0)
  ) AS sofa_renal_24h_forward

FROM grid g
LEFT JOIN sofa_renal_creat_24h_current cc USING (stay_id, grid_ts)
LEFT JOIN sofa_renal_creat_24h_forward cf USING (stay_id, grid_ts)
LEFT JOIN sofa_renal_uo_current uc USING (stay_id, grid_ts)
LEFT JOIN sofa_renal_uo_forward uf USING (stay_id, grid_ts)
ORDER BY subject_id, stay_id, grid_ts;
