/* ===============================================================
   SQL14_sofa_resp.sql - MIMIC-IV SOFA Respiratory Component

   Analogue to AUMCdb SQL14_Sofa_resp.sql

   Uses PaO2/FiO2 ratio + mechanical ventilation status

   SOFA thresholds for PaO2/FiO2 (mmHg):
   - 0: >= 400
   - 1: 300-399
   - 2: 200-299 (without vent) or < 200 (without vent)
   - 3: 100-199 (with vent)
   - 4: < 100 (with vent)
   =============================================================== */

CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.sofa_resp_24h` AS
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
   2) EVENTS (PaO2, FiO2)
   =============================================================== */
pao2_events AS (
  SELECT
    stay_id,
    ts,
    value_as_number AS pao2
  FROM `windy-forge-475207-e3.${DATASET}.cohort_measurements_window_stat_valid`
  WHERE var = 'pao2'
    AND value_as_number IS NOT NULL
),

fio2_events AS (
  SELECT
    stay_id,
    ts,
    value_as_number AS fio2
  FROM `windy-forge-475207-e3.${DATASET}.cohort_measurements_window_stat_valid`
  WHERE var = 'fio2'
    AND value_as_number IS NOT NULL
),

/* ===============================================================
   3) MECHANICAL VENTILATION INTERVALS (from raw chartevents)

   Key itemids for ventilator settings:
   - 223848: Ventilator Mode
   - 223849: Vent Mode (Hamilton)
   - 229314: Ventilator Mode (PB)
   - 220210: Respiratory Rate (set)
   - 224700: Total PEEP Level
   =============================================================== */
vent_itemids AS (
  SELECT itemid FROM UNNEST([
    223848, 223849, 229314, 220210, 224700,
    224687, 224684, 224685, 224686, 223835,
    224697, 224695, 224696
  ]) AS itemid
),

vent_events AS (
  SELECT
    ce.stay_id,
    TIMESTAMP(ce.charttime) AS ts
  FROM `physionet-data.mimiciv_3_1_icu.chartevents` ce
  WHERE ce.stay_id IN (SELECT DISTINCT stay_id FROM grid)
    AND ce.itemid IN (SELECT itemid FROM vent_itemids)
    AND ce.valuenum IS NOT NULL
),

/* ===============================================================
   3a) CURRENT WINDOW (grid_ts - 24h, grid_ts]
   =============================================================== */
pao2_worst_24h_current AS (
  SELECT
    g.stay_id,
    g.grid_ts,
    MIN(e.pao2) AS pao2_worst_24h_current
  FROM grid g
  LEFT JOIN pao2_events e
    ON e.stay_id = g.stay_id
   AND e.ts > TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR)
   AND e.ts <= g.grid_ts
  GROUP BY g.stay_id, g.grid_ts
),

fio2_worst_24h_current AS (
  SELECT
    g.stay_id,
    g.grid_ts,
    MAX(e.fio2) AS fio2_worst_24h_current
  FROM grid g
  LEFT JOIN fio2_events e
    ON e.stay_id = g.stay_id
   AND e.ts > TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR)
   AND e.ts <= g.grid_ts
  GROUP BY g.stay_id, g.grid_ts
),

mech_vent_24h_current AS (
  SELECT
    g.stay_id,
    g.grid_ts,
    CASE WHEN COUNT(e.ts) > 0 THEN 1 ELSE 0 END AS mech_vent_24h_current
  FROM grid g
  LEFT JOIN vent_events e
    ON e.stay_id = g.stay_id
   AND e.ts > TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR)
   AND e.ts <= g.grid_ts
  GROUP BY g.stay_id, g.grid_ts
),

/* ===============================================================
   3b) FORWARD WINDOW (grid_ts + step - 24h, grid_ts + step]
   =============================================================== */
pao2_worst_24h_forward AS (
  SELECT
    g.stay_id,
    g.grid_ts,
    MIN(e.pao2) AS pao2_worst_24h_forward
  FROM grid g
  CROSS JOIN cfg
  LEFT JOIN pao2_events e
    ON e.stay_id = g.stay_id
   AND e.ts > TIMESTAMP_SUB(
                TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR),
                INTERVAL 24 HOUR)
   AND e.ts <= TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR)
  GROUP BY g.stay_id, g.grid_ts
),

fio2_worst_24h_forward AS (
  SELECT
    g.stay_id,
    g.grid_ts,
    MAX(e.fio2) AS fio2_worst_24h_forward
  FROM grid g
  CROSS JOIN cfg
  LEFT JOIN fio2_events e
    ON e.stay_id = g.stay_id
   AND e.ts > TIMESTAMP_SUB(
                TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR),
                INTERVAL 24 HOUR)
   AND e.ts <= TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR)
  GROUP BY g.stay_id, g.grid_ts
),

mech_vent_24h_forward AS (
  SELECT
    g.stay_id,
    g.grid_ts,
    CASE WHEN COUNT(e.ts) > 0 THEN 1 ELSE 0 END AS mech_vent_24h_forward
  FROM grid g
  CROSS JOIN cfg
  LEFT JOIN vent_events e
    ON e.stay_id = g.stay_id
   AND e.ts > TIMESTAMP_SUB(
                TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR),
                INTERVAL 24 HOUR)
   AND e.ts <= TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR)
  GROUP BY g.stay_id, g.grid_ts
),

/* ===============================================================
   4) SOFA RESP CALCULATION - CURRENT
   =============================================================== */
sofa_resp_current AS (
  SELECT
    g.subject_id,
    g.hadm_id,
    g.stay_id,
    g.grid_ts,

    pc.pao2_worst_24h_current,
    fc.fio2_worst_24h_current,
    mc.mech_vent_24h_current,

    SAFE_DIVIDE(pc.pao2_worst_24h_current, fc.fio2_worst_24h_current / 100.0) AS pafi_worst_24h_current,

    CASE
      WHEN pc.pao2_worst_24h_current IS NULL OR fc.fio2_worst_24h_current IS NULL THEN NULL
      WHEN SAFE_DIVIDE(pc.pao2_worst_24h_current, fc.fio2_worst_24h_current / 100.0) >= 400 THEN 0
      WHEN SAFE_DIVIDE(pc.pao2_worst_24h_current, fc.fio2_worst_24h_current / 100.0) >= 300 THEN 1
      WHEN SAFE_DIVIDE(pc.pao2_worst_24h_current, fc.fio2_worst_24h_current / 100.0) >= 200 THEN 2
      WHEN SAFE_DIVIDE(pc.pao2_worst_24h_current, fc.fio2_worst_24h_current / 100.0) < 200
        AND IFNULL(mc.mech_vent_24h_current, 0) = 0 THEN 2
      WHEN SAFE_DIVIDE(pc.pao2_worst_24h_current, fc.fio2_worst_24h_current / 100.0) >= 100
        AND IFNULL(mc.mech_vent_24h_current, 0) = 1 THEN 3
      WHEN SAFE_DIVIDE(pc.pao2_worst_24h_current, fc.fio2_worst_24h_current / 100.0) < 100
        AND IFNULL(mc.mech_vent_24h_current, 0) = 1 THEN 4
      ELSE NULL
    END AS sofa_resp_24h_current

  FROM grid g
  LEFT JOIN pao2_worst_24h_current pc USING (stay_id, grid_ts)
  LEFT JOIN fio2_worst_24h_current fc USING (stay_id, grid_ts)
  LEFT JOIN mech_vent_24h_current mc USING (stay_id, grid_ts)
),

/* ===============================================================
   5) SOFA RESP CALCULATION - FORWARD
   =============================================================== */
sofa_resp_forward AS (
  SELECT
    g.stay_id,
    g.grid_ts,

    pf.pao2_worst_24h_forward,
    ff.fio2_worst_24h_forward,
    mf.mech_vent_24h_forward,

    SAFE_DIVIDE(pf.pao2_worst_24h_forward, ff.fio2_worst_24h_forward / 100.0) AS pafi_worst_24h_forward,

    CASE
      WHEN pf.pao2_worst_24h_forward IS NULL OR ff.fio2_worst_24h_forward IS NULL THEN NULL
      WHEN SAFE_DIVIDE(pf.pao2_worst_24h_forward, ff.fio2_worst_24h_forward / 100.0) >= 400 THEN 0
      WHEN SAFE_DIVIDE(pf.pao2_worst_24h_forward, ff.fio2_worst_24h_forward / 100.0) >= 300 THEN 1
      WHEN SAFE_DIVIDE(pf.pao2_worst_24h_forward, ff.fio2_worst_24h_forward / 100.0) >= 200 THEN 2
      WHEN SAFE_DIVIDE(pf.pao2_worst_24h_forward, ff.fio2_worst_24h_forward / 100.0) < 200
        AND IFNULL(mf.mech_vent_24h_forward, 0) = 0 THEN 2
      WHEN SAFE_DIVIDE(pf.pao2_worst_24h_forward, ff.fio2_worst_24h_forward / 100.0) >= 100
        AND IFNULL(mf.mech_vent_24h_forward, 0) = 1 THEN 3
      WHEN SAFE_DIVIDE(pf.pao2_worst_24h_forward, ff.fio2_worst_24h_forward / 100.0) < 100
        AND IFNULL(mf.mech_vent_24h_forward, 0) = 1 THEN 4
      ELSE NULL
    END AS sofa_resp_24h_forward

  FROM grid g
  LEFT JOIN pao2_worst_24h_forward pf USING (stay_id, grid_ts)
  LEFT JOIN fio2_worst_24h_forward ff USING (stay_id, grid_ts)
  LEFT JOIN mech_vent_24h_forward mf USING (stay_id, grid_ts)
)

/* ===============================================================
   FINAL
   =============================================================== */
SELECT
  g.subject_id,
  g.hadm_id,
  g.stay_id,
  g.grid_ts,
  c.sofa_resp_24h_current,
  f.sofa_resp_24h_forward,
  c.pafi_worst_24h_current,
  f.pafi_worst_24h_forward,
  c.pao2_worst_24h_current,
  f.pao2_worst_24h_forward,
  c.fio2_worst_24h_current,
  f.fio2_worst_24h_forward,
  c.mech_vent_24h_current,
  f.mech_vent_24h_forward
FROM grid g
LEFT JOIN sofa_resp_current c USING (subject_id, hadm_id, stay_id, grid_ts)
LEFT JOIN sofa_resp_forward f USING (stay_id, grid_ts)
ORDER BY subject_id, stay_id, grid_ts;
