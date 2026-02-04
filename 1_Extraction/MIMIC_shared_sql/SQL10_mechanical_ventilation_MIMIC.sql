/* ===============================================================
   SQL10_mechanical_ventilation.sql - MIMIC-IV Mechanical Ventilation

   Analogue to AUMCdb SQL10_Mechanical_ventilation.sql

   Uses MIMIC-IV derived.ventilation table which identifies
   ventilation status with start/end times.

   ventilation_status values:
   - 'InvasiveVent' -> mechanical_ventilation = 1
   - 'Tracheostomy' -> mechanical_ventilation = 1
   - 'NonInvasiveVent' -> can be included as mech vent (BIPAP/CPAP)
   - 'SupplementalOxygen', 'HFNC', 'None' -> not mech vent

   Source: physionet-data.mimiciv_3_1_derived.ventilation
   =============================================================== */

CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.cohort_mechanical_ventilation_grid` AS
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
   2) VENTILATION INTERVALS FROM DERIVED TABLE

   physionet-data.mimiciv_3_1_derived.ventilation has:
   - stay_id, starttime, endtime, ventilation_status

   Map ventilation_status to mechanical ventilation:
   - InvasiveVent: ETT/tracheostomy with mechanical ventilator
   - Tracheostomy: often still on mechanical vent
   - NonInvasiveVent: BIPAP/CPAP (include as mech vent per Python code)
   =============================================================== */
vent_intervals AS (
  SELECT
    v.stay_id,
    TIMESTAMP(v.starttime) AS starttime,
    TIMESTAMP(v.endtime) AS endtime,
    v.ventilation_status,
    CASE
      WHEN v.ventilation_status IN ('InvasiveVent', 'Tracheostomy', 'NonInvasiveVent') THEN 1
      ELSE 0
    END AS is_mechanical_vent
  FROM `physionet-data.mimiciv_3_1_derived.ventilation` v
  WHERE v.stay_id IN (SELECT DISTINCT stay_id FROM grid)
    AND v.starttime IS NOT NULL
    AND v.endtime IS NOT NULL
    AND v.endtime > v.starttime
),

/* ===============================================================
   3) FILTER TO MECHANICAL VENTILATION ONLY
   =============================================================== */
mv_intervals AS (
  SELECT
    stay_id,
    starttime,
    endtime
  FROM vent_intervals
  WHERE is_mechanical_vent = 1
),

/* ===============================================================
   4) OVERLAP WITH GRID (past timestep)
   =============================================================== */
mv_overlap AS (
  SELECT
    g.subject_id,
    g.hadm_id,
    g.stay_id,
    g.grid_ts
  FROM grid g
  CROSS JOIN cfg
  JOIN mv_intervals m
    ON m.stay_id = g.stay_id
   -- overlap with (grid_ts - dt, grid_ts]
   AND m.endtime   > TIMESTAMP_SUB(g.grid_ts, INTERVAL cfg.grid_step_hours HOUR)
   AND m.starttime <= g.grid_ts
),

/* ===============================================================
   5) AGGREGATION PER GRID
   =============================================================== */
mv_at_grid AS (
  SELECT
    subject_id,
    hadm_id,
    stay_id,
    grid_ts,
    1 AS mechanical_ventilation_in_use
  FROM mv_overlap
  GROUP BY subject_id, hadm_id, stay_id, grid_ts
)

/* ===============================================================
   FINAL
   =============================================================== */
SELECT
  g.subject_id,
  g.hadm_id,
  g.stay_id,
  g.grid_ts,
  COALESCE(m.mechanical_ventilation_in_use, 0) AS mechanical_ventilation_in_use
FROM grid g
LEFT JOIN mv_at_grid m
  USING (subject_id, hadm_id, stay_id, grid_ts)
ORDER BY subject_id, stay_id, grid_ts;
