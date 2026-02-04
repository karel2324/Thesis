/* ===============================================================
   SQL9_vasopressors.sql - MIMIC-IV Vasopressors at Grid

   Analogue to AUMCdb SQL9_Vasopressors.sql

   Uses MIMIC-IV derived.norepinephrine_equivalent_dose table
   which pre-computes NE equivalent dose for all vasopressors

   Source: physionet-data.mimiciv_3_1_derived.norepinephrine_equivalent_dose
   Columns: stay_id, starttime, endtime, norepinephrine_equivalent_dose
   =============================================================== */

CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.cohort_vasopressors_grid` AS
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
   2) VASOPRESSOR EVENTS FROM DERIVED TABLE

   physionet-data.mimiciv_3_1_derived.norepinephrine_equivalent_dose
   already calculates NE equivalent for:
   - Norepinephrine
   - Epinephrine
   - Dopamine
   - Phenylephrine
   - Vasopressin

   The dose is in mcg/kg/min
   =============================================================== */
vaso_events AS (
  SELECT
    ne.stay_id,
    TIMESTAMP(ne.starttime) AS starttime,
    TIMESTAMP(ne.endtime) AS endtime,
    ne.norepinephrine_equivalent_dose AS ne_equiv_mcgkgmin
  FROM `physionet-data.mimiciv_3_1_derived.norepinephrine_equivalent_dose` ne
  WHERE ne.stay_id IN (SELECT DISTINCT stay_id FROM grid)
    AND ne.norepinephrine_equivalent_dose IS NOT NULL
    AND ne.norepinephrine_equivalent_dose > 0
    AND ne.endtime > ne.starttime
),

/* ===============================================================
   3) OVERLAP WITH GRID (past timestep)
   =============================================================== */
vas_overlap AS (
  SELECT
    g.subject_id,
    g.hadm_id,
    g.stay_id,
    g.grid_ts,
    MAX(v.ne_equiv_mcgkgmin) AS max_ne_equiv
  FROM grid g
  CROSS JOIN cfg
  JOIN vaso_events v
    ON v.stay_id = g.stay_id
   -- overlap with (grid_ts - dt, grid_ts]
   AND v.endtime   > TIMESTAMP_SUB(g.grid_ts, INTERVAL cfg.grid_step_hours HOUR)
   AND v.starttime <= g.grid_ts
  GROUP BY g.subject_id, g.hadm_id, g.stay_id, g.grid_ts
)

/* ===============================================================
   FINAL
   =============================================================== */
SELECT
  g.subject_id,
  g.hadm_id,
  g.stay_id,
  g.grid_ts,
  CASE WHEN v.max_ne_equiv > 0 THEN 1 ELSE 0 END AS vasopressor_in_use,
  COALESCE(v.max_ne_equiv, 0) AS ne_equiv_mcgkgmin
FROM grid g
LEFT JOIN vas_overlap v
  USING (subject_id, hadm_id, stay_id, grid_ts)
ORDER BY subject_id, stay_id, grid_ts;
