/* ===============================================================
   SQL4_static.sql - MIMIC-IV Static Variables

   Analogue to AUMCdb SQL4_Static_variables.sql

   Collects static (time-invariant) variables per stay:
   - Demographics (age, gender)
   - Weight
   - Cohort info (t0, terminal_ts, etc.)
   =============================================================== */

CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.cohort_static_variables` AS
WITH
cohort AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.cohort_index`
),

weight_effective AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.weight_effective_stay`
),

weight_first AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.weight_first_stay`
)

SELECT
  -- Identifiers
  c.subject_id,
  c.hadm_id,
  c.stay_id,

  -- Timestamps
  c.t0,
  c.terminal_ts,
  c.death_dt,
  c.crrt_start_dt,
  c.icu_intime,
  c.icu_outtime,

  -- Terminal info
  c.terminal_event,
  c.inclusion_applied,

  -- Demographics
  c.gender,
  c.age_years,

  -- Weight (cohort level = first measurement near t0)
  wf.weight_kg_first AS weight_kg_cohort,
  we.weight_used_kg AS weight_used_kg_cohort,

  -- Weight (stay level)
  wf.weight_kg_first,
  we.weight_used_kg AS weight_used_kg_stay,
  we.weight_source AS weight_source_stay

FROM cohort c
LEFT JOIN weight_effective we ON we.stay_id = c.stay_id
LEFT JOIN weight_first wf ON wf.stay_id = c.stay_id
;
