CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.cohort_static_variables` AS
WITH
cohort AS (
  SELECT
    person_id,
    visit_occurrence_id,
    t0,
    terminal_ts,
    death_dt,
    inclusion_applied,
    weight_kg_first  AS weight_kg_cohort,
    weight_used_kg   AS weight_used_kg_cohort
  FROM `windy-forge-475207-e3.${DATASET}.cohort_index`
  WHERE t0 IS NOT NULL
    AND terminal_ts IS NOT NULL
    AND t0 <= terminal_ts
),

dem AS (
  SELECT
    person_id,
    gender,
    birth_date
  FROM `windy-forge-475207-e3.${DATASET}.demographics`
),

vt AS (
  SELECT
    person_id,
    visit_occurrence_id,
    admit_dt,
    discharge_dt
  FROM `windy-forge-475207-e3.${DATASET}.visit_times`
),

w_person AS (
  SELECT
    person_id,
    weight_kg_first,
    weight_used_kg AS weight_used_kg_person,
    weight_source  AS weight_source_person
  FROM `windy-forge-475207-e3.${DATASET}.weight_effective_person`
)

SELECT
  c.person_id,
  c.visit_occurrence_id,

  -- tijdsankers
  c.t0,
  c.terminal_ts,
  c.death_dt,

  -- demografie
  d.gender,
  d.birth_date,
  DATE_DIFF(DATE(c.t0), d.birth_date, YEAR) AS age_years,

  -- gewicht (cohort / visit-level als aanwezig)
  c.weight_kg_cohort,
  c.weight_used_kg_cohort,

  -- gewicht (person-level canonical uit SQL1)
  wp.weight_kg_first,
  wp.weight_used_kg_person,
  wp.weight_source_person,

  -- visit info
  vt.admit_dt,
  vt.discharge_dt,

  -- cohort metadata
  c.inclusion_applied

FROM cohort c
LEFT JOIN dem d
  ON d.person_id = c.person_id
LEFT JOIN w_person wp
  ON wp.person_id = c.person_id
LEFT JOIN vt
  ON vt.person_id = c.person_id
 AND vt.visit_occurrence_id = c.visit_occurrence_id
ORDER BY c.person_id, c.visit_occurrence_id
;
