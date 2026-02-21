CREATE OR REPLACE TABLE
  `windy-forge-475207-e3.derived.cohort_vasopressors_grid` AS
WITH
/* ===============================================================
   0) CONFIG (SQL0)
   =============================================================== */
cfg AS (
  SELECT *
  FROM `windy-forge-475207-e3.derived.cfg_params`
  LIMIT 1
),

/* ===============================================================
   1) DRUG EXPOSURES (alleen vasopressoren uit cfg.vas_ids_aumcdb,
      EXCLUSIEF bolus-epinephrine)
   Uses unified cfg_params with vas_ids_aumcdb
   =============================================================== */
drug_raw AS (
  SELECT
    d.person_id,
    d.visit_occurrence_id,

    COALESCE(
      d.drug_exposure_start_datetime,
      TIMESTAMP(d.drug_exposure_start_date)
    ) AS start_ts,

    COALESCE(
      d.drug_exposure_end_datetime,
      TIMESTAMP_ADD(
        COALESCE(
          d.drug_exposure_start_datetime,
          TIMESTAMP(d.drug_exposure_start_date)
        ),
        INTERVAL cfg.grid_step_hours HOUR
      )
    ) AS end_ts
  FROM `amsterdamumcdb.version1_5_0.drug_exposure` d
  CROSS JOIN cfg
  WHERE d.drug_concept_id IN UNNEST(cfg.vas_ids_aumcdb)
    -- sluit bolus adrenaline expliciet uit
    AND d.drug_concept_id != 40038600
),

/* ===============================================================
   2) OVERLAP MET GRID (afgelopen timestep)
   =============================================================== */
vas_overlap AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts
  FROM `windy-forge-475207-e3.derived.cohort_grid` g
  CROSS JOIN cfg
  JOIN drug_raw dr
    ON dr.person_id = g.person_id
   AND dr.visit_occurrence_id = g.visit_occurrence_id
   -- overlap met (grid_ts - Δt, grid_ts]
   AND dr.end_ts   > TIMESTAMP_SUB(g.grid_ts, INTERVAL cfg.grid_step_hours HOUR)
   AND dr.start_ts <= g.grid_ts
),

/* ===============================================================
   3) AGGREGATIE PER GRID
   =============================================================== */
vas_at_grid AS (
  SELECT
    person_id,
    visit_occurrence_id,
    grid_ts,
    1 AS vasopressor_in_use
  FROM vas_overlap
  GROUP BY person_id, visit_occurrence_id, grid_ts
)

/* ===============================================================
   FINAL
   =============================================================== */
SELECT
  g.person_id,
  g.visit_occurrence_id,
  g.grid_ts,
  COALESCE(v.vasopressor_in_use, 0) AS vasopressor_in_use
FROM `windy-forge-475207-e3.derived.cohort_grid` g
LEFT JOIN vas_at_grid v
  USING (person_id, visit_occurrence_id, grid_ts)
ORDER BY person_id, visit_occurrence_id, grid_ts;
