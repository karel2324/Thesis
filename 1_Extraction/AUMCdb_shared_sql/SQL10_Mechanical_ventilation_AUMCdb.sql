CREATE OR REPLACE TABLE
  `windy-forge-475207-e3.${DATASET}.cohort_mechanical_ventilation_grid` AS
WITH
/* ===============================================================
   0) CONFIG (SQL0)
   =============================================================== */
cfg AS (
  SELECT *
  FROM `windy-forge-475207-e3.${DATASET}.cfg_params`
  LIMIT 1
),

/* ===============================================================
   1) MECHANICAL VENTILATION EVENTS
      (using vent_ids.aumcdb_ids from unified cfg_params)
   =============================================================== */
mv_raw AS (
  SELECT
    m.person_id,
    m.visit_occurrence_id,
    COALESCE(
      m.measurement_datetime,
      TIMESTAMP(m.measurement_date)
    ) AS ts
  FROM `amsterdamumcdb.version1_5_0.measurement` m
  CROSS JOIN cfg
  WHERE m.measurement_concept_id IN UNNEST(cfg.vent_ids.aumcdb_ids)
    AND m.provider_id BETWEEN 0 AND 99
),

/* ===============================================================
   2) OVERLAP MET GRID (afgelopen timestep)
   =============================================================== */
mv_overlap AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts
  FROM `windy-forge-475207-e3.${DATASET}.cohort_grid` g
  CROSS JOIN cfg
  JOIN mv_raw m
    ON m.person_id = g.person_id
   AND m.visit_occurrence_id = g.visit_occurrence_id
   -- event binnen (grid_ts - Δt, grid_ts]
   AND m.ts >  TIMESTAMP_SUB(g.grid_ts, INTERVAL cfg.grid_step_hours HOUR)
   AND m.ts <= g.grid_ts
),

/* ===============================================================
   3) AGGREGATIE PER GRID
   =============================================================== */
mv_at_grid AS (
  SELECT
    person_id,
    visit_occurrence_id,
    grid_ts,
    1 AS mechanical_ventilation_in_use
  FROM mv_overlap
  GROUP BY person_id, visit_occurrence_id, grid_ts
)

/* ===============================================================
   FINAL
   =============================================================== */
SELECT
  g.person_id,
  g.visit_occurrence_id,
  g.grid_ts,
  COALESCE(m.mechanical_ventilation_in_use, 0)
    AS mechanical_ventilation_in_use
FROM `windy-forge-475207-e3.${DATASET}.cohort_grid` g
LEFT JOIN mv_at_grid m
  USING (person_id, visit_occurrence_id, grid_ts)
ORDER BY person_id, visit_occurrence_id, grid_ts;
