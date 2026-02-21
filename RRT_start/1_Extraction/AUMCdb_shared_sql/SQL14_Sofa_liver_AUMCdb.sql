CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.sofa_liver_24h` AS
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
   1) GRID (single source of truth)
   =============================================================== */
grid AS (
  SELECT
    person_id,
    visit_occurrence_id,
    grid_ts
  FROM `windy-forge-475207-e3.${DATASET}.cohort_grid`
),

/* ===============================================================
   2) BILIRUBIN EVENTS → SOFA LIVER SCORE
   =============================================================== */
sofa_liver_events AS (
  SELECT
    m.person_id,
    m.visit_occurrence_id,
    m.ts,

    CASE
      WHEN m.value_as_number <  20  THEN 0
      WHEN m.value_as_number <  33  THEN 1
      WHEN m.value_as_number < 102  THEN 2
      WHEN m.value_as_number < 204  THEN 3
      WHEN m.value_as_number >= 204 THEN 4
      ELSE NULL
    END AS sofa_liver
  FROM `windy-forge-475207-e3.${DATASET}.cohort_measurements_window_stat_valid` m
  WHERE m.var = 'bilirubin'
    AND m.value_as_number IS NOT NULL
),

/* ===============================================================
   3a) WORST SOFA LIVER - CURRENT WINDOW (grid_ts - 24h, grid_ts]
   =============================================================== */
sofa_liver_24h_current AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,
    MAX(e.sofa_liver) AS sofa_liver_24h_current
  FROM grid g
  LEFT JOIN sofa_liver_events e
    ON e.person_id = g.person_id
   AND e.visit_occurrence_id = g.visit_occurrence_id
   AND e.ts > TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR)
   AND e.ts <= g.grid_ts
  GROUP BY g.person_id, g.visit_occurrence_id, g.grid_ts
),

/* ===============================================================
   3b) WORST SOFA LIVER - FORWARD WINDOW (grid_ts + step - 24h, grid_ts + step]
   =============================================================== */
sofa_liver_24h_forward AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,
    MAX(e.sofa_liver) AS sofa_liver_24h_forward
  FROM grid g
  CROSS JOIN cfg
  LEFT JOIN sofa_liver_events e
    ON e.person_id = g.person_id
   AND e.visit_occurrence_id = g.visit_occurrence_id
   AND e.ts > TIMESTAMP_SUB(
                TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR),
                INTERVAL 24 HOUR)
   AND e.ts <= TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR)
  GROUP BY g.person_id, g.visit_occurrence_id, g.grid_ts
)

/* ===============================================================
   FINAL
   =============================================================== */
SELECT
  g.person_id,
  g.visit_occurrence_id,
  g.grid_ts,
  c.sofa_liver_24h_current,
  f.sofa_liver_24h_forward
FROM grid g
LEFT JOIN sofa_liver_24h_current c USING (person_id, visit_occurrence_id, grid_ts)
LEFT JOIN sofa_liver_24h_forward f USING (person_id, visit_occurrence_id, grid_ts)
ORDER BY person_id, visit_occurrence_id, grid_ts;
