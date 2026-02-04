CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.sofa_neuro_24h` AS
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
   2) GCS METINGEN (measurement → component + score)
   =============================================================== */
gcs_meas AS (
  SELECT
    m.person_id,
    m.visit_occurrence_id,
    COALESCE(m.measurement_datetime, TIMESTAMP(m.measurement_date)) AS ts,
    gm.component,
    gm.score
  FROM `amsterdamumcdb.version1_5_0.measurement` m
  JOIN `windy-forge-475207-e3.${DATASET}.gcs_map` gm
    ON gm.value_as_concept_id = m.value_as_concept_id
  WHERE m.value_as_concept_id IS NOT NULL
    AND (m.provider_id BETWEEN 0 AND 99 OR m.provider_id IS NULL)
),

/* ===============================================================
   3) DEDUPLICATIE (1 component per ts)
   =============================================================== */
gcs_dedup AS (
  SELECT
    person_id,
    visit_occurrence_id,
    ts,
    component,
    score
  FROM (
    SELECT
      person_id,
      visit_occurrence_id,
      ts,
      component,
      score,
      ROW_NUMBER() OVER (
        PARTITION BY person_id, visit_occurrence_id, ts, component
        ORDER BY score DESC
      ) AS rn
    FROM gcs_meas
  )
  WHERE rn = 1
),

/* ===============================================================
   4) COMPLETE GCS EVENTS + TOTAALSCORE
   =============================================================== */
gcs_complete_events AS (
  SELECT
    person_id,
    visit_occurrence_id,
    ts,

    MAX(IF(component = 'eye',    score, NULL)) AS eye,
    MAX(IF(component = 'motor',  score, NULL)) AS motor,
    MAX(IF(component = 'verbal', score, NULL)) AS verbal,

    (
      GREATEST(MAX(IF(component = 'eye',   score, NULL)), 1) +
      GREATEST(MAX(IF(component = 'motor', score, NULL)), 1) +
      GREATEST(IFNULL(MAX(IF(component = 'verbal', score, NULL)), 1), 1)
    ) AS gcs_total
  FROM gcs_dedup
  GROUP BY person_id, visit_occurrence_id, ts
  HAVING
    eye   IS NOT NULL
    AND motor IS NOT NULL
),

/* ===============================================================
   5a) WORST GCS - CURRENT WINDOW (grid_ts - 24h, grid_ts]
   =============================================================== */
gcs_current AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,
    MIN(e.gcs_total) AS gcs_worst_24h_current
  FROM grid g
  LEFT JOIN gcs_complete_events e
    ON e.person_id = g.person_id
   AND e.visit_occurrence_id = g.visit_occurrence_id
   AND e.ts > TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR)
   AND e.ts <= g.grid_ts
  GROUP BY g.person_id, g.visit_occurrence_id, g.grid_ts
),

/* ===============================================================
   5b) WORST GCS - FORWARD WINDOW (grid_ts + step - 24h, grid_ts + step]
   =============================================================== */
gcs_forward AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,
    MIN(e.gcs_total) AS gcs_worst_24h_forward
  FROM grid g
  CROSS JOIN cfg
  LEFT JOIN gcs_complete_events e
    ON e.person_id = g.person_id
   AND e.visit_occurrence_id = g.visit_occurrence_id
   AND e.ts > TIMESTAMP_SUB(
                TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR),
                INTERVAL 24 HOUR)
   AND e.ts <= TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR)
  GROUP BY g.person_id, g.visit_occurrence_id, g.grid_ts
),

/* ===============================================================
   6) SOFA NEURO SCORE - CURRENT
   =============================================================== */
sofa_neuro_current AS (
  SELECT
    person_id,
    visit_occurrence_id,
    grid_ts,
    gcs_worst_24h_current,

    CASE
      WHEN gcs_worst_24h_current >= 13 THEN 1
      WHEN gcs_worst_24h_current >= 10 THEN 2
      WHEN gcs_worst_24h_current >=  6 THEN 3
      WHEN gcs_worst_24h_current <   6 THEN 4
      ELSE NULL
    END AS sofa_neuro_24h_current
  FROM gcs_current
),

/* ===============================================================
   7) SOFA NEURO SCORE - FORWARD
   =============================================================== */
sofa_neuro_forward AS (
  SELECT
    person_id,
    visit_occurrence_id,
    grid_ts,
    gcs_worst_24h_forward,

    CASE
      WHEN gcs_worst_24h_forward IS NULL THEN NULL
      WHEN gcs_worst_24h_forward >= 13 THEN 1
      WHEN gcs_worst_24h_forward >= 10 THEN 2
      WHEN gcs_worst_24h_forward >=  6 THEN 3
      WHEN gcs_worst_24h_forward <   6 THEN 4
      ELSE NULL
    END AS sofa_neuro_24h_forward
  FROM gcs_forward
)

/* ===============================================================
   FINAL
   =============================================================== */
SELECT
  g.person_id,
  g.visit_occurrence_id,
  g.grid_ts,
  c.sofa_neuro_24h_current,
  f.sofa_neuro_24h_forward,
  c.gcs_worst_24h_current,
  f.gcs_worst_24h_forward
FROM grid g
LEFT JOIN sofa_neuro_current c USING (person_id, visit_occurrence_id, grid_ts)
LEFT JOIN sofa_neuro_forward f USING (person_id, visit_occurrence_id, grid_ts)
ORDER BY person_id, visit_occurrence_id, grid_ts;
