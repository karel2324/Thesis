CREATE OR REPLACE TABLE
  `windy-forge-475207-e3.${DATASET}.cohort_gcs_grid` AS
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
   1) GCS METINGEN (measurement → component + score)
   =============================================================== */
gcs_meas AS (
  SELECT
    m.person_id,
    m.visit_occurrence_id,
    COALESCE(
      m.measurement_datetime,
      TIMESTAMP(m.measurement_date)
    ) AS ts,
    gm.component,
    gm.score
  FROM `amsterdamumcdb.version1_5_0.measurement` m
  JOIN `windy-forge-475207-e3.${DATASET}.gcs_map` gm
    ON gm.value_as_concept_id = m.value_as_concept_id
  WHERE m.value_as_concept_id IS NOT NULL
    AND m.provider_id BETWEEN 0 AND 99
),

/* ===============================================================
   2) DEDUPLICATIE (1 score per component per timestamp)
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
        ORDER BY ts ASC
      ) AS rn
    FROM gcs_meas
  )
  WHERE rn = 1
),

/* ===============================================================
   3) COMPLETE GCS EVENTS (eye + motor vereist)
   =============================================================== */
gcs_complete_events AS (
  SELECT
    person_id,
    visit_occurrence_id,
    ts,

    MAX(IF(component = 'eye',    score, NULL)) AS eye,
    MAX(IF(component = 'motor',  score, NULL)) AS motor,
    MAX(IF(component = 'verbal', score, NULL)) AS verbal
  FROM gcs_dedup
  GROUP BY person_id, visit_occurrence_id, ts
  HAVING
    eye   IS NOT NULL
    AND motor IS NOT NULL
),

/* ===============================================================
   4) LAATSTE COMPLETE GCS BINNEN LOOKBACK PER GRID
   =============================================================== */
gcs_last_complete AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,

    c.ts AS gcs_ts,

    c.eye    AS gcs_eye_last,
    c.motor  AS gcs_motor_last,
    c.verbal AS gcs_verbal_last,

    -- SOFA-conforme total (verbal mag ontbreken)
    (
      GREATEST(c.eye, 1)
      + GREATEST(c.motor, 1)
      + GREATEST(IFNULL(c.verbal, 1), 1)
    ) AS gcs_total_last,

    TIMESTAMP_DIFF(g.grid_ts, c.ts, HOUR)
      AS gcs_total_last_hours,

  FROM `windy-forge-475207-e3.${DATASET}.cohort_grid` g
  CROSS JOIN cfg
  JOIN gcs_complete_events c
    ON c.person_id = g.person_id
   AND c.visit_occurrence_id = g.visit_occurrence_id
   AND c.ts <= g.grid_ts
   AND c.ts > TIMESTAMP_SUB(
         g.grid_ts,
         INTERVAL cfg.gcs_lb_hours HOUR
       )
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY g.person_id, g.visit_occurrence_id, g.grid_ts
    ORDER BY c.ts DESC
  ) = 1
)

/* ===============================================================
   FINAL
   =============================================================== */
SELECT
  g.person_id,
  g.visit_occurrence_id,
  g.grid_ts,

  c.gcs_eye_last,
  c.gcs_motor_last,
  c.gcs_verbal_last,
  c.gcs_total_last,

  c.gcs_total_last_hours,

FROM `windy-forge-475207-e3.${DATASET}.cohort_grid` g
LEFT JOIN gcs_last_complete c
  USING (person_id, visit_occurrence_id, grid_ts)
ORDER BY person_id, visit_occurrence_id, grid_ts;
