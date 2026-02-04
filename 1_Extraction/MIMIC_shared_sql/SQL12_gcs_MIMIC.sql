/* ===============================================================
   SQL12_gcs.sql - MIMIC-IV GCS at Grid

   Analogue to AUMCdb SQL12_GCS.sql

   Key differences:
   - MIMIC GCS values are numeric in chartevents (not coded concepts)
   - Uses gcs_map from cfg_params
   =============================================================== */

CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.cohort_gcs_grid` AS
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
   1) GCS MAP (from utils)
   =============================================================== */
gcs_map AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.gcs_map`
),

/* ===============================================================
   2) GCS MEASUREMENTS (chartevents -> component + numeric score)
   =============================================================== */
gcs_meas AS (
  SELECT
    ce.stay_id,
    TIMESTAMP(ce.charttime) AS ts,
    gm.component,
    CAST(ce.valuenum AS INT64) AS score
  FROM `physionet-data.mimiciv_3_1_icu.chartevents` ce
  JOIN gcs_map gm ON gm.itemid = ce.itemid
  JOIN `windy-forge-475207-e3.${DATASET}.cohort_grid` g ON g.stay_id = ce.stay_id
  WHERE ce.valuenum IS NOT NULL
    AND ce.valuenum > 0
),

/* ===============================================================
   3) DEDUPLICATION (1 score per component per timestamp)
   =============================================================== */
gcs_dedup AS (
  SELECT
    stay_id,
    ts,
    component,
    score
  FROM (
    SELECT
      stay_id,
      ts,
      component,
      score,
      ROW_NUMBER() OVER (
        PARTITION BY stay_id, ts, component
        ORDER BY ts ASC
      ) AS rn
    FROM gcs_meas
  )
  WHERE rn = 1
),

/* ===============================================================
   4) COMPLETE GCS EVENTS (eye + motor required)
   =============================================================== */
gcs_complete_events AS (
  SELECT
    stay_id,
    ts,

    MAX(IF(component = 'eye',    score, NULL)) AS eye,
    MAX(IF(component = 'motor',  score, NULL)) AS motor,
    MAX(IF(component = 'verbal', score, NULL)) AS verbal
  FROM gcs_dedup
  GROUP BY stay_id, ts
  HAVING
    eye   IS NOT NULL
    AND motor IS NOT NULL
),

/* ===============================================================
   5) LAST COMPLETE GCS WITHIN LOOKBACK PER GRID
   =============================================================== */
grid AS (
  SELECT
    subject_id,
    hadm_id,
    stay_id,
    grid_ts
  FROM `windy-forge-475207-e3.${DATASET}.cohort_grid`
),

gcs_last_complete AS (
  SELECT
    g.subject_id,
    g.hadm_id,
    g.stay_id,
    g.grid_ts,

    c.ts AS gcs_ts,

    c.eye    AS gcs_eye_last,
    c.motor  AS gcs_motor_last,
    c.verbal AS gcs_verbal_last,

    -- SOFA-compliant total (verbal may be missing)
    (
      GREATEST(c.eye, 1)
      + GREATEST(c.motor, 1)
      + GREATEST(IFNULL(c.verbal, 1), 1)
    ) AS gcs_total_last,

    TIMESTAMP_DIFF(g.grid_ts, c.ts, HOUR) AS gcs_total_last_hours

  FROM grid g
  CROSS JOIN cfg
  JOIN gcs_complete_events c
    ON c.stay_id = g.stay_id
   AND c.ts <= g.grid_ts
   AND c.ts > TIMESTAMP_SUB(g.grid_ts, INTERVAL cfg.gcs_lb_hours HOUR)
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY g.stay_id, g.grid_ts
    ORDER BY c.ts DESC
  ) = 1
)

/* ===============================================================
   FINAL
   =============================================================== */
SELECT
  g.subject_id,
  g.hadm_id,
  g.stay_id,
  g.grid_ts,

  c.gcs_eye_last,
  c.gcs_motor_last,
  c.gcs_verbal_last,
  c.gcs_total_last,
  c.gcs_total_last_hours

FROM grid g
LEFT JOIN gcs_last_complete c
  USING (subject_id, hadm_id, stay_id, grid_ts)
ORDER BY subject_id, stay_id, grid_ts;
