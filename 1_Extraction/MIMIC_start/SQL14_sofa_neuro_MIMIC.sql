/* ===============================================================
   SQL14_sofa_neuro.sql - MIMIC-IV SOFA Neurological Component

   Analogue to AUMCdb SQL14_Sofa_neuro.sql

   Uses GCS total score
   SOFA thresholds for GCS:
   - 0: 15
   - 1: 13-14
   - 2: 10-12
   - 3: 6-9
   - 4: <6
   =============================================================== */

CREATE OR REPLACE TABLE `windy-forge-475207-e3.derived_mimic.sofa_neuro_24h` AS
WITH
/* ===============================================================
   0) CONFIG
   =============================================================== */
cfg AS (
  SELECT *
  FROM `windy-forge-475207-e3.derived_mimic.cfg_params`
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
  FROM `windy-forge-475207-e3.derived_mimic.cohort_grid`
),

/* ===============================================================
   2) GCS MAP (from utils)
   =============================================================== */
gcs_map AS (
  SELECT * FROM `windy-forge-475207-e3.derived_mimic.gcs_map`
),

/* ===============================================================
   3) GCS MEASUREMENTS (chartevents -> component + numeric score)
   =============================================================== */
gcs_meas AS (
  SELECT
    ce.stay_id,
    TIMESTAMP(ce.charttime) AS ts,
    gm.component,
    CAST(ce.valuenum AS INT64) AS score
  FROM `physionet-data.mimiciv_3_1_icu.chartevents` ce
  JOIN gcs_map gm ON gm.itemid = ce.itemid
  JOIN grid g ON g.stay_id = ce.stay_id
  WHERE ce.valuenum IS NOT NULL
    AND ce.valuenum > 0
),

/* ===============================================================
   4) DEDUPLICATION (1 component per ts)
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
        ORDER BY score DESC
      ) AS rn
    FROM gcs_meas
  )
  WHERE rn = 1
),

/* ===============================================================
   5) COMPLETE GCS EVENTS + TOTAL SCORE
   =============================================================== */
gcs_complete_events AS (
  SELECT
    stay_id,
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
  GROUP BY stay_id, ts
  HAVING
    eye   IS NOT NULL
    AND motor IS NOT NULL
),

/* ===============================================================
   6a) WORST GCS - CURRENT WINDOW (grid_ts - 24h, grid_ts]
   =============================================================== */
gcs_current AS (
  SELECT
    g.stay_id,
    g.grid_ts,
    MIN(e.gcs_total) AS gcs_worst_24h_current
  FROM grid g
  LEFT JOIN gcs_complete_events e
    ON e.stay_id = g.stay_id
   AND e.ts > TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR)
   AND e.ts <= g.grid_ts
  GROUP BY g.stay_id, g.grid_ts
),

/* ===============================================================
   6b) WORST GCS - FORWARD WINDOW (grid_ts + step - 24h, grid_ts + step]
   =============================================================== */
gcs_forward AS (
  SELECT
    g.stay_id,
    g.grid_ts,
    MIN(e.gcs_total) AS gcs_worst_24h_forward
  FROM grid g
  CROSS JOIN cfg
  LEFT JOIN gcs_complete_events e
    ON e.stay_id = g.stay_id
   AND e.ts > TIMESTAMP_SUB(
                TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR),
                INTERVAL 24 HOUR)
   AND e.ts <= TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR)
  GROUP BY g.stay_id, g.grid_ts
),

/* ===============================================================
   7) SOFA NEURO SCORE - CURRENT
   =============================================================== */
sofa_neuro_current AS (
  SELECT
    stay_id,
    grid_ts,
    gcs_worst_24h_current,

    CASE
      WHEN gcs_worst_24h_current = 15 THEN 0
      WHEN gcs_worst_24h_current >= 13 THEN 1
      WHEN gcs_worst_24h_current >= 10 THEN 2
      WHEN gcs_worst_24h_current >=  6 THEN 3
      WHEN gcs_worst_24h_current <   6 THEN 4
      ELSE NULL
    END AS sofa_neuro_24h_current
  FROM gcs_current
),

/* ===============================================================
   8) SOFA NEURO SCORE - FORWARD
   =============================================================== */
sofa_neuro_forward AS (
  SELECT
    stay_id,
    grid_ts,
    gcs_worst_24h_forward,

    CASE
      WHEN gcs_worst_24h_forward IS NULL THEN NULL
      WHEN gcs_worst_24h_forward = 15 THEN 0
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
  g.subject_id,
  g.hadm_id,
  g.stay_id,
  g.grid_ts,
  c.sofa_neuro_24h_current,
  f.sofa_neuro_24h_forward,
  c.gcs_worst_24h_current,
  f.gcs_worst_24h_forward
FROM grid g
LEFT JOIN sofa_neuro_current c USING (stay_id, grid_ts)
LEFT JOIN sofa_neuro_forward f USING (stay_id, grid_ts)
ORDER BY subject_id, stay_id, grid_ts;
