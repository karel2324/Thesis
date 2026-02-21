CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.sofa_resp_24h` AS
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
   2) EVENTS (PaO2, FiO2, Mechanical Ventilation)
   =============================================================== */
pao2_events AS (
  SELECT
    m.person_id,
    m.visit_occurrence_id,
    m.ts,
    m.value_as_number AS pao2
  FROM `windy-forge-475207-e3.${DATASET}.cohort_measurements_window_stat_valid` m
  WHERE m.var = 'pao2'
    AND m.value_as_number IS NOT NULL
),

fio2_events AS (
  SELECT
    m.person_id,
    m.visit_occurrence_id,
    m.ts,
    m.value_as_number AS fio2
  FROM `windy-forge-475207-e3.${DATASET}.cohort_measurements_window_stat_valid` m
  WHERE m.var = 'fio2'
    AND m.value_as_number IS NOT NULL
),

mech_vent_events AS (
  SELECT
    m.person_id,
    m.visit_occurrence_id,
    COALESCE(m.measurement_datetime, TIMESTAMP(m.measurement_date)) AS ts,
    IF(m.value_as_number > 0, 1, 0) AS mech_vent
  FROM `amsterdamumcdb.version1_5_0.measurement` m
  CROSS JOIN cfg
  WHERE m.measurement_concept_id IN UNNEST(cfg.vent_ids.aumcdb_ids)
    AND m.value_as_number IS NOT NULL
    AND m.provider_id BETWEEN 0 AND 99
),

/* ===============================================================
   3a) CURRENT WINDOW (grid_ts - 24h, grid_ts]
   =============================================================== */
pao2_worst_24h_current AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,
    MIN(e.pao2) AS pao2_worst_24h_current
  FROM grid g
  LEFT JOIN pao2_events e
    ON e.person_id = g.person_id
   AND e.visit_occurrence_id = g.visit_occurrence_id
   AND e.ts > TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR)
   AND e.ts <= g.grid_ts
  GROUP BY 1,2,3
),

fio2_worst_24h_current AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,
    MAX(e.fio2) AS fio2_worst_24h_current
  FROM grid g
  LEFT JOIN fio2_events e
    ON e.person_id = g.person_id
   AND e.visit_occurrence_id = g.visit_occurrence_id
   AND e.ts > TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR)
   AND e.ts <= g.grid_ts
  GROUP BY 1,2,3
),

mech_vent_24h_current AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,
    MAX(e.mech_vent) AS mech_vent_24h_current
  FROM grid g
  LEFT JOIN mech_vent_events e
    ON e.person_id = g.person_id
   AND e.visit_occurrence_id = g.visit_occurrence_id
   AND e.ts > TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR)
   AND e.ts <= g.grid_ts
  GROUP BY 1,2,3
),

/* ===============================================================
   3b) FORWARD WINDOW (grid_ts + step - 24h, grid_ts + step]
   =============================================================== */
pao2_worst_24h_forward AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,
    MIN(e.pao2) AS pao2_worst_24h_forward
  FROM grid g
  CROSS JOIN cfg
  LEFT JOIN pao2_events e
    ON e.person_id = g.person_id
   AND e.visit_occurrence_id = g.visit_occurrence_id
   AND e.ts > TIMESTAMP_SUB(
                TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR),
                INTERVAL 24 HOUR)
   AND e.ts <= TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR)
  GROUP BY 1,2,3
),

fio2_worst_24h_forward AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,
    MAX(e.fio2) AS fio2_worst_24h_forward
  FROM grid g
  CROSS JOIN cfg
  LEFT JOIN fio2_events e
    ON e.person_id = g.person_id
   AND e.visit_occurrence_id = g.visit_occurrence_id
   AND e.ts > TIMESTAMP_SUB(
                TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR),
                INTERVAL 24 HOUR)
   AND e.ts <= TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR)
  GROUP BY 1,2,3
),

mech_vent_24h_forward AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,
    MAX(e.mech_vent) AS mech_vent_24h_forward
  FROM grid g
  CROSS JOIN cfg
  LEFT JOIN mech_vent_events e
    ON e.person_id = g.person_id
   AND e.visit_occurrence_id = g.visit_occurrence_id
   AND e.ts > TIMESTAMP_SUB(
                TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR),
                INTERVAL 24 HOUR)
   AND e.ts <= TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR)
  GROUP BY 1,2,3
),

/* ===============================================================
   4) SOFA RESP CALCULATION - CURRENT
   =============================================================== */
sofa_resp_current AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,

    pc.pao2_worst_24h_current,
    fc.fio2_worst_24h_current,
    mc.mech_vent_24h_current,

    SAFE_DIVIDE(pc.pao2_worst_24h_current, fc.fio2_worst_24h_current / 100.0) AS pafi_worst_24h_current,

    CASE
      WHEN pc.pao2_worst_24h_current IS NULL OR fc.fio2_worst_24h_current IS NULL THEN NULL
      WHEN SAFE_DIVIDE(pc.pao2_worst_24h_current, fc.fio2_worst_24h_current / 100.0) >= 400 THEN 0
      WHEN SAFE_DIVIDE(pc.pao2_worst_24h_current, fc.fio2_worst_24h_current / 100.0) >= 300 THEN 1
      WHEN SAFE_DIVIDE(pc.pao2_worst_24h_current, fc.fio2_worst_24h_current / 100.0) >= 200 THEN 2
      WHEN SAFE_DIVIDE(pc.pao2_worst_24h_current, fc.fio2_worst_24h_current / 100.0) < 200
        AND IFNULL(mc.mech_vent_24h_current, 0) = 0 THEN 2
      WHEN SAFE_DIVIDE(pc.pao2_worst_24h_current, fc.fio2_worst_24h_current / 100.0) >= 100
        AND IFNULL(mc.mech_vent_24h_current, 0) = 1 THEN 3
      WHEN SAFE_DIVIDE(pc.pao2_worst_24h_current, fc.fio2_worst_24h_current / 100.0) < 100
        AND IFNULL(mc.mech_vent_24h_current, 0) = 1 THEN 4
      ELSE NULL
    END AS sofa_resp_24h_current

  FROM grid g
  LEFT JOIN pao2_worst_24h_current pc USING (person_id, visit_occurrence_id, grid_ts)
  LEFT JOIN fio2_worst_24h_current fc USING (person_id, visit_occurrence_id, grid_ts)
  LEFT JOIN mech_vent_24h_current mc USING (person_id, visit_occurrence_id, grid_ts)
),

/* ===============================================================
   5) SOFA RESP CALCULATION - FORWARD
   =============================================================== */
sofa_resp_forward AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,

    pf.pao2_worst_24h_forward,
    ff.fio2_worst_24h_forward,
    mf.mech_vent_24h_forward,

    SAFE_DIVIDE(pf.pao2_worst_24h_forward, ff.fio2_worst_24h_forward / 100.0) AS pafi_worst_24h_forward,

    CASE
      WHEN pf.pao2_worst_24h_forward IS NULL OR ff.fio2_worst_24h_forward IS NULL THEN NULL
      WHEN SAFE_DIVIDE(pf.pao2_worst_24h_forward, ff.fio2_worst_24h_forward / 100.0) >= 400 THEN 0
      WHEN SAFE_DIVIDE(pf.pao2_worst_24h_forward, ff.fio2_worst_24h_forward / 100.0) >= 300 THEN 1
      WHEN SAFE_DIVIDE(pf.pao2_worst_24h_forward, ff.fio2_worst_24h_forward / 100.0) >= 200 THEN 2
      WHEN SAFE_DIVIDE(pf.pao2_worst_24h_forward, ff.fio2_worst_24h_forward / 100.0) < 200
        AND IFNULL(mf.mech_vent_24h_forward, 0) = 0 THEN 2
      WHEN SAFE_DIVIDE(pf.pao2_worst_24h_forward, ff.fio2_worst_24h_forward / 100.0) >= 100
        AND IFNULL(mf.mech_vent_24h_forward, 0) = 1 THEN 3
      WHEN SAFE_DIVIDE(pf.pao2_worst_24h_forward, ff.fio2_worst_24h_forward / 100.0) < 100
        AND IFNULL(mf.mech_vent_24h_forward, 0) = 1 THEN 4
      ELSE NULL
    END AS sofa_resp_24h_forward

  FROM grid g
  LEFT JOIN pao2_worst_24h_forward pf USING (person_id, visit_occurrence_id, grid_ts)
  LEFT JOIN fio2_worst_24h_forward ff USING (person_id, visit_occurrence_id, grid_ts)
  LEFT JOIN mech_vent_24h_forward mf USING (person_id, visit_occurrence_id, grid_ts)
)

/* ===============================================================
   FINAL
   =============================================================== */
SELECT
  g.person_id,
  g.visit_occurrence_id,
  g.grid_ts,
  c.sofa_resp_24h_current,
  f.sofa_resp_24h_forward,
  c.pafi_worst_24h_current,
  f.pafi_worst_24h_forward,
  c.pao2_worst_24h_current,
  f.pao2_worst_24h_forward,
  c.fio2_worst_24h_current,
  f.fio2_worst_24h_forward,
  c.mech_vent_24h_current,
  f.mech_vent_24h_forward
FROM grid g
LEFT JOIN sofa_resp_current c USING (person_id, visit_occurrence_id, grid_ts)
LEFT JOIN sofa_resp_forward f USING (person_id, visit_occurrence_id, grid_ts)
ORDER BY person_id, visit_occurrence_id, grid_ts;
