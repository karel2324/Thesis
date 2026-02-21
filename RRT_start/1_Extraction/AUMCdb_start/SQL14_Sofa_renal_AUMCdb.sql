CREATE OR REPLACE TABLE `windy-forge-475207-e3.derived.sofa_renal_24h` AS
WITH
/* ===============================================================
   0) CONFIG
   =============================================================== */
cfg AS (
  SELECT *
  FROM `windy-forge-475207-e3.derived.cfg_params`
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
  FROM `windy-forge-475207-e3.derived.cohort_grid`
),

/* ===============================================================
   2) CREATININE EVENTS → SOFA RENAL (CREAT)
   =============================================================== */
sofa_renal_creat_events AS (
  SELECT
    m.person_id,
    m.visit_occurrence_id,
    m.ts,

    CASE
      WHEN m.value_as_number < 110 THEN 0
      WHEN m.value_as_number < 171 THEN 1
      WHEN m.value_as_number < 300 THEN 2
      WHEN m.value_as_number < 440 THEN 3
      WHEN m.value_as_number >= 440 THEN 4
    END AS sofa_renal_creat
  FROM `windy-forge-475207-e3.derived.cohort_measurements_window_stat_valid` m
  WHERE m.var = 'creat'
    AND m.value_as_number IS NOT NULL
),

/* ===============================================================
   3a) WORST CREATININE SOFA - CURRENT WINDOW (grid_ts - 24h, grid_ts]
   =============================================================== */
sofa_renal_creat_24h_current AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,
    MAX(e.sofa_renal_creat) AS sofa_renal_creat_24h_current
  FROM grid g
  LEFT JOIN sofa_renal_creat_events e
    ON e.person_id = g.person_id
   AND e.visit_occurrence_id = g.visit_occurrence_id
   AND e.ts > TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR)
   AND e.ts <= g.grid_ts
  GROUP BY g.person_id, g.visit_occurrence_id, g.grid_ts
),

/* ===============================================================
   3b) WORST CREATININE SOFA - FORWARD WINDOW (grid_ts + step - 24h, grid_ts + step]
   =============================================================== */
sofa_renal_creat_24h_forward AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,
    MAX(e.sofa_renal_creat) AS sofa_renal_creat_24h_forward
  FROM grid g
  CROSS JOIN cfg
  LEFT JOIN sofa_renal_creat_events e
    ON e.person_id = g.person_id
   AND e.visit_occurrence_id = g.visit_occurrence_id
   AND e.ts > TIMESTAMP_SUB(
                TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR),
                INTERVAL 24 HOUR)
   AND e.ts <= TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR)
  GROUP BY g.person_id, g.visit_occurrence_id, g.grid_ts
),

/* ===============================================================
   4) UO RATES (HERGEBRUIKT)
   =============================================================== */
uo_rates AS (
  SELECT
    person_id,
    visit_occurrence_id,
    t_start,
    t_end,
    rate_ml_per_h
  FROM `windy-forge-475207-e3.derived.uo_rates`
),

/* ===============================================================
   5a) UO INTEGRATIE - CURRENT WINDOW: 24h VÓÓR grid_ts
   =============================================================== */
uo_24h_current AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,

    SUM(
      GREATEST(
        0.0,
        TIMESTAMP_DIFF(
          LEAST(g.grid_ts, r.t_end),
          GREATEST(TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR), r.t_start),
          MINUTE
        ) / 60.0
      ) * r.rate_ml_per_h
    ) AS uo_24h_ml_current

  FROM grid g
  LEFT JOIN uo_rates r
    ON r.person_id = g.person_id
   AND r.visit_occurrence_id = g.visit_occurrence_id
   AND r.t_end > TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR)
   AND r.t_start < g.grid_ts

  GROUP BY g.person_id, g.visit_occurrence_id, g.grid_ts
),

/* ===============================================================
   5b) UO INTEGRATIE - FORWARD WINDOW: 24h VÓÓR (grid_ts + step)
   =============================================================== */
uo_24h_forward AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,

    SUM(
      GREATEST(
        0.0,
        TIMESTAMP_DIFF(
          LEAST(
            TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR),
            r.t_end
          ),
          GREATEST(
            TIMESTAMP_SUB(
              TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR),
              INTERVAL 24 HOUR
            ),
            r.t_start
          ),
          MINUTE
        ) / 60.0
      ) * r.rate_ml_per_h
    ) AS uo_24h_ml_forward

  FROM grid g
  CROSS JOIN cfg
  LEFT JOIN uo_rates r
    ON r.person_id = g.person_id
   AND r.visit_occurrence_id = g.visit_occurrence_id
   AND r.t_end > TIMESTAMP_SUB(
                   TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR),
                   INTERVAL 24 HOUR)
   AND r.t_start < TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR)

  GROUP BY g.person_id, g.visit_occurrence_id, g.grid_ts
),

/* ===============================================================
   6a) UO → SOFA RENAL - CURRENT
   =============================================================== */
sofa_renal_uo_current AS (
  SELECT
    person_id,
    visit_occurrence_id,
    grid_ts,
    CASE
      WHEN uo_24h_ml_current < 200 THEN 4
      WHEN uo_24h_ml_current < 500 THEN 3
      ELSE 0
    END AS sofa_renal_uo_current
  FROM uo_24h_current
),

/* ===============================================================
   6b) UO → SOFA RENAL - FORWARD
   =============================================================== */
sofa_renal_uo_forward AS (
  SELECT
    person_id,
    visit_occurrence_id,
    grid_ts,
    CASE
      WHEN uo_24h_ml_forward < 200 THEN 4
      WHEN uo_24h_ml_forward < 500 THEN 3
      ELSE 0
    END AS sofa_renal_uo_forward
  FROM uo_24h_forward
)

/* ===============================================================
   FINAL: COMBINE CREAT + UO FOR BOTH WINDOWS
   =============================================================== */
SELECT
  g.person_id,
  g.visit_occurrence_id,
  g.grid_ts,

  GREATEST(
    IFNULL(cc.sofa_renal_creat_24h_current, 0),
    IFNULL(uc.sofa_renal_uo_current, 0)
  ) AS sofa_renal_24h_current,

  GREATEST(
    IFNULL(cf.sofa_renal_creat_24h_forward, 0),
    IFNULL(uf.sofa_renal_uo_forward, 0)
  ) AS sofa_renal_24h_forward

FROM grid g
LEFT JOIN sofa_renal_creat_24h_current cc USING (person_id, visit_occurrence_id, grid_ts)
LEFT JOIN sofa_renal_creat_24h_forward cf USING (person_id, visit_occurrence_id, grid_ts)
LEFT JOIN sofa_renal_uo_current uc USING (person_id, visit_occurrence_id, grid_ts)
LEFT JOIN sofa_renal_uo_forward uf USING (person_id, visit_occurrence_id, grid_ts)
ORDER BY person_id, visit_occurrence_id, grid_ts;
