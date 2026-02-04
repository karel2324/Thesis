CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.cohort_grid` AS
WITH
/* ===============================================================
   0) CONFIG - ${COHORT_NAME} (7 dagen observatie)
   =============================================================== */
cfg AS (
  SELECT *
  FROM `windy-forge-475207-e3.${DATASET}.cfg_params`
  LIMIT 1
),

/* ===============================================================
   1) COHORT BASIS
   =============================================================== */
cohort AS (
  SELECT
    person_id,
    visit_occurrence_id,
    t0,
    terminal_ts,
    death_dt,
    admit_dt,
    discharge_dt,
    bloodflow_dt,
    window_end
  FROM `windy-forge-475207-e3.${DATASET}.cohort_index`
  WHERE t0 IS NOT NULL
),

/* ===============================================================
   2) GRID GENERATIE (gebruikt cfg.obs_days = 7 dagen)
   =============================================================== */
base_grid AS (
  SELECT
    c.person_id,
    c.visit_occurrence_id,
    c.t0,
    c.terminal_ts,
    c.death_dt,
    c.admit_dt,
    c.discharge_dt,
    c.bloodflow_dt,
    c.window_end,
    ts AS grid_ts
  FROM cohort c
  CROSS JOIN cfg
  CROSS JOIN UNNEST(
    GENERATE_TIMESTAMP_ARRAY(
      c.t0,
      TIMESTAMP_ADD(c.t0, INTERVAL cfg.obs_days DAY),
      INTERVAL cfg.grid_step_hours HOUR
    )
  ) AS ts
),

/* ===============================================================
   3) TERMINAL LOGICA OP GRID
   =============================================================== */
grid AS (
  SELECT
    b.*,

    -- geldig binnen episode
    (b.terminal_ts IS NOT NULL AND b.grid_ts <= b.terminal_ts) AS included,

    -- laatste geldige grid voor of op terminal
    MAX(
      IF(
        b.terminal_ts IS NOT NULL
        AND b.grid_ts <= b.terminal_ts,
        b.grid_ts,
        NULL
      )
    ) OVER (
      PARTITION BY b.person_id, b.visit_occurrence_id
    ) AS last_valid_ts,

    -- expliciete terminal step
    (
      b.terminal_ts IS NOT NULL
      AND b.grid_ts =
        MAX(
          IF(
            b.terminal_ts IS NOT NULL
            AND b.grid_ts <= b.terminal_ts,
            b.grid_ts,
            NULL
          )
        ) OVER (PARTITION BY b.person_id, b.visit_occurrence_id)
    ) AS is_terminal_step,

    -- type terminal event (only on terminal step)
    CASE
      WHEN b.terminal_ts IS NULL THEN NULL
      -- Only populate on terminal step (grid_ts = last valid grid ts)
      WHEN b.grid_ts < MAX(IF(b.terminal_ts IS NOT NULL AND b.grid_ts <= b.terminal_ts, b.grid_ts, NULL))
           OVER (PARTITION BY b.person_id, b.visit_occurrence_id) THEN NULL
      WHEN b.terminal_ts = b.death_dt            THEN 'death'
      WHEN b.terminal_ts = b.discharge_dt        THEN 'discharge'
      WHEN b.terminal_ts = b.bloodflow_dt        THEN 'rrt_start'
      WHEN b.terminal_ts = b.window_end          THEN 'window_end'
      ELSE 'window_end'
    END AS terminal_event

  FROM base_grid b
)

/* ===============================================================
   FINAL
   =============================================================== */
SELECT
  person_id,
  visit_occurrence_id,
  t0,
  grid_ts,
  terminal_ts,
  window_end,
  -- episode flags
  included,
  is_terminal_step,
  terminal_event,

  -- timing helpers
  TIMESTAMP_DIFF(grid_ts, t0, HOUR) AS hours_since_t0,
  TIMESTAMP_DIFF(terminal_ts, grid_ts, HOUR) AS hours_to_terminal,

  last_valid_ts,
  CASE
    WHEN is_terminal_step
     AND terminal_event = 'rrt_start'
    THEN 1
    ELSE 0
  END AS action_rrt

FROM grid
WHERE included
ORDER BY person_id, visit_occurrence_id, grid_ts;
