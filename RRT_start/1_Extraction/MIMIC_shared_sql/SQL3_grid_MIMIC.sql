/* ===============================================================
   SQL3_grid.sql - MIMIC-IV Time Grid

   Analogue to AUMCdb SQL3_grid.sql

   Creates a time grid from t0 to terminal_ts with configurable step size.
   Marks terminal steps and RRT actions.
   =============================================================== */

CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.cohort_grid` AS
WITH
cfg AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.cfg_params` LIMIT 1
),

cohort AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.cohort_index`
),

/* ===============================================================
   1) Generate grid timestamps (t0 to terminal_ts, step = grid_step_hours)
   =============================================================== */
max_steps AS (
  SELECT
    c.subject_id,
    c.hadm_id,
    c.stay_id,
    c.t0,
    c.terminal_ts,
    c.terminal_event,
    c.crrt_start_dt,
    -- Calculate max number of steps needed
    CAST(CEIL(
      TIMESTAMP_DIFF(c.terminal_ts, c.t0, HOUR) / cfg.grid_step_hours
    ) AS INT64) + 1 AS n_steps
  FROM cohort c
  CROSS JOIN cfg
),

-- Generate step numbers (0 to max observed)
step_numbers AS (
  SELECT step_num
  FROM UNNEST(GENERATE_ARRAY(0, 500)) AS step_num  -- max ~167 days at 8h steps
),

grid_raw AS (
  SELECT
    ms.subject_id,
    ms.hadm_id,
    ms.stay_id,
    ms.t0,
    ms.terminal_ts,
    ms.terminal_event,
    ms.crrt_start_dt,
    sn.step_num,
    TIMESTAMP_ADD(ms.t0, INTERVAL sn.step_num * cfg.grid_step_hours HOUR) AS grid_ts
  FROM max_steps ms
  CROSS JOIN cfg
  CROSS JOIN step_numbers sn
  WHERE sn.step_num < ms.n_steps
),

/* ===============================================================
   2) Add max grid_ts per stay using window function
   =============================================================== */
grid_with_max AS (
  SELECT
    g.*,
    MAX(g.grid_ts) OVER (PARTITION BY g.stay_id) AS max_grid_ts_per_stay
  FROM grid_raw g
)

/* ===============================================================
   FINAL: Add terminal flags and action
   =============================================================== */
SELECT
  g.subject_id,
  g.hadm_id,
  g.stay_id,
  g.grid_ts,
  g.step_num,
  g.t0, 

  -- Is this the terminal step?
  (g.grid_ts = g.max_grid_ts_per_stay) AS is_terminal_step,

  -- Terminal event (only on terminal step)
  CASE
    WHEN g.grid_ts = g.max_grid_ts_per_stay
    THEN g.terminal_event
    ELSE NULL
  END AS terminal_event,

  -- Action: was RRT started in the NEXT interval?
  -- action_rrt = 1 if CRRT started between current grid_ts and next grid_ts
  CASE
    WHEN g.crrt_start_dt IS NOT NULL
         AND g.crrt_start_dt >= g.grid_ts
         AND g.crrt_start_dt < TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_hours HOUR)
    THEN 1
    ELSE 0
  END AS action_rrt,

    -- timing helpers
  TIMESTAMP_DIFF(g.grid_ts, g.t0, HOUR) AS hours_since_t0,

FROM grid_with_max g
CROSS JOIN cfg
ORDER BY g.subject_id, g.stay_id, g.grid_ts
;
