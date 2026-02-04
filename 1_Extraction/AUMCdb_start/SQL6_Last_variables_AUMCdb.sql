CREATE OR REPLACE TABLE `windy-forge-475207-e3.derived.grid_last_var_long` AS
WITH
/* ===============================================================
   1) VAR MAP (utils, gede-dupliceerd, geen gcs)
   =============================================================== */
var_map AS (
  SELECT DISTINCT
    var,
    lb_h
  FROM `windy-forge-475207-e3.derived.var_map`
  WHERE var != 'gcs'
),

/* ===============================================================
   2) GRID (single source of truth)
   =============================================================== */
grid AS (
  SELECT
    person_id,
    visit_occurrence_id,
    grid_ts
  FROM `windy-forge-475207-e3.derived.cohort_grid`
),

/* ===============================================================
   3) CANDIDATE METINGEN BINNEN LOOKBACK
   =============================================================== */
candidates AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,

    m.var,
    m.value_as_number AS val,
    m.ts
  FROM grid g
  JOIN `windy-forge-475207-e3.derived.cohort_measurements_window_stat_valid` m
    ON m.person_id = g.person_id
   AND m.visit_occurrence_id = g.visit_occurrence_id
   AND m.ts <= g.grid_ts
  JOIN var_map vm
    ON vm.var = m.var
  WHERE m.source_type = 'state_var'
    AND m.ts > TIMESTAMP_SUB(g.grid_ts, INTERVAL vm.lb_h HOUR)
),

/* ===============================================================
   4) LAATSTE METING PER VAR
   =============================================================== */
last_var AS (
  SELECT
    person_id,
    visit_occurrence_id,
    grid_ts,
    var,

    val AS val_last,
    ts  AS val_ts,

    -- age since measurement (uren)
    TIMESTAMP_DIFF(grid_ts, ts, MINUTE) / 60.0 AS val_age_hours,

    FORMAT('%s_last', var) AS feature_name
  FROM (
    SELECT
      *,
      ROW_NUMBER() OVER (
        PARTITION BY person_id, visit_occurrence_id, grid_ts, var
        ORDER BY ts DESC
      ) AS rn
    FROM candidates
  )
  WHERE rn = 1
)

SELECT *
FROM last_var
ORDER BY person_id, visit_occurrence_id, grid_ts, var;
