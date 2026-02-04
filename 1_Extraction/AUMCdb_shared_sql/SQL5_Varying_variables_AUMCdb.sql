
CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.cohort_measurements_window` AS
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
   1) CONCEPT MAP (ALLEEN uit utils / cfg)
   =============================================================== */
concept_map AS (
  -- state variables (labs / vitals)
  SELECT
    vm.var,
    vm.lb_h,
    vm.measurement_concept_id,
    vm.min_val_clin,
    vm.max_val_clin,
    vm.min_val_stat,
    vm.max_val_stat,
    'state_var' AS source_type
  FROM `windy-forge-475207-e3.${DATASET}.var_map` vm

  UNION ALL

  -- urine output (uses unified cfg_params with aumcdb_ids)
  SELECT
    'urine_output' AS var,
    24 AS lb_h,
    cid AS measurement_concept_id,
    0 AS min_val_clin,
    10000 AS max_val_clin,
    0 AS min_val_stat,
    10000 AS max_val_stat,
    'urine_output' AS source_type
  FROM cfg, UNNEST(cfg.uo_ids.aumcdb_ids) cid

  UNION ALL

  -- fluid out (uses unified cfg_params with aumcdb_ids)
  SELECT
    'fluid_out' AS var,
    cfg.fb_lb_hours AS lb_h,
    cid AS measurement_concept_id,
    0 AS min_val_clin,
    10000 AS max_val_clin,
    0 AS min_val_stat,
    10000 AS max_val_stat,
    'fluid_out' AS source_type
  FROM cfg, UNNEST(cfg.fluid_out_ids.aumcdb_ids) cid
),

/* ===============================================================
   2) COHORT + TIJDSWINDOW
   =============================================================== */
cohort_window AS (
  SELECT
    c.person_id,
    c.visit_occurrence_id,
    c.t0,
    c.terminal_ts,

    -- maximale lookback over ALLE concepten (+24h buffer)
    TIMESTAMP_SUB(
      c.t0,
      INTERVAL 168 HOUR
    ) AS window_start_ts,

    -- iets voorbij terminal_ts voor laatste grid / SOFA
    TIMESTAMP_ADD(
      c.terminal_ts,
      INTERVAL cfg.grid_step_sofa_hours HOUR
    ) AS window_end_ts
  FROM `windy-forge-475207-e3.${DATASET}.cohort_static_variables` c
  CROSS JOIN cfg
),

/* ===============================================================
   3) METINGEN BINNEN WINDOW
   =============================================================== */
meas_in_window AS (
  SELECT
    m.person_id,
    m.visit_occurrence_id,
    COALESCE(m.measurement_datetime, TIMESTAMP(m.measurement_date)) AS ts,

    m.measurement_concept_id,
    cm.var,
    cm.source_type,
    cm.lb_h,

    m.value_as_number,
    m.value_as_concept_id,
    m.unit_concept_id,

    cm.min_val_stat,
    cm.max_val_stat

  FROM `amsterdamumcdb.version1_5_0.measurement` m
  JOIN concept_map cm
    ON cm.measurement_concept_id = m.measurement_concept_id
  JOIN cohort_window cw
    ON cw.person_id = m.person_id
   AND cw.visit_occurrence_id = m.visit_occurrence_id
   AND COALESCE(m.measurement_datetime, TIMESTAMP(m.measurement_date))
       BETWEEN cw.window_start_ts AND cw.window_end_ts
  WHERE m.person_id IS NOT NULL
    AND m.value_as_number IS NOT NULL
    AND ((m.provider_id BETWEEN 0 AND 99) OR m.provider_id IS NULL)
    AND m.value_as_number BETWEEN cm.min_val_clin AND cm.max_val_clin
)

/* ===============================================================
   FINAL
   =============================================================== */
SELECT *
FROM meas_in_window
ORDER BY person_id, visit_occurrence_id, ts;
