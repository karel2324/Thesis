/* ===============================================================
   SQL5_varying_variables.sql - MIMIC-IV Measurements Window

   Analogue to AUMCdb SQL5_Varying_variables.sql

   Key differences:
   - Labs from labevents, vitals from chartevents
   - Uses itemid instead of measurement_concept_id
   - Uses stay_id as primary identifier
   =============================================================== */

CREATE OR REPLACE TABLE `windy-forge-475207-e3.derived_mimic.cohort_measurements_window` AS
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
   1) VAR MAP (from utils)
   =============================================================== */
var_map AS (
  SELECT *
  FROM `windy-forge-475207-e3.derived_mimic.var_map`
),

/* ===============================================================
   2) COHORT + TIME WINDOW
   =============================================================== */
cohort_window AS (
  SELECT
    c.subject_id,
    c.hadm_id,
    c.stay_id,
    c.t0,
    c.terminal_ts,

    -- max lookback over ALL concepts 
    TIMESTAMP_SUB(
      c.t0,
      INTERVAL 168 HOUR
    ) AS window_start_ts,

    -- slightly past terminal_ts for last grid/SOFA
    TIMESTAMP_ADD(
      c.terminal_ts,
      INTERVAL cfg.grid_step_sofa_hours HOUR
    ) AS window_end_ts
  FROM `windy-forge-475207-e3.derived_mimic.cohort_static_variables` c
  CROSS JOIN cfg
),

/* ===============================================================
   3a) LABEVENTS (source = 'lab')
   Convert MIMIC units to SI units (AUMCdb standard)
   =============================================================== */
lab_meas AS (
  SELECT
    cw.subject_id,
    cw.hadm_id,
    cw.stay_id,
    TIMESTAMP(le.charttime) AS ts,
    vm.var,
    vm.source,
    vm.lb_h,
    -- Convert to SI units: multiply by conv_factor (or apply formula for temperature)
    CASE
      WHEN vm.var = 'temperature' THEN (le.valuenum - 32) * 5 / 9  -- Fahrenheit to Celsius
      WHEN vm.conv_factor IS NOT NULL AND vm.conv_factor != 1.0 THEN le.valuenum * vm.conv_factor
      ELSE le.valuenum
    END AS value_as_number,
    -- Use SI bounds for downstream validation
    vm.min_val_clin_si AS min_val_clin,
    vm.max_val_clin_si AS max_val_clin,
    vm.min_val_stat_si AS min_val_stat,
    vm.max_val_stat_si AS max_val_stat,
    'state_var' AS source_type

  FROM `physionet-data.mimiciv_3_1_hosp.labevents` le
  JOIN var_map vm ON vm.itemid = le.itemid AND vm.source = 'lab'
  JOIN cohort_window cw
    ON cw.subject_id = le.subject_id
   AND cw.hadm_id = le.hadm_id
   AND TIMESTAMP(le.charttime) BETWEEN cw.window_start_ts AND cw.window_end_ts
  WHERE le.valuenum IS NOT NULL
    -- Validate in MIMIC units before conversion
    AND le.valuenum BETWEEN vm.min_val_clin AND vm.max_val_clin
),

/* ===============================================================
   3b) CHARTEVENTS (source = 'chart')
   Convert MIMIC units to SI units (AUMCdb standard)
   =============================================================== */
chart_meas AS (
  SELECT
    cw.subject_id,
    cw.hadm_id,
    cw.stay_id,
    TIMESTAMP(ce.charttime) AS ts,
    vm.var,
    vm.source,
    vm.lb_h,
    -- Convert to SI units: multiply by conv_factor (or apply formula for temperature)
    CASE
      WHEN vm.var = 'temperature' THEN (ce.valuenum - 32) * 5 / 9  -- Fahrenheit to Celsius
      WHEN vm.conv_factor IS NOT NULL AND vm.conv_factor != 1.0 THEN ce.valuenum * vm.conv_factor
      ELSE ce.valuenum
    END AS value_as_number,
    -- Use SI bounds for downstream validation
    vm.min_val_clin_si AS min_val_clin,
    vm.max_val_clin_si AS max_val_clin,
    vm.min_val_stat_si AS min_val_stat,
    vm.max_val_stat_si AS max_val_stat,
    'state_var' AS source_type

  FROM `physionet-data.mimiciv_3_1_icu.chartevents` ce
  JOIN var_map vm ON vm.itemid = ce.itemid AND vm.source = 'chart'
  JOIN cohort_window cw
    ON cw.stay_id = ce.stay_id
   AND TIMESTAMP(ce.charttime) BETWEEN cw.window_start_ts AND cw.window_end_ts
  WHERE ce.valuenum IS NOT NULL
    -- Validate in MIMIC units before conversion
    AND ce.valuenum BETWEEN vm.min_val_clin AND vm.max_val_clin
),

/* ===============================================================
   3c) URINE OUTPUT (from outputevents)
   =============================================================== */
uo_meas AS (
  SELECT
    cw.subject_id,
    cw.hadm_id,
    cw.stay_id,
    TIMESTAMP(oe.charttime) AS ts,
    'urine_output' AS var,
    'output' AS source,
    24 AS lb_h,
    SAFE_CAST(oe.value AS FLOAT64) AS value_as_number,
    0 AS min_val_clin,
    10000 AS max_val_clin,
    0 AS min_val_stat,
    10000 AS max_val_stat,
    'urine_output' AS source_type

  FROM `physionet-data.mimiciv_3_1_icu.outputevents` oe
  CROSS JOIN cfg
  JOIN cohort_window cw
    ON cw.stay_id = oe.stay_id
   AND TIMESTAMP(oe.charttime) BETWEEN cw.window_start_ts AND cw.window_end_ts
  WHERE oe.itemid IN UNNEST(cfg.uo_ids.mimic_ids)
    AND oe.value IS NOT NULL
    AND oe.value > 0
)

/* ===============================================================
   FINAL: UNION ALL SOURCES
   =============================================================== */
SELECT * FROM lab_meas
UNION ALL
SELECT * FROM chart_meas
UNION ALL
SELECT * FROM uo_meas
ORDER BY subject_id, stay_id, ts;
