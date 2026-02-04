/* ===============================================================
   SQL5_1_varying_variables_censored.sql - MIMIC-IV Stat Valid Measurements

   Analogue to AUMCdb SQL5_1_Varying_variables_censored.sql

   Applies statistical validity filters (min_val_stat, max_val_stat)

   Note: Values are already converted to SI units (AUMCdb standard) in SQL5
   Bounds are also in SI units for consistent validation
   =============================================================== */

CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.cohort_measurements_window_stat_valid` AS
SELECT
  subject_id,
  hadm_id,
  stay_id,
  ts,
  var,
  source,
  lb_h,
  value_as_number,
  source_type,
  min_val_stat,
  max_val_stat

FROM `windy-forge-475207-e3.${DATASET}.cohort_measurements_window`
WHERE value_as_number BETWEEN min_val_stat AND max_val_stat
ORDER BY subject_id, stay_id, ts;
