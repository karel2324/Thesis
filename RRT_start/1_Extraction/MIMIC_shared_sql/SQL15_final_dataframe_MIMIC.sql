/* ===============================================================
   SQL15_final_dataframe.sql - MIMIC-IV Final Master Table

   Analogue to AUMCdb SQL15_Final_dataframe.sql

   Joins all derived tables into final feature matrix
   =============================================================== */

CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.grid_master_all_features` AS
WITH
/* ===============================================================
   0) GRID (SQL3_grid = single source of truth)
   =============================================================== */
grid AS (
  SELECT
    subject_id,
    hadm_id,
    stay_id,
    grid_ts,
    is_terminal_step,
    terminal_event,
    action_rrt
  FROM `windy-forge-475207-e3.${DATASET}.cohort_grid`
),

/* ===============================================================
   1) SOFA COMPONENTS (CURRENT + FORWARD)
   =============================================================== */
sofa_components AS (
  SELECT
    g.subject_id,
    g.hadm_id,
    g.stay_id,
    g.grid_ts,

    /* CURRENT */
    CAST(sc.sofa_cardio_24h_current AS INT64) AS sofa_cardio_24h_current,
    CAST(sr.sofa_renal_24h_current  AS INT64) AS sofa_renal_24h_current,
    CAST(ss.sofa_resp_24h_current   AS INT64) AS sofa_resp_24h_current,
    CAST(sn.sofa_neuro_24h_current  AS INT64) AS sofa_neuro_24h_current,
    CAST(sco.sofa_coag_24h_current  AS INT64) AS sofa_coag_24h_current,
    CAST(sl.sofa_liver_24h_current  AS INT64) AS sofa_liver_24h_current,

    /* FORWARD */
    CAST(sc.sofa_cardio_24h_forward AS INT64) AS sofa_cardio_24h_forward,
    CAST(sr.sofa_renal_24h_forward  AS INT64) AS sofa_renal_24h_forward,
    CAST(ss.sofa_resp_24h_forward   AS INT64) AS sofa_resp_24h_forward,
    CAST(sn.sofa_neuro_24h_forward  AS INT64) AS sofa_neuro_24h_forward,
    CAST(sco.sofa_coag_24h_forward  AS INT64) AS sofa_coag_24h_forward,
    CAST(sl.sofa_liver_24h_forward  AS INT64) AS sofa_liver_24h_forward

  FROM grid g
  LEFT JOIN `windy-forge-475207-e3.${DATASET}.sofa_cardio_24h` sc
    USING (subject_id, hadm_id, stay_id, grid_ts)
  LEFT JOIN `windy-forge-475207-e3.${DATASET}.sofa_renal_24h` sr
    USING (subject_id, hadm_id, stay_id, grid_ts)
  LEFT JOIN `windy-forge-475207-e3.${DATASET}.sofa_resp_24h` ss
    USING (subject_id, hadm_id, stay_id, grid_ts)
  LEFT JOIN `windy-forge-475207-e3.${DATASET}.sofa_neuro_24h` sn
    USING (subject_id, hadm_id, stay_id, grid_ts)
  LEFT JOIN `windy-forge-475207-e3.${DATASET}.sofa_coag_24h` sco
    USING (subject_id, hadm_id, stay_id, grid_ts)
  LEFT JOIN `windy-forge-475207-e3.${DATASET}.sofa_liver_24h` sl
    USING (subject_id, hadm_id, stay_id, grid_ts)
),

/* ===============================================================
   2) SOFA TOTALS (CURRENT + FORWARD)
   =============================================================== */
sofa_all AS (
  SELECT
    *,

    /* CURRENT TOTAL */
      IFNULL(sofa_cardio_24h_current, 0)
    + IFNULL(sofa_renal_24h_current,  0)
    + IFNULL(sofa_resp_24h_current,   0)
    + IFNULL(sofa_neuro_24h_current,  0)
    + IFNULL(sofa_coag_24h_current,   0)
    + IFNULL(sofa_liver_24h_current,  0)
      AS sofa_total_24h_current,

    /* FORWARD TOTAL */
      IFNULL(sofa_cardio_24h_forward, 0)
    + IFNULL(sofa_renal_24h_forward,  0)
    + IFNULL(sofa_resp_24h_forward,   0)
    + IFNULL(sofa_neuro_24h_forward,  0)
    + IFNULL(sofa_coag_24h_forward,   0)
    + IFNULL(sofa_liver_24h_forward,  0)
      AS sofa_total_24h_forward

  FROM sofa_components
),

/* ===============================================================
   3) LAST VAR -> WIDE (STATE VARS)
   =============================================================== */
last_var_wide AS (
  SELECT
    stay_id,
    grid_ts,

    MAX(IF(var='creat',        val_last, NULL)) AS creat_last,
    MAX(IF(var='creat',        val_age_hours, NULL)) AS creat_age_h,

    MAX(IF(var='urea',         val_last, NULL)) AS urea_last,
    MAX(IF(var='urea',         val_age_hours, NULL)) AS urea_age_h,

    MAX(IF(var='lactate',      val_last, NULL)) AS lactate_last,
    MAX(IF(var='lactate',      val_age_hours, NULL)) AS lactate_age_h,

    MAX(IF(var='potassium',    val_last, NULL)) AS potassium_last,
    MAX(IF(var='potassium',    val_age_hours, NULL)) AS potassium_age_h,

    MAX(IF(var='ph',           val_last, NULL)) AS ph_last,
    MAX(IF(var='ph',           val_age_hours, NULL)) AS ph_age_h,

    MAX(IF(var='bicarb',       val_last, NULL)) AS bicarb_last,
    MAX(IF(var='bicarb',       val_age_hours, NULL)) AS bicarb_age_h,

    MAX(IF(var='sodium',       val_last, NULL)) AS sodium_last,
    MAX(IF(var='sodium',       val_age_hours, NULL)) AS sodium_age_h,

    MAX(IF(var='chloride',     val_last, NULL)) AS chloride_last,
    MAX(IF(var='chloride',     val_age_hours, NULL)) AS chloride_age_h,

    MAX(IF(var='map',          val_last, NULL)) AS map_last,
    MAX(IF(var='map',          val_age_hours, NULL)) AS map_age_h,

    MAX(IF(var='phosphate',    val_last, NULL)) AS phosphate_last,
    MAX(IF(var='phosphate',    val_age_hours, NULL)) AS phosphate_age_h,

    MAX(IF(var='calcium',      val_last, NULL)) AS calcium_last,
    MAX(IF(var='calcium',      val_age_hours, NULL)) AS calcium_age_h,

    MAX(IF(var='magnesium',    val_last, NULL)) AS magnesium_last,
    MAX(IF(var='magnesium',    val_age_hours, NULL)) AS magnesium_age_h,

    MAX(IF(var='base_excess',  val_last, NULL)) AS base_excess_last,
    MAX(IF(var='base_excess',  val_age_hours, NULL)) AS base_excess_age_h,

    MAX(IF(var='anion_gap',    val_last, NULL)) AS anion_gap_last,
    MAX(IF(var='anion_gap',    val_age_hours, NULL)) AS anion_gap_age_h,

    MAX(IF(var='hemoglobin',   val_last, NULL)) AS hemoglobin_last,
    MAX(IF(var='hemoglobin',   val_age_hours, NULL)) AS hemoglobin_age_h,

    MAX(IF(var='bilirubin',    val_last, NULL)) AS bilirubin_last,
    MAX(IF(var='bilirubin',    val_age_hours, NULL)) AS bilirubin_age_h,

    MAX(IF(var='pao2',         val_last, NULL)) AS pao2_last,
    MAX(IF(var='pao2',         val_age_hours, NULL)) AS pao2_age_h,

    MAX(IF(var='fio2',         val_last, NULL)) AS fio2_last,
    MAX(IF(var='fio2',         val_age_hours, NULL)) AS fio2_age_h,

    MAX(IF(var='platelets',    val_last, NULL)) AS platelets_last,
    MAX(IF(var='platelets',    val_age_hours, NULL)) AS platelets_age_h,

    MAX(IF(var='heartrate',    val_last, NULL)) AS heartrate_last,
    MAX(IF(var='heartrate',    val_age_hours, NULL)) AS heartrate_age_h,

    MAX(IF(var='temperature',  val_last, NULL)) AS temperature_last,
    MAX(IF(var='temperature',  val_age_hours, NULL)) AS temperature_age_h,

    MAX(IF(var='albumin',      val_last, NULL)) AS albumin_last,
    MAX(IF(var='albumin',      val_age_hours, NULL)) AS albumin_age_h,

    MAX(IF(var='crp',          val_last, NULL)) AS crp_last,
    MAX(IF(var='crp',          val_age_hours, NULL)) AS crp_age_h,

    MAX(IF(var='glucose',      val_last, NULL)) AS glucose_last,
    MAX(IF(var='glucose',      val_age_hours, NULL)) AS glucose_age_h,

    MAX(IF(var='hematocrit',   val_last, NULL)) AS hematocrit_last,
    MAX(IF(var='hematocrit',   val_age_hours, NULL)) AS hematocrit_age_h,

    MAX(IF(var='wbc',          val_last, NULL)) AS wbc_last,
    MAX(IF(var='wbc',          val_age_hours, NULL)) AS wbc_age_h

  FROM `windy-forge-475207-e3.${DATASET}.grid_last_var_long`
  GROUP BY stay_id, grid_ts
)

/* ===============================================================
   4) FINAL TABLE
   =============================================================== */
SELECT
  g.subject_id as person_id,
  g.stay_id as visit_occurrence_id,
  g.grid_ts,

  /* ---- terminal flags (from SQL3_grid) ---- */
  g.is_terminal_step,
  g.terminal_event,
  g.action_rrt,

  /* ---- cohort static ---- */
  cs.t0,
  cs.terminal_ts,
  cs.death_dt,
  cs.gender,
  cs.age_years,
  cs.weight_kg_cohort,
  cs.weight_used_kg_cohort,
  cs.weight_kg_first,
  cs.weight_used_kg_stay as weight_used_kg_person,
  cs.weight_source_stay as weight_source_person,
  cs.icu_intime as admit_dt,
  cs.icu_outtime as discharge_dt,
  cs.inclusion_applied,

  /* ---- SOFA CURRENT ---- */
  s.sofa_cardio_24h_current,
  s.sofa_renal_24h_current,
  s.sofa_resp_24h_current,
  s.sofa_neuro_24h_current,
  s.sofa_coag_24h_current,
  s.sofa_liver_24h_current,
  s.sofa_total_24h_current,

  /* ---- SOFA FORWARD ---- */
  s.sofa_cardio_24h_forward,
  s.sofa_renal_24h_forward,
  s.sofa_resp_24h_forward,
  s.sofa_neuro_24h_forward,
  s.sofa_coag_24h_forward,
  s.sofa_liver_24h_forward,
  s.sofa_total_24h_forward,

  /* ---- urine output (ml/kg/h) ---- */
  uo.uo_6h_mlkgh,
  uo.uo_12h_mlkgh,
  uo.uo_24h_mlkgh,
  uo.uo_24h_ml,

  /* ---- renal trends ---- */
  rt.creat_abs_inc_48h,
  IF(rt.creat_abs_inc_48h IS NULL, 1, 0) AS creat_abs_inc_48h_missing,

  rt.urea_abs_inc_48h,
  IF(rt.urea_abs_inc_48h IS NULL, 1, 0) AS urea_abs_inc_48h_missing,

  rt.creat_rel_inc_7d,

  rt.kdigo_stage,

  /* ---- support ---- */
  vp.vasopressor_in_use,
  mv.mechanical_ventilation_in_use,

  /* ---- fluid balance ---- */
  fb.fluid_in_fblb_ml,
  fb.fluid_out_fblb_ml,
  fb.fluid_balance_fblb_ml,

  /* ---- GCS ---- */
  gcs.gcs_eye_last,
  gcs.gcs_motor_last,
  gcs.gcs_verbal_last,
  gcs.gcs_total_last,
  gcs.gcs_total_last_hours,
  IF(gcs.gcs_total_last IS NULL, 1, 0) AS gcs_total_last_missing,


  /* ---- state vars ---- */

  lv.creat_last,
  lv.creat_age_h,
  IF(lv.creat_last IS NULL, 1, 0) AS creat_last_missing,

  lv.urea_last,
  lv.urea_age_h,
  IF(lv.urea_last IS NULL, 1, 0) AS urea_last_missing,

  lv.lactate_last,
  lv.lactate_age_h,
  IF(lv.lactate_last IS NULL, 1, 0) AS lactate_last_missing,

  lv.potassium_last,
  lv.potassium_age_h,
  IF(lv.potassium_last IS NULL, 1, 0) AS potassium_last_missing,

  lv.ph_last,
  lv.ph_age_h,
  IF(lv.ph_last IS NULL, 1, 0) AS ph_last_missing,

  lv.bicarb_last,
  lv.bicarb_age_h,
  IF(lv.bicarb_last IS NULL, 1, 0) AS bicarb_last_missing,

  lv.sodium_last,
  lv.sodium_age_h,
  IF(lv.sodium_last IS NULL, 1, 0) AS sodium_last_missing,

  lv.chloride_last,
  lv.chloride_age_h,
  IF(lv.chloride_last IS NULL, 1, 0) AS chloride_last_missing,

  lv.map_last,
  lv.map_age_h,
  IF(lv.map_last IS NULL, 1, 0) AS map_last_missing,

  lv.phosphate_last,
  lv.phosphate_age_h,
  IF(lv.phosphate_last IS NULL, 1, 0) AS phosphate_last_missing,

  lv.calcium_last,
  lv.calcium_age_h,
  IF(lv.calcium_last IS NULL, 1, 0) AS calcium_last_missing,

  lv.magnesium_last,
  lv.magnesium_age_h,
  IF(lv.magnesium_last IS NULL, 1, 0) AS magnesium_last_missing,

  lv.base_excess_last,
  lv.base_excess_age_h,
  IF(lv.base_excess_last IS NULL, 1, 0) AS base_excess_last_missing,

  lv.anion_gap_last,
  lv.anion_gap_age_h,
  IF(lv.anion_gap_last IS NULL, 1, 0) AS anion_gap_last_missing,

  lv.hemoglobin_last,
  lv.hemoglobin_age_h,
  IF(lv.hemoglobin_last IS NULL, 1, 0) AS hemoglobin_last_missing,

  lv.bilirubin_last,
  lv.bilirubin_age_h,
  IF(lv.bilirubin_last IS NULL, 1, 0) AS bilirubin_last_missing,

  lv.pao2_last,
  lv.pao2_age_h,
  IF(lv.pao2_last IS NULL, 1, 0) AS pao2_last_missing,

  lv.fio2_last,
  lv.fio2_age_h,
  IF(lv.fio2_last IS NULL, 1, 0) AS fio2_last_missing,

  lv.platelets_last,
  lv.platelets_age_h,
  IF(lv.platelets_last IS NULL, 1, 0) AS platelets_last_missing,

  lv.heartrate_last,
  lv.heartrate_age_h,
  IF(lv.heartrate_last IS NULL, 1, 0) AS heartrate_last_missing,

  lv.temperature_last,
  lv.temperature_age_h,
  IF(lv.temperature_last IS NULL, 1, 0) AS temperature_last_missing,

  lv.albumin_last,
  lv.albumin_age_h,
  IF(lv.albumin_last IS NULL, 1, 0) AS albumin_last_missing,

  lv.crp_last,
  lv.crp_age_h,
  IF(lv.crp_last IS NULL, 1, 0) AS crp_last_missing,

  lv.glucose_last,
  lv.glucose_age_h,
  IF(lv.glucose_last IS NULL, 1, 0) AS glucose_last_missing,

  lv.hematocrit_last,
  lv.hematocrit_age_h,
  IF(lv.hematocrit_last IS NULL, 1, 0) AS hematocrit_last_missing,

  lv.wbc_last,
  lv.wbc_age_h,
  IF(lv.wbc_last IS NULL, 1, 0) AS wbc_last_missing,

  /* weight is static (first measurement per stay) */
  IF(cs.weight_kg_first IS NULL, 1, 0) AS weight_kg_first_missing

FROM grid g
LEFT JOIN `windy-forge-475207-e3.${DATASET}.cohort_static_variables` cs
  USING (subject_id, hadm_id, stay_id)
LEFT JOIN sofa_all s
  USING (subject_id, hadm_id, stay_id, grid_ts)
LEFT JOIN `windy-forge-475207-e3.${DATASET}.grid_urine_output` uo
  USING (subject_id, hadm_id, stay_id, grid_ts)
LEFT JOIN `windy-forge-475207-e3.${DATASET}.grid_renal_trends` rt
  USING (subject_id, hadm_id, stay_id, grid_ts)
LEFT JOIN `windy-forge-475207-e3.${DATASET}.cohort_vasopressors_grid` vp
  USING (subject_id, hadm_id, stay_id, grid_ts)
LEFT JOIN `windy-forge-475207-e3.${DATASET}.cohort_mechanical_ventilation_grid` mv
  USING (subject_id, hadm_id, stay_id, grid_ts)
LEFT JOIN `windy-forge-475207-e3.${DATASET}.grid_fluid_balance` fb
  USING (subject_id, hadm_id, stay_id, grid_ts)
LEFT JOIN `windy-forge-475207-e3.${DATASET}.cohort_gcs_grid` gcs
  USING (subject_id, hadm_id, stay_id, grid_ts)
LEFT JOIN last_var_wide lv
  USING (stay_id, grid_ts)
ORDER BY g.subject_id, g.stay_id, g.grid_ts;
