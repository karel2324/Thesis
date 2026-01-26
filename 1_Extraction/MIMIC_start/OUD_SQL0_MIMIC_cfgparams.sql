CREATE OR REPLACE TABLE
  `windy-forge-475207-e3.derived_mimic.cfg_params` AS
SELECT

/* ===============================================================
   1) GLOBAL PARAMETERS
   =============================================================== */

  42  AS seed,

  8   AS grid_step_hours,
  8   AS grid_step_sofa_hours,

  24  AS sofa_lb_hours,
  48  AS gcs_lb_hours,
  24  AS fb_lb_hours,

  28  AS obs_days,

  'rrt_start' AS decision_problem,


/* ===============================================================
   2) STATE VARIABLES (labs / vitals)
   =============================================================== */

  [
    -- RENAL
    STRUCT('creat' AS var, [50912] AS itemids, 48 AS lb_h,
           0 AS min_val_clin, 20 AS max_val_clin,
           0 AS min_val_stat, 20 AS max_val_stat),

    STRUCT('urea' AS var, [51006] AS itemids, 48,
           0, 200, 0, 200),

    STRUCT('lactate' AS var, [50813] AS itemids, 48,
           0, 200, 0, 200),

    STRUCT('potassium' AS var, [50822, 50971] AS itemids, 48,
           0, 200, 0, 200),

    STRUCT('ph' AS var, [50820] AS itemids, 48,
           0, 200, 0, 200),

    STRUCT('bicarb' AS var, [50803, 50882] AS itemids, 48,
           0, 200, 0, 200),

    STRUCT('sodium' AS var, [50824,50983] AS itemids, 48,
           100, 200, 100, 200),

    STRUCT('chloride' AS var, [50806,50902] AS itemids, 48,
           100, 200, 100, 200),

    STRUCT('map' AS var, [220052,225312,220181] AS itemids, 48,
           10, 200, 10, 200),

    STRUCT('phosphate' AS var, [50970] AS itemids, 48,
           10, 200, 10, 200),

    STRUCT('calcium' AS var, [50808] AS itemids, 48,
          10, 200, 10, 200),
    
    STRUCT('magnesium' AS var, [50960] AS itemids, 48,
          10, 200, 10, 200),

     STRUCT('base_excess' AS var, [50802] AS itemids, 48,
          10, 200, 10, 200),

     STRUCT('anion_gap' AS var, [52500,50868] AS itemids, 48,
           10, 200, 10, 200),

    STRUCT('hemoglobin' AS var, [50811,51222] AS itemids, 48,
           10, 200, 10, 200),

    STRUCT('bilirubin' AS var, [50885] AS itemids, 48,
           0, 50, 0, 50),

     --STRUCT('pao2' AS var, [220052,225312,220181] AS itemids, 48,
        --   10, 200, 10, 200),

     STRUCT('fio2' AS var, [50816] AS itemids, 48,
           10, 200, 10, 200),

    STRUCT('platelets' AS var, [51265] AS itemids, 48,
           0, 2000, 0, 2000),

    STRUCT('heartrate' AS var, [220045] AS itemids, 48,
           0, 300, 0, 300),

    -- STRUCT('vent' AS var, [50813] AS itemids, 48,
       --    0, 30, 0, 30),

    STRUCT('weight' AS var, [762,226512] AS itemids, 48,
          6.8, 7.8, 6.8, 7.8)
  ] AS var_defs,


/* ===============================================================
   3) URINE OUTPUT / FLUID BALANCE
   =============================================================== */
            [226559 -- Foley
            , 226560 -- Void
            , 226561 -- Condom Cath
            , 226584 -- Ileoconduit
            , 226563 -- Suprapubic
            , 226564 -- R Nephrostomy
            , 226565 -- L Nephrostomy
            , 226567 -- Straight Cath
            , 226557 -- R Ureteral Stent
            , 226558 -- L Ureteral Stent
            , 227488 -- GU Irrigant Volume In
            , 227489  -- GU Irrigant/Urine Volume Out
            ]
        AS uo_ids,

  [
    'inputevents',
    'outputevents'
  ] AS fluid_balance_tables,


/* ===============================================================
   4) ACTION VARIABLES
   =============================================================== */

  -- RRT / CRRT
  [
    'crrt'
  ] AS rrt_tables,

  -- vasopressor exposure (binary / dose later)
  [
    'norepinephrine_equivalent_dose'
  ] AS vasopressor_tables,


/* ===============================================================
   5) VASOPRESSOR NAME MAP (SEMANTIC)
   =============================================================== */

  [
    STRUCT('norepinephrine' AS vas_name),
    STRUCT('epinephrine'),
    STRUCT('phenylephrine'),
    STRUCT('dopamine'),
    STRUCT('dobutamine')
  ] AS vas_dict,


/* ===============================================================
   6) GCS MAP (chartevents itemid → component)
   =============================================================== */

  [
    STRUCT(220739 AS itemid, 'eye'    AS component),
    STRUCT(223900 AS itemid, 'verbal' AS component),
    STRUCT(223901 AS itemid, 'motor'  AS component)
  ] AS gcs_map,


/* ===============================================================
   7) INCLUSION / EXCLUSION LOGIC
   =============================================================== */

  'kdigo2' AS inclusion,

  [
    STRUCT(
      'minimum_icu_stay_hours' AS rule,
      24 AS hours
    )
  ] AS exclusion_defs
;
