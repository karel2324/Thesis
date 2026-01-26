CREATE OR REPLACE TABLE derived.cfg_params AS
SELECT
  ------------------------------------------------------------------
  -- 1. VARIABELE DEFINITIES (labs / vitals / states)
  ------------------------------------------------------------------
  [
  STRUCT('creat'       AS var, [3020564]                    AS ids, 48     AS lb_h,
         0     AS min_val_clin, 3000     AS max_val_clin,
         0     AS min_val_stat, 3000     AS max_val_stat),

  STRUCT('urea'        AS var, [43534077]                   AS ids, 48     AS lb_h,
         0     AS min_val_clin, 300      AS max_val_clin,
         0     AS min_val_stat, 300      AS max_val_stat),

  STRUCT('lactate'     AS var, [3047181,3014111]            AS ids, 48     AS lb_h,
         0     AS min_val_clin, 30       AS max_val_clin,
         0     AS min_val_stat, 30       AS max_val_stat),

  STRUCT('potassium'   AS var, [3005456,3023103]            AS ids, 48     AS lb_h,
         1     AS min_val_clin, 10       AS max_val_clin,
         1     AS min_val_stat, 10       AS max_val_stat),

  STRUCT('ph'          AS var, [3010421]                    AS ids, 48     AS lb_h,
         6.5   AS min_val_clin, 8        AS max_val_clin,
         6.5   AS min_val_stat, 8        AS max_val_stat),

  STRUCT('bicarb'      AS var, [3006576]                    AS ids, 48     AS lb_h,
         0     AS min_val_clin, 60       AS max_val_clin,
         0     AS min_val_stat, 60       AS max_val_stat),

  STRUCT('sodium'      AS var, [3000285,3019550]            AS ids, 48     AS lb_h,
         100   AS min_val_clin, 200      AS max_val_clin,
         100   AS min_val_stat, 200      AS max_val_stat),

  STRUCT('chloride'    AS var, [3018572,3014576]            AS ids, 48     AS lb_h,
         60    AS min_val_clin, 150      AS max_val_clin,
         60    AS min_val_stat, 150      AS max_val_stat),

  STRUCT('map'         AS var, [21490852,21492241,21490673] AS ids, 48     AS lb_h,
         10    AS min_val_clin, 250      AS max_val_clin,
         10    AS min_val_stat, 250      AS max_val_stat),

  STRUCT('phosphate'   AS var, [3003458]                    AS ids, 48     AS lb_h,
         0     AS min_val_clin, 6        AS max_val_clin,
         0     AS min_val_stat, 6        AS max_val_stat),

  STRUCT('calcium'     AS var, [3048816]                    AS ids, 48     AS lb_h,
         0.5   AS min_val_clin, 4        AS max_val_clin,
         0.5   AS min_val_stat, 4        AS max_val_stat),

  STRUCT('magnesium'   AS var, [3012095,3033836]            AS ids, 48     AS lb_h,
         0     AS min_val_clin, 4        AS max_val_clin,
         0     AS min_val_stat, 4        AS max_val_stat),

  STRUCT('base_excess' AS var, [3012501]                    AS ids, 48     AS lb_h,
         -40   AS min_val_clin, 40       AS max_val_clin,
         -40   AS min_val_stat, 40       AS max_val_stat),

  STRUCT('anion_gap'   AS var, [3039000]                    AS ids, 48     AS lb_h,
         0     AS min_val_clin, 60       AS max_val_clin,
         0     AS min_val_stat, 60       AS max_val_stat),

  STRUCT('hemoglobin'  AS var, [40762351]                   AS ids, 48     AS lb_h,
         1     AS min_val_clin, 25       AS max_val_clin,
         1     AS min_val_stat, 25       AS max_val_stat),

  STRUCT('bilirubin'   AS var, [3006140,40757494]           AS ids, 48     AS lb_h,
         0     AS min_val_clin, 1000     AS max_val_clin,
         0     AS min_val_stat, 1000     AS max_val_stat),

  STRUCT('pao2'        AS var, [3027315]                    AS ids, 48     AS lb_h,
         0     AS min_val_clin, 80       AS max_val_clin,
         0     AS min_val_stat, 80       AS max_val_stat),

  STRUCT('fio2'        AS var, [42869590,3024882,8554]      AS ids, 48     AS lb_h,
         21    AS min_val_clin, 100      AS max_val_clin,
         21    AS min_val_stat, 100      AS max_val_stat),

  STRUCT('platelets'   AS var, [3007461]                    AS ids, 48     AS lb_h,
         0     AS min_val_clin, 2000     AS max_val_clin,
         0     AS min_val_stat, 2000     AS max_val_stat),

  STRUCT('heartrate'   AS var, [3027018,21490872]           AS ids, 48     AS lb_h,
         10   AS min_val_clin, 300      AS max_val_clin,
         10  AS min_val_stat, 300      AS max_val_stat),

  STRUCT('temperature' AS var, [21490586,3022060,3025085]   AS ids, 48     AS lb_h,
         30   AS min_val_clin, 45       AS max_val_clin,
         30   AS min_val_stat, 45       AS max_val_stat),

  STRUCT('albumin'     AS var, [3028286,3024561]            AS ids, 48     AS lb_h,
         10   AS min_val_clin, 60       AS max_val_clin,
         10   AS min_val_stat, 60       AS max_val_stat),

  STRUCT('crp'         AS var, [3020460]                    AS ids, 48     AS lb_h,
         0    AS min_val_clin, 500      AS max_val_clin,
         0    AS min_val_stat, 500      AS max_val_stat),

  STRUCT('glucose'     AS var, [3020491]                    AS ids, 48     AS lb_h,
         1    AS min_val_clin, 50       AS max_val_clin,
         1    AS min_val_stat, 50       AS max_val_stat),

  STRUCT('hematocrit'  AS var, [42869588]                   AS ids, 48     AS lb_h,
         0.10 AS min_val_clin, 0.70     AS max_val_clin,
         0.10 AS min_val_stat, 0.70     AS max_val_stat),

  STRUCT('wbc'         AS var, [3010813]                    AS ids, 48     AS lb_h,
         0    AS min_val_clin, 100      AS max_val_clin,
         0    AS min_val_stat, 100      AS max_val_stat)
         
] AS var_defs,


  ------------------------------------------------------------------
  -- 2. SPECIALE CONCEPT-ID SETS
  ------------------------------------------------------------------
  [1988610] AS bloodflow_ids,

  [3026600,3013762,3025315,3023166] AS weight_ids,

  [3014315,3016267,21491173] AS uo_ids,

  [3022875,21490855] as vent_ids,

  [
    3025727,3006376,3018767,21491184,3039001,3026556,3023454,
    3020433,3017152,21491180,3008793,4092499,3037836,3028277,
    21491183,3011087,1033562,36032760,3557196,40458249,
    3014315,40481610,3016267,4265527,36033316,21493221,
    21491173,4264378,36032656,3007123
  ] AS fluid_out_ids,

  [
    19076867,40038600, 21088391,36411287, 
    2907531, 40074086
  ] AS vas_ids,

  ------------------------------------------------------------------
  -- 3. VASOPRESSOR NAAMMAPPING
  ------------------------------------------------------------------
  [
    STRUCT(19076867 AS cid, 'epinephrine'      AS vas_name),
    STRUCT(40038600 AS cid, 'epinephrine'),
-- STRUCT(36812278 AS cid, 'isoproterenol'),
--  STRUCT(44097196 AS cid, 'isoproterenol injection'),
    STRUCT(21088391 AS cid, 'dobutamine'),
    STRUCT(36411287 AS cid, 'dopamine'),
--  STRUCT(41064805 AS cid, 'enoximone'),
    STRUCT(2907531  AS cid, 'norepinephrine'),
--  STRUCT(40072394 AS cid, 'norepinephrine'),
--  STRUCT(40102804 AS cid, 'terlipressin'),
--  STRUCT(44109975 AS cid, 'methylene_blue'),
    STRUCT(40074086 AS cid, 'phenylephrine')
  ] AS vas_dict,

  ------------------------------------------------------------------
  -- 4. GCS VALUE_AS_CONCEPT_ID → SCORE
  ------------------------------------------------------------------
  [
    STRUCT(45880466 AS value_as_concept_id, 4 AS score, 'eye'    AS component),
    STRUCT(45881530, 4, 'eye'),
    STRUCT(45876254, 3, 'eye'),
    STRUCT(45884209, 2, 'eye'),
    STRUCT(45877537, 1, 'eye'),

    STRUCT(45877602, 5, 'verbal'),
    STRUCT(45883906, 4, 'verbal'),
    STRUCT(45877601, 3, 'verbal'),
    STRUCT(45883352, 2, 'verbal'),
    STRUCT(45877384, 1, 'verbal'),

    STRUCT(45880468, 6, 'motor'),
    STRUCT(45880467, 5, 'motor'),
    STRUCT(45882047, 4, 'motor'),
    STRUCT(45879885, 3, 'motor'),
    STRUCT(45878993, 2, 'motor'),
    STRUCT(45878992, 1, 'motor')
  ] AS gcs_map,

  ------------------------------------------------------------------
  -- 5. GLOBALE TIJDSPARAMETERS
  ------------------------------------------------------------------
  28  AS obs_days,
  8   AS grid_step_hours,
  8   AS grid_step_sofa_hours,
  24  AS fb_lb_hours,
  48  AS gcs_lb_hours, 
  24  AS intox_fw_hours, 

  ------------------------------------------------------------------
  -- 6. INCLUSIELOGICA
  ------------------------------------------------------------------
  'kdigo2' AS inclusion,

  ------------------------------------------------------------------
  -- 7. EXCLUSIELOGICA
  ------------------------------------------------------------------
  [
  STRUCT(
    'Intoxication' AS var,
    [
      45769428,  -- Illicit drug overdose
      4176983,   -- Overdose of antidepressant drug
      4174619,   -- Sedative overdose
      4166498,   -- Overdose of analgesic drug
      607208,    -- Acute alcohol intoxication
      602985     -- Alcohol intoxication
    ] AS ids,
    48 AS lb_h
  )
] AS intox_defs;

