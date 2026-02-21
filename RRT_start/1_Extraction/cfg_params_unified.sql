/*
================================================================================
UNIFIED CONFIGURATION - AUMCdb - ${COHORT_NAME} (${OBS_DAYS} dagen observatie)
================================================================================

Verschil met AUMCdb_start:
  - Inclusie: KDIGO stage 3 (i.p.v. KDIGO 2)
  - Observatieperiode: 7 dagen (i.p.v. 28 dagen)
  - Doel: Hogere actiefrequentie voor RL

Structuur:
  - global_params: Tijdsparameters (obs_days=7, grid_step, etc.)
  - inclusion/exclusion: Cohort criteria (kdigo3)
  - var_defs: Variable definities met IDs + units + conversie
  - special_ids: Weight, urine output, fluid balance, etc.
  - gcs_map: GCS score mappings
  - vas_defs: Vasopressor mappings

================================================================================
*/

CREATE OR REPLACE TABLE ${DATASET}.cfg_params AS
SELECT

  -- ============================================================================
  -- 1. GLOBAL TIME PARAMETERS
  -- ============================================================================
  ${OBS_DAYS}  AS obs_days,           -- CHANGED: 28 -> 7 dagen
  ${GRID_STEP_HOURS}  AS grid_step_hours,
  ${GRID_STEP_HOURS}  AS grid_step_sofa_hours,
  24  AS fb_lb_hours,
  72  AS gcs_lb_hours,
  48  AS intox_fw_hours,

  -- ============================================================================
  -- 2. INCLUSION CRITERIA
  -- ============================================================================
  '${INCLUSION}'  AS inclusion_default,  -- CHANGED: 'kdigo2' -> 'kdigo3'

  -- ============================================================================
  -- 3. EXCLUSION CRITERIA
  -- ============================================================================
  ${MIN_ICU_HOURS} AS min_icu_stay_hours,

  -- Intoxication exclusion (AUMCdb condition_concept_ids)
  [45769428, 4176983, 4174619, 4166498, 607208, 602985] AS intox_ids_aumcdb,
  48 AS intox_lb_hours,

  -- Intoxication exclusion (MIMIC ICD prefixes - handled in SQL logic)
  ['T40', 'T42', 'T43', 'T51', 'F10'] AS intox_icd_prefixes_mimic,

  -- ============================================================================
  -- 4. VARIABLE DEFINITIONS
  -- ============================================================================
  -- Structure: var, category, lb_h (lookback hours),
  --            bounds (min/max clinical and statistical - in AUMCdb units),
  --            aumcdb_ids, aumcdb_unit,
  --            mimic_ids, mimic_unit,
  --            conv_factor (mimic * factor = aumcdb), conv_formula
  -- ============================================================================
  [
    -- RENAL / METABOLIC
    STRUCT(
      'creat' AS var, 'renal' AS category, 72 AS lb_h,
      0 AS min_clin, 3000 AS max_clin, 0 AS min_stat, 3000 AS max_stat,
      [3020564] AS aumcdb_ids, 'umol/L' AS aumcdb_unit,
      [50912] AS mimic_ids, 'mg/dL' AS mimic_unit,
      88.4 AS conv_factor, 'mimic * 88.4 = aumcdb' AS conv_formula
    ),
    STRUCT(
      'urea', 'renal', 72,
      0, 300, 0, 300,
      [43534077], 'mmol/L',
      [51006], 'mg/dL',
      0.357, 'mimic * 0.357 = aumcdb (BUN to Urea)'
    ),
    STRUCT(
      'lactate', 'metabolic', 72,
      0, 30, 0, 30,
      [3047181, 3014111], 'mmol/L',
      [50813], 'mmol/L',
      1.0, NULL
    ),
    STRUCT(
      'glucose', 'metabolic', 72,
      1, 50, 1, 50,
      [3020491], 'mmol/L',
      [50809, 50931, 225664, 220621, 226537], 'mg/dL',
      0.0556, 'mimic * 0.0556 = aumcdb'
    ),

    -- ELECTROLYTES
    STRUCT(
      'potassium', 'electrolyte', 72,
      1, 10, 1, 10,
      [3005456, 3023103], 'mmol/L',
      [50822, 50971], 'mEq/L',
      1.0, NULL
    ),
    STRUCT(
      'sodium', 'electrolyte', 72,
      100, 200, 100, 200,
      [3000285, 3019550], 'mmol/L',
      [50824, 50983], 'mEq/L',
      1.0, NULL
    ),
    STRUCT(
      'chloride', 'electrolyte', 72,
      60, 150, 60, 150,
      [3018572, 3014576], 'mmol/L',
      [50806, 50902], 'mEq/L',
      1.0, NULL
    ),
    STRUCT(
      'calcium', 'electrolyte', 72,
      0.5, 4, 0.5, 4,
      [3048816], 'mmol/L',
      [50808], 'mmol/L',
      1.0, NULL
    ),
    STRUCT(
      'phosphate', 'electrolyte', 72,
      0, 6, 0, 6,
      [3003458], 'mmol/L',
      [50970], 'mg/dL',
      0.323, 'mimic * 0.323 = aumcdb'
    ),
    STRUCT(
      'magnesium', 'electrolyte', 72,
      0, 4, 0, 4,
      [3012095, 3033836], 'mmol/L',
      [50960], 'mg/dL',
      0.411, 'mimic * 0.411 = aumcdb'
    ),

    -- ACID-BASE
    STRUCT(
      'ph', 'acid_base', 72,
      6.5, 8.0, 6.5, 8.0,
      [3010421], 'pH',
      [50820], 'pH',
      1.0, NULL
    ),
    STRUCT(
      'bicarb', 'acid_base', 72,
      0, 60, 0, 60,
      [3006576], 'mmol/L',
      [50803, 50882], 'mEq/L',
      1.0, NULL
    ),
    STRUCT(
      'base_excess', 'acid_base', 72,
      -40, 40, -40, 40,
      [3012501], 'mmol/L',
      [50802], 'mEq/L',
      1.0, NULL
    ),
    STRUCT(
      'anion_gap', 'acid_base', 72,
      0, 60, 0, 60,
      [3039000], 'mmol/L',
      [50868], 'mEq/L',
      1.0, NULL
    ),

    -- RESPIRATORY
    STRUCT(
      'pao2', 'respiratory', 72,
      0, 600, 0, 600,
      [3027315], 'mmHg',
      [50821], 'mmHg',
      1.0, NULL
    ),
    STRUCT(
      'fio2', 'respiratory', 72,
      21, 100, 21, 100,
      [42869590, 3024882, 8554], '%',
      [50816, 223835], '%',
      1.0, NULL
    ),

    -- HEMATOLOGY
    STRUCT(
      'hemoglobin', 'hematology', 72,
      1, 16, 1, 16,
      [40762351], 'mmol/L',
      [50811, 51222], 'g/dL',
      0.62, 'mimic * 0.62 = aumcdb'
    ),
    STRUCT(
      'hematocrit', 'hematology', 72,
      0.10, 0.70, 0.10, 0.70,
      [42869588], 'fraction',
      [51221], '%',
      0.01, 'mimic * 0.01 = aumcdb (% to fraction)'
    ),
    STRUCT(
      'platelets', 'hematology', 72,
      0, 2000, 0, 2000,
      [3007461], '10^9/L',
      [51265], 'K/uL',
      1.0, NULL
    ),
    STRUCT(
      'wbc', 'hematology', 72,
      0, 100, 0, 100,
      [3010813], '10^9/L',
      [51301], 'K/uL',
      1.0, NULL
    ),

    -- LIVER
    STRUCT(
      'bilirubin', 'liver', 72,
      0, 1000, 0, 1000,
      [3006140, 40757494], 'umol/L',
      [50885], 'mg/dL',
      17.1, 'mimic * 17.1 = aumcdb'
    ),
    STRUCT(
      'albumin', 'liver', 72,
      10, 60, 10, 60,
      [3028286, 3024561], 'g/L',
      [50862], 'g/dL',
      10.0, 'mimic * 10 = aumcdb'
    ),

    -- INFLAMMATION
    STRUCT(
      'crp', 'inflammation', 72,
      0, 500, 0, 500,
      [3020460], 'mg/L',
      [50889], 'mg/L',
      1.0, NULL
    ),

    -- VITALS
    STRUCT(
      'map', 'vital', 72,
      10, 250, 10, 250,
      [21490852, 21492241, 21490673], 'mmHg',
      [220052, 220181, 225312], 'mmHg',
      1.0, NULL
    ),
    STRUCT(
      'heartrate', 'vital', 72,
      10, 300, 10, 300,
      [3027018, 21490872], 'bpm',
      [220045], 'bpm',
      1.0, NULL
    ),
    STRUCT(
      'temperature', 'vital', 72,
      30, 45, 30, 45,
      [21490586, 21490588, 21490870, 3022060, 3025163,3006322, 3025085], 'C',
      [223761], 'F',
      CAST(NULL AS FLOAT64), '(F-32)*5/9=C'
    )
  ] AS var_defs,

  -- ============================================================================
  -- 5. SPECIAL ID MAPPINGS
  -- ============================================================================

  -- Weight
  STRUCT(
    [3026600, 3013762, 3025315, 3023166] AS aumcdb_ids,
    [224639, 226512] AS mimic_ids,
    'kg' AS unit
  ) AS weight_ids,

  -- Urine output
  STRUCT(
    [3014315, 3016267, 21491173] AS aumcdb_ids,
    [226559, 226560, 226561, 226584, 226563, 226564,
     226565, 226567, 226557, 226558, 227488, 227489] AS mimic_ids,
    'mL' AS unit
  ) AS uo_ids,

  -- Fluid output (all)
  STRUCT(
    [3025727, 3006376, 3018767, 21491184, 3039001, 3026556, 3023454,
     3020433, 3017152, 21491180, 3008793, 4092499, 3037836, 3028277,
     21491183, 3011087, 1033562, 36032760, 3557196, 40458249,
     3014315, 40481610, 3016267, 4265527, 36033316, 21493221,
     21491173, 4264378, 36032656, 3007123] AS aumcdb_ids,
    CAST(NULL AS ARRAY<INT64>) AS mimic_ids,  -- All outputevents
    'mL' AS unit
  ) AS fluid_out_ids,

  -- Mechanical ventilation
  STRUCT(
    [3022875, 21490855] AS aumcdb_ids,
    CAST(NULL AS ARRAY<INT64>) AS mimic_ids  -- Use ${DATASET}.ventilation
  ) AS vent_ids,

  -- Bloodflow (CRRT)
  STRUCT(
    [1988610] AS aumcdb_ids,
    CAST(NULL AS ARRAY<INT64>) AS mimic_ids  -- Use ${DATASET}.crrt
  ) AS bloodflow_ids,

  -- ============================================================================
  -- 6. GCS MAPPINGS
  -- ============================================================================

  -- AUMCdb: value_as_concept_id -> score mapping
  [
    STRUCT(45880466 AS value_as_concept_id, 4 AS score, 'eye' AS component),
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
  ] AS gcs_map_aumcdb,

  -- MIMIC: itemids for GCS components (values are numeric 1-6)
  STRUCT(
    220739 AS eye_itemid,
    223900 AS verbal_itemid,
    223901 AS motor_itemid
  ) AS gcs_itemids_mimic,

  -- ============================================================================
  -- 7. VASOPRESSOR MAPPINGS
  -- ============================================================================

  -- AUMCdb vasopressors (concept_ids)
  [
    STRUCT('norepinephrine' AS vas_name, [2907531] AS concept_ids),
    STRUCT('epinephrine', [19076867, 40038600]),
    STRUCT('phenylephrine', [40074086]),
    STRUCT('dopamine', [36411287]),
    STRUCT('dobutamine', [21088391])
  ] AS vas_defs_aumcdb,

  -- MIMIC vasopressors (itemids)
  [
    STRUCT('norepinephrine' AS vas_name, 221906 AS itemid),
    STRUCT('epinephrine', 221289),
    STRUCT('phenylephrine', 221749),
    STRUCT('dopamine', 221662),
    STRUCT('dobutamine', 221653),
    STRUCT('vasopressin', 222315)
  ] AS vas_defs_mimic,

  -- All vasopressor concept_ids (AUMCdb) - flat list for filtering
  [19076867, 40038600, 21088391, 36411287, 2907531, 40074086] AS vas_ids_aumcdb

;
