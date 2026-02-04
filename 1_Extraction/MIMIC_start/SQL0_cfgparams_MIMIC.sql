/* ===============================================================
   SQL0_cfgparams.sql - MIMIC-IV Configuration Parameters

   Analogue to AUMCdb SQL0_cfgparams.SQL

   Key differences with AUMCdb:
   - Uses itemid instead of concept_id
   - Labs in labevents, vitals in chartevents
   - Has derived tables (kdigo_stages, crrt, norepinephrine_equivalent_dose)
   - Uses stay_id as primary ICU identifier
   =============================================================== */

CREATE OR REPLACE TABLE `windy-forge-475207-e3.derived_mimic.cfg_params` AS
SELECT
  ------------------------------------------------------------------
  -- 1. VARIABELE DEFINITIES (labs / vitals)
  --     source: 'lab' = labevents, 'chart' = chartevents
  ------------------------------------------------------------------
  [
    -- RENAL / METABOLIC (labevents)
    STRUCT('creat' AS var, [50912] AS itemids, 'lab' AS source, 48 AS lb_h,
           0 AS min_val_clin, 25 AS max_val_clin,      -- mg/dL in MIMIC
           0 AS min_val_stat, 25 AS max_val_stat),

    STRUCT('urea' AS var, [51006] AS itemids, 'lab' AS source, 48 AS lb_h,
           0 AS min_val_clin, 200 AS max_val_clin, 0 AS min_val_stat, 200 AS max_val_stat),  -- BUN mg/dL

    STRUCT('lactate' AS var, [50813] AS itemids, 'lab' AS source, 48 AS lb_h,
           0 AS min_val_clin, 30 AS max_val_clin, 0 AS min_val_stat, 30 AS max_val_stat),   -- mmol/L

    STRUCT('potassium' AS var, [50822, 50971] AS itemids, 'lab' AS source, 48 AS lb_h,
           1 AS min_val_clin, 10 AS max_val_clin, 1 AS min_val_stat, 10 AS max_val_stat),   -- mEq/L

    STRUCT('ph' AS var, [50820] AS itemids, 'lab' AS source, 48 AS lb_h,
           6.5 AS min_val_clin, 8.0 AS max_val_clin, 6.5 AS min_val_stat, 8.0 AS max_val_stat),

    STRUCT('bicarb' AS var, [50803, 50882] AS itemids, 'lab' AS source, 48 AS lb_h,
           0 AS min_val_clin, 60 AS max_val_clin, 0 AS min_val_stat, 60 AS max_val_stat),   -- mEq/L

    STRUCT('sodium' AS var, [50824, 50983] AS itemids, 'lab' AS source, 48 AS lb_h,
           100 AS min_val_clin, 180 AS max_val_clin, 100 AS min_val_stat, 180 AS max_val_stat),  -- mEq/L

    STRUCT('chloride' AS var, [50806, 50902] AS itemids, 'lab' AS source, 48 AS lb_h,
           60 AS min_val_clin, 150 AS max_val_clin, 60 AS min_val_stat, 150 AS max_val_stat),  -- mEq/L

    STRUCT('phosphate' AS var, [50970] AS itemids, 'lab' AS source, 48 AS lb_h,
           0 AS min_val_clin, 15 AS max_val_clin, 0 AS min_val_stat, 15 AS max_val_stat),   -- mg/dL

    STRUCT('calcium' AS var, [50808] AS itemids, 'lab' AS source, 48 AS lb_h,
           4 AS min_val_clin, 15 AS max_val_clin, 4 AS min_val_stat, 15 AS max_val_stat),   -- mg/dL (total)

    STRUCT('magnesium' AS var, [50960] AS itemids, 'lab' AS source, 48 AS lb_h,
           0 AS min_val_clin, 5 AS max_val_clin, 0 AS min_val_stat, 5 AS max_val_stat),     -- mg/dL

    STRUCT('base_excess' AS var, [50802] AS itemids, 'lab' AS source, 48 AS lb_h,
           -30 AS min_val_clin, 30 AS max_val_clin, -30 AS min_val_stat, 30 AS max_val_stat),  -- mEq/L

    STRUCT('anion_gap' AS var, [50868] AS itemids, 'lab' AS source, 48 AS lb_h,
           0 AS min_val_clin, 40 AS max_val_clin, 0 AS min_val_stat, 40 AS max_val_stat),   -- mEq/L

    STRUCT('hemoglobin' AS var, [50811, 51222] AS itemids, 'lab' AS source, 48 AS lb_h,
           3 AS min_val_clin, 20 AS max_val_clin, 3 AS min_val_stat, 20 AS max_val_stat),   -- g/dL

    STRUCT('bilirubin' AS var, [50885] AS itemids, 'lab' AS source, 48 AS lb_h,
           0 AS min_val_clin, 50 AS max_val_clin, 0 AS min_val_stat, 50 AS max_val_stat),   -- mg/dL (total)

    STRUCT('pao2' AS var, [50821] AS itemids, 'lab' AS source, 48 AS lb_h,
           0 AS min_val_clin, 600 AS max_val_clin, 0 AS min_val_stat, 600 AS max_val_stat), -- mmHg

    STRUCT('fio2' AS var, [50816, 223835] AS itemids, 'lab' AS source, 48 AS lb_h,  -- 50816=lab, 223835=chart
           21 AS min_val_clin, 100 AS max_val_clin, 21 AS min_val_stat, 100 AS max_val_stat),  -- %

    STRUCT('platelets' AS var, [51265] AS itemids, 'lab' AS source, 48 AS lb_h,
           0 AS min_val_clin, 1500 AS max_val_clin, 0 AS min_val_stat, 1500 AS max_val_stat),  -- K/uL

    STRUCT('albumin' AS var, [50862] AS itemids, 'lab' AS source, 48 AS lb_h,
           0 AS min_val_clin, 6 AS max_val_clin, 0 AS min_val_stat, 6 AS max_val_stat),     -- g/dL

    STRUCT('crp' AS var, [50889] AS itemids, 'lab' AS source, 48 AS lb_h,
           0 AS min_val_clin, 500 AS max_val_clin, 0 AS min_val_stat, 500 AS max_val_stat), -- mg/L

    STRUCT('glucose' AS var, [50809, 50931] AS itemids, 'lab' AS source, 48 AS lb_h,
           0 AS min_val_clin, 1000 AS max_val_clin, 0 AS min_val_stat, 1000 AS max_val_stat),  -- mg/dL

    STRUCT('hematocrit' AS var, [51221] AS itemids, 'lab' AS source, 48 AS lb_h,
           10 AS min_val_clin, 60 AS max_val_clin, 10 AS min_val_stat, 60 AS max_val_stat), -- %

    STRUCT('wbc' AS var, [51301] AS itemids, 'lab' AS source, 48 AS lb_h,
           0 AS min_val_clin, 100 AS max_val_clin, 0 AS min_val_stat, 100 AS max_val_stat), -- K/uL

    -- VITALS (chartevents)
    STRUCT('map' AS var, [220052, 220181, 225312] AS itemids, 'chart' AS source, 48 AS lb_h,
           20 AS min_val_clin, 200 AS max_val_clin, 20 AS min_val_stat, 200 AS max_val_stat),  -- mmHg

    STRUCT('heartrate' AS var, [220045] AS itemids, 'chart' AS source, 48 AS lb_h,
           20 AS min_val_clin, 250 AS max_val_clin, 20 AS min_val_stat, 250 AS max_val_stat),  -- bpm

    STRUCT('temperature' AS var, [223761, 223762] AS itemids, 'chart' AS source, 48 AS lb_h,
           90 AS min_val_clin, 110 AS max_val_clin, 90 AS min_val_stat, 110 AS max_val_stat)   -- Fahrenheit (223761) / Celsius (223762)

  ] AS var_defs,


  ------------------------------------------------------------------
  -- 2. WEIGHT ITEMIDS (chartevents)
  ------------------------------------------------------------------
  [762, 763, 224639, 226512, 226531] AS weight_ids,    -- Daily Weight, Admit Wt, etc.


  ------------------------------------------------------------------
  -- 3. URINE OUTPUT ITEMIDS (outputevents)
  ------------------------------------------------------------------
  [
    226559,  -- Foley
    226560,  -- Void
    226561,  -- Condom Cath
    226584,  -- Ileoconduit
    226563,  -- Suprapubic
    226564,  -- R Nephrostomy
    226565,  -- L Nephrostomy
    226567,  -- Straight Cath
    226557,  -- R Ureteral Stent
    226558,  -- L Ureteral Stent
    227488,  -- GU Irrigant Volume In
    227489   -- GU Irrigant/Urine Volume Out
  ] AS uo_ids,


  ------------------------------------------------------------------
  -- 4. GCS ITEMIDS (chartevents) - waarden zijn numeriek in MIMIC
  ------------------------------------------------------------------
  [
    STRUCT(220739 AS itemid, 'eye'    AS component),  -- GCS - Eye Opening
    STRUCT(223900 AS itemid, 'verbal' AS component),  -- GCS - Verbal Response
    STRUCT(223901 AS itemid, 'motor'  AS component)   -- GCS - Motor Response
  ] AS gcs_map,


  ------------------------------------------------------------------
  -- 5. VASOPRESSOR DRUG ITEMIDS (voor inputevents indien nodig)
  --     NB: We gebruiken primair derived.norepinephrine_equivalent_dose
  ------------------------------------------------------------------
  [
    STRUCT(221906 AS itemid, 'norepinephrine' AS vas_name),
    STRUCT(221289 AS itemid, 'epinephrine' AS vas_name),
    STRUCT(221749 AS itemid, 'phenylephrine' AS vas_name),
    STRUCT(221662 AS itemid, 'dopamine' AS vas_name),
    STRUCT(221653 AS itemid, 'dobutamine' AS vas_name),
    STRUCT(222315 AS itemid, 'vasopressin' AS vas_name)
  ] AS vas_map,


  ------------------------------------------------------------------
  -- 6. GLOBALE TIJDSPARAMETERS
  ------------------------------------------------------------------
  28  AS obs_days,               -- Max observatiedagen na t0
  8   AS grid_step_hours,        -- Tijdstap voor grid
  8   AS grid_step_sofa_hours,   -- Tijdstap voor SOFA berekening
  24  AS fb_lb_hours,            -- Fluid balance lookback
  48  AS gcs_lb_hours,           -- GCS lookback


  ------------------------------------------------------------------
  -- 7. INCLUSIELOGICA
  --     'kdigo1' = KDIGO stage 1+
  --     'kdigo2' = KDIGO stage 2+
  ------------------------------------------------------------------
  'kdigo2' AS inclusion,


  ------------------------------------------------------------------
  -- 8. EXCLUSIELOGICA
  ------------------------------------------------------------------
  24 AS min_icu_stay_hours,      -- Minimum ICU verblijf

  -- ICD codes voor intoxicatie exclusie (ICD-10)
  [
    'T40',   -- Poisoning by narcotics
    'T42',   -- Poisoning by antiepileptic, sedative-hypnotic
    'T43',   -- Poisoning by psychotropic drugs
    'T51',   -- Toxic effect of alcohol
    'F10'    -- Alcohol related disorders
  ] AS intox_icd_prefixes
;
