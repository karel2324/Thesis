/* ===============================================================
   SQL14_sofa_cardio.sql - MIMIC-IV SOFA Cardiovascular Component

   Analogue to AUMCdb SQL14_Sofa_cardio.sql

   Uses MAP + vasopressor dose (calculated from inputevents)

   SOFA thresholds:
   - 0: MAP >= 70
   - 1: MAP < 70
   - 2: Dopamine <= 5 or any Dobutamine
   - 3: Dopamine > 5 or NE/Epi <= 0.1
   - 4: Dopamine > 15 or NE/Epi > 0.1
   =============================================================== */

CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.sofa_cardio_24h` AS
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
   1) GRID
   =============================================================== */
grid AS (
  SELECT subject_id, hadm_id, stay_id, grid_ts
  FROM `windy-forge-475207-e3.${DATASET}.cohort_grid`
),

/* ===============================================================
   2a) MAP - CURRENT WINDOW (grid_ts - 24h, grid_ts]
   =============================================================== */
map_24h_current AS (
  SELECT
    g.stay_id,
    g.grid_ts,
    MIN(m.value_as_number) AS map_lowest_24h_current
  FROM grid g
  LEFT JOIN `windy-forge-475207-e3.${DATASET}.cohort_measurements_window_stat_valid` m
    ON m.stay_id = g.stay_id
   AND m.var = 'map'
   AND m.ts > TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR)
   AND m.ts <= g.grid_ts
  GROUP BY g.stay_id, g.grid_ts
),

/* ===============================================================
   2b) MAP - FORWARD WINDOW (grid_ts + step - 24h, grid_ts + step]
   =============================================================== */
map_24h_forward AS (
  SELECT
    g.stay_id,
    g.grid_ts,
    MIN(m.value_as_number) AS map_lowest_24h_forward
  FROM grid g
  CROSS JOIN cfg
  LEFT JOIN `windy-forge-475207-e3.${DATASET}.cohort_measurements_window_stat_valid` m
    ON m.stay_id = g.stay_id
   AND m.var = 'map'
   AND m.ts > TIMESTAMP_SUB(
                TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR),
                INTERVAL 24 HOUR)
   AND m.ts <= TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR)
  GROUP BY g.stay_id, g.grid_ts
),

/* ===============================================================
   3) WEIGHT (for dose calculation)
   =============================================================== */
weight AS (
  SELECT stay_id, weight_used_kg
  FROM `windy-forge-475207-e3.${DATASET}.weight_effective_stay`
),

/* ===============================================================
   4) VASOPRESSOR EVENTS FROM INPUTEVENTS

   itemid mapping:
   - 221906: Norepinephrine
   - 221289: Epinephrine
   - 221662: Dopamine
   - 221653: Dobutamine
   - 222315: Vasopressin
   - 221749: Phenylephrine
   =============================================================== */
vaso_raw AS (
  SELECT
    ie.stay_id,
    TIMESTAMP(ie.starttime) AS starttime,
    TIMESTAMP(ie.endtime) AS endtime,
    ie.itemid,
    ie.rate,
    ie.rateuom,
    CASE
      WHEN ie.itemid = 221906 THEN 'norepinephrine'
      WHEN ie.itemid = 221289 THEN 'epinephrine'
      WHEN ie.itemid = 221662 THEN 'dopamine'
      WHEN ie.itemid = 221653 THEN 'dobutamine'
      WHEN ie.itemid = 222315 THEN 'vasopressin'
      WHEN ie.itemid = 221749 THEN 'phenylephrine'
    END AS drug_name
  FROM `physionet-data.mimiciv_3_1_icu.inputevents` ie
  WHERE ie.itemid IN (221906, 221289, 221662, 221653, 222315, 221749)
    AND ie.stay_id IN (SELECT DISTINCT stay_id FROM grid)
    AND ie.rate IS NOT NULL
    AND ie.rate > 0
    AND ie.endtime > ie.starttime
),

/* ===============================================================
   5) CALCULATE SOFA CARDIO FROM VASOPRESSORS

   Based on SOFA cardiovascular criteria:
   - NE/Epi > 0.1 mcg/kg/min = SOFA 4
   - NE/Epi <= 0.1 or Dopamine > 5 = SOFA 3
   - Dopamine <= 5 or Dobutamine = SOFA 2
   =============================================================== */
vaso_events AS (
  SELECT
    v.stay_id,
    v.starttime,
    v.endtime,
    v.drug_name,
    v.rate,
    v.rateuom,
    w.weight_used_kg,

    -- Calculate standardized dose
    CASE
      -- NE/Epi in mcg/kg/min
      WHEN v.drug_name IN ('norepinephrine', 'epinephrine') AND v.rateuom = 'mcg/kg/min' THEN v.rate
      WHEN v.drug_name IN ('norepinephrine', 'epinephrine') AND v.rateuom = 'mcg/min' THEN v.rate / NULLIF(w.weight_used_kg, 0)

      -- Dopamine in mcg/kg/min
      WHEN v.drug_name = 'dopamine' AND v.rateuom = 'mcg/kg/min' THEN v.rate
      WHEN v.drug_name = 'dopamine' AND v.rateuom = 'mcg/min' THEN v.rate / NULLIF(w.weight_used_kg, 0)

      ELSE NULL
    END AS dose_mcgkgmin,

    -- SOFA score for this instant
    CASE
      -- NE/Epi > 0.1 = SOFA 4
      WHEN v.drug_name IN ('norepinephrine', 'epinephrine')
           AND CASE
                 WHEN v.rateuom = 'mcg/kg/min' THEN v.rate
                 WHEN v.rateuom = 'mcg/min' THEN v.rate / NULLIF(w.weight_used_kg, 0)
               END > 0.1
      THEN 4

      -- NE/Epi <= 0.1 = SOFA 3
      WHEN v.drug_name IN ('norepinephrine', 'epinephrine')
           AND CASE
                 WHEN v.rateuom = 'mcg/kg/min' THEN v.rate
                 WHEN v.rateuom = 'mcg/min' THEN v.rate / NULLIF(w.weight_used_kg, 0)
               END <= 0.1
      THEN 3

      -- Dopamine > 15 = SOFA 4
      WHEN v.drug_name = 'dopamine'
           AND CASE
                 WHEN v.rateuom = 'mcg/kg/min' THEN v.rate
                 WHEN v.rateuom = 'mcg/min' THEN v.rate / NULLIF(w.weight_used_kg, 0)
               END > 15
      THEN 4

      -- Dopamine > 5 = SOFA 3
      WHEN v.drug_name = 'dopamine'
           AND CASE
                 WHEN v.rateuom = 'mcg/kg/min' THEN v.rate
                 WHEN v.rateuom = 'mcg/min' THEN v.rate / NULLIF(w.weight_used_kg, 0)
               END > 5
      THEN 3

      -- Dopamine <= 5 or Dobutamine = SOFA 2
      WHEN v.drug_name = 'dopamine' THEN 2
      WHEN v.drug_name = 'dobutamine' THEN 2

      -- Phenylephrine/Vasopressin = SOFA 3 (treated as vasopressor use)
      WHEN v.drug_name IN ('phenylephrine', 'vasopressin') THEN 3

      ELSE 0
    END AS sofa_cardio_vaso_inst

  FROM vaso_raw v
  LEFT JOIN weight w ON w.stay_id = v.stay_id
),

/* ===============================================================
   6a) Worst vasopressor SOFA - CURRENT WINDOW
   =============================================================== */
vaso_24h_current AS (
  SELECT
    g.stay_id,
    g.grid_ts,
    MAX(v.sofa_cardio_vaso_inst) AS sofa_cardio_vaso_24h_current
  FROM grid g
  LEFT JOIN vaso_events v
    ON v.stay_id = g.stay_id
   AND v.endtime > TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR)
   AND v.starttime <= g.grid_ts
  GROUP BY g.stay_id, g.grid_ts
),

/* ===============================================================
   6b) Worst vasopressor SOFA - FORWARD WINDOW
   =============================================================== */
vaso_24h_forward AS (
  SELECT
    g.stay_id,
    g.grid_ts,
    MAX(v.sofa_cardio_vaso_inst) AS sofa_cardio_vaso_24h_forward
  FROM grid g
  CROSS JOIN cfg
  LEFT JOIN vaso_events v
    ON v.stay_id = g.stay_id
   AND v.endtime > TIMESTAMP_SUB(
                    TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR),
                    INTERVAL 24 HOUR)
   AND v.starttime <= TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR)
  GROUP BY g.stay_id, g.grid_ts
)

/* ===============================================================
   FINAL: combine MAP + vasopressors for BOTH windows
   =============================================================== */
SELECT
  g.subject_id,
  g.hadm_id,
  g.stay_id,
  g.grid_ts,

  /* CURRENT SOFA CARDIO */
  GREATEST(
    IFNULL(vc.sofa_cardio_vaso_24h_current, 0),
    CASE
      WHEN mc.map_lowest_24h_current < 70 THEN 1
      ELSE 0
    END
  ) AS sofa_cardio_24h_current,

  /* FORWARD SOFA CARDIO */
  GREATEST(
    IFNULL(vf.sofa_cardio_vaso_24h_forward, 0),
    CASE
      WHEN mf.map_lowest_24h_forward < 70 THEN 1
      ELSE 0
    END
  ) AS sofa_cardio_24h_forward

FROM grid g
LEFT JOIN map_24h_current mc USING (stay_id, grid_ts)
LEFT JOIN map_24h_forward mf USING (stay_id, grid_ts)
LEFT JOIN vaso_24h_current vc USING (stay_id, grid_ts)
LEFT JOIN vaso_24h_forward vf USING (stay_id, grid_ts)
ORDER BY subject_id, stay_id, grid_ts;
