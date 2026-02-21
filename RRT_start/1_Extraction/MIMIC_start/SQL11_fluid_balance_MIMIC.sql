/* ===============================================================
   SQL11_fluid_balance.sql - MIMIC-IV Fluid Balance

   Analogue to AUMCdb SQL11_Fluid_balance.sql

   Key differences:
   - Uses inputevents for IV fluids
   - Uses outputevents for fluid output
   =============================================================== */

CREATE OR REPLACE TABLE `windy-forge-475207-e3.derived_mimic.grid_fluid_balance` AS
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
   1) GRID
   =============================================================== */
grid AS (
  SELECT
    subject_id,
    hadm_id,
    stay_id,
    grid_ts
  FROM `windy-forge-475207-e3.derived_mimic.cohort_grid`
),

/* ===============================================================
   2) FLUID OUTPUT (outputevents - all outputs including urine)
   =============================================================== */
fluid_out_raw AS (
  SELECT DISTINCT
    oe.stay_id,
    TIMESTAMP(oe.charttime) AS ts,
    SAFE_CAST(oe.value AS FLOAT64) AS vol_ml
  FROM `physionet-data.mimiciv_3_1_icu.outputevents` oe
  JOIN grid g ON g.stay_id = oe.stay_id
  WHERE oe.value IS NOT NULL
    AND oe.value > 0
),

fluid_out_lb_at_grid AS (
  SELECT
    g.subject_id,
    g.hadm_id,
    g.stay_id,
    g.grid_ts,
    IFNULL(SUM(fo.vol_ml), 0.0) AS fluid_out_fblb_ml
  FROM grid g
  CROSS JOIN cfg
  LEFT JOIN fluid_out_raw fo
    ON fo.stay_id = g.stay_id
   AND fo.ts >  TIMESTAMP_SUB(g.grid_ts, INTERVAL cfg.fb_lb_hours HOUR)
   AND fo.ts <= g.grid_ts
  GROUP BY g.subject_id, g.hadm_id, g.stay_id, g.grid_ts
),

/* ===============================================================
   3) FLUID INPUT (inputevents - IV fluids)
      Uses amountuom = 'ml' for volume-based inputs
   =============================================================== */
fluid_in_raw AS (
  SELECT
    ie.stay_id,
    TIMESTAMP(ie.starttime) AS start_ts,
    TIMESTAMP(ie.endtime) AS end_ts,
    ie.amount AS vol_ml
  FROM `physionet-data.mimiciv_3_1_icu.inputevents` ie
  JOIN grid g ON g.stay_id = ie.stay_id
  WHERE ie.amount IS NOT NULL
    AND ie.amount > 0
    AND LOWER(ie.amountuom) = 'ml'
    AND ie.starttime IS NOT NULL
    AND ie.endtime IS NOT NULL
),

fluid_in_lb_at_grid AS (
  SELECT
    g.subject_id,
    g.hadm_id,
    g.stay_id,
    g.grid_ts,

    IFNULL(
      SUM(
        GREATEST(
          0.0,
          TIMESTAMP_DIFF(
            LEAST(g.grid_ts, e.end_ts),
            GREATEST(
              TIMESTAMP_SUB(g.grid_ts, INTERVAL cfg.fb_lb_hours HOUR),
              e.start_ts
            ),
            MINUTE
          ) / 60.0
        )
        / NULLIF(
            TIMESTAMP_DIFF(e.end_ts, e.start_ts, MINUTE) / 60.0,
            0.0
          )
        * e.vol_ml
      ),
      0.0
    ) AS fluid_in_fblb_ml
  FROM grid g
  CROSS JOIN cfg
  LEFT JOIN fluid_in_raw e
    ON e.stay_id = g.stay_id
   AND e.end_ts   > TIMESTAMP_SUB(g.grid_ts, INTERVAL cfg.fb_lb_hours HOUR)
   AND e.start_ts < g.grid_ts
  GROUP BY g.subject_id, g.hadm_id, g.stay_id, g.grid_ts
)

/* ===============================================================
   FINAL: FLUID BALANCE
   =============================================================== */
SELECT
  g.subject_id,
  g.hadm_id,
  g.stay_id,
  g.grid_ts,

  fi.fluid_in_fblb_ml,
  fo.fluid_out_fblb_ml,

  fi.fluid_in_fblb_ml - fo.fluid_out_fblb_ml AS fluid_balance_fblb_ml

FROM grid g
LEFT JOIN fluid_in_lb_at_grid fi
  USING (subject_id, hadm_id, stay_id, grid_ts)
LEFT JOIN fluid_out_lb_at_grid fo
  USING (subject_id, hadm_id, stay_id, grid_ts)
ORDER BY subject_id, stay_id, grid_ts;
