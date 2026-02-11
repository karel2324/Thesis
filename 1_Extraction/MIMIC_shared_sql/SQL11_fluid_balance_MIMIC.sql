/* ===============================================================
   SQL11_fluid_balance.sql - MIMIC-IV Fluid Balance (IMPROVED)

   Key improvements over previous version:
   1. Unit conversion: handles both 'ml' AND 'L' (×1000)
   2. Bolus smoothing: instant boluses (≤1 min) spread over 1 hour
   3. Excludes cancelled/rewritten orders
   4. Uses SECOND precision for proportional allocation
   5. Fixed JOIN duplication bug: old query JOIN'd grid (multiple rows
      per stay) causing N× overcounting of fluid input (e.g. 21× for
      obs7/grid8 → +60L fluid balances). Now uses WHERE IN subquery.

   Approach: proportional allocation within lookback window.
   Mathematically equivalent to rate-based integration for totals.
   =============================================================== */

CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.grid_fluid_balance` AS
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
  SELECT
    subject_id,
    hadm_id,
    stay_id,
    grid_ts
  FROM `windy-forge-475207-e3.${DATASET}.cohort_grid`
),

/* ===============================================================
   2) FLUID OUTPUT (outputevents - all outputs including urine)
   =============================================================== */
cohort_stays AS (
  SELECT DISTINCT stay_id FROM grid
),

fluid_out_raw AS (
  SELECT DISTINCT
    oe.stay_id,
    TIMESTAMP(oe.charttime) AS ts,
    SAFE_CAST(oe.value AS FLOAT64) AS vol_ml
  FROM `physionet-data.mimiciv_3_1_icu.outputevents` oe
  WHERE oe.stay_id IN (SELECT stay_id FROM cohort_stays)
    AND oe.value IS NOT NULL
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
   3) FLUID INPUT (inputevents)

   Fixes:
   - Unit conversion: 'ml' as-is, 'L' × 1000
   - Bolus smoothing: if duration ≤ 1 minute, assume given over 1 hour
     (prevents division by zero and unrealistic spikes)
   - Exclude cancelled/rewritten orders
   =============================================================== */
fluid_in_raw AS (
  SELECT
    ie.stay_id,

    -- Bolus smoothing: if near-instantaneous, spread over 1 hour
    CASE
      WHEN TIMESTAMP_DIFF(TIMESTAMP(ie.endtime), TIMESTAMP(ie.starttime), MINUTE) <= 1
      THEN TIMESTAMP_SUB(TIMESTAMP(ie.endtime), INTERVAL 1 HOUR)
      ELSE TIMESTAMP(ie.starttime)
    END AS start_ts,

    TIMESTAMP(ie.endtime) AS end_ts,

    -- Unit conversion to ml
    CASE
      WHEN LOWER(ie.amountuom) = 'l'  THEN ie.amount * 1000.0
      WHEN LOWER(ie.amountuom) = 'ml' THEN ie.amount
      ELSE NULL
    END AS vol_ml

  FROM `physionet-data.mimiciv_3_1_icu.inputevents` ie
  WHERE ie.stay_id IN (SELECT stay_id FROM cohort_stays)
    AND ie.amount IS NOT NULL
    AND ie.amount > 0
    AND LOWER(ie.amountuom) IN ('ml', 'l')
    AND ie.starttime IS NOT NULL
    AND ie.endtime IS NOT NULL
    -- Exclude cancelled/rewritten orders
    AND LOWER(COALESCE(ie.statusdescription, '')) != 'rewritten'
),

fluid_in_lb_at_grid AS (
  SELECT
    g.subject_id,
    g.hadm_id,
    g.stay_id,
    g.grid_ts,

    IFNULL(
      SUM(
        -- Overlap fraction: how much of the infusion falls in the lookback window
        GREATEST(
          0.0,
          TIMESTAMP_DIFF(
            LEAST(g.grid_ts, e.end_ts),                                      -- window end
            GREATEST(TIMESTAMP_SUB(g.grid_ts, INTERVAL cfg.fb_lb_hours HOUR), e.start_ts), -- window start
            SECOND
          )
        )
        / NULLIF(
            TIMESTAMP_DIFF(e.end_ts, e.start_ts, SECOND),
            0
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
  WHERE e.vol_ml IS NOT NULL  -- Filter out NULL from unit conversion
     OR e.stay_id IS NULL     -- Keep grid rows without any input (LEFT JOIN)
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
