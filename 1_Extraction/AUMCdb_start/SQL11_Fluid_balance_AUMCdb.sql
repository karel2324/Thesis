CREATE OR REPLACE TABLE `windy-forge-475207-e3.derived.grid_fluid_balance` AS
WITH
/* ===============================================================
   0) CONFIG -- get configuration
   =============================================================== */
cfg AS (
  SELECT *
  FROM `windy-forge-475207-e3.derived.cfg_params`
  LIMIT 1
),

/* ===============================================================
   1) GRID (single source of truth) -- timesteps per person / visit
   =============================================================== */
grid AS (
  SELECT
    person_id,
    visit_occurrence_id,
    grid_ts
  FROM `windy-forge-475207-e3.derived.cohort_grid`
),

/* ===============================================================
   2) FLUID OUTPUT (urine + drains)
   =============================================================== */
fluid_out_raw AS ( -- Retrieve all fluid outputs
  SELECT DISTINCT -- Distinct to deduplicate
    m.person_id,
    m.visit_occurrence_id,
    m.ts,
    SAFE_CAST(m.value_as_number AS FLOAT64) AS vol_ml
  FROM `windy-forge-475207-e3.derived.cohort_measurements_window_stat_valid` m
  WHERE m.source_type IN ('fluid_out') -- All concept_ids of fluid output
),

fluid_out_lb_at_grid AS ( -- Find fluid output for each timestep, with certain lookbackperiod
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts, -- Timestep
    IFNULL(SUM(m.vol_ml), 0.0) AS fluid_out_fblb_ml -- Sum over all fluid outputs
  FROM grid g
  CROSS JOIN cfg
  LEFT JOIN fluid_out_raw m
    ON m.person_id          = g.person_id
   AND m.visit_occurrence_id = g.visit_occurrence_id
   AND m.ts >  TIMESTAMP_SUB(g.grid_ts, INTERVAL cfg.fb_lb_hours HOUR) -- Output should be after timestep - lookbackperiod
   AND m.ts <= g.grid_ts -- Output should before timestep
  GROUP BY g.person_id, g.visit_occurrence_id, g.grid_ts
),

/* ===============================================================
   3) FLUID INPUT (IV fluids via drug_exposure, time-weighted)
   =============================================================== */
fluid_in_raw AS (
  SELECT
    d.person_id,
    d.visit_occurrence_id,
    d.drug_exposure_start_datetime AS start_ts,
    d.drug_exposure_end_datetime   AS end_ts,

    CASE
      -- absolute volumes in mL
      WHEN REGEXP_CONTAINS(LOWER(d.sig), r'[0-9]+(?:\.[0-9]+)?\s*ml\b') THEN
        SAFE_CAST(
          REGEXP_EXTRACT(
            LOWER(d.sig),
            r'([0-9]+(?:\.[0-9]+)?)\s*ml\b'
          ) AS FLOAT64
        )

      -- absolute volumes in liters
      WHEN REGEXP_CONTAINS(LOWER(d.sig), r'[0-9]+(?:\.[0-9]+)?\s*l\b')
       AND NOT REGEXP_CONTAINS(LOWER(d.sig), r'l\s*/') THEN
        SAFE_CAST(
          REGEXP_EXTRACT(
            LOWER(d.sig),
            r'([0-9]+(?:\.[0-9]+)?)\s*l\b'
          ) AS FLOAT64
        ) * 1000

      ELSE NULL
    END AS vol_ml

  FROM `amsterdamumcdb.version1_5_0.drug_exposure` d
  WHERE d.sig IS NOT NULL
    AND d.drug_exposure_start_datetime IS NOT NULL
    AND d.drug_exposure_end_datetime   IS NOT NULL
    AND LOWER(d.dose_unit_source_value) IN ('ml', 'l')
),

fluid_in_lb_at_grid AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,

    IFNULL( -- Goal: find ml of infusion withing lookback period.
      SUM( -- Sum over all relevant fluid inputs
        GREATEST( -- Find total hours of infusion
          0.0,
          TIMESTAMP_DIFF( -- Find total hours of infusion
            LEAST(g.grid_ts, e.end_ts), -- Find end point within lookback period, if endpoint of infusion is after timestep, censor at timestep
            GREATEST( -- Find starting point within lookback period, censor at maximum lookback
              TIMESTAMP_SUB(g.grid_ts, INTERVAL cfg.fb_lb_hours HOUR),
              e.start_ts
            ),
            MINUTE
          ) / 60.0
        )
        / NULLIF( -- Total infusion time of drug
            TIMESTAMP_DIFF(e.end_ts, e.start_ts, MINUTE) / 60.0,
            0.0
          )
        * e.vol_ml -- * volume of drug infusion
      ),
      0.0
    ) AS fluid_in_fblb_ml
  FROM grid g
  CROSS JOIN cfg
  LEFT JOIN fluid_in_raw e
    ON e.person_id          = g.person_id
   AND e.visit_occurrence_id = g.visit_occurrence_id
   AND e.end_ts   > TIMESTAMP_SUB(g.grid_ts, INTERVAL cfg.fb_lb_hours HOUR)
   AND e.start_ts < g.grid_ts
  GROUP BY g.person_id, g.visit_occurrence_id, g.grid_ts
)

/* ===============================================================
   FINAL: FLUID BALANCE
   =============================================================== */
SELECT
  g.person_id,
  g.visit_occurrence_id,
  g.grid_ts,

  fi.fluid_in_fblb_ml,
  fo.fluid_out_fblb_ml,

  fi.fluid_in_fblb_ml
    - fo.fluid_out_fblb_ml AS fluid_balance_fblb_ml

FROM grid g
LEFT JOIN fluid_in_lb_at_grid  fi
  USING (person_id, visit_occurrence_id, grid_ts)
LEFT JOIN fluid_out_lb_at_grid fo
  USING (person_id, visit_occurrence_id, grid_ts)
ORDER BY g.person_id, g.visit_occurrence_id, g.grid_ts;
