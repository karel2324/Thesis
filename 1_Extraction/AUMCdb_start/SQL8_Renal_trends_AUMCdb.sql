CREATE OR REPLACE TABLE `windy-forge-475207-e3.derived.grid_renal_trends` AS
WITH
/* ===============================================================
   1) GRID (single source of truth)
   =============================================================== */
grid AS (
  SELECT
    person_id,
    visit_occurrence_id,
    grid_ts
  FROM `windy-forge-475207-e3.derived.cohort_grid`
),

/* ===============================================================
   2) LABS (creat & urea, raw)
   =============================================================== */
labs AS (
  SELECT
    person_id,
    visit_occurrence_id,
    ts,
    var,
    value_as_number AS val
  FROM `windy-forge-475207-e3.derived.cohort_measurements_window_stat_valid`
  WHERE var IN ('creat', 'urea')
),

/* ===============================================================
   3) MINIMUM BINNEN 48h EN 7d VOOR GRID
   =============================================================== */
min_windows AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,

    -- 48h minima
    MIN(IF(l.var = 'creat' AND l.ts > TIMESTAMP_SUB(g.grid_ts, INTERVAL 48 HOUR), l.val, NULL)) AS creat_min_48h,
    MIN(IF(l.var = 'urea'  AND l.ts > TIMESTAMP_SUB(g.grid_ts, INTERVAL 48 HOUR), l.val, NULL)) AS urea_min_48h,

    -- 7d baseline (for KDIGO ratio criteria)
    MIN(IF(l.var = 'creat', l.val, NULL)) AS creat_min_7d

  FROM grid g
  LEFT JOIN labs l
    ON l.person_id = g.person_id
   AND l.visit_occurrence_id = g.visit_occurrence_id
   AND l.ts > TIMESTAMP_SUB(g.grid_ts, INTERVAL 7 DAY)
   AND l.ts <= g.grid_ts

  GROUP BY g.person_id, g.visit_occurrence_id, g.grid_ts
),

/* ===============================================================
   4) LAATSTE WAARDE OP GRID
   =============================================================== */
last_vals AS (
  SELECT
    person_id,
    visit_occurrence_id,
    grid_ts,

    MAX(IF(var = 'creat', val_last, NULL)) AS creat_last,
    MAX(IF(var = 'urea',  val_last, NULL)) AS urea_last

  FROM `windy-forge-475207-e3.derived.grid_last_var_long`
  WHERE var IN ('creat', 'urea')
  GROUP BY person_id, visit_occurrence_id, grid_ts
),

/* ===============================================================
   5) URINE OUTPUT RATES (from uo_rates util)
   =============================================================== */
uo_rates AS (
  SELECT * FROM `windy-forge-475207-e3.derived.uo_rates`
),

/* ===============================================================
   6) URINE OUTPUT VOLUMES PER GRID (6h, 12h, 24h)
   =============================================================== */
uo_vol_at_grid AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,

    -- Volume in 6h window (ml/kg)
    SUM(
      GREATEST(0.0,
        TIMESTAMP_DIFF(
          LEAST(g.grid_ts, r.t_end),
          GREATEST(TIMESTAMP_SUB(g.grid_ts, INTERVAL 6 HOUR), r.t_start),
          MINUTE
        ) / 60.0
      ) * r.rate_ml_per_kg_per_h
    ) AS vol_6h_mlkg,

    -- Volume in 12h window (ml/kg)
    SUM(
      GREATEST(0.0,
        TIMESTAMP_DIFF(
          LEAST(g.grid_ts, r.t_end),
          GREATEST(TIMESTAMP_SUB(g.grid_ts, INTERVAL 12 HOUR), r.t_start),
          MINUTE
        ) / 60.0
      ) * r.rate_ml_per_kg_per_h
    ) AS vol_12h_mlkg,

    -- Volume in 24h window (ml/kg)
    SUM(
      GREATEST(0.0,
        TIMESTAMP_DIFF(
          LEAST(g.grid_ts, r.t_end),
          GREATEST(TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR), r.t_start),
          MINUTE
        ) / 60.0
      ) * r.rate_ml_per_kg_per_h
    ) AS vol_24h_mlkg,

    -- Check coverage: earliest t_start in each window
    MIN(IF(r.t_end > TIMESTAMP_SUB(g.grid_ts, INTERVAL 6 HOUR)  AND r.t_start < g.grid_ts, r.t_start, NULL)) AS min_tstart_6h,
    MIN(IF(r.t_end > TIMESTAMP_SUB(g.grid_ts, INTERVAL 12 HOUR) AND r.t_start < g.grid_ts, r.t_start, NULL)) AS min_tstart_12h,
    MIN(IF(r.t_end > TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR) AND r.t_start < g.grid_ts, r.t_start, NULL)) AS min_tstart_24h

  FROM grid g
  JOIN uo_rates r
    ON r.person_id = g.person_id
   AND r.visit_occurrence_id = g.visit_occurrence_id
   AND r.t_end > TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR)
   AND r.t_start < g.grid_ts
  GROUP BY g.person_id, g.visit_occurrence_id, g.grid_ts
),

/* ===============================================================
   7) URINE OUTPUT AVERAGES (only if full window coverage)
   =============================================================== */
uo_avg AS (
  SELECT
    person_id,
    visit_occurrence_id,
    grid_ts,

    CASE
      WHEN min_tstart_6h IS NOT NULL
       AND min_tstart_6h <= TIMESTAMP_SUB(grid_ts, INTERVAL 6 HOUR)
      THEN SAFE_DIVIDE(vol_6h_mlkg, 6.0)
    END AS avg_uo_6h,

    CASE
      WHEN min_tstart_12h IS NOT NULL
       AND min_tstart_12h <= TIMESTAMP_SUB(grid_ts, INTERVAL 12 HOUR)
      THEN SAFE_DIVIDE(vol_12h_mlkg, 12.0)
    END AS avg_uo_12h,

    CASE
      WHEN min_tstart_24h IS NOT NULL
       AND min_tstart_24h <= TIMESTAMP_SUB(grid_ts, INTERVAL 24 HOUR)
      THEN SAFE_DIVIDE(vol_24h_mlkg, 24.0)
    END AS avg_uo_24h

  FROM uo_vol_at_grid
)

/* ===============================================================
   FINAL
   =============================================================== */
SELECT
  g.person_id,
  g.visit_occurrence_id,
  g.grid_ts,

  -- absolute levels
  lv.creat_last,
  lv.urea_last,

  -- minima
  mw.creat_min_48h,
  mw.urea_min_48h,
  mw.creat_min_7d,

  -- absolute increase (48h)
  (lv.creat_last - mw.creat_min_48h) AS creat_abs_inc_48h,

  -- relative increase (48h-min based)
  SAFE_DIVIDE(lv.creat_last, mw.creat_min_48h) AS creat_rel_inc_48h,
  SAFE_DIVIDE(lv.urea_last,  mw.urea_min_48h)  AS urea_rel_inc_48h,

  -- relative increase (7d baseline)
  SAFE_DIVIDE(lv.creat_last, mw.creat_min_7d) AS creat_rel_inc_7d,

  -- KDIGO stage (0 = no AKI, 1-3 = stage)
  CASE
    -- Stage 3 (highest priority)
    WHEN (lv.creat_last >= 353.6 AND (lv.creat_last - mw.creat_min_48h) >= 26.5)
      OR SAFE_DIVIDE(lv.creat_last, mw.creat_min_7d) >= 3.0
      OR uo.avg_uo_24h < 0.3
      OR uo.avg_uo_12h = 0.0
    THEN 3

    -- Stage 2
    WHEN SAFE_DIVIDE(lv.creat_last, mw.creat_min_7d) >= 2.0
      OR uo.avg_uo_12h < 0.5
    THEN 2

    -- Stage 1
    WHEN (lv.creat_last - mw.creat_min_48h) >= 26.5
      OR SAFE_DIVIDE(lv.creat_last, mw.creat_min_7d) >= 1.5
      OR uo.avg_uo_6h < 0.5
    THEN 1

    ELSE 0
  END AS kdigo_stage

FROM grid g
LEFT JOIN last_vals lv
  USING (person_id, visit_occurrence_id, grid_ts)
LEFT JOIN min_windows mw
  USING (person_id, visit_occurrence_id, grid_ts)
LEFT JOIN uo_avg uo
  USING (person_id, visit_occurrence_id, grid_ts)
ORDER BY person_id, visit_occurrence_id, grid_ts;
