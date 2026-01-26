CREATE OR REPLACE TABLE `windy-forge-475207-e3.derived.grid_urine_output` AS
WITH
/* ===============================================================
   1) GRID
   =============================================================== */
grid AS (
  SELECT
    person_id,
    visit_occurrence_id,
    grid_ts
  FROM `windy-forge-475207-e3.derived.cohort_grid`
),

/* ===============================================================
   2) UO RATES (uit SQL2)
   =============================================================== */
uo_rates AS (
  SELECT
    person_id,
    visit_occurrence_id,
    t_start,
    t_end,
    rate_ml_per_kg_per_h,
    rate_ml_per_h
  FROM `windy-forge-475207-e3.derived.uo_rates`
),

/* ===============================================================
   3) INTEGRATIE NAAR GRID
   =============================================================== */
uo_at_grid AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,

    -- ml/kg/h (6h)
    SUM(
      GREATEST(
        0.0,
        TIMESTAMP_DIFF(
          LEAST(g.grid_ts, r.t_end),
          GREATEST(TIMESTAMP_SUB(g.grid_ts, INTERVAL 6 HOUR), r.t_start),
          MINUTE
        ) / 60.0
      ) * r.rate_ml_per_kg_per_h
    ) AS uo_6h_mlkg,

    -- ml/kg/h (12h)
    SUM(
      GREATEST(
        0.0,
        TIMESTAMP_DIFF(
          LEAST(g.grid_ts, r.t_end),
          GREATEST(TIMESTAMP_SUB(g.grid_ts, INTERVAL 12 HOUR), r.t_start),
          MINUTE
        ) / 60.0
      ) * r.rate_ml_per_kg_per_h
    ) AS uo_12h_mlkg,

    -- ml/kg/h (24h)
    SUM(
      GREATEST(
        0.0,
        TIMESTAMP_DIFF(
          LEAST(g.grid_ts, r.t_end),
          GREATEST(TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR), r.t_start),
          MINUTE
        ) / 60.0
      ) * r.rate_ml_per_kg_per_h
    ) AS uo_24h_mlkg,

    -- absolute ml (24h)
    SUM(
      GREATEST(
        0.0,
        TIMESTAMP_DIFF(
          LEAST(g.grid_ts, r.t_end),
          GREATEST(TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR), r.t_start),
          MINUTE
        ) / 60.0
      ) * r.rate_ml_per_h
    ) AS uo_24h_ml

  FROM grid g
  LEFT JOIN uo_rates r
    ON r.person_id = g.person_id
   AND r.visit_occurrence_id = g.visit_occurrence_id
   AND r.t_end   > TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR)
   AND r.t_start < g.grid_ts

  GROUP BY
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts
)

SELECT *
FROM uo_at_grid
ORDER BY person_id, visit_occurrence_id, grid_ts;
