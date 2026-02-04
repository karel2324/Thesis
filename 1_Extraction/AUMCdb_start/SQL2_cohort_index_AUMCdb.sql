CREATE OR REPLACE TABLE `windy-forge-475207-e3.derived.cohort_index` AS
WITH
/* ===============================================================
   0) CONFIG
   =============================================================== */
cfg AS (
  SELECT *
  FROM `windy-forge-475207-e3.derived.cfg_params`
  LIMIT 1
),

/* ===============================================================
   1) INPUTS UIT SQL1_UTILS (single source of truth)
   =============================================================== */
visit_times AS (
  SELECT * FROM `windy-forge-475207-e3.derived.visit_times`
),
death AS (
  SELECT * FROM `windy-forge-475207-e3.derived.death`
),
dem AS (
  SELECT * FROM `windy-forge-475207-e3.derived.demographics`
),

weight_effective AS (
  SELECT * FROM `windy-forge-475207-e3.derived.weight_effective_person`
),
weight_first AS (
  SELECT * FROM `windy-forge-475207-e3.derived.weight_first_person`
),
uo_rates AS (
  SELECT * FROM `windy-forge-475207-e3.derived.uo_rates`
),

/* ===============================================================
   2) KDIGO – CREATININE EVENTS (zelfde als eerder, maar bron = meas_state util)
   =============================================================== */
creat_cids AS (
  SELECT cid
  FROM cfg
  CROSS JOIN UNNEST(cfg.var_defs) v
  CROSS JOIN UNNEST(v.aumcdb_ids) cid
  WHERE v.var = 'creat'
),

creat_raw AS (
  SELECT
    m.person_id,
    m.visit_occurrence_id,
    COALESCE(m.measurement_datetime, TIMESTAMP(m.measurement_date)) AS ts,
    SAFE_CAST(m.value_as_number AS FLOAT64) AS creat_umol,
    UNIX_SECONDS(COALESCE(m.measurement_datetime, TIMESTAMP(m.measurement_date))) AS ts_s
  FROM `amsterdamumcdb.version1_5_0.measurement` m
  WHERE m.person_id IS NOT NULL
    AND m.value_as_number IS NOT NULL
    AND ((m.provider_id BETWEEN 0 AND 99) OR m.provider_id IS NULL)
    AND m.measurement_concept_id IN (SELECT cid FROM creat_cids)
),

creat_win AS (
  SELECT
    *,
    MIN(creat_umol) OVER (
      PARTITION BY person_id, visit_occurrence_id
      ORDER BY ts_s
      RANGE BETWEEN 172800 PRECEDING AND 1 PRECEDING
    ) AS min_creat_48h,
    MIN(creat_umol) OVER (
      PARTITION BY person_id, visit_occurrence_id
      ORDER BY ts_s
      RANGE BETWEEN 604800 PRECEDING AND 1 PRECEDING
    ) AS min_creat_7d
  FROM creat_raw
),

creat_final_kdigo AS (
  SELECT
    person_id,
    visit_occurrence_id,
    ts,
    creat_umol,
    CASE
      WHEN min_creat_48h IS NULL THEN NULL
      ELSE creat_umol - min_creat_48h
    END AS abs_inc_creat_48h,
    SAFE_DIVIDE(creat_umol, min_creat_7d) AS rel_inc_creat_7d
  FROM creat_win
),

/* ===============================================================
   3) KDIGO – URINE OUTPUT EVENTS (bron = derived.uo_rates util)
   =============================================================== */
uo_events AS (
  SELECT DISTINCT
    person_id,
    visit_occurrence_id,
    t_end AS ts
  FROM uo_rates
),

all_events AS (
  SELECT person_id, visit_occurrence_id, ts FROM uo_events
  UNION DISTINCT
  SELECT person_id, visit_occurrence_id, ts FROM creat_final_kdigo
),

uo_vol_at_event AS (
  SELECT
    e.person_id,
    e.visit_occurrence_id,
    e.ts,

    SUM(
      GREATEST(
        0.0,
        TIMESTAMP_DIFF(
          LEAST(e.ts, r.t_end),
          GREATEST(TIMESTAMP_SUB(e.ts, INTERVAL 6 HOUR), r.t_start),
          MINUTE
        ) / 60.0
      ) * r.rate_ml_per_kg_per_h
    ) AS vol_6h_mlkg,

    SUM(
      GREATEST(
        0.0,
        TIMESTAMP_DIFF(
          LEAST(e.ts, r.t_end),
          GREATEST(TIMESTAMP_SUB(e.ts, INTERVAL 12 HOUR), r.t_start),
          MINUTE
        ) / 60.0
      ) * r.rate_ml_per_kg_per_h
    ) AS vol_12h_mlkg,

    SUM(
      GREATEST(
        0.0,
        TIMESTAMP_DIFF(
          LEAST(e.ts, r.t_end),
          GREATEST(TIMESTAMP_SUB(e.ts, INTERVAL 24 HOUR), r.t_start),
          MINUTE
        ) / 60.0
      ) * r.rate_ml_per_kg_per_h
    ) AS vol_24h_mlkg,

    MIN(IF(r.t_end > TIMESTAMP_SUB(e.ts, INTERVAL 6 HOUR)  AND r.t_start < e.ts, r.t_start, NULL))  AS min_tstart_6h,
    MIN(IF(r.t_end > TIMESTAMP_SUB(e.ts, INTERVAL 12 HOUR) AND r.t_start < e.ts, r.t_start, NULL)) AS min_tstart_12h,
    MIN(IF(r.t_end > TIMESTAMP_SUB(e.ts, INTERVAL 24 HOUR) AND r.t_start < e.ts, r.t_start, NULL)) AS min_tstart_24h


  FROM all_events e
  JOIN uo_rates r
    ON r.person_id = e.person_id
   AND r.visit_occurrence_id = e.visit_occurrence_id
   AND r.t_end   > TIMESTAMP_SUB(e.ts, INTERVAL 24 HOUR)
   AND r.t_start < e.ts
  GROUP BY e.person_id, e.visit_occurrence_id, e.ts
),

uo_avg_final AS (
  SELECT
    person_id,
    visit_occurrence_id,
    ts,

    CASE
      WHEN min_tstart_6h IS NOT NULL
       AND min_tstart_6h <= TIMESTAMP_SUB(ts, INTERVAL 6 HOUR)
      THEN SAFE_DIVIDE(vol_6h_mlkg, 6.0)
    END AS avg_uo_6h_ml_per_kg_per_h,

    CASE
      WHEN min_tstart_12h IS NOT NULL
       AND min_tstart_12h <= TIMESTAMP_SUB(ts, INTERVAL 12 HOUR)
      THEN SAFE_DIVIDE(vol_12h_mlkg, 12.0)
    END AS avg_uo_12h_ml_per_kg_per_h,

    CASE
      WHEN min_tstart_24h IS NOT NULL
       AND min_tstart_24h <= TIMESTAMP_SUB(ts, INTERVAL 24 HOUR)
      THEN SAFE_DIVIDE(vol_24h_mlkg, 24.0)
    END AS avg_uo_24h_ml_per_kg_per_h

  FROM uo_vol_at_event
),

/* ===============================================================
   4) KDIGO FLAGS + INDEX (t0)
   =============================================================== */
timeline AS (
  SELECT
    ev.person_id,
    ev.visit_occurrence_id,
    ev.ts,
    ua.avg_uo_6h_ml_per_kg_per_h,
    ua.avg_uo_12h_ml_per_kg_per_h,
    ua.avg_uo_24h_ml_per_kg_per_h,
    cf.creat_umol,
    cf.abs_inc_creat_48h,
    cf.rel_inc_creat_7d
  FROM all_events ev
  LEFT JOIN uo_avg_final ua
    USING (person_id, visit_occurrence_id, ts)
  LEFT JOIN creat_final_kdigo cf
    USING (person_id, visit_occurrence_id, ts)
),

flags AS (
  SELECT
    *,
    (
      (abs_inc_creat_48h >= 26.5)
      OR (rel_inc_creat_7d >= 1.5)
      OR (avg_uo_6h_ml_per_kg_per_h < 0.5)
    ) AS kdigo1_flag,
    (
      (avg_uo_12h_ml_per_kg_per_h < 0.5)
      OR (rel_inc_creat_7d >= 2.0)
    ) AS kdigo2_flag,

    (
      (creat_umol >= 353.6 AND abs_inc_creat_48h >= 26.5)
      OR (rel_inc_creat_7d >= 3)
      OR (avg_uo_24h_ml_per_kg_per_h < 0.3)
      OR (avg_uo_12h_ml_per_kg_per_h = 0.0)
    ) AS kdigo3_flag

  FROM timeline
),

kdigo1_index AS (
  SELECT
    f.person_id,
    f.visit_occurrence_id,
    MIN(f.ts) AS t0
  FROM flags f
  JOIN visit_times v
    ON v.person_id = f.person_id
   AND v.visit_occurrence_id = f.visit_occurrence_id
  WHERE f.kdigo1_flag
    AND v.admit_dt <= f.ts
  GROUP BY f.person_id, f.visit_occurrence_id
),

kdigo2_index AS (
  SELECT
    f.person_id,
    f.visit_occurrence_id,
    MIN(f.ts) AS t0
  FROM flags f
  JOIN visit_times v
    ON v.person_id = f.person_id
   AND v.visit_occurrence_id = f.visit_occurrence_id
  WHERE f.kdigo2_flag
    AND v.admit_dt <= f.ts
  GROUP BY f.person_id, f.visit_occurrence_id
),

kdigo3_index AS (
  SELECT
    f.person_id,
    f.visit_occurrence_id,
    MIN(f.ts) AS t0
  FROM flags f
  JOIN visit_times v
    ON v.person_id = f.person_id
   AND v.visit_occurrence_id = f.visit_occurrence_id
  WHERE f.kdigo3_flag
    AND v.admit_dt <= f.ts
  GROUP BY f.person_id, f.visit_occurrence_id
),

chosen_index AS (
  SELECT k1.*
  FROM kdigo1_index k1
  JOIN cfg ON cfg.inclusion_default = 'kdigo1'
  UNION ALL
  SELECT k2.*
  FROM kdigo2_index k2
  JOIN cfg ON cfg.inclusion_default = 'kdigo2'
    UNION ALL
  SELECT k3.*
  FROM kdigo3_index k3
  JOIN cfg ON cfg.inclusion_default = 'kdigo3'
),

/* ===============================================================
   5) TERMINAL STATE (death, discharge, bloodflow)
   =============================================================== */
bloodflow_raw AS (
  SELECT
    m.person_id,
    m.visit_occurrence_id,
    COALESCE(m.measurement_datetime, TIMESTAMP(m.measurement_date)) AS ts
  FROM `amsterdamumcdb.version1_5_0.measurement` m
  CROSS JOIN cfg
  WHERE m.person_id IS NOT NULL
    AND ((m.provider_id BETWEEN 0 AND 99) OR m.provider_id IS NULL)
    AND m.measurement_concept_id IN UNNEST(cfg.bloodflow_ids.aumcdb_ids)
),

bloodflow_first AS (
  SELECT
    ci.person_id,
    ci.visit_occurrence_id,
    MIN(bf.ts) AS bloodflow_dt
  FROM chosen_index ci
  JOIN bloodflow_raw bf
    ON bf.person_id = ci.person_id
   AND bf.visit_occurrence_id = ci.visit_occurrence_id
   AND bf.ts >= ci.t0
  GROUP BY ci.person_id, ci.visit_occurrence_id
),

terminal AS (
  SELECT
    ci.person_id,
    ci.visit_occurrence_id,
    ci.t0,
    dth.death_dt,
    vt.admit_dt,
    vt.discharge_dt,
    bf.bloodflow_dt,
    LEAST(
      COALESCE(dth.death_dt,    TIMESTAMP '9999-12-31'),
      COALESCE(vt.discharge_dt, TIMESTAMP '9999-12-31'),
      COALESCE(bf.bloodflow_dt, TIMESTAMP '9999-12-31')
    ) AS terminal_ts
  FROM chosen_index ci
  LEFT JOIN death dth
    ON dth.person_id = ci.person_id
  LEFT JOIN visit_times vt
    ON vt.person_id = ci.person_id
   AND vt.visit_occurrence_id = ci.visit_occurrence_id
  LEFT JOIN bloodflow_first bf
    ON bf.person_id = ci.person_id
   AND bf.visit_occurrence_id = ci.visit_occurrence_id
),

/* ===============================================================
   6) EXCLUSIONS
   =============================================================== */
-- Exclude: ICU stay too short
short_stay AS (
  SELECT v.person_id, v.visit_occurrence_id
  FROM visit_times v
  CROSS JOIN cfg
  WHERE TIMESTAMP_DIFF(v.discharge_dt, v.admit_dt, HOUR) < cfg.min_icu_stay_hours
),

-- Exclude: intoxication (cfg.intox_ids_aumcdb)
intox_events AS (
  SELECT
    c.person_id,
    c.visit_occurrence_id,
    COALESCE(c.condition_start_datetime, TIMESTAMP(c.condition_start_date)) AS intox_ts
  FROM `amsterdamumcdb.version1_5_0.condition_occurrence` c
  JOIN chosen_index ci
    ON ci.person_id = c.person_id
   AND ci.visit_occurrence_id = c.visit_occurrence_id
  JOIN cfg ON TRUE
  WHERE c.condition_concept_id IN UNNEST(cfg.intox_ids_aumcdb)
)

/* ===============================================================
   FINAL
   =============================================================== */
SELECT
  t.person_id,
  t.visit_occurrence_id,
  t.admit_dt,
  t.discharge_dt,
  t.t0,
  t.terminal_ts,
  t.death_dt,
  t.bloodflow_dt,

  dem.gender,
  dem.birth_date,

  wf.weight_kg_first  AS weight_kg_first,
  we.weight_used_kg   AS weight_used_kg,
  we.weight_source    AS weight_source,

  cfg.inclusion_default AS inclusion_applied
FROM terminal t
LEFT JOIN dem
  ON dem.person_id = t.person_id
LEFT JOIN weight_effective we
  ON we.person_id = t.person_id
LEFT JOIN weight_first wf
  ON wf.person_id = t.person_id
CROSS JOIN cfg
WHERE t.t0 <= t.terminal_ts
  AND NOT EXISTS (
    SELECT 1
    FROM short_stay ss
    WHERE ss.person_id = t.person_id
      AND ss.visit_occurrence_id = t.visit_occurrence_id
  )
  AND NOT EXISTS (
    SELECT 1
    FROM intox_events e
    WHERE e.person_id = t.person_id
      AND e.visit_occurrence_id = t.visit_occurrence_id
      AND e.intox_ts < TIMESTAMP_ADD(t.terminal_ts, INTERVAL cfg.intox_fw_hours HOUR)
  )
  AND NOT EXISTS (
    SELECT 1
    FROM bloodflow_raw bf
    WHERE bf.person_id = t.person_id
      AND bf.visit_occurrence_id = t.visit_occurrence_id
      AND bf.ts < t.t0
  )
;