CREATE OR REPLACE TABLE `windy-forge-475207-e3.${DATASET}.cohort_index` AS
WITH
/* ===============================================================
   0) CONFIG
   =============================================================== */
cfg AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.cfg_params` LIMIT 1
),

/* ===============================================================
   1) INPUT TABLES
   =============================================================== */
visit_times AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.visit_times`
),
death AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.death`
),
dem AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.demographics`
),
weight_first AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.weight_first_stay`
),
weight_effective AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.weight_effective_stay`
),
age_info AS (
  SELECT * FROM `windy-forge-475207-e3.${DATASET}.icu_admission_age`
),

/* ===============================================================
   2) CKD IDENTIFICATION (ICD codes)
   =============================================================== */
ckd_stays AS (
  SELECT DISTINCT ie.stay_id
  FROM `physionet-data.mimiciv_3_1_icu.icustays` ie
  JOIN `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` dx
    ON ie.hadm_id = dx.hadm_id
  WHERE STARTS_WITH(dx.icd_code, '585')
     OR STARTS_WITH(dx.icd_code, 'N18')
),

/* ===============================================================
   3) KDIGO - CREATININE (converted to µmol/L at source)
   =============================================================== */
creat_raw AS (
  SELECT
    kc.stay_id,
    TIMESTAMP(kc.charttime) AS ts,
    kc.creat * 88.4 AS creat_umol,
    UNIX_SECONDS(TIMESTAMP(kc.charttime)) AS ts_s
  FROM `physionet-data.mimiciv_3_1_derived.kdigo_creatinine` kc
  WHERE kc.creat IS NOT NULL
),

creat_win AS (
  SELECT
    *,
    MIN(creat_umol) OVER (
      PARTITION BY stay_id
      ORDER BY ts_s
      RANGE BETWEEN 172800 PRECEDING AND 1 PRECEDING
    ) AS min_creat_48h,
    MIN(CASE WHEN creat_umol >= (SELECT v.min_clin
          FROM cfg, UNNEST(cfg.var_defs) v WHERE v.var = 'creat')
        THEN creat_umol END) OVER (
      PARTITION BY stay_id
      ORDER BY ts_s
      RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS min_creat_alltime
  FROM creat_raw
),

creat_win_adj AS (
  SELECT
    cw.stay_id,
    cw.ts,
    cw.creat_umol,
    cw.ts_s,
    cw.min_creat_48h,
    COALESCE(
      cw.min_creat_alltime,
      CASE
        WHEN ckd.stay_id IS NOT NULL THEN NULL
        ELSE POWER(
          (175.0 / 75.0)
          * POWER(GREATEST(COALESCE(ai.age_at_icu_admission, 65.0), 18.0), -0.203)
          * IF(COALESCE(d.gender, 'M') = 'F', 0.742, 1.0),
          1.0 / 1.154
        ) * 88.4
      END
    ) AS min_creat_baseline
  FROM creat_win cw
  LEFT JOIN ckd_stays ckd ON ckd.stay_id = cw.stay_id
  LEFT JOIN visit_times v ON v.stay_id = cw.stay_id
  LEFT JOIN dem d ON d.subject_id = v.subject_id
  LEFT JOIN age_info ai ON ai.stay_id = cw.stay_id
),

creat_final_kdigo AS (
  SELECT
    stay_id,
    ts,
    creat_umol,
    CASE
      WHEN min_creat_48h IS NULL THEN NULL
      ELSE creat_umol - min_creat_48h
    END AS abs_inc_creat_48h,
    SAFE_DIVIDE(creat_umol, min_creat_baseline) AS rel_inc_creat_7d
  FROM creat_win_adj
),

/* ===============================================================
   4) KDIGO - URINE OUTPUT (from MIMIC kdigo_stages built-in)
   =============================================================== */
uo_events AS (
  SELECT DISTINCT
    ks.stay_id,
    TIMESTAMP(ks.charttime) AS ts,
    ks.aki_stage_uo
  FROM `physionet-data.mimiciv_3_1_derived.kdigo_stages` ks
  WHERE ks.aki_stage_uo IS NOT NULL
),

/* ===============================================================
   5) COMBINED TIMELINE + FLAGS
   =============================================================== */
all_events AS (
  SELECT stay_id, ts FROM creat_final_kdigo
  UNION DISTINCT
  SELECT stay_id, ts FROM uo_events
),

timeline AS (
  SELECT
    ev.stay_id,
    ev.ts,
    cf.creat_umol,
    cf.abs_inc_creat_48h,
    cf.rel_inc_creat_7d,
    uo.aki_stage_uo
  FROM all_events ev
  LEFT JOIN creat_final_kdigo cf
    ON cf.stay_id = ev.stay_id AND cf.ts = ev.ts
  LEFT JOIN uo_events uo
    ON uo.stay_id = ev.stay_id AND uo.ts = ev.ts
),

flags AS (
  SELECT
    *,
    (
      (abs_inc_creat_48h >= 26.5)
      OR (rel_inc_creat_7d >= 1.5)
      OR (aki_stage_uo >= 1)
    ) AS kdigo1_flag,
    (
      (rel_inc_creat_7d >= 2.0)
      OR (aki_stage_uo >= 2)
    ) AS kdigo2_flag,
    (
      (creat_umol >= 353.6 AND abs_inc_creat_48h >= 26.5)
      OR (rel_inc_creat_7d >= 3.0)
      OR (aki_stage_uo >= 3)
    ) AS kdigo3_flag
  FROM timeline
),

/* ===============================================================
   6) KDIGO INDEX (monotone: K3 ⊂ K2 ⊂ K1)
   =============================================================== */
kdigo1_index AS (
  SELECT f.stay_id, MIN(f.ts) AS t0
  FROM flags f
  JOIN visit_times v ON v.stay_id = f.stay_id
  WHERE (f.kdigo1_flag OR f.kdigo2_flag OR f.kdigo3_flag)
    AND f.ts >= v.icu_intime
  GROUP BY f.stay_id
),

kdigo2_index AS (
  SELECT f.stay_id, MIN(f.ts) AS t0
  FROM flags f
  JOIN visit_times v ON v.stay_id = f.stay_id
  WHERE (f.kdigo2_flag OR f.kdigo3_flag)
    AND f.ts >= v.icu_intime
  GROUP BY f.stay_id
),

kdigo3_index AS (
  SELECT f.stay_id, MIN(f.ts) AS t0
  FROM flags f
  JOIN visit_times v ON v.stay_id = f.stay_id
  WHERE f.kdigo3_flag
    AND f.ts >= v.icu_intime
  GROUP BY f.stay_id
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
   7) RRT EVENTS — derived.rrt (all modalities, active only)
   =============================================================== */
rrt_raw AS (
  SELECT DISTINCT
    stay_id,
    TIMESTAMP(charttime) AS ts,
    dialysis_type AS rrt_modality
  FROM `physionet-data.mimiciv_3_1_derived.rrt`
  WHERE dialysis_active = 1
),

rrt_first AS (
  SELECT
    ci.stay_id,
    MIN(r.ts) AS rrt_start_dt
  FROM chosen_index ci
  JOIN rrt_raw r ON r.stay_id = ci.stay_id
    AND r.ts >= ci.t0
  GROUP BY ci.stay_id
),

/* ===============================================================
   8) TERMINAL STATE (death, discharge, RRT start, window_end)
   =============================================================== */
terminal AS (
  SELECT
    ci.stay_id,
    ci.t0,
    v.subject_id,
    v.hadm_id,
    TIMESTAMP(v.icu_intime) AS icu_intime,
    TIMESTAMP(v.icu_outtime) AS icu_outtime,
    TIMESTAMP(d.death_dt) AS death_dt,
    rf.rrt_start_dt,

    TIMESTAMP_ADD(ci.t0, INTERVAL cfg.obs_days DAY) AS window_end,

    LEAST(
      COALESCE(TIMESTAMP(d.death_dt), TIMESTAMP '9999-12-31'),
      COALESCE(TIMESTAMP(v.icu_outtime), TIMESTAMP '9999-12-31'),
      COALESCE(rf.rrt_start_dt, TIMESTAMP '9999-12-31'),
      TIMESTAMP_ADD(ci.t0, INTERVAL cfg.obs_days DAY)
    ) AS terminal_ts,

    CASE
      WHEN rf.rrt_start_dt IS NOT NULL
           AND rf.rrt_start_dt <= COALESCE(TIMESTAMP(d.death_dt), TIMESTAMP '9999-12-31')
           AND rf.rrt_start_dt <= COALESCE(TIMESTAMP(v.icu_outtime), TIMESTAMP '9999-12-31')
           AND rf.rrt_start_dt <= TIMESTAMP_ADD(ci.t0, INTERVAL cfg.obs_days DAY)
      THEN 'rrt_start'
      WHEN d.death_dt IS NOT NULL
           AND TIMESTAMP(d.death_dt) <= COALESCE(TIMESTAMP(v.icu_outtime), TIMESTAMP '9999-12-31')
           AND TIMESTAMP(d.death_dt) <= TIMESTAMP_ADD(ci.t0, INTERVAL cfg.obs_days DAY)
      THEN 'death'
      WHEN TIMESTAMP(v.icu_outtime) <= TIMESTAMP_ADD(ci.t0, INTERVAL cfg.obs_days DAY)
      THEN 'discharge'
      ELSE 'window_end'
    END AS terminal_event

  FROM chosen_index ci
  JOIN visit_times v ON v.stay_id = ci.stay_id
  LEFT JOIN death d ON d.subject_id = v.subject_id
  LEFT JOIN rrt_first rf ON rf.stay_id = ci.stay_id
  CROSS JOIN cfg
),

/* ===============================================================
   9) EXCLUSIONS
   =============================================================== */
prior_rrt AS (
  SELECT DISTINCT r.stay_id
  FROM rrt_raw r
  JOIN chosen_index ci ON ci.stay_id = r.stay_id
  WHERE r.ts < ci.t0
),

short_stay AS (
  SELECT v.stay_id
  FROM visit_times v
  CROSS JOIN cfg
  WHERE TIMESTAMP_DIFF(v.icu_outtime, v.icu_intime, HOUR) < cfg.min_icu_stay_hours
),

intox_stays AS (
  SELECT DISTINCT ie.stay_id
  FROM `physionet-data.mimiciv_3_1_icu.icustays` ie
  JOIN `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` dx
    ON ie.hadm_id = dx.hadm_id
  CROSS JOIN cfg
  CROSS JOIN UNNEST(cfg.intox_icd_prefixes_mimic) AS prefix
  WHERE STARTS_WITH(dx.icd_code, prefix)
)

/* ===============================================================
   FINAL OUTPUT
   =============================================================== */
SELECT
  t.subject_id,
  t.hadm_id,
  t.stay_id,
  t.icu_intime,
  t.icu_outtime,
  t.t0,
  t.terminal_ts,
  t.terminal_event,
  t.death_dt,
  t.rrt_start_dt,
  t.window_end,

  dem.gender,
  dem.anchor_age,
  age_info.age_at_icu_admission AS age_years,

  wf.weight_kg_first,
  we.weight_used_kg,
  we.weight_source,

  IF(ckd.stay_id IS NOT NULL, 1, 0) AS has_ckd,

  cfg.inclusion_default AS inclusion_applied

FROM terminal t
JOIN dem ON dem.subject_id = t.subject_id
JOIN age_info ON age_info.stay_id = t.stay_id
LEFT JOIN weight_first wf ON wf.stay_id = t.stay_id
LEFT JOIN weight_effective we ON we.stay_id = t.stay_id
LEFT JOIN ckd_stays ckd ON ckd.stay_id = t.stay_id
CROSS JOIN cfg

WHERE t.t0 <= t.terminal_ts
  AND t.stay_id NOT IN (SELECT stay_id FROM prior_rrt)
  AND t.stay_id NOT IN (SELECT stay_id FROM short_stay)
  AND t.stay_id NOT IN (SELECT stay_id FROM intox_stays)
;