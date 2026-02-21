/* ===============================================================
   SQL2_cohort.sql - MIMIC-IV Cohort Selection

   Analogue to AUMCdb SQL2_cohort_index.sql

   Key differences:
   - Uses physionet-data.mimiciv_3_1_derived.kdigo_stages (pre-computed)
   - Uses stay_id as primary identifier
   - Uses physionet-data.mimiciv_3_1_derived.crrt for RRT events

   Cohort criteria:
   - KDIGO stage 2+ (configurable)
   - No prior RRT before t0
   - Minimum ICU stay
   - No intoxication (optional)
   =============================================================== */

CREATE OR REPLACE TABLE `windy-forge-475207-e3.derived_mimic.cohort_index` AS
WITH
/* ===============================================================
   0) CONFIG
   =============================================================== */
cfg AS (
  SELECT * FROM `windy-forge-475207-e3.derived_mimic.cfg_params` LIMIT 1
),

/* ===============================================================
   1) INPUT TABLES
   =============================================================== */
visit_times AS (
  SELECT * FROM `windy-forge-475207-e3.derived_mimic.visit_times`
),
death AS (
  SELECT * FROM `windy-forge-475207-e3.derived_mimic.death`
),
dem AS (
  SELECT * FROM `windy-forge-475207-e3.derived_mimic.demographics`
),
weight_first AS (
  SELECT * FROM `windy-forge-475207-e3.derived_mimic.weight_first_stay`
),
weight_effective AS (
  SELECT * FROM `windy-forge-475207-e3.derived_mimic.weight_effective_stay`
),
age_info AS (
  SELECT * FROM `windy-forge-475207-e3.derived_mimic.icu_admission_age`
),

/* ===============================================================
   2) KDIGO STAGES (from MIMIC-IV 3.1 derived)
      physionet-data.mimiciv_3_1_derived.kdigo_stages has:
      - stay_id, charttime, aki_stage_creat, aki_stage_uo, aki_stage
   =============================================================== */
kdigo_raw AS (
  SELECT
    k.stay_id,
    TIMESTAMP(k.charttime) AS ts,
    k.aki_stage,
    k.aki_stage_creat,
    k.aki_stage_uo
  FROM `physionet-data.mimiciv_3_1_derived.kdigo_stages` k
  WHERE k.stay_id IS NOT NULL
    AND k.charttime IS NOT NULL
),

/* ===============================================================
   3) KDIGO INDEX (t0) - first time reaching KDIGO threshold
   =============================================================== */
kdigo1_index AS (
  SELECT
    k.stay_id,
    TIMESTAMP(MIN(k.ts)) AS t0
  FROM kdigo_raw k
  JOIN visit_times v ON v.stay_id = k.stay_id
  WHERE k.aki_stage >= 1
    AND k.ts >= v.icu_intime
  GROUP BY k.stay_id
),

kdigo2_index AS (
  SELECT
    k.stay_id,
    TIMESTAMP(MIN(k.ts)) AS t0
  FROM kdigo_raw k
  JOIN visit_times v ON v.stay_id = k.stay_id
  WHERE k.aki_stage >= 2
    AND k.ts >= v.icu_intime
  GROUP BY k.stay_id
),

-- Choose based on config
chosen_index AS (
  SELECT k1.stay_id, k1.t0
  FROM kdigo1_index k1
  CROSS JOIN cfg
  WHERE cfg.inclusion_default = 'kdigo1'

  UNION ALL

  SELECT k2.stay_id, k2.t0
  FROM kdigo2_index k2
  CROSS JOIN cfg
  WHERE cfg.inclusion_default = 'kdigo2'
),

/* ===============================================================
   4) CRRT / RRT EVENTS (from MIMIC-IV derived)
   =============================================================== */
crrt_raw AS (
  SELECT
    stay_id,
    TIMESTAMP(charttime) AS ts
  FROM `physionet-data.mimiciv_3_1_derived.crrt`
  WHERE stay_id IS NOT NULL
    AND charttime IS NOT NULL
),

crrt_first AS (
  SELECT
    ci.stay_id,
    MIN(cr.ts) AS crrt_start_dt
  FROM chosen_index ci
  JOIN crrt_raw cr ON cr.stay_id = ci.stay_id
    AND cr.ts >= ci.t0
  GROUP BY ci.stay_id
),

/* ===============================================================
   5) TERMINAL STATE (death, discharge, CRRT start)
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
    TIMESTAMP(cf.crrt_start_dt) AS crrt_start_dt,

    -- Terminal timestamp = earliest of death/discharge/CRRT
    LEAST(
      COALESCE(TIMESTAMP(d.death_dt), TIMESTAMP '9999-12-31'),
      COALESCE(TIMESTAMP(v.icu_outtime), TIMESTAMP '9999-12-31'),
      COALESCE(TIMESTAMP(cf.crrt_start_dt), TIMESTAMP '9999-12-31')
    ) AS terminal_ts,

    -- Terminal event type
    CASE
      WHEN cf.crrt_start_dt IS NOT NULL
           AND TIMESTAMP(cf.crrt_start_dt) <= COALESCE(TIMESTAMP(d.death_dt), TIMESTAMP '9999-12-31')
           AND TIMESTAMP(cf.crrt_start_dt) <= COALESCE(TIMESTAMP(v.icu_outtime), TIMESTAMP '9999-12-31')
      THEN 'rrt_start'
      WHEN d.death_dt IS NOT NULL
           AND TIMESTAMP(d.death_dt) <= COALESCE(TIMESTAMP(v.icu_outtime), TIMESTAMP '9999-12-31')
      THEN 'death'
      ELSE 'discharge'
    END AS terminal_event

  FROM chosen_index ci
  JOIN visit_times v ON v.stay_id = ci.stay_id
  LEFT JOIN death d ON d.subject_id = v.subject_id
  LEFT JOIN crrt_first cf ON cf.stay_id = ci.stay_id
),

/* ===============================================================
   6) EXCLUSIONS
   =============================================================== */
-- Exclude: prior CRRT before t0
prior_crrt AS (
  SELECT DISTINCT cr.stay_id
  FROM crrt_raw cr
  JOIN chosen_index ci ON ci.stay_id = cr.stay_id
  WHERE cr.ts < ci.t0
),

-- Exclude: ICU stay too short
short_stay AS (
  SELECT v.stay_id
  FROM visit_times v
  CROSS JOIN cfg
  WHERE TIMESTAMP_DIFF(v.icu_outtime, v.icu_intime, HOUR) < cfg.min_icu_stay_hours
),

-- Exclude: intoxication (based on ICD codes)
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
  t.crrt_start_dt,

  dem.gender,
  dem.anchor_age,
  age_info.age_at_icu_admission AS age_years,

  wf.weight_kg_first,
  we.weight_used_kg,
  we.weight_source,

  cfg.inclusion_default AS inclusion_applied

FROM terminal t
JOIN dem ON dem.subject_id = t.subject_id
JOIN age_info ON age_info.stay_id = t.stay_id
LEFT JOIN weight_first wf ON wf.stay_id = t.stay_id
LEFT JOIN weight_effective we ON we.stay_id = t.stay_id
CROSS JOIN cfg

-- Apply exclusions
WHERE t.t0 <= t.terminal_ts
  AND t.stay_id NOT IN (SELECT stay_id FROM prior_crrt)
  AND t.stay_id NOT IN (SELECT stay_id FROM short_stay)
  AND t.stay_id NOT IN (SELECT stay_id FROM intox_stays)
;
