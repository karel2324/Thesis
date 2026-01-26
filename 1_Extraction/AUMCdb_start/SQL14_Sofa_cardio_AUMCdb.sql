CREATE OR REPLACE TABLE `windy-forge-475207-e3.derived.sofa_cardio_24h` AS
WITH
/* =========================
   CONFIG
   ========================= */
cfg AS (
  SELECT *
  FROM `windy-forge-475207-e3.derived.cfg_params`
  LIMIT 1
),

/* =========================
   GRID
   ========================= */
grid AS (
  SELECT person_id, visit_occurrence_id, grid_ts
  FROM `windy-forge-475207-e3.derived.cohort_grid`
),

/* =========================
   1a) MAP - CURRENT WINDOW (grid_ts - 24h, grid_ts]
   ========================= */
map_24h_current AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,
    MIN(m.value_as_number) AS map_lowest_24h_current
  FROM grid g
  LEFT JOIN `windy-forge-475207-e3.derived.cohort_measurements_window_stat_valid` m
    ON m.person_id = g.person_id
   AND m.visit_occurrence_id = g.visit_occurrence_id
   AND m.var = 'map'
   AND m.ts > TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR)
   AND m.ts <= g.grid_ts
  GROUP BY g.person_id, g.visit_occurrence_id, g.grid_ts
),

/* =========================
   1b) MAP - FORWARD WINDOW (grid_ts + step - 24h, grid_ts + step]
   ========================= */
map_24h_forward AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,
    MIN(m.value_as_number) AS map_lowest_24h_forward
  FROM grid g
  CROSS JOIN cfg
  LEFT JOIN `windy-forge-475207-e3.derived.cohort_measurements_window_stat_valid` m
    ON m.person_id = g.person_id
   AND m.visit_occurrence_id = g.visit_occurrence_id
   AND m.var = 'map'
   AND m.ts > TIMESTAMP_SUB(
                TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR),
                INTERVAL 24 HOUR)
   AND m.ts <= TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR)
  GROUP BY g.person_id, g.visit_occurrence_id, g.grid_ts
),

/* =========================
   2) Vasopressoren (vas_defs_aumcdb + spuitpomp)
   Uses unified cfg_params with vas_defs_aumcdb
   ========================= */
vaso_raw AS (
  SELECT
    d.person_id,
    d.visit_occurrence_id,
    vd.vas_name,
    d.drug_exposure_start_datetime AS start_ts,
    d.drug_exposure_end_datetime   AS end_ts,
    d.sig
  FROM `amsterdamumcdb.version1_5_0.drug_exposure` d
  CROSS JOIN cfg
  CROSS JOIN UNNEST(cfg.vas_defs_aumcdb) vd
  CROSS JOIN UNNEST(vd.concept_ids) cid
  WHERE d.drug_concept_id = cid
    AND REGEXP_CONTAINS(d.route_source_value, r'^2\. Spuitpompen$')
),

/* =========================
   3) Dosis uit sig + gewicht
   ========================= */
vaso_dose AS (
  SELECT
    v.*,

    -- mg/uur uit sig
    SAFE_CAST(
      REPLACE(
        REGEXP_EXTRACT(
          LOWER(v.sig),
          r'@\s*([0-9]+(?:[.,][0-9]+)?)\s*mg\s*/\s*(?:u|uur|h)'
        ),
        ',','.'
      ) AS FLOAT64
    ) AS dose_mg_per_h,

    (
      SELECT w.weight_used_kg
      FROM `windy-forge-475207-e3.derived.weight_effective_person` w
      WHERE w.person_id = v.person_id
    ) AS weight_used_kg
  FROM vaso_raw v
),

/* =========================
   4) mg/uur → µg/kg/min
   ========================= */
vaso_dose_final AS (
  SELECT
    *,
    SAFE_DIVIDE(dose_mg_per_h * 1000.0, weight_used_kg * 60.0)
      AS dose_mcgkgmin
  FROM vaso_dose
),

/* =========================
   5) NE-equivalent dose
      phenylephrine → ÷10
   ========================= */
vaso_dose_ne_equiv AS (
  SELECT
    *,
    CASE
      WHEN vas_name = 'phenylephrine'
        THEN dose_mcgkgmin / 10.0
      ELSE dose_mcgkgmin
    END AS dose_ne_equiv_mcgkgmin
  FROM vaso_dose_final
),

/* =========================
   6) SOFA per vasopressor-event
   ========================= */
vaso_sofa AS (
  SELECT
    person_id,
    visit_occurrence_id,
    start_ts,
    end_ts,

    CASE
      -- dopamine
      WHEN vas_name = 'dopamine'
           AND dose_ne_equiv_mcgkgmin > 15 THEN 4
      WHEN vas_name = 'dopamine'
           AND dose_ne_equiv_mcgkgmin > 5  THEN 3
      WHEN vas_name = 'dopamine'           THEN 2

      -- dobutamine
      WHEN vas_name = 'dobutamine'         THEN 2

      -- norepi / epi / phenylephrine (NE-equivalent)
      WHEN vas_name IN ('norepinephrine','epinephrine','phenylephrine')
           AND dose_ne_equiv_mcgkgmin > 0.1 THEN 4
      WHEN vas_name IN ('norepinephrine','epinephrine','phenylephrine')
                                              THEN 3
      ELSE 0
    END AS sofa_cardio_vaso_inst
  FROM vaso_dose_ne_equiv
),

/* =========================
   7a) Worst vasopressor SOFA - CURRENT WINDOW (grid_ts - 24h, grid_ts]
   ========================= */
vaso_24h_current AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,
    MAX(v.sofa_cardio_vaso_inst) AS sofa_cardio_vaso_24h_current
  FROM grid g
  LEFT JOIN vaso_sofa v
    ON v.person_id = g.person_id
   AND v.visit_occurrence_id = g.visit_occurrence_id
   AND v.end_ts > TIMESTAMP_SUB(g.grid_ts, INTERVAL 24 HOUR)
   AND v.start_ts <= g.grid_ts
  GROUP BY g.person_id, g.visit_occurrence_id, g.grid_ts
),

/* =========================
   7b) Worst vasopressor SOFA - FORWARD WINDOW (grid_ts + step - 24h, grid_ts + step]
   ========================= */
vaso_24h_forward AS (
  SELECT
    g.person_id,
    g.visit_occurrence_id,
    g.grid_ts,
    MAX(v.sofa_cardio_vaso_inst) AS sofa_cardio_vaso_24h_forward
  FROM grid g
  CROSS JOIN cfg
  LEFT JOIN vaso_sofa v
    ON v.person_id = g.person_id
   AND v.visit_occurrence_id = g.visit_occurrence_id
   AND v.end_ts > TIMESTAMP_SUB(
                    TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR),
                    INTERVAL 24 HOUR)
   AND v.start_ts <= TIMESTAMP_ADD(g.grid_ts, INTERVAL cfg.grid_step_sofa_hours HOUR)
  GROUP BY g.person_id, g.visit_occurrence_id, g.grid_ts
)

/* =========================
   FINAL: combine MAP + vasopressors for BOTH windows
   ========================= */
SELECT
  g.person_id,
  g.visit_occurrence_id,
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
LEFT JOIN map_24h_current mc USING (person_id, visit_occurrence_id, grid_ts)
LEFT JOIN map_24h_forward mf USING (person_id, visit_occurrence_id, grid_ts)
LEFT JOIN vaso_24h_current vc USING (person_id, visit_occurrence_id, grid_ts)
LEFT JOIN vaso_24h_forward vf USING (person_id, visit_occurrence_id, grid_ts)
ORDER BY person_id, visit_occurrence_id, grid_ts;
