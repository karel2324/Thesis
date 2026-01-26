CREATE OR REPLACE TABLE
  `windy-forge-475207-e3.derived.cohort_measurements_window_stat_valid` AS
SELECT
  *
FROM `windy-forge-475207-e3.derived.cohort_measurements_window`
WHERE
  -- ondergrens (indien gedefinieerd)
  (
    min_val_stat IS NULL
    OR value_as_number >= min_val_stat
  )
  AND
  -- bovengrens (indien gedefinieerd)
  (
    max_val_stat IS NULL
    OR value_as_number <= max_val_stat
  )
ORDER BY
  person_id,
  visit_occurrence_id,
  ts;
