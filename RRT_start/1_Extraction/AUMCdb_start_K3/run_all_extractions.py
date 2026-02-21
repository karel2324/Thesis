"""
Run all extraction combinations automatically.

Combinations:
- Database: MIMIC, AUMCdb (2)
- OBS days: 3, 5, 7, 14, 28 (5)
- GRID: 8, 12, 24 (3)
- INCLUSION: kdigo2, kdigo3 (2)

Total: 2 * 5 * 3 * 2 = 60 combinations
"""

from google.cloud import bigquery
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from itertools import product
import json
import re
import traceback

# =============================================================================
# CONFIGURATION - All combinations to run
# =============================================================================
DATABASES = ["MIMIC", "AUMCdb"]
OBS_DAYS_LIST = [7, 14, 28]
GRID_STEP_LIST = [8, 12]
INCLUSIONS = ["kdigo2", "kdigo3"]

MIN_ICU_HOURS = 24
OUTPUT_BASE = r"C:\Users\karel\Desktop\data\Thesis\Data"
PROJECT = "windy-forge-475207-e3"

# Only run specific SQL numbers (e.g., [7, 15]). Empty list = run all.
SQL_ONLY = [7,8, 15]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_sort_key(filename: str) -> tuple:
    """Extract SQL number for sorting (e.g., SQL5_1 -> (5, 1), SQL14 -> (14, 0))."""
    match = re.search(r'SQL(\d+)(?:_(\d+))?', filename)
    if match:
        main = int(match.group(1))
        sub = int(match.group(2)) if match.group(2) else 0
        return (main, sub)
    return (999, 0)


def run_extraction(database: str, obs_days: int, grid_step: int, inclusion: str):
    """Run a single extraction for the given parameters."""

    # Setup
    region = "EU" if database == "AUMCdb" else "US"
    dataset_name = f"derived_{database}_obs{obs_days}_grid{grid_step}_{inclusion}"

    config = {
        "PROJECT": PROJECT,
        "DATASET": dataset_name,
        "REGION": region,
        "COHORT_NAME": f"{database} - {inclusion}, ({obs_days}d, {grid_step}h)",
        "OBS_DAYS": obs_days,
        "GRID_STEP_HOURS": grid_step,
        "INCLUSION": inclusion,
        "MIN_ICU_HOURS": MIN_ICU_HOURS,
        "OUTPUT_DIR": f"{OUTPUT_BASE}\\{dataset_name}\\Full_dataset",
    }

    print(f"\n{'='*70}")
    print(f"STARTING: {config['COHORT_NAME']}")
    print(f"Dataset:  {dataset_name}")
    print(f"{'='*70}")

    # Paths
    base_dir = Path(__file__).parent.parent
    config_file = base_dir / "cfg_params_unified.sql"
    sql_dir = base_dir / f"{database}_shared_sql"

    # BigQuery client
    client = bigquery.Client(project=PROJECT, location=region)

    # Create dataset
    dataset_id = f"{PROJECT}.{dataset_name}"
    dataset = bigquery.Dataset(dataset_id)
    dataset.location = region
    client.create_dataset(dataset, exists_ok=True)
    print(f"Dataset {dataset_name} ready ({region} region)")

    # Run SQL function
    def run_sql(sql_path: Path):
        sql = sql_path.read_text(encoding="utf-8")
        for key, value in config.items():
            placeholder = f"${{{key}}}"
            sql = sql.replace(placeholder, str(value))
        job = client.query(sql)
        job.result()
        print(f"  {sql_path.name} - Done")

    # Run SQL files
    sql_files = sorted(sql_dir.glob("*.sql"), key=lambda f: get_sort_key(f.name))

    if SQL_ONLY:
        sql_files = [f for f in sql_files if get_sort_key(f.name)[0] in SQL_ONLY]
        print(f"\nRunning {len(sql_files)} SQL files (filtered: {SQL_ONLY})...")
    else:
        print(f"\nRunning {len(sql_files) + 1} SQL files...")
        run_sql(config_file)

    for sql_file in sql_files:
        run_sql(sql_file)

    print("SQL execution complete!")

    # Load data and calculate stats
    print("\nCalculating statistics...")
    query = f"""
    SELECT *
    FROM `{PROJECT}.{dataset_name}.grid_master_all_features`
    """
    df = client.query(query).to_dataframe()

    # Basic stats
    total_rows = len(df)
    total_actions = df['action_rrt'].sum()
    action_freq = (total_actions / total_rows) * 100 if total_rows > 0 else 0
    n_stays = df['visit_occurrence_id'].nunique()
    n_persons = df['person_id'].nunique()

    # Terminal events distribution
    df['hours_since_t0'] = (df['grid_ts'] - df['t0']).dt.total_seconds() / 3600
    terminal_df = df[df['is_terminal_step'] == True].copy()

    terminal_dist = terminal_df.groupby(['hours_since_t0', 'terminal_event']).size().unstack(fill_value=0)
    terminal_dist_dict = terminal_dist.to_dict(orient='index')
    terminal_dist_json = {str(int(k)): v for k, v in terminal_dist_dict.items()}

    # Create general features
    general_features = {
        "dataset_name": dataset_name,
        "database": database,
        "inclusion": inclusion,
        "obs_days": obs_days,
        "grid_step_hours": grid_step,
        "min_icu_hours": MIN_ICU_HOURS,
        "unique_stays": int(n_stays),
        "unique_persons": int(n_persons),
        "total_rows": int(total_rows),
        "total_actions": int(total_actions),
        "action_frequency_pct": round(action_freq, 4),
        "terminal_events_total": terminal_df['terminal_event'].value_counts().to_dict(),
        "terminal_events_by_grid_step": terminal_dist_json,
        "extraction_timestamp": datetime.now(timezone.utc).isoformat()
    }

    # Save locally as JSON
    output_dir = Path(config["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{dataset_name}_general_features.json"
    with open(json_path, 'w') as f:
        json.dump(general_features, f, indent=2)
    print(f"General features saved: {json_path}")

    # Save to BigQuery
    features_df = pd.DataFrame([{
        "dataset_name": dataset_name,
        "database": database,
        "inclusion": inclusion,
        "obs_days": obs_days,
        "grid_step_hours": grid_step,
        "min_icu_hours": MIN_ICU_HOURS,
        "unique_stays": n_stays,
        "unique_persons": n_persons,
        "total_rows": total_rows,
        "total_actions": total_actions,
        "action_frequency_pct": round(action_freq, 4),
        "terminal_death": int(terminal_df[terminal_df['terminal_event'] == 'death'].shape[0]),
        "terminal_discharge": int(terminal_df[terminal_df['terminal_event'] == 'discharge'].shape[0]),
        "terminal_rrt_start": int(terminal_df[terminal_df['terminal_event'] == 'rrt_start'].shape[0]),
        "terminal_window_end": int(terminal_df[terminal_df['terminal_event'] == 'window_end'].shape[0]) if 'window_end' in terminal_df['terminal_event'].values else 0,
        "extraction_timestamp": datetime.now(timezone.utc)
    }])

    table_id = f"{PROJECT}.{dataset_name}.general_features"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    client.load_table_from_dataframe(features_df, table_id, job_config=job_config).result()
    print(f"General features saved to BigQuery: {table_id}")

    # Save dataset as parquet
    output_path = output_dir / f"{database.lower()}_rrt_raw.parquet"
    df.to_parquet(output_path)
    print(f"Dataset saved: {output_path}")

    # Summary
    print(f"\n--- SUMMARY ---")
    print(f"Unique stays:     {n_stays:,}")
    print(f"Unique persons:   {n_persons:,}")
    print(f"Total rows:       {total_rows:,}")
    print(f"Action frequency: {action_freq:.2f}%")
    print(f"Terminal events:  {terminal_df['terminal_event'].value_counts().to_dict()}")

    return {
        "dataset_name": dataset_name,
        "status": "success",
        "n_stays": n_stays,
        "total_rows": total_rows,
        "action_freq": action_freq
    }


def main():
    """Run all combinations."""

    # Generate all combinations
    combinations = list(product(DATABASES, OBS_DAYS_LIST, GRID_STEP_LIST, INCLUSIONS))
    total = len(combinations)

    print(f"{'#'*70}")
    print(f"# BATCH EXTRACTION - {total} combinations")
    print(f"#")
    print(f"# Databases:  {DATABASES}")
    print(f"# OBS days:   {OBS_DAYS_LIST}")
    print(f"# Grid steps: {GRID_STEP_LIST}")
    print(f"# Inclusions: {INCLUSIONS}")
    print(f"{'#'*70}")

    results = []
    failed = []

    for i, (database, obs_days, grid_step, inclusion) in enumerate(combinations, 1):
        print(f"\n\n{'*'*70}")
        print(f"* COMBINATION {i}/{total}")
        print(f"* {database} | obs={obs_days}d | grid={grid_step}h | {inclusion}")
        print(f"{'*'*70}")

        try:
            result = run_extraction(database, obs_days, grid_step, inclusion)
            results.append(result)
        except Exception as e:
            error_msg = f"{database}_obs{obs_days}_grid{grid_step}_{inclusion}: {str(e)}"
            print(f"\n!!! ERROR: {error_msg}")
            traceback.print_exc()
            failed.append(error_msg)

    # Final summary
    print(f"\n\n{'='*70}")
    print(f"BATCH EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"Successful: {len(results)}/{total}")
    print(f"Failed:     {len(failed)}/{total}")

    if failed:
        print(f"\nFailed extractions:")
        for f in failed:
            print(f"  - {f}")

    # Save summary
    summary_path = Path(OUTPUT_BASE) / "extraction_summary.json"
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_combinations": total,
        "successful": len(results),
        "failed": len(failed),
        "results": results,
        "errors": failed
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
