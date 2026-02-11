"""
Re-run fluid balance (SQL11) + final dataframe (SQL15) for ALL MIMIC combinations.

Combinations: obs 3/5/7/14/28 x grid 8/12/24 x kdigo2/kdigo3 = 30 datasets
"""

from google.cloud import bigquery
from pathlib import Path
from itertools import product
import traceback

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT = "windy-forge-475207-e3"
DATABASE = "MIMIC"
MIN_ICU_HOURS = 24

OBS_DAYS_LIST = [3, 5, 7, 14, 28]
GRID_STEP_LIST = [8, 12, 24]
INCLUSIONS = ["kdigo2", "kdigo3"]

# =============================================================================
# PATHS
# =============================================================================
base_dir = Path(__file__).parent.parent  # Code/1_Extraction/
config_file = base_dir / "cfg_params_unified.sql"
sql_dir = base_dir / f"{DATABASE}_shared_sql"
sql11 = sql_dir / "SQL11_fluid_balance_MIMIC.sql"
sql15 = sql_dir / "SQL15_final_dataframe_MIMIC.sql"

# =============================================================================
# RUN
# =============================================================================
def run_sql(client, sql_path: Path, cfg: dict):
    """Read SQL file, substitute config placeholders, execute."""
    sql = sql_path.read_text(encoding="utf-8")
    for key, value in cfg.items():
        sql = sql.replace(f"${{{key}}}", str(value))
    print(f"    {sql_path.name}...", end=" ", flush=True)
    job = client.query(sql)
    job.result()
    print("Done")


def main():
    combinations = list(product(OBS_DAYS_LIST, GRID_STEP_LIST, INCLUSIONS))
    total = len(combinations)

    print(f"{'='*60}")
    print(f"MIMIC FLUID BALANCE RE-RUN: {total} combinations")
    print(f"  obs_days:   {OBS_DAYS_LIST}")
    print(f"  grid_steps: {GRID_STEP_LIST}")
    print(f"  inclusions: {INCLUSIONS}")
    print(f"{'='*60}")

    client = bigquery.Client(project=PROJECT, location="US")
    failed = []

    for i, (obs_days, grid_step, inclusion) in enumerate(combinations, 1):
        dataset_name = f"derived_{DATABASE}_obs{obs_days}_grid{grid_step}_{inclusion}"

        cfg = {
            "PROJECT": PROJECT,
            "DATASET": dataset_name,
            "REGION": "US",
            "COHORT_NAME": f"{DATABASE} - {inclusion}, ({obs_days}d, {grid_step}h)",
            "OBS_DAYS": obs_days,
            "GRID_STEP_HOURS": grid_step,
            "INCLUSION": inclusion,
            "MIN_ICU_HOURS": MIN_ICU_HOURS,
        }

        print(f"\n[{i}/{total}] {dataset_name}")
        try:
            run_sql(client, config_file, cfg)
            run_sql(client, sql11, cfg)
            run_sql(client, sql15, cfg)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            traceback.print_exc()
            failed.append(dataset_name)

    print(f"\n{'='*60}")
    print(f"DONE: {total - len(failed)}/{total} successful")
    if failed:
        print(f"FAILED ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
