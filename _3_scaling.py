"""
Scaling Pipeline

Applies StandardScaler (or MinMaxScaler) to continuous variables.
Binary and discrete variables are NOT scaled.

Which variables to scale is determined by the 'scale' parameter in config.yaml.
Missing indicators (_missing columns) are NEVER scaled.
Time features (_age_h, _hours) follow the scale setting of their parent variable.

Settings controlled via config.yaml scaling section.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def main():
    """Main entry point for scaling."""
    from utils import load_config, get_data_paths

    config = load_config()
    all_paths = get_data_paths(config)

    # Look whether AUMC is enabled 
    aumc_enabled = config['databases']['aumc'] 

    # Look whether MIMIC is enabled 
    mimic_enabled = config['databases']['mimic'] 

    # If both enabled, then MIMIC is scaled on scaling info of AUMCdb
    if aumc_enabled and mimic_enabled:
        # Both enabled: fit on AUMCdb, apply to MIMIC
        print("Both databases enabled: fitting scaler on AUMCdb, applying to both")
        scaler_info = run_scaling_for_db(all_paths['aumc'], config, scaler_info=None)
        run_scaling_for_db(all_paths['mimic'], config, scaler_info=scaler_info)

    # If only AUMC enabled, then just scale AUMCdb
    elif aumc_enabled and not mimic_enabled:
        # Only AUMCdb
        run_scaling_for_db(all_paths['aumc'], config, scaler_info=None)

    # If only MIMIC enabled, then just scale MIMIC
    elif mimic_enabled and not aumc_enabled:
        # Only MIMIC
        run_scaling_for_db(all_paths['mimic'], config, scaler_info=None)

    # If both not enabled
    else:
        print("No databases enabled or data not found")


def get_columns_to_scale(df: pd.DataFrame, config: dict) -> list:
    """
    Determine which columns should be scaled based on config.

    Rules:
    - Variables with scale: true in config -> scale
    - Variables with scale: false in config -> don't scale
    - Missing indicators (_missing) -> NEVER scale
    - Time features (_age_h, _hours) -> ALWAYS scale (continuous time values)
    """
    variables = config.get('variables', {})
    columns_to_scale = []

    # Columns that should NEVER be scaled
    skip_cols = {
        # IDs and metadata
        'person_id', 'visit_occurrence_id', 'time_step', 'icustay_id', 'hadm_id',
        'grid_ts', 't0', 'terminal_ts', 'death_dt', 'birth_date', 'admit_dt', 'discharge_dt',
        # MDP components (action, terminal, rewards)
        'action', 'action_rrt', 'terminal', 'is_terminal_step', 'died_in_hosp',
        'reward', 'reward_terminal', 'reward_intermediate', 'reward_full',
        'reward_terminal_visitid', 'sofa_change', 'icu_free_days', 'icu_days_hor',
        # Categorical
        'gender', 'terminal_event', 'inclusion_applied',
    }

    for col in df.columns:
        # Skip non-feature columns
        if col in skip_cols:
            continue

        # Skip non-numeric columns (strings, objects, etc.)
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Missing indicators are NEVER scaled
        if col.endswith('_missing'):
            continue

        # Check if this is a time feature
        is_time_feature = col.endswith('_age_h') or col.endswith('_hours')

        if is_time_feature:
            # Time features are ALWAYS scaled (continuous time values)
            columns_to_scale.append(col)
        else:
            # Check if variable is in config
            var_config = variables.get(col, {})
            if var_config.get('scale'):
                columns_to_scale.append(col)

    return columns_to_scale


def run_scaling_for_db(db_paths: dict, config: dict, scaler_info: dict = None):
    """
    Run scaling for one database.

    Args:
        db_paths: Database paths dict
        config: Pipeline config
        scaler_info: Optional pre-fitted scaler dict. If provided, uses this scaler
                     instead of fitting a new one (for external validation).

    Returns:
        scaler_info dict (for reuse on other databases)
    """
    db_name = db_paths['name']
    scaling_cfg = config.get('scaling', {})

    print(f"\n{'='*50}")
    print(f"SCALING: {db_name}")
    print(f"{'='*50}")

    # Load data
    if not db_paths['reward_path'].exists():
        raise FileNotFoundError(f"Reward data not found: {db_paths['reward_path']}")

    df = pd.read_parquet(db_paths['reward_path'])
    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} cols")

    # Get columns to scale
    columns_to_scale = get_columns_to_scale(df, config)
    columns_not_scaled = [col for col in df.columns if col not in columns_to_scale]
    method = scaling_cfg.get('method')

    # Print scaling summary
    print(f"\nSCALED ({len(columns_to_scale)} columns):")
    print(f"  {columns_to_scale}")

    print(f"\nNOT SCALED ({len(columns_not_scaled)} columns):")
    print(f"  {columns_not_scaled}")

    # 1. If there is no database scaled before:
    if scaler_info is None:
        # Fit new scaler
        scaler = StandardScaler() if method == 'standard' else NameError(f"Unknown scaling method: {method}")
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        print(f"Fit + transformed {len(columns_to_scale)} columns ({method})")
        print(f"To check the scaling is equal: scaler means: {scaler.mean_[:5]}")

        scaler_info = {
            'scaler': scaler,
            'columns_to_scale': columns_to_scale,
            'method': method,
        }
    
    # 2. If another database has been scaled before
    elif scaler_info is not None:
        # Use pre-fitted scaler (external validation)
        scaler = scaler_info['scaler']
        df[columns_to_scale] = scaler.transform(df[columns_to_scale])
        print(f"Transformed {len(columns_to_scale)} columns (using pre-fitted scaler)")
        print(f"To check the scaling is equal: same scaler means?: {scaler.mean_[:5]}")

    # Save (create directory if needed)
    db_paths['reward_dir'].mkdir(parents=True, exist_ok=True)
    df.to_parquet(db_paths['scaled_path'])
    joblib.dump(scaler_info, db_paths['scaler_path'])
    print(f"Saved: {db_paths['scaled_path'].name}")

    return scaler_info


if __name__ == "__main__":
    main()
