"""
MDP Preparation - Build MDPDatasets

Creates MDPDatasets for each configuration (all data, not yet split).
"""

import pandas as pd
import numpy as np
from d3rlpy.dataset import MDPDataset
import joblib

def main():
    """Main entry point for MDP preparation."""
    from utils import load_config, get_data_paths

    config = load_config()
    all_paths = get_data_paths(config)

    # Check whether to run AUMCdb
    if config['databases']['aumc'] == True:
        if all_paths['aumc']['scaled_path'].exists():
            run_mdp_prep_for_db(all_paths['aumc'], config)
        else:
            print(f"Skipping {all_paths['aumc']['name']}: scaled data not found")
    
    # Check whether to run MIMIC
    if config['databases']['mimic'] == True:
        if all_paths['mimic']['scaled_path'].exists():
            run_mdp_prep_for_db(all_paths['mimic'], config)
        else:
            print(f"Skipping {all_paths['mimic']['name']}: scaled data not found")


def get_feature_cols(df, config):
    """Identify base, missing, and time columns from config.yaml."""

    variables = config.get('variables', {}) # Retrieve variables from config.yaml
    df_cols = set(df.columns) # Look all columns in the dataframe
    base, missing, time = [], [], [] # Make different parts for the different MDPs

    # Loop over all variable names and their configurations
    for var_name, var_cfg in variables.items():

        # 1. If it should be added in the basic features --> add to base
        if var_cfg.get('basic_feature') and var_name in df_cols:
            # Skip non-numeric columns (like gender with 'M'/'F')
            if pd.api.types.is_numeric_dtype(df[var_name]):
                base.append(var_name)
            # If non-numeric, check for encoded version (e.g., gender -> gender_encoded)
            elif f"{var_name}_encoded" in df_cols:
                base.append(f"{var_name}_encoded")

        # 2. If it should be added in the missing indicators and present in df --> add to missing indicators
        if var_cfg.get('missing_indicator') and f"{var_name}_missing" in df_cols:
            missing.append(f"{var_name}_missing")

        # 3. If it should be added in the time features --> add to time features
        if var_cfg.get('time_feature'):
            # Try match first, then strip _last/_first suffix
            # e.g., creat_last -> creat_age_h, gcs_total_last -> gcs_total_last_hours
            base_name = var_name.replace('_last', '').replace('_first', '')
            for col_name in [f"{var_name}_age_h", f"{base_name}_age_h", f"{var_name}_hours", f"{base_name}_hours"]:
                if col_name in df_cols:
                    time.append(col_name)
                    break

    # Add hours_since_t0 if present and not already added
    if 'hours_since_t0' in df_cols and 'hours_since_t0' not in base:
        base.append('hours_since_t0')

    return sorted(base), sorted(missing), sorted(time)


def build_mdp(df, state_cols, reward_col, mdp_cfg, terminal_only=False):
    """Build MDPDataset from data."""
    sort_cols = mdp_cfg['episode_id_cols'] + [mdp_cfg['time_col']]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    obs = df[state_cols].values.astype(np.float32)
    acts = df[mdp_cfg['action_col']].values.astype(np.int32)
    terms = df[mdp_cfg['terminal_col']].values.astype(bool)
    rews = df[reward_col].values.astype(np.float32)
    if terminal_only:
        rews = np.where(terms, rews, 0.0)

    return MDPDataset(observations=obs, actions=acts, rewards=rews, terminals=terms)


def run_mdp_prep_for_db(db_paths: dict, config: dict):
    """Build MDPDatasets for each configuration."""

    # INPUT DATA
    input_path = db_paths.get('scaled_path') 
    df = pd.read_parquet(input_path)

    # OUTPUT DATA
    output_dir = db_paths['mdp_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    # GENERAL CONFIGURATION
    db_name = db_paths['name'] # Name of current processed database
    mdp_cfg = config['mdp'] # Look at mdp preparation

    print(f"\n{'='*50}")
    print(f"MDP PREPARATION: {db_name}")
    print(f"{'='*50}")

    # RETRIEVE FEATURES
    base_cols, missing_cols, time_cols = get_feature_cols(df, config)
    all_feature_cols = set(base_cols + missing_cols + time_cols)
    not_included = [col for col in df.columns if col not in all_feature_cols]

    # Last check to look at all features
    print(f"\nBASE FEATURES ({len(base_cols)}):")
    print(f"  {base_cols}")

    print(f"\nMISSING INDICATORS ({len(missing_cols)}):")
    print(f"  {missing_cols}")

    print(f"\nTIME FEATURES ({len(time_cols)}):")
    print(f"  {time_cols}")

    print(f"\nNOT INCLUDED ({len(not_included)}):")
    print(f"  {not_included}")

    # BUILD MDP FOR EACH MDP CONFIG
    for mdp_name in mdp_cfg['configurations']:

        # Get spefic settings for this MDP
        mdp_settings = mdp_cfg['configurations'][mdp_name]

        # Get state features fot this MDP
        feature_set = mdp_settings['state_features']
        if feature_set == 'base':
            state_cols = base_cols
        elif feature_set == 'base+missing':
            state_cols = base_cols + missing_cols
        elif feature_set == 'base+time':
            state_cols = base_cols + time_cols
        elif feature_set == 'base+missing+time':
            state_cols = base_cols + missing_cols + time_cols

        # Get reward column
        reward_type = mdp_settings['reward']
        reward_col = 'reward_terminal' if reward_type == 'terminal' else 'reward_full'
        terminal_only = (reward_type == 'terminal')

        # Build MDP
        ds = build_mdp(df= df, 
                       state_cols = state_cols, 
                       reward_col = reward_col, 
                       mdp_cfg = mdp_cfg, 
                       terminal_only = terminal_only)
        ds.dump(str(output_dir / f"{mdp_name}_full.h5"))

        # Save specifi configuration
        joblib.dump({
            'name': mdp_name, 
            'state_cols': state_cols, 
            'n_states': len(state_cols),
            'n_actions': 2, 
            'n_episodes': len(ds.episodes),
            'reward_col': reward_col, 
            'terminal_only': terminal_only,
        }, output_dir / f"{mdp_name}_config.joblib")

        print(f"  {mdp_name.upper()}: {len(state_cols)} features, {len(ds.episodes)} episodes")

    # Save outcomes
    episode_col = mdp_cfg['episode_id_cols'][-1]
    df[mdp_cfg['episode_id_cols'] + ['death_dt', 'terminal_event']].drop_duplicates(episode_col).to_parquet(output_dir / "outcomes.parquet")

    print(f"Saved to: {output_dir}")

if __name__ == "__main__":
    main()
