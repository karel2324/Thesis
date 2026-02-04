"""
Reward Calculation Pipeline

Calculates for each database:
1. Terminal reward: ICU-free days / horizon (normalized to [0, 1])
2. Intermediate reward: SOFA change / max_sofa
3. Combined reward: intermediate + terminal
"""

import pandas as pd
import numpy as np
import gc


def main():
    """Main entry point for reward calculation."""
    from utils import load_config, get_data_paths

    config = load_config()
    all_paths = get_data_paths(config)

    # Check whether to run AUMCdb
    if config['databases']['aumc'] == True:
        if all_paths['aumc']['imputed_path'].exists():
            run_reward_for_db(all_paths['aumc'], config)
        else:
            print(f"Skipping {all_paths['aumc']['name']}: imputed data not found")
    
    # Check whether to run MIMIC
    if config['databases']['mimic'] == True:
        if all_paths['mimic']['imputed_path'].exists():
            run_reward_for_db(all_paths['mimic'], config)
        else:
            print(f"Skipping {all_paths['mimic']['name']}: imputed data not found")

def calculate_terminal_reward(df: pd.DataFrame, config:dict) -> pd.DataFrame:
    """Calculate terminal reward based on ICU-free days."""
    
    # Reward configurations
    reward_config = config.get('reward', {})
    horizon_days = reward_config.get('horizon_days') # Horizin days for ICUFD
    discount = reward_config.get('discount') # Discount factor from end episode --> min(discharge, death, window_end)
    mortality_penalty = reward_config.get('mortality_penalty') # How to use mortality 
    grid_step_hours = config.get('dataset', {}).get('grid_step_hours', 8) # Find grid step hours (useful for discount factor)

    print(f"\nCalculating terminal reward (ICU-free days / {horizon_days})...")

    # Episode-level aggregates
    episodes = df.groupby(['person_id', 'visit_occurrence_id']).agg(
        admit_dt=('admit_dt', 'first'),
        discharge_dt=('discharge_dt', 'first'),
        death_dt=('death_dt', 'first'),
        t0=('t0', 'first'),
        terminal_ts=('terminal_ts', 'first')
    ).reset_index()

    print(f"Episodes: {len(episodes):,}")

    # Convert to datetime
    for col in ['admit_dt', 'discharge_dt', 'death_dt', 't0', 'terminal_ts']:
        episodes[col] = pd.to_datetime(episodes[col])

    # Horizon = t0 + horizon_days
    episodes['horizon'] = episodes['t0'] + pd.Timedelta(days=horizon_days)

    # Retrieve minimum of hirzon and death
    episodes['final_horizon'] = episodes[['horizon', 'death_dt']].min(axis=1)
    
    # Retrieve end of ICU stay: discharge / end horizon / death
    episodes['icu_end_hor'] = episodes[['discharge_dt', 'horizon', 'death_dt']].min(axis=1)

    # Death within horizon observed
    episodes['death_within_obs'] = (
            episodes['death_dt'].notna() & 
            (episodes['death_dt'] <= episodes['t0'] + pd.Timedelta(days=horizon_days))
        )
    
    # ICU days within horizon
    episodes['icu_days_hor'] = (
        (episodes['icu_end_hor'] - episodes['t0']).dt.total_seconds() / 86400
    ).clip(lower=0, upper=horizon_days)

    # Total days in horizon
    total_horizon = (
        (episodes['final_horizon'] - episodes['t0']).dt.total_seconds() / 86400
    ).clip(lower=0, upper=horizon_days)

    # ICU-free days: end of maximum horizon - ICU days in horizon
    episodes['icu_free_days'] = total_horizon - episodes['icu_days_hor']


    # Discount factor
    episodes['hours_to_end_icu']= ((episodes['icu_end_hor'] - episodes['terminal_ts']).dt.total_seconds() / 3600).clip(lower=0)
    episodes['steps_to_end_icu']= (episodes['hours_to_end_icu'] / grid_step_hours).astype(int)
    episodes['discount_factor']= discount ** episodes['steps_to_end_icu']

    # 4 different outcomes (depending on config of reward function)
    # 1. Simple ICUFD:
    if mortality_penalty == 'no':
        episodes['reward_terminal_visitid'] = episodes['icu_free_days'] * episodes['discount_factor'] / horizon_days


    # 2. Mortality in ICUFD
    if mortality_penalty == 'zero':
        episodes['reward_terminal_visitid_raw'] = episodes['icu_free_days'] # ICU free days
        episodes['reward_terminal_visitid_raw2'] = np.where( # If death --> 0 ICU free days
            episodes['death_within_obs'], 
            0,
            episodes['reward_terminal_visitid_raw'])
        episodes['reward_terminal_visitid'] = episodes['reward_terminal_visitid_raw2'] * episodes['discount_factor'] / horizon_days

    # 3. Only mortality penalty
    if mortality_penalty == 'penalty':
        episodes['reward_terminal_visitid_raw'] = episodes['icu_free_days']  # ICU free days
        episodes['mortality_penalty'] = np.where( # If death --> penalty (on scale of observation horizon)
            episodes['death_within_obs'], 
            horizon_days,
            0)
        episodes['reward_terminal_visitid_raw2'] = episodes['reward_terminal_visitid_raw'] - episodes['mortality_penalty']
        episodes['reward_terminal_visitid'] = episodes['reward_terminal_visitid_raw2'] * episodes['discount_factor'] / horizon_days

    # 4. Both
    if mortality_penalty == 'both':
        episodes['reward_terminal_visitid_raw'] = episodes['icu_free_days'] # ICU free days
        episodes['reward_terminal_visitid_raw2'] = np.where( # If death --> 0 ICU free days
            episodes['death_within_obs'], 
            0,
            episodes['reward_terminal_visitid_raw'])
        episodes['mortality_penalty'] = np.where( # If death --> penalty (on scale of observation horizon)
            episodes['death_within_obs'], 
            horizon_days,
            0)
        episodes['reward_terminal_visitid_raw3'] = episodes['reward_terminal_visitid_raw2'] - episodes['mortality_penalty']
        episodes['reward_terminal_visitid'] = episodes['reward_terminal_visitid_raw3'] * episodes['discount_factor'] / horizon_days
   

    print(f"Terminal reward: [{episodes['reward_terminal_visitid'].min():.4f}, {episodes['reward_terminal_visitid'].max():.4f}]")

    # Merge back
    merge_cols = ['person_id', 'visit_occurrence_id', 'reward_terminal_visitid', 'icu_free_days', 'icu_days_hor']
    df = df.merge(episodes[merge_cols], on=['person_id', 'visit_occurrence_id'], how='left')

    # Terminal reward only on terminal step
    df['reward_terminal'] = np.where(df['is_terminal_step'], df['reward_terminal_visitid'], 0.0)

    # Clean up
    del episodes
    gc.collect()

    # Return final dataframe
    return df


def calculate_intermediate_reward(df: pd.DataFrame, config:dict) -> pd.DataFrame:
    """Calculate intermediate reward based on SOFA change."""

    # Configuration for intermediate reward
    reward_config = config.get('reward', {})
    max_sofa = reward_config.get('max_sofa') # Maximum sofa difference
    scale = reward_config.get('intermediate_reward_scale') # Scale of intermediate reward to terminal reward
    include_in_terminal_enabled = reward_config.get('intermediate_in_terminal') # Whether to include the intermediate reward in the terminal reward

    print(f"\nCalculating intermediate reward (SOFA change / {max_sofa})...")

    # Calculate intermediate reward
    df['sofa_change'] =  df['sofa_total_24h_current'] - df['sofa_total_24h_forward'] #Difference in SOFA score
    df['reward_intermediate'] = df['sofa_change'] / max_sofa * scale # Scaled to maximum sofa (and intermediate 'scale')
    df['reward_intermediate'] = df['reward_intermediate'].fillna(0) # If missing, then zero

    # If intermediate rewards should not be included in the terminal timestep
    if not include_in_terminal_enabled:
        df.loc[df['is_terminal_step'], 'reward_intermediate'] = 0

    print(f"Intermediate reward: [{df['reward_intermediate'].min():.4f}, {df['reward_intermediate'].max():.4f}]")

    # Output
    return df


def calculate_combined_reward(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate combined reward."""
    print("\nCalculating combined reward...")
    df['reward_full'] = df['reward_intermediate'] + df['reward_terminal']
    print(f"Combined reward: [{df['reward_full'].min():.4f}, {df['reward_full'].max():.4f}]")
    return df


def run_reward_for_db(db_paths: dict, config: dict):
    """Run reward calculation for one database."""

    # NAME OF DATABASE ON WHICH REWARD IS ADDED
    db_name = db_paths['name']
    
    print(f"\n{'='*50}")
    print(f"REWARD CALCULATION: {db_name}")
    print(f"{'='*50}")

    # OOUTPUT
    output_path = db_paths['reward_path']
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # INPUT 
    input_path = db_paths['imputed_path']
    print(f"Loading: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Imputed data not found: {input_path}")

    # Load data
    df = pd.read_parquet(input_path)
    print(f"Shape: {df.shape}, Visits: {df['visit_occurrence_id'].nunique()}")

    # CALCULATING REWARD
    df = calculate_terminal_reward(df, config)
    df = calculate_intermediate_reward(df, config)
    df = calculate_combined_reward(df)

    # Verification if correct assignment of rewards
    print("\nVerification:")
    non_term = ~df['is_terminal_step'] # Find all intermediate steps
    term = df['is_terminal_step'] # Find all terminal steps
    print(f"  reward_terminal on non-terminal: all zero = {(df.loc[non_term, 'reward_terminal'] == 0).all()}")
    print(f"  reward_intermediate on terminal: all zero = {(df.loc[term, 'reward_intermediate'] == 0).all()}")

    # Save
    print(f"\nSaving: {output_path}")
    df.to_parquet(output_path, index=False)
    print(f"Saved: {df.shape} ({output_path.stat().st_size / 1024**2:.1f} MB)")

    # Clean up
    gc.collect()

if __name__ == "__main__":
    main()
