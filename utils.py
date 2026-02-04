"""
Utility functions for the RRT pipeline.
Handles config loading and path construction for BOTH databases.
"""

import yaml
import numpy as np
from pathlib import Path
from d3rlpy.dataset import MDPDataset, ReplayBuffer, InfiniteBuffer


def load_config(config_path: str = None) -> dict:
    """Load the pipeline configuration from YAML."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


# Based on configuration --> retrieve right datapath
def get_data_paths(config: dict) -> dict:
    """
    Construct data paths for BOTH AUMCdb and MIMIC based on config settings.

    Folder structure:
        derived_AUMCdb_obs7_grid4_sepsis/
        ├── Full_dataset/                    # Shared (pre-reward)
        │   ├── aumc_rrt_raw.parquet
        │   ├── aumc_rrt_imputed.parquet
        │   └── mice_imputer.joblib
        └── Reward_mpen-no_disc-1/           # Reward-specific
            ├── aumc_rrt_rewards.parquet
            ├── aumc_rrt_scaled.parquet
            ├── scaler.joblib
            └── MDP/

    Returns dict with paths for both databases:
        - aumc: dict with raw_path, imputed_path, rewards_path, mdp_dir
        - mimic: dict with raw_path, imputed_path, rewards_path, mdp_dir
    """
    # Dataset config
    dataset = config['dataset']
    obs_days = dataset['obs_days']
    grid_hours = dataset['grid_step_hours']
    inclusion = dataset['inclusion']

    # Reward config for subfolder naming
    reward_cfg = config.get('reward', {})
    m_pen = reward_cfg.get('mortality_penalty')
    rew_discount = reward_cfg.get('discount', 1.0)
    disc_str = str(rew_discount).replace('.', '')  # 1.0 → "10", 0.99 → "099"

    # Base data directory (hardcoded)
    data_dir = Path(r"c:\Users\karel\Desktop\data\Thesis\Data")

    # Reward subfolder name
    reward_folder = f"Reward_mpen-{m_pen}_disc-{disc_str}"

    # === AUMCdb paths ===
    aumc_base = data_dir / f"derived_AUMCdb_obs{obs_days}_grid{grid_hours}_{inclusion}"
    aumc_shared = aumc_base / "Full_dataset"
    aumc_reward = aumc_base / reward_folder

    aumc_paths = {
        'name': 'AUMCdb',
        # Basis path
        'base_dir': aumc_base,
        # Shared (pre-reward)
        'pre_reward_dir': aumc_shared,
        'raw_path': aumc_shared / "aumcdb_rrt_raw.parquet",
        'imputed_path': aumc_shared / "aumcdb_rrt_imputed.parquet",
        'imputer_path': aumc_shared / "mice_imputer.joblib",
        # Reward-specific
        'reward_dir': aumc_reward,
        'reward_path': aumc_reward / "aumcdb_rrt_rewards.parquet",
        'scaled_path': aumc_reward / "aumcdb_rrt_scaled.parquet",
        'scaler_path': aumc_reward / "scaler.joblib",
        'mdp_dir': aumc_reward / "MDP",
    }

    # === MIMIC paths ===
    mimic_base = data_dir / f"derived_MIMIC_obs{obs_days}_grid{grid_hours}_{inclusion}"
    mimic_shared = mimic_base / "Full_dataset"
    mimic_reward = mimic_base / reward_folder

    mimic_paths = {
        'name': 'MIMIC',
        # Basis path
        'base_dir': mimic_base,
        # Shared (pre-reward)
        'pre_reward_dir': mimic_shared,
        'raw_path': mimic_shared / "mimic_rrt_raw.parquet",
        'imputed_path': mimic_shared / "mimic_rrt_imputed.parquet",
        'imputer_path': mimic_shared / "mice_imputer.joblib",
        # Reward-specific
        'reward_dir': mimic_reward,
        'reward_path': mimic_reward / "mimic_rrt_rewards.parquet",
        'scaled_path': mimic_reward / "mimic_rrt_scaled.parquet",
        'scaler_path': mimic_reward / "scaler.joblib",
        'mdp_dir': mimic_reward / "MDP",
    }

    return {
        'aumc': aumc_paths,
        'mimic': mimic_paths,
    }


def load_mdp(db_paths: dict, mdp_name: str, split: str = 'train'):
    """
    Load MDP dataset (d3rlpy 2.x compatible).

    Args:
        db_paths: Database paths dict
        mdp_name: MDP configuration name (e.g., 'mdp1')
        split: 'train', 'val', 'test', or 'full'
    """
    path = db_paths['mdp_dir'] / f"{mdp_name}_{split}.h5"
    return ReplayBuffer.load(str(path), InfiniteBuffer())

def episodes_to_mdp(episodes) -> MDPDataset:
    """Convert list of episodes back to MDPDataset."""
    obs = np.concatenate([e.observations for e in episodes])
    acts = np.concatenate([e.actions for e in episodes])
    rews = np.concatenate([e.rewards for e in episodes])
    terms = np.concatenate([[False] * (len(e) - 1) + [True] for e in episodes])
    return MDPDataset(observations=obs, actions=acts, rewards=rews, terminals=terms)

if __name__ == "__main__":
    # Test the utilities
    config = load_config()
    paths = get_data_paths(config)

    print("=" * 60)
    print("CONFIG LOADED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nDataset settings:")
    print(f"  obs_days: {config['dataset']['obs_days']}")
    print(f"  grid_hours: {config['dataset']['grid_step_hours']}")
    print(f"  inclusion: {config['dataset']['inclusion']}")

    print(f"\nReward settings:")
    print(f"  mortality_penalty: {config['reward']['mortality_penalty']}")
    print(f"  discount: {config['reward']['discount']}")

    for db_key, db_paths in paths.items():
        print(f"\n{db_paths['name']}:")
        print(f"  [Shared folder]: {db_paths['base_dir'].parent.name}")
        print(f"  [Reward folder]: {db_paths['reward_dir'].name}")
        for name, path in db_paths.items():
            if name in ['name', 'reward_dir']:
                continue
            exists = "EXISTS" if Path(path).exists() else "NOT FOUND"
            print(f"  {name}: {path.name} [{exists}]")
