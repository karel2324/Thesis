"""
Run Metadata Generation for Full Traceability

Provides functions to generate comprehensive metadata for each pipeline run,
including timestamps, git info, environment details, and configuration snapshots.
"""

from datetime import datetime
import subprocess
import platform
import hashlib
import json
from pathlib import Path

# Lazy imports to avoid circular dependencies
_torch = None
_d3rlpy = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _get_d3rlpy():
    global _d3rlpy
    if _d3rlpy is None:
        import d3rlpy
        _d3rlpy = d3rlpy
    return _d3rlpy


def get_git_commit() -> str:
    """Get current git commit hash (8 characters)."""
    try:
        result = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        )
        return result.decode().strip()[:8]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def get_git_branch() -> str:
    """Get current git branch name."""
    try:
        result = subprocess.check_output(
            ['git', 'branch', '--show-current'],
            stderr=subprocess.DEVNULL
        )
        return result.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def get_environment_info() -> dict:
    """Get environment information (Python, libraries, device)."""
    torch = _get_torch()
    d3rlpy = _get_d3rlpy()

    env_info = {
        'python_version': platform.python_version(),
        'd3rlpy_version': d3rlpy.__version__,
        'torch_version': torch.__version__,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    if torch.cuda.is_available():
        env_info['gpu_name'] = torch.cuda.get_device_name(0)
        env_info['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)

    return env_info


def get_run_metadata(config: dict, db_name: str) -> dict:
    """
    Generate complete run metadata for traceability.

    Args:
        config: Full configuration dictionary from config.yaml
        db_name: Database name (e.g., 'AUMCdb', 'MIMIC')

    Returns:
        Dictionary with all run metadata
    """
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    return {
        # Run identification
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),

        # Git info
        'git_commit': get_git_commit(),
        'git_branch': get_git_branch(),

        # Environment
        'environment': get_environment_info(),

        # Database & data
        'database': db_name,
        'dataset': {
            'obs_days': config['dataset']['obs_days'],
            'grid_step_hours': config['dataset']['grid_step_hours'],
            'inclusion': config['dataset']['inclusion'],
        },

        # Reward configuration
        'reward': config['reward'],

        # Processing
        'processing': {
            'random_seed': config['processing']['random_state'],
            'train_ratio': config['processing']['train_ratio'],
            'val_ratio': config['processing']['val_ratio'],
            'test_ratio': config['processing']['test_ratio'],
        },
    }


def get_config_hash(metadata: dict) -> str:
    """
    Generate hash from relevant config parameters for comparison.

    Used to determine if two runs have the same configuration.
    """
    relevant = {
        'database': metadata.get('database'),
        'dataset': metadata.get('dataset'),
        'reward': metadata.get('reward'),
    }
    return hashlib.md5(
        json.dumps(relevant, sort_keys=True).encode()
    ).hexdigest()[:8]


def should_overwrite(output_dir: Path, current_metadata: dict) -> tuple[bool, str]:
    """
    Check if existing output should be overwritten.

    Returns:
        Tuple of (should_overwrite, reason)
        - If configs match: (True, 'same_config')
        - If no existing config: (True, 'no_existing')
        - If configs differ: (False, config_hash) - use hash as suffix
    """
    config_files = list(output_dir.glob('run_*_config.json'))

    if not config_files:
        return True, 'no_existing'

    # Check most recent config
    latest_config = sorted(config_files)[-1]
    try:
        with open(latest_config) as f:
            existing = json.load(f)

        current_hash = get_config_hash(current_metadata)
        existing_hash = get_config_hash(existing)

        if current_hash == existing_hash:
            return True, 'same_config'
        else:
            return False, current_hash
    except (json.JSONDecodeError, KeyError):
        return True, 'invalid_existing'


def save_run_config(output_dir: Path, metadata: dict, training_config: dict = None, eval_config: dict = None):
    """
    Save run configuration to JSON file.

    Args:
        output_dir: Directory to save config file
        metadata: Run metadata from get_run_metadata()
        training_config: Optional training-specific config (algorithm, hyperparams)
        eval_config: Optional evaluation-specific config (FQE settings)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_data = {**metadata}

    if training_config:
        config_data['training'] = training_config

    if eval_config:
        config_data['evaluation'] = eval_config

    config_path = output_dir / f"run_{metadata['run_id']}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2, default=str)

    return config_path


def print_run_header(metadata: dict, title: str = "RUN"):
    """Print formatted run header to console."""
    env = metadata['environment']
    dataset = metadata['dataset']
    reward = metadata['reward']

    print("=" * 80)
    print(f"{title}: {metadata['run_id']} | {metadata['database']} | {metadata['git_branch']} ({metadata['git_commit']})")
    print("=" * 80)

    # Environment
    device_str = env['device']
    if env['device'] == 'cuda' and 'gpu_name' in env:
        device_str = f"cuda ({env['gpu_name']})"
    print(f"Device: {device_str}")
    print(f"d3rlpy: {env['d3rlpy_version']} | PyTorch: {env['torch_version']} | Python: {env['python_version']}")

    # Dataset
    print(f"\nDATASET:")
    print(f"  obs_days: {dataset['obs_days']} | grid_hours: {dataset['grid_step_hours']} | inclusion: {dataset['inclusion']}")

    # Reward
    print(f"\nREWARD CONFIG:")
    print(f"  structure: {reward.get('structure', 'N/A')} | mortality_penalty: {reward.get('mortality_penalty', 'N/A')} | discount: {reward.get('discount', 'N/A')}")

    print("=" * 80)


def print_training_config(n_steps: int, batch_size: int, gamma: float, hidden_units: list):
    """Print training configuration."""
    print(f"\nTRAINING CONFIG:")
    print(f"  n_steps: {n_steps} | batch_size: {batch_size} | gamma: {gamma}")
    print(f"  hidden_units: {hidden_units}")


def add_metadata_to_dict(result: dict, metadata: dict) -> dict:
    """Add standard metadata columns to a result dictionary."""
    return {
        'run_id': metadata['run_id'],
        'timestamp': metadata['timestamp'],
        'database': metadata['database'],
        'git_commit': metadata['git_commit'],
        **result
    }


def add_metadata_to_df(df, metadata: dict):
    """Add standard metadata columns to a DataFrame."""
    import pandas as pd

    # Insert at beginning
    df.insert(0, 'git_commit', metadata['git_commit'])
    df.insert(0, 'database', metadata['database'])
    df.insert(0, 'timestamp', metadata['timestamp'])
    df.insert(0, 'run_id', metadata['run_id'])

    return df
