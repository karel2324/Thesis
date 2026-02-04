"""
Optuna HPO - Simplified version using FQE as objective.
Reuses functions from _7_hpo.py.
"""

import numpy as np
import pandas as pd
import joblib
import torch
import d3rlpy
import optuna
from optuna.samplers import TPESampler
from d3rlpy.algos import DiscreteCQLConfig, DoubleDQNConfig, DiscreteBCQConfig, NFQConfig
from d3rlpy.ope import FQEConfig
from d3rlpy.models.encoders import VectorEncoderFactory
from utils import load_config, get_data_paths, load_mdp
from rl_training import function_fqe, get_action_frequency_per_episode

def main():
    """Main entry point for HPO study."""
    from utils import load_config, get_data_paths

    config = load_config()
    all_paths = get_data_paths(config)

    # Check whether to run AUMCdb
    if config['hpo']['databases']['aumc'] == True:
        print("AUMCdb is enabled for HPO study")
        if all_paths['aumc']['mdp_dir'].exists():
            run_optuna_hpo_for_db(all_paths['aumc'], config)
        else:
            print(f"Skipping AUMCdb: MDP directory not found")
    
    # Check whether to run MIMIC
    if config['hpo']['databases']['mimic'] == True:     
        print("MIMIC is enabled for HPO study")   
        if all_paths['mimic']['mdp_dir'].exists():
            run_optuna_hpo_for_db(all_paths['mimic'], config)
        else:
            print(f"Skipping MIMIC: MDP directory not found")



############################################################################################################
# ALGORITHM FACTORY

def create_algo(algo_name: str, trial: optuna.Trial, gamma: float, hidden_units: list, device: str):
    """Create algorithm with Optuna-suggested hyperparameters."""

    if algo_name == 'cql':
        return DiscreteCQLConfig(
            alpha=trial.suggest_float('alpha', 0.01, 10.0, log=True),
            learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            batch_size=trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
            n_critics=trial.suggest_int('n_critics', 1, 4),
            gamma=gamma,
            encoder_factory=VectorEncoderFactory(hidden_units=hidden_units),
        ).create(device=device)

    elif algo_name == 'ddqn':
        return DoubleDQNConfig(
            target_update_interval=trial.suggest_int('target_update_interval', 100, 10000),
            learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            batch_size=trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
            gamma=gamma,
            encoder_factory=VectorEncoderFactory(hidden_units=hidden_units),
        ).create(device=device)

    elif algo_name == 'bcq':
        return DiscreteBCQConfig(
            action_flexibility=trial.suggest_float('action_flexibility', 0.1, 0.5),
            beta=trial.suggest_float('beta', 0.0, 1.0),
            learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            batch_size=trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
            gamma=gamma,
            encoder_factory=VectorEncoderFactory(hidden_units=hidden_units),
        ).create(device=device)

    elif algo_name == 'nfq':
        return NFQConfig(
            learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            batch_size=trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
            gamma=gamma,
            encoder_factory=VectorEncoderFactory(hidden_units=hidden_units),
        ).create(device=device)

############################################################################################################
# MAIN

def run_optuna_hpo_for_db(db_paths: dict, config: dict):
    """Run Optuna HPO for one database."""

    db_name = db_paths['name']
    db_key = 'aumc' if 'aumc' in db_name.lower() else 'mimic'

    # Get configuration
    hpo_cfg = config.get('hpo', {})
    optuna_cfg = hpo_cfg.get('optuna', {})
    training_cfg = hpo_cfg.get(f'training_{db_key}', {})

    # Settings
    ## Optuna
    n_trials = optuna_cfg.get('n_trials', 30) # Number of trials per algorithm
    n_steps_cfg = optuna_cfg.get('n_steps', {}) 
    training_n_steps = n_steps_cfg.get('training', 10000) # Number of steps for training
    fqe_n_steps = n_steps_cfg.get('validation', 10000) # Number of steps for FQE evaluation

    ## General settings
    gamma = training_cfg.get('gamma', 0.99) # Discount factor
    hidden_units = training_cfg.get('hidden_units', [256, 256]) # Hidden units
    seed = config.get('processing', {}).get('random_state', 42) # Seed
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Get enabled algorithms
    enabled_algos = [a for a, e in hpo_cfg.get('algorithms', {}).items() if e]
    if not enabled_algos:
        print("No algorithms enabled")
        return

    print(f"\n{'='*60}\nOPTUNA HPO: {db_name}\n{'='*60}")
    print(f"Algorithms: {enabled_algos}, Trials: {n_trials}")

    # Setup
    output_dir = db_paths['reward_dir'] / "Optuna_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    mdp_name = hpo_cfg.get('mdp', 'mdp1')
    train_ds = load_mdp(db_paths, mdp_name, 'train')
    val_ds = load_mdp(db_paths, mdp_name, 'val')
    fqe_cfg = FQEConfig(learning_rate=1e-4, gamma=gamma)

    # Run for each algorithm
    results = {}
    for algo_name in enabled_algos:
        print(f"\n--- {algo_name.upper()} ---")

        def objective(trial):
            d3rlpy.seed(seed + trial.number)

            # Create and train
            algo = create_algo(algo_name, trial, gamma, hidden_units, device)
            algo.fit(train_ds, n_steps=training_n_steps, n_steps_per_epoch=2000, show_progress=True)

            # Evaluate with FQE
            fqe_isv = function_fqe(algo, val_ds, val_ds, fqe_cfg, fqe_n_steps, fqe_n_steps, device, seed)

            trial.set_user_attr('model', algo)
            trial.set_user_attr('action_freq', get_action_frequency_per_episode(algo, val_ds))
            print(f"  Trial {trial.number}: FQE={fqe_isv:.4f}")
            return fqe_isv

        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials)

        results[algo_name] = study
        study.best_trial.user_attrs['model'].save(str(output_dir / f"{algo_name}_best.d3"))
        print(f"Best: {study.best_value:.4f} | {study.best_params}")

    # Save summary
    rows = []
    for algo_name, study in results.items():
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                rows.append({'algorithm': algo_name, 'trial': t.number, 'fqe_isv': t.value, **t.params})

    pd.DataFrame(rows).sort_values('fqe_isv', ascending=False).to_csv(output_dir / "optuna_results.csv", index=False)
    print(f"\nSaved to: {output_dir}")

if __name__ == "__main__":
    main()
