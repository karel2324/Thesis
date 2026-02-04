"""
External Validation Pipeline

Validates trained models on a different database (cross-database validation).
e.g., Train on AUMCdb, validate on MIMIC (or vice versa).

Compares best RL algorithm with Behavior Cloning.
Includes subgroup analysis for clinically relevant populations.

Settings controlled via config.yaml external_validation section.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import h5py
import torch
import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteBCConfig
from d3rlpy.ope import DiscreteFQE, FQEConfig
from d3rlpy.metrics import InitialStateValueEstimationEvaluator
from d3rlpy.models.encoders import VectorEncoderFactory


def load_h5_dataset(path: Path) -> MDPDataset:
    """Load dataset from h5 file (d3rlpy 2.x format)."""
    with h5py.File(path, 'r') as f:
        indices = sorted(set(
            int(k.split('_')[-1]) for k in f.keys() if k.startswith('observations_')
        ))

        obs_list, act_list, rew_list, term_list = [], [], [], []
        for i in indices:
            obs = f[f'observations_{i}'][:]
            act = f[f'actions_{i}'][:]
            rew = f[f'rewards_{i}'][:]
            term_ds = f[f'terminated_{i}']
            term = np.array([term_ds[()]]) if term_ds.shape == () else term_ds[:]

            obs_list.append(obs)
            act_list.append(act)
            rew_list.append(rew)
            term_arr = np.zeros(len(obs), dtype=bool)
            term_arr[-1] = bool(term.any() if hasattr(term, 'any') else term)
            term_list.append(term_arr)

        observations = np.concatenate(obs_list).astype(np.float32)
        actions = np.concatenate(act_list).astype(np.int32).flatten()
        rewards = np.concatenate(rew_list).astype(np.float32).flatten()
        terminals = np.concatenate(term_list).astype(bool)

    return MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )


def evaluate_with_fqe(policy, dataset, fqe_lr: float, fqe_steps: int, gamma: float, device: str):
    """Evaluate policy using FQE. Returns Initial State Value."""
    fqe_config = FQEConfig(learning_rate=fqe_lr, gamma=gamma)
    fqe = DiscreteFQE(algo=policy, config=fqe_config, device=device)

    fqe.fit(
        dataset,
        n_steps=fqe_steps,
        n_steps_per_epoch=fqe_steps,
        experiment_name="fqe_external",
        show_progress=False,
    )

    isv_evaluator = InitialStateValueEstimationEvaluator(episodes=dataset.episodes)
    return isv_evaluator(fqe, dataset)


def bootstrap_fqe(policy, dataset, fqe_config, n_bootstrap: int, fqe_steps: int,
                  device: str, seed: int = 42):
    """Bootstrap FQE following Hao et al. (2022)."""
    np.random.seed(seed)
    episodes = dataset.episodes
    n_episodes = len(episodes)

    # Train FQE on original data
    fqe_original = DiscreteFQE(algo=policy, config=fqe_config, device=device)
    fqe_original.fit(dataset, n_steps=fqe_steps, n_steps_per_epoch=fqe_steps,
                     experiment_name="fqe_ext_boot_orig", show_progress=False)
    isv_evaluator = InitialStateValueEstimationEvaluator(episodes=episodes)
    v_original = isv_evaluator(fqe_original, dataset)

    # Bootstrap
    errors = []
    for i in range(n_bootstrap):
        idx = np.random.choice(n_episodes, size=n_episodes, replace=True)
        sampled_episodes = [episodes[j] for j in idx]

        obs = np.concatenate([ep.observations for ep in sampled_episodes])
        acts = np.concatenate([ep.actions for ep in sampled_episodes])
        rews = np.concatenate([ep.rewards for ep in sampled_episodes])
        terms = np.concatenate([np.append(np.zeros(len(ep)-1, dtype=bool), True)
                               for ep in sampled_episodes])
        bootstrap_ds = MDPDataset(observations=obs, actions=acts, rewards=rews, terminals=terms)

        fqe_boot = DiscreteFQE(algo=policy, config=fqe_config, device=device)
        fqe_boot.fit(bootstrap_ds, n_steps=fqe_steps, n_steps_per_epoch=fqe_steps,
                     experiment_name=f"fqe_ext_boot_{i}", show_progress=False)

        v_boot = isv_evaluator(fqe_boot, dataset)
        errors.append(v_boot - v_original)

    q_low, q_high = np.percentile(errors, [2.5, 97.5])
    ci_low = v_original - q_high
    ci_high = v_original - q_low

    return v_original, ci_low, ci_high


def compute_metrics(policy, dataset, metrics_cfg: dict, feature_df: pd.DataFrame = None):
    """Compute metrics for a policy on a dataset."""
    obs = np.concatenate([ep.observations for ep in dataset.episodes])
    actions = np.concatenate([ep.actions for ep in dataset.episodes])

    predictions = policy.predict(obs)

    result = {}

    if metrics_cfg.get('action_match', True):
        result['action_match'] = (predictions == actions).mean()

    if metrics_cfg.get('rrt_rate', True):
        result['rrt_rate'] = (predictions == 1).mean()
        result['data_rrt_rate'] = (actions == 1).mean()

    if metrics_cfg.get('q_values', True):
        # Get Q-values for analysis
        q_values = policy.predict_value(obs)
        result['mean_q_no_rrt'] = q_values[:, 0].mean() if q_values.ndim > 1 else q_values.mean()
        result['mean_q_rrt'] = q_values[:, 1].mean() if q_values.ndim > 1 else q_values.mean()

    return result, predictions


def run_external_validation(config: dict, all_paths: dict):
    """Run external validation across databases."""
    ext_cfg = config.get('external_validation', {})
    proc_cfg = config.get('processing', {})

    print(f"\n{'='*60}")
    print("EXTERNAL VALIDATION")
    print(f"{'='*60}")

    # Get settings
    source_cfg = ext_cfg.get('source', {})
    target_cfg = ext_cfg.get('target', {})
    algo_cfg = ext_cfg.get('algorithms', {})
    metrics_cfg = ext_cfg.get('metrics', {})
    fqe_cfg = ext_cfg.get('fqe', {})
    bootstrap_cfg = ext_cfg.get('bootstrap', {})
    subgroup_cfg = ext_cfg.get('subgroup_analysis', {})
    output_cfg = ext_cfg.get('output', {})

    source_db = source_cfg.get('database', 'aumc')
    source_mdp = source_cfg.get('mdp', 'mdp1')
    target_db = target_cfg.get('database', 'mimic')
    target_splits = target_cfg.get('eval_splits', ['val', 'test'])

    print(f"\nSource: {source_db.upper()} ({source_mdp})")
    print(f"Target: {target_db.upper()} ({target_splits})")

    seed = proc_cfg.get('random_state', 42)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Get paths
    source_paths = all_paths.get(source_db)
    target_paths = all_paths.get(target_db)

    if source_paths is None:
        raise ValueError(f"Source database '{source_db}' not found in paths")
    if target_paths is None:
        raise ValueError(f"Target database '{target_db}' not found in paths")

    # Output directory
    output_dir = target_paths['reward_dir'] / "External_validation_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # FQE settings
    fqe_enabled = fqe_cfg.get('enabled', True)
    fqe_lr = fqe_cfg.get('learning_rate', 1e-4)
    fqe_steps = fqe_cfg.get('n_steps', 10000)
    gamma = fqe_cfg.get('gamma', 0.99)

    # Load models from source
    models = {}
    model_names = {}

    # Load best RL model
    best_rl_cfg = algo_cfg.get('best_rl', {})
    rl_algo = best_rl_cfg.get('algorithm', 'cql')
    model_source = best_rl_cfg.get('model_source', 'hpo')

    if model_source == 'hpo':
        model_dir = source_paths['reward_dir'] / "HPO_results"
        model_path = model_dir / f"best_{rl_algo}_model.d3"
    else:  # ablation
        model_dir = source_paths['reward_dir'] / "Ablation_results"
        model_path = model_dir / f"{rl_algo}_{source_mdp}_model.d3"

    if model_path.exists():
        models['rl'] = d3rlpy.load_learnable(str(model_path), device=device)
        model_names['rl'] = f"{rl_algo.upper()} ({source_db.upper()})"
        print(f"\nLoaded RL model: {rl_algo.upper()} from {model_source}")
    else:
        print(f"Warning: RL model not found at {model_path}")

    # Load BC model from source
    bc_cfg_algo = algo_cfg.get('bc', {})
    if bc_cfg_algo.get('enabled', True):
        bc_dir = source_paths['reward_dir'] / "BC_results"
        bc_path = bc_dir / f"bc_{source_mdp}_model.d3"

        if bc_path.exists():
            models['bc_source'] = d3rlpy.load_learnable(str(bc_path), device=device)
            model_names['bc_source'] = f"BC ({source_db.upper()})"
            print(f"Loaded BC model from source ({source_db.upper()})")
        else:
            print(f"Warning: BC model not found at {bc_path}")

    if not models:
        print("No models found for external validation!")
        return

    # Evaluate on target database
    all_results = []

    for split in target_splits:
        print(f"\n{'='*50}")
        print(f"Evaluating on {target_db.upper()} - {split.upper()}")
        print(f"{'='*50}")

        # Load target dataset
        target_mdp_dir = target_paths['mdp_dir']
        ds_path = target_mdp_dir / f"{source_mdp}_{split}.h5"

        if not ds_path.exists():
            print(f"Dataset not found: {ds_path}")
            continue

        dataset = load_h5_dataset(ds_path)
        print(f"Episodes: {len(dataset.episodes)}")

        # Optionally train BC on target
        if bc_cfg_algo.get('retrain_on_target', False):
            print(f"\n  Training BC on target data...")
            bc_hp = config.get('behavior_cloning', {}).get('hyperparameters', {})

            train_path = target_mdp_dir / f"{source_mdp}_train.h5"
            if train_path.exists():
                train_ds = load_h5_dataset(train_path)

                bc_target = DiscreteBCConfig(
                    learning_rate=bc_hp.get('learning_rate', 1e-4),
                    batch_size=bc_hp.get('batch_size', 256),
                    beta=bc_hp.get('beta', 0.5),
                    encoder_factory=VectorEncoderFactory(
                        hidden_units=bc_hp.get('hidden_units', [256, 256])
                    ),
                ).create(device=device)

                d3rlpy.seed(seed)
                for epoch, metrics in bc_target.fit(
                    train_ds,
                    n_steps=bc_hp.get('n_steps', 10000),
                    n_steps_per_epoch=bc_hp.get('n_steps_per_epoch', 2000),
                    show_progress=False,
                    experiment_name="bc_target",
                ):
                    pass

                models['bc_target'] = bc_target
                model_names['bc_target'] = f"BC ({target_db.upper()})"
                print(f"  Trained BC on target")

        # Evaluate each model
        for model_key, model in models.items():
            model_name = model_names[model_key]
            print(f"\n  {model_name}:")

            # Compute metrics
            metrics, predictions = compute_metrics(model, dataset, metrics_cfg)

            result = {
                'source': source_db,
                'target': target_db,
                'split': split,
                'model': model_key,
                'model_name': model_name,
                **metrics
            }

            if metrics_cfg.get('action_match', True):
                print(f"    Action Match: {metrics['action_match']:.1%}")

            if metrics_cfg.get('rrt_rate', True):
                print(f"    RRT Rate: {metrics['rrt_rate']:.1%} (data: {metrics['data_rrt_rate']:.1%})")

            if metrics_cfg.get('q_values', True) and 'mean_q_rrt' in metrics:
                print(f"    Mean Q (No RRT): {metrics['mean_q_no_rrt']:.4f}")
                print(f"    Mean Q (RRT): {metrics['mean_q_rrt']:.4f}")

            # FQE evaluation
            if fqe_enabled:
                print(f"    Running FQE...")

                if bootstrap_cfg.get('enabled', True):
                    n_bootstrap = bootstrap_cfg.get('n_bootstrap', 10)
                    fqe_config_obj = FQEConfig(learning_rate=fqe_lr, gamma=gamma)
                    fqe_isv, ci_low, ci_high = bootstrap_fqe(
                        model, dataset, fqe_config_obj, n_bootstrap, fqe_steps, device, seed
                    )
                    result['fqe_isv'] = fqe_isv
                    result['fqe_ci_low'] = ci_low
                    result['fqe_ci_high'] = ci_high
                    print(f"    FQE ISV: {fqe_isv:.4f} [{ci_low:.4f}, {ci_high:.4f}]")
                else:
                    fqe_isv = evaluate_with_fqe(model, dataset, fqe_lr, fqe_steps, gamma, device)
                    result['fqe_isv'] = fqe_isv
                    print(f"    FQE ISV: {fqe_isv:.4f}")

            all_results.append(result)

            # Save predictions if requested
            if output_cfg.get('save_predictions', True):
                pred_df = pd.DataFrame({
                    'prediction': predictions,
                    'actual': np.concatenate([ep.actions for ep in dataset.episodes])
                })
                pred_df.to_csv(output_dir / f"predictions_{model_key}_{split}.csv", index=False)

    # Create summary DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)

        if output_cfg.get('save_metrics', True):
            results_df.to_csv(output_dir / "external_validation_results.csv", index=False)

        # Print summary
        print(f"\n{'='*60}")
        print("EXTERNAL VALIDATION SUMMARY")
        print(f"{'='*60}")
        display_cols = ['model_name', 'split', 'action_match', 'rrt_rate', 'fqe_isv']
        display_cols = [c for c in display_cols if c in results_df.columns]
        print(results_df[display_cols].to_string(index=False))

        # Create plots
        if output_cfg.get('save_plots', True):
            create_external_validation_plots(results_df, output_dir, source_db, target_db)

    print(f"\n{'='*60}")
    print(f"External validation complete! Results saved to: {output_dir}")
    print(f"{'='*60}")


def create_external_validation_plots(results_df: pd.DataFrame, output_dir: Path,
                                      source_db: str, target_db: str):
    """Create external validation comparison plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    models = results_df['model_name'].unique()
    splits = results_df['split'].unique()

    colors = {'rl': '#1f77b4', 'bc_source': '#ff7f0e', 'bc_target': '#2ca02c'}

    # Plot 1: FQE ISV
    ax1 = axes[0]
    if 'fqe_isv' in results_df.columns:
        width = 0.35
        x = np.arange(len(splits))

        for i, model in enumerate(results_df['model'].unique()):
            model_df = results_df[results_df['model'] == model]
            values = [model_df[model_df['split'] == s]['fqe_isv'].values[0]
                     for s in splits if len(model_df[model_df['split'] == s]) > 0]

            offset = (i - len(results_df['model'].unique())/2 + 0.5) * width
            model_name = model_df['model_name'].iloc[0]

            if 'fqe_ci_low' in results_df.columns:
                ci_lows = [model_df[model_df['split'] == s]['fqe_ci_low'].values[0]
                          for s in splits if len(model_df[model_df['split'] == s]) > 0]
                ci_highs = [model_df[model_df['split'] == s]['fqe_ci_high'].values[0]
                           for s in splits if len(model_df[model_df['split'] == s]) > 0]
                errors = [np.array(values) - np.array(ci_lows),
                         np.array(ci_highs) - np.array(values)]
                ax1.bar(x + offset, values, width, yerr=errors, capsize=3,
                       label=model_name, color=colors.get(model, 'gray'), alpha=0.8)
            else:
                ax1.bar(x + offset, values, width, label=model_name,
                       color=colors.get(model, 'gray'), alpha=0.8)

        ax1.set_xticks(x)
        ax1.set_xticklabels([s.upper() for s in splits])
        ax1.set_ylabel('FQE ISV')
        ax1.set_title(f'Policy Value on {target_db.upper()}')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Action Match
    ax2 = axes[1]
    if 'action_match' in results_df.columns:
        x = np.arange(len(splits))

        for i, model in enumerate(results_df['model'].unique()):
            model_df = results_df[results_df['model'] == model]
            values = [model_df[model_df['split'] == s]['action_match'].values[0] * 100
                     for s in splits if len(model_df[model_df['split'] == s]) > 0]

            offset = (i - len(results_df['model'].unique())/2 + 0.5) * width
            model_name = model_df['model_name'].iloc[0]

            ax2.bar(x + offset, values, width, label=model_name,
                   color=colors.get(model, 'gray'), alpha=0.8)

        ax2.set_xticks(x)
        ax2.set_xticklabels([s.upper() for s in splits])
        ax2.set_ylabel('Action Match (%)')
        ax2.set_title(f'Clinician Agreement on {target_db.upper()}')
        ax2.legend()
        ax2.set_ylim(0, 105)
        ax2.grid(axis='y', alpha=0.3)

    # Plot 3: RRT Rate comparison
    ax3 = axes[2]
    if 'rrt_rate' in results_df.columns:
        # For one split
        split = splits[0]
        split_df = results_df[results_df['split'] == split]

        x = np.arange(len(split_df))
        model_names = split_df['model_name'].values
        rrt_rates = split_df['rrt_rate'].values * 100
        data_rate = split_df['data_rrt_rate'].values[0] * 100

        ax3.bar(x, rrt_rates, color=[colors.get(m, 'gray') for m in split_df['model'].values], alpha=0.8)
        ax3.axhline(y=data_rate, color='red', linestyle='--', label=f'Clinician ({data_rate:.1f}%)')

        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names, rotation=15, ha='right')
        ax3.set_ylabel('RRT Rate (%)')
        ax3.set_title(f'RRT Initiation Rate ({split.upper()})')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

    plt.suptitle(f'External Validation: {source_db.upper()} → {target_db.upper()}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'external_validation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: external_validation_comparison.png")


def main():
    """Main entry point for external validation."""
    from utils import load_config, get_data_paths

    config = load_config()
    all_paths = get_data_paths(config)

    run_external_validation(config, all_paths)


if __name__ == "__main__":
    main()
