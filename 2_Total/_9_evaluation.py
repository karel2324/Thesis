"""
Evaluation Pipeline - Compare trained models on test data.

Reuses FQE functions from _7_hpo.py.
Settings from config.yaml evaluation section.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import d3rlpy
from d3rlpy.ope import FQEConfig

from utils import load_config, get_data_paths, load_mdp
from rl_training import function_fqe, bootstrap_fqe, get_action_frequency_per_episode

def main():
    """Main entry point for evaluation."""
    from utils import load_config, get_data_paths

    config = load_config()
    all_paths = get_data_paths(config)

    # Check whether to run AUMCdb
    if config['evaluation']['source']['aumc'] == True:
        print("AUMCdb is enabled for evaluation")
        if all_paths['aumc']['mdp_dir'].exists():
            run_evaluation_for_db(all_paths['aumc'], config)
        else:
            print(f"Skipping AUMCdb: MDP directory not found")
    
    # Check whether to run MIMIC
    if config['evaluation']['source']['mimic'] == True:
        print("MIMIC is enabled for evaluation")
        if all_paths['aumimicmc']['mdp_dir'].exists():
            run_evaluation_for_db(all_paths['mimic'], config)
        else:
            print(f"Skipping MIMIC: MDP directory not found")


############################################################################################################
# METRICS

def compute_metrics(algo, dataset, metrics_cfg: dict) -> dict:
    """Compute action match and RRT rate metrics."""
    obs = np.concatenate([ep.observations for ep in dataset.episodes])
    actions = np.concatenate([ep.actions for ep in dataset.episodes])
    predictions = algo.predict(obs)

    result = {}
    if metrics_cfg.get('action_match', True):
        result['action_match'] = (predictions == actions).mean()
    if metrics_cfg.get('rrt_rate', True):
        result['rrt_rate'] = (predictions == 1).mean()
        result['data_rrt_rate'] = (actions == 1).mean()
    return result

############################################################################################################
# PLOTTING

def create_plots(results_df: pd.DataFrame, output_dir: Path):
    """Create evaluation comparison plots."""
    colors = {'cql': '#1f77b4', 'ddqn': '#ff7f0e', 'bcq': '#2ca02c', 'nfq': '#d62728', 'bc': '#9467bd'}

    for split in results_df['split'].unique():
        df = results_df[results_df['split'] == split]
        algos = df['algorithm'].unique()
        x = np.arange(len(algos))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # FQE ISV
        if 'fqe_isv' in df.columns:
            vals = [df[df['algorithm'] == a]['fqe_isv'].mean() for a in algos]
            bar_colors = [colors.get(a, 'gray') for a in algos]

            if 'fqe_ci_low' in df.columns:
                ci_lo = [df[df['algorithm'] == a]['fqe_ci_low'].mean() for a in algos]
                ci_hi = [df[df['algorithm'] == a]['fqe_ci_high'].mean() for a in algos]
                err = [np.array(vals) - np.array(ci_lo), np.array(ci_hi) - np.array(vals)]
                axes[0].bar(x, vals, yerr=err, capsize=5, color=bar_colors, alpha=0.8)
            else:
                axes[0].bar(x, vals, color=bar_colors, alpha=0.8)

            axes[0].set_xticks(x)
            axes[0].set_xticklabels([a.upper() for a in algos])
            axes[0].set_ylabel('FQE ISV')
            axes[0].set_title(f'Policy Value ({split.upper()})')
            axes[0].grid(axis='y', alpha=0.3)

        # Action Match
        if 'action_match' in df.columns:
            vals = [df[df['algorithm'] == a]['action_match'].mean() * 100 for a in algos]
            bar_colors = [colors.get(a, 'gray') for a in algos]
            axes[1].bar(x, vals, color=bar_colors, alpha=0.8)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([a.upper() for a in algos])
            axes[1].set_ylabel('Action Match (%)')
            axes[1].set_title(f'Clinician Agreement ({split.upper()})')
            axes[1].set_ylim(0, 105)
            axes[1].grid(axis='y', alpha=0.3)
            for i, v in enumerate(vals):
                axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / f'evaluation_{split}.png', dpi=150)
        plt.close()
        print(f"  Saved: evaluation_{split}.png")

############################################################################################################
# MAIN

def run_evaluation_for_db(db_paths: dict, config: dict):
    """Run evaluation for one database."""

    db_name = db_paths['name']
    db_key = 'aumc' if 'aumc' in db_name.lower() else 'mimic'
    eval_cfg = config.get('evaluation', {})

    # Check if this database should be evaluated
    if not eval_cfg.get('target', {}).get(db_key, True):
        print(f"Skipping {db_name}: disabled in evaluation.target")
        return

    print(f"\n{'='*60}\nEVALUATION: {db_name}\n{'='*60}")

    # Setup
    hpo_dir = db_paths['reward_dir'] / "HPO_results"
    bc_dir = db_paths['reward_dir'] / "BC_results"
    output_dir = db_paths['reward_dir'] / "Evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = config.get('processing', {}).get('random_state', 42)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Get settings from config
    mdps = eval_cfg.get('mdps', ['mdp1'])
    target_splits = [s for s, e in eval_cfg.get('target_split', {}).items() if e]
    algorithms = eval_cfg.get('algorithms', {})
    metrics_cfg = eval_cfg.get('metrics', {})
    fqe_cfg = eval_cfg.get('fqe', {})
    output_cfg = eval_cfg.get('output', {})

    # FQE settings
    gamma = config.get('hpo', {}).get(f'training_{db_key}', {}).get('gamma', 0.99)
    fqe_config = FQEConfig(learning_rate=fqe_cfg.get('learning_rate', 1e-4), gamma=gamma)
    fqe_n_steps = fqe_cfg.get('n_steps', 10000)
    bootstrap_cfg = fqe_cfg.get('bootstrap', {})

    print(f"MDPs: {mdps}, Splits: {target_splits}")
    print(f"Algorithms: {[k for k, v in algorithms.items() if v]}")

    all_results = []

    for mdp_name in mdps:
        print(f"\n--- {mdp_name.upper()} ---")

        # Load models
        models = {}
        for algo_name in ['cql', 'ddqn', 'bcq', 'nfq']:
            if algorithms.get(algo_name, False):
                path = hpo_dir / f"best_{algo_name}_model.d3"
                if path.exists():
                    models[algo_name] = d3rlpy.load_learnable(str(path), device=device)
                    print(f"  Loaded: {algo_name.upper()}")

        if algorithms.get('bc', False):
            path = bc_dir / f"bc_{mdp_name}_model.d3"
            if path.exists():
                models['bc'] = d3rlpy.load_learnable(str(path), device=device)
                print(f"  Loaded: BC")

        if not models:
            print(f"  No models found, skipping")
            continue

        # Evaluate on each split
        for split in target_splits:
            print(f"\n  {split.upper()} split:")
            dataset = load_mdp(db_paths, mdp_name, split)
            print(f"  Episodes: {len(dataset.episodes)}")

            for algo_name, model in models.items():
                d3rlpy.seed(seed)
                result = {'mdp': mdp_name, 'split': split, 'algorithm': algo_name}

                # Basic metrics
                metrics = compute_metrics(model, dataset, metrics_cfg)
                result.update(metrics)

                if metrics_cfg.get('action_match'):
                    print(f"    {algo_name.upper()}: Match={metrics['action_match']:.1%}", end='')
                if metrics_cfg.get('rrt_rate'):
                    print(f", RRT={metrics['rrt_rate']:.1%} (data: {metrics['data_rrt_rate']:.1%})", end='')

                # FQE evaluation
                if fqe_cfg.get('enabled', True) and metrics_cfg.get('fqe_isv', True):
                    if bootstrap_cfg.get('enabled', False):
                        n_boot = bootstrap_cfg.get('n_bootstrap', 10)
                        boot_steps = bootstrap_cfg.get('n_steps', fqe_n_steps)
                        CI = bootstrap_cfg.get('confidence_level', 0.95)
                        fqe_isv, ci_lo, ci_hi = bootstrap_fqe(
                            model, dataset, fqe_config, n_boot, boot_steps, boot_steps, device, seed, CI
                        )
                        result.update({'fqe_isv': fqe_isv, 'fqe_ci_low': ci_lo, 'fqe_ci_high': ci_hi})
                        print(f", FQE={fqe_isv:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
                    else:
                        fqe_isv = function_fqe(model, dataset, dataset, fqe_config, fqe_n_steps, fqe_n_steps, device, seed)
                        result['fqe_isv'] = fqe_isv
                        print(f", FQE={fqe_isv:.4f}")
                else:
                    print()

                all_results.append(result)

    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
        print(results_df.to_string(index=False))

        if output_cfg.get('save_metrics', True):
            results_df.to_csv(output_dir / "evaluation_results.csv", index=False)

        if output_cfg.get('save_plots', True):
            create_plots(results_df, output_dir)

    print(f"\nResults saved to: {output_dir}")



if __name__ == "__main__":
    main()
