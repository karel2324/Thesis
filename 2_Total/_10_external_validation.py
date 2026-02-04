"""
External Validation - Evaluate models across databases.

Tests models trained on source database (e.g., AUMCdb) on target database (e.g., MIMIC).
Uses shared evaluation utilities from rl_utils.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from utils import load_config, get_data_paths, load_mdp, load_model
from rl_utils import evaluate_algo, compute_mc_return, compute_metrics_algo_vs_algo


def main():
    """Main entry point for external validation."""
    config = load_config()
    all_paths = get_data_paths(config)

    ext_cfg = config['external_validation']
    source_db = ext_cfg['source_database']
    target_db = ext_cfg['target_database']

    # Validate databases exist
    if not all_paths[source_db]['mdp_dir'].exists():
        print(f"Source database {source_db} MDP directory not found")
        return
    if not all_paths[target_db]['mdp_dir'].exists():
        print(f"Target database {target_db} MDP directory not found")
        return

    run_external_validation(config=config, all_paths=all_paths)


def run_external_validation(config: dict, all_paths: dict):
    """
    Run cross-database validation.

    Args:
        config: Full configuration dictionary
        all_paths: Dictionary with paths for all databases
    """
    ext_cfg = config['external_validation']

    source_db = ext_cfg['source_database']
    target_db = ext_cfg['target_database']
    source_paths = all_paths[source_db]
    target_paths = all_paths[target_db]

    print(f"\n{'='*60}")
    print(f"EXTERNAL VALIDATION: {source_db.upper()} → {target_db.upper()}")
    print(f"{'='*60}")

    # Settings
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = config['processing']['random_state']
    gamma = 0.99

    # Model directories (from source database)
    model_source = ext_cfg['model_source']
    if model_source == 'hpo':
        model_dir = source_paths['reward_dir'] / "HPO_results"
    else:
        model_dir = source_paths['reward_dir'] / "Ablation_results"
    bc_dir = source_paths['reward_dir'] / "BC_results"

    # Output directory (in target database)
    output_dir = target_paths['reward_dir'] / "External_validation_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for mdp_name in ext_cfg['mdps']:
        print(f"\n--- {mdp_name.upper()} ---")

        for split in ext_cfg['splits']:
            # Load TARGET dataset (this is the key difference from _9_evaluation)
            dataset = load_mdp(db_paths=target_paths, mdp_name=mdp_name, split=split)

            # Monte Carlo return from target data
            mc_mean, mc_std = compute_mc_return(episodes=dataset.episodes, gamma=gamma)
            print(f"\n  {split.upper()} ({len(dataset.episodes)} episodes) - MC Return: {mc_mean:.4f} (±{mc_std:.4f})")

            # Evaluate HPO models (CQL, DDQN, BCQ, NFQ) trained on SOURCE
            for algo_name in ['cql', 'ddqn', 'bcq', 'nfq']:
                if ext_cfg['algorithms'][algo_name]:
                    if model_source == 'hpo':
                        model_path = model_dir / f"best_{algo_name}_model.d3"
                    else:
                        model_path = model_dir / f"{algo_name}_{mdp_name}_model.d3"

                    model = load_model(model_path=model_path, device=device)
                    if model is not None:
                        result = evaluate_algo(
                            algo=model,
                            algo_name=algo_name,
                            dataset=dataset,
                            device=device,
                            seed=seed,
                            mc_mean=mc_mean,
                            mc_std=mc_std,
                            fqe_enabled=ext_cfg['fqe']['enabled'],
                            fqe_learning_rate=ext_cfg['fqe']['learning_rate'],
                            fqe_n_steps=ext_cfg['fqe']['n_steps'],
                            fqe_bootstrap_enabled=ext_cfg['fqe']['bootstrap']['enabled'],
                            fqe_bootstrap_n_bootstrap=ext_cfg['fqe']['bootstrap']['n_bootstrap'],
                            fqe_bootstrap_n_steps=ext_cfg['fqe']['bootstrap']['n_steps'],
                            fqe_bootstrap_confidence_level=ext_cfg['fqe']['bootstrap']['confidence_level'],
                            gamma=gamma
                        )
                        result['source'] = source_db
                        result['target'] = target_db
                        result['mdp'] = mdp_name
                        result['split'] = split
                        all_results.append(result)
                    else:
                        print(f"    {algo_name.upper()}: Model not found at {model_path}")

            # Evaluate BC model trained on SOURCE
            if ext_cfg['algorithms']['bc']:
                model_path = bc_dir / f"bc_{mdp_name}_model.d3"
                model = load_model(model_path=model_path, device=device)
                if model is not None:
                    result = evaluate_algo(
                        algo=model,
                        algo_name='bc',
                        dataset=dataset,
                        device=device,
                        seed=seed,
                        mc_mean=mc_mean,
                        mc_std=mc_std,
                        fqe_enabled=ext_cfg['fqe']['enabled'],
                        fqe_learning_rate=ext_cfg['fqe']['learning_rate'],
                        fqe_n_steps=ext_cfg['fqe']['n_steps'],
                        fqe_bootstrap_enabled=ext_cfg['fqe']['bootstrap']['enabled'],
                        fqe_bootstrap_n_bootstrap=ext_cfg['fqe']['bootstrap']['n_bootstrap'],
                        fqe_bootstrap_n_steps=ext_cfg['fqe']['bootstrap']['n_steps'],
                        fqe_bootstrap_confidence_level=ext_cfg['fqe']['bootstrap']['confidence_level'],
                        gamma=gamma
                    )
                    result['source'] = source_db
                    result['target'] = target_db
                    result['mdp'] = mdp_name
                    result['split'] = split
                    all_results.append(result)
                else:
                    print(f"    BC: Model not found at {model_path}")

    # Save results vs behaviour policy
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(f"\n{'='*60}\nSUMMARY (vs Behaviour Policy)\n{'='*60}")
        print(results_df.to_string(index=False))

        if ext_cfg['output']['save_metrics']:
            results_df.to_csv(output_dir / "external_validation_results.csv", index=False)

        if ext_cfg['output']['save_plots']:
            create_plots(results_df=results_df, output_dir=output_dir, source_db=source_db, target_db=target_db)

    # =========================================================================
    # ALGO VS ALGO COMPARISON (each RL algo vs BC)
    # =========================================================================
    if ext_cfg['algorithms']['bc']:
        print(f"\n{'='*60}\nALGO VS BC COMPARISON\n{'='*60}")

        algo_vs_bc_results = []

        for mdp_name in ext_cfg['mdps']:
            print(f"\n--- {mdp_name.upper()} ---")

            # Load BC model once per MDP (from SOURCE database)
            bc_path = bc_dir / f"bc_{mdp_name}_model.d3"
            bc_model = load_model(model_path=bc_path, device=device)

            if bc_model is None:
                print(f"  BC model not found, skipping algo-vs-BC comparison")
                continue

            for split in ext_cfg['splits']:
                # Load TARGET dataset
                dataset = load_mdp(db_paths=target_paths, mdp_name=mdp_name, split=split)
                print(f"\n  {split.upper()} ({len(dataset.episodes)} episodes)")

                # Compare each RL algo vs BC
                for algo_name in ['cql', 'ddqn', 'bcq', 'nfq']:
                    if ext_cfg['algorithms'][algo_name]:
                        if model_source == 'hpo':
                            model_path = model_dir / f"best_{algo_name}_model.d3"
                        else:
                            model_path = model_dir / f"{algo_name}_{mdp_name}_model.d3"

                        rl_model = load_model(model_path=model_path, device=device)

                        if rl_model is not None:
                            metrics = compute_metrics_algo_vs_algo(
                                algo1=rl_model,
                                algo2=bc_model,
                                dataset=dataset
                            )
                            metrics['algo1'] = algo_name
                            metrics['algo2'] = 'bc'
                            metrics['source'] = source_db
                            metrics['target'] = target_db
                            metrics['mdp'] = mdp_name
                            metrics['split'] = split
                            algo_vs_bc_results.append(metrics)

                            print(f"    {algo_name.upper()} vs BC: "
                                  f"Match={metrics['action_match']:.1%}, "
                                  f"Earlier={metrics['rrt_timing_earlier']:.1%}, "
                                  f"Same={metrics['rrt_timing_same']:.1%}")

        # Save algo-vs-BC results
        if algo_vs_bc_results:
            algo_vs_bc_df = pd.DataFrame(algo_vs_bc_results)
            print(f"\n{'='*60}\nSUMMARY (RL Algos vs BC)\n{'='*60}")
            print(algo_vs_bc_df.to_string(index=False))

            if ext_cfg['output']['save_metrics']:
                algo_vs_bc_df.to_csv(output_dir / "external_validation_algo_vs_bc.csv", index=False)

    print(f"\nResults saved to: {output_dir}")


def create_plots(results_df: pd.DataFrame, output_dir: Path, source_db: str, target_db: str):
    """
    Create external validation comparison plots.

    Args:
        results_df: DataFrame with evaluation results
        output_dir: Directory to save plots
        source_db: Source database name
        target_db: Target database name
    """
    colors = {'cql': '#1f77b4', 'ddqn': '#ff7f0e', 'bcq': '#2ca02c', 'nfq': '#d62728', 'bc': '#9467bd'}

    for split in results_df['split'].unique():
        df = results_df[results_df['split'] == split]
        algos = df['algorithm'].unique()
        x = np.arange(len(algos))
        bar_colors = [colors.get(a, 'gray') for a in algos]

        # Create figure with 2x3 subplots
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        fig.suptitle(f'External Validation: {source_db.upper()} → {target_db.upper()} ({split.upper()} Split)',
                     fontsize=14, fontweight='bold')

        # 1. FQE ISV (top-left)
        ax = axes[0, 0]
        if 'fqe_isv' in df.columns:
            vals = [df[df['algorithm'] == a]['fqe_isv'].mean() for a in algos]
            if 'fqe_ci_low' in df.columns:
                ci_lo = [df[df['algorithm'] == a]['fqe_ci_low'].mean() for a in algos]
                ci_hi = [df[df['algorithm'] == a]['fqe_ci_high'].mean() for a in algos]
                err = [np.array(vals) - np.array(ci_lo), np.array(ci_hi) - np.array(vals)]
                ax.bar(x, vals, yerr=err, capsize=5, color=bar_colors, alpha=0.8)
            else:
                ax.bar(x, vals, color=bar_colors, alpha=0.8)
            mc_return = df['mc_return_mean'].iloc[0]
            ax.axhline(y=mc_return, color='red', linestyle='--', label=f'MC Return: {mc_return:.3f}')
            ax.legend()
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in algos])
        ax.set_ylabel('FQE ISV')
        ax.set_title('Policy Value (FQE)')
        ax.grid(axis='y', alpha=0.3)

        # 2. Action Match (top-middle)
        ax = axes[0, 1]
        vals = [df[df['algorithm'] == a]['action_match'].mean() * 100 for a in algos]
        ax.bar(x, vals, color=bar_colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in algos])
        ax.set_ylabel('Action Match (%)')
        ax.set_title('Clinician Agreement')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)

        # 3. RRT Rate per State (top-right)
        ax = axes[0, 2]
        vals_algo = [df[df['algorithm'] == a]['rrt_rate_per_state_algo'].mean() * 100 for a in algos]
        data_rate = df['rrt_rate_per_state_data'].iloc[0] * 100
        ax.bar(x, vals_algo, color=bar_colors, alpha=0.8)
        ax.axhline(y=data_rate, color='red', linestyle='--', label=f'Data: {data_rate:.1f}%')
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in algos])
        ax.set_ylabel('RRT Rate (%)')
        ax.set_title('RRT Rate per State')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 4. RRT Rate per Episode (bottom-left)
        ax = axes[1, 0]
        vals_algo = [df[df['algorithm'] == a]['rrt_rate_per_episode_algo'].mean() * 100 for a in algos]
        data_rate = df['rrt_rate_per_episode_data'].iloc[0] * 100
        ax.bar(x, vals_algo, color=bar_colors, alpha=0.8)
        ax.axhline(y=data_rate, color='red', linestyle='--', label=f'Data: {data_rate:.1f}%')
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in algos])
        ax.set_ylabel('RRT Rate (%)')
        ax.set_title('RRT Rate per Episode')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 5. RRT Timing (bottom-middle)
        ax = axes[1, 1]
        timing_metrics = ['rrt_timing_earlier', 'rrt_timing_same', 'rrt_timing_later',
                         'rrt_timing_algo_only', 'rrt_timing_data_only', 'rrt_timing_neither']
        timing_labels = ['Earlier', 'Same', 'Later', 'Algo only', 'Data only', 'Neither']
        timing_colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728', '#7f7f7f']

        bottom = np.zeros(len(algos))
        for metric, label, color in zip(timing_metrics, timing_labels, timing_colors):
            vals = [df[df['algorithm'] == a][metric].mean() * 100 for a in algos]
            ax.bar(x, vals, bottom=bottom, label=label, color=color, alpha=0.8)
            bottom += np.array(vals)

        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in algos])
        ax.set_ylabel('Percentage (%)')
        ax.set_title('RRT Timing vs Data')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)

        # 6. Summary table (bottom-right)
        ax = axes[1, 2]
        ax.axis('off')
        summary_data = []
        for a in algos:
            row = df[df['algorithm'] == a].iloc[0]
            summary_data.append([
                a.upper(),
                f"{row['action_match']*100:.1f}%",
                f"{row['rrt_timing_earlier']*100:.1f}%",
                f"{row['rrt_timing_same']*100:.1f}%",
                f"{row.get('fqe_isv', 0):.3f}" if 'fqe_isv' in row else 'N/A'
            ])
        table = ax.table(
            cellText=summary_data,
            colLabels=['Algo', 'Match', 'Earlier', 'Same', 'FQE'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax.set_title('Summary', pad=20)

        plt.tight_layout()
        plt.savefig(output_dir / f'external_validation_{split}.png', dpi=150)
        plt.close()
        print(f"  Saved: external_validation_{split}.png")


if __name__ == "__main__":
    main()
