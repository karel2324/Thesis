"""
Evaluation Pipeline - Compare trained models on test data.

Compares HPO best models (CQL, DDQN, BCQ, NFQ) with BC baseline
using FQE policy evaluation and basic metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from utils import load_config, get_data_paths, load_mdp, load_model
from rl_utils import evaluate_algo, compute_mc_return, compute_metrics_algo_vs_algo


def main():
    """Main entry point for evaluation."""
    config = load_config()
    all_paths = get_data_paths(config)

    for db_key in ['aumc', 'mimic']:
        if config['evaluation']['databases'][db_key]:
            if all_paths[db_key]['mdp_dir'].exists():
                run_evaluation_for_db(db_paths=all_paths[db_key], config=config)
            else:
                print(f"Skipping {db_key}: MDP directory not found")


def run_evaluation_for_db(db_paths: dict, config: dict):
    """
    Run evaluation for one database.

    Args:
        db_paths: Dictionary with database paths
        config: Full configuration dictionary
    """
    db_name = db_paths['name']

    print(f"\n{'='*60}\nEVALUATION: {db_name}\n{'='*60}")

    # Setup paths
    hpo_dir = db_paths['reward_dir'] / "HPO_results"
    bc_dir = db_paths['reward_dir'] / "BC_results"
    output_dir = db_paths['reward_dir'] / "Evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Settings
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = config['processing']['random_state']
    gamma = 0.99

    all_results = []

    for mdp_name in config['evaluation']['mdps']:
        print(f"\n--- {mdp_name.upper()} ---")

        for split in config['evaluation']['splits']:
            dataset = load_mdp(db_paths=db_paths, mdp_name=mdp_name, split=split)

            # Monte Carlo return from data (behavior policy)
            mc_mean, mc_std = compute_mc_return(episodes=dataset.episodes, gamma=gamma)
            print(f"\n  {split.upper()} ({len(dataset.episodes)} episodes) - MC Return: {mc_mean:.4f} (±{mc_std:.4f})")

            # Evaluate HPO models (CQL, DDQN, BCQ, NFQ)
            for algo_name in ['cql', 'ddqn', 'bcq', 'nfq']:
                if config['evaluation']['algorithms'][algo_name]:
                    model_path = hpo_dir / f"best_{algo_name}_model.d3"
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
                            fqe_enabled=config['evaluation']['fqe']['enabled'],
                            fqe_learning_rate=config['evaluation']['fqe']['learning_rate'],
                            fqe_n_steps=config['evaluation']['fqe']['n_steps'],
                            fqe_bootstrap_enabled=config['evaluation']['fqe']['bootstrap']['enabled'],
                            fqe_bootstrap_n_bootstrap=config['evaluation']['fqe']['bootstrap']['n_bootstrap'],
                            fqe_bootstrap_n_steps=config['evaluation']['fqe']['bootstrap']['n_steps'],
                            fqe_bootstrap_confidence_level=config['evaluation']['fqe']['bootstrap']['confidence_level'],
                            gamma=gamma
                        )
                        result['mdp'] = mdp_name
                        result['split'] = split
                        all_results.append(result)

            # Evaluate BC model
            if config['evaluation']['algorithms']['bc']:
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
                        fqe_enabled=config['evaluation']['fqe']['enabled'],
                        fqe_learning_rate=config['evaluation']['fqe']['learning_rate'],
                        fqe_n_steps=config['evaluation']['fqe']['n_steps'],
                        fqe_bootstrap_enabled=config['evaluation']['fqe']['bootstrap']['enabled'],
                        fqe_bootstrap_n_bootstrap=config['evaluation']['fqe']['bootstrap']['n_bootstrap'],
                        fqe_bootstrap_n_steps=config['evaluation']['fqe']['bootstrap']['n_steps'],
                        fqe_bootstrap_confidence_level=config['evaluation']['fqe']['bootstrap']['confidence_level'],
                        gamma=gamma
                    )
                    result['mdp'] = mdp_name
                    result['split'] = split
                    all_results.append(result)

    # Save results vs behaviour policy
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(f"\n{'='*60}\nSUMMARY (vs Behaviour Policy)\n{'='*60}")
        print(results_df.to_string(index=False))

        if config['evaluation']['output']['save_metrics']:
            results_df.to_csv(output_dir / "evaluation_results.csv", index=False)

        if config['evaluation']['output']['save_plots']:
            create_plots_vs_behaviour_policy(results_df=results_df, output_dir=output_dir)

    # =========================================================================
    # ALGO VS ALGO COMPARISON (each RL algo vs BC)
    # =========================================================================
    if config['evaluation']['algorithms']['bc']:
        print(f"\n{'='*60}\nALGO VS BC COMPARISON\n{'='*60}")

        algo_vs_bc_results = []

        for mdp_name in config['evaluation']['mdps']:
            print(f"\n--- {mdp_name.upper()} ---")

            # Load BC model once per MDP
            bc_path = bc_dir / f"bc_{mdp_name}_model.d3"
            bc_model = load_model(model_path=bc_path, device=device)

            if bc_model is None:
                print(f"  BC model not found, skipping algo-vs-BC comparison")
                continue

            for split in config['evaluation']['splits']:
                dataset = load_mdp(db_paths=db_paths, mdp_name=mdp_name, split=split)
                print(f"\n  {split.upper()} ({len(dataset.episodes)} episodes)")

                # Compare each RL algo vs BC
                for algo_name in ['cql', 'ddqn', 'bcq', 'nfq']:
                    if config['evaluation']['algorithms'][algo_name]:
                        model_path = hpo_dir / f"best_{algo_name}_model.d3"
                        rl_model = load_model(model_path=model_path, device=device)

                        if rl_model is not None:
                            metrics = compute_metrics_algo_vs_algo(
                                algo1=rl_model,
                                algo2=bc_model,
                                dataset=dataset
                            )
                            metrics['algo1'] = algo_name
                            metrics['algo2'] = 'bc'
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

            if config['evaluation']['output']['save_metrics']:
                algo_vs_bc_df.to_csv(output_dir / "evaluation_algo_vs_bc.csv", index=False)

            if config['evaluation']['output']['save_plots']:
                # Rename columns to match create_plots_vs_behaviour_policy format
                plot_df = algo_vs_bc_df.rename(columns={
                    'algo1': 'algorithm',
                    'rrt_rate_per_state_algo1': 'rrt_rate_per_state_algo',
                    'rrt_rate_per_state_algo2': 'rrt_rate_per_state_data',
                    'rrt_rate_per_episode_algo1': 'rrt_rate_per_episode_algo',
                    'rrt_rate_per_episode_algo2': 'rrt_rate_per_episode_data',
                    'rrt_timing_algo1_only': 'rrt_timing_algo_only',
                    'rrt_timing_algo2_only': 'rrt_timing_data_only',
                })
                create_plots_algo_vs_bc(results_df=plot_df, output_dir=output_dir)

    print(f"\nResults saved to: {output_dir}")


def create_plots_vs_behaviour_policy(results_df: pd.DataFrame, output_dir: Path):
    """
    Create evaluation comparison plots with all metrics.

    Args:
        results_df: DataFrame with evaluation results
        output_dir: Directory to save plots
    """
    colors = {'cql': '#1f77b4', 'ddqn': '#ff7f0e', 'bcq': '#2ca02c', 'nfq': '#d62728', 'bc': '#9467bd'}

    for split in results_df['split'].unique():
        df = results_df[results_df['split'] == split]
        algos = df['algorithm'].unique()
        x = np.arange(len(algos))
        bar_colors = [colors.get(a, 'gray') for a in algos]

        # Create figure with 2x3 subplots for all metrics
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        fig.suptitle(f'Evaluation Results - {split.upper()} Split', fontsize=14, fontweight='bold')

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
            # Add MC return reference line
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

        # 5. RRT Timing (bottom-middle) - stacked bar
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
        plt.savefig(output_dir / f'evaluation_{split}.png', dpi=150)
        plt.close()
        print(f"  Saved: evaluation_{split}.png")


def create_plots_algo_vs_bc(results_df: pd.DataFrame, output_dir: Path):
    """
    Create comparison plots for RL algorithms vs BC.

    Args:
        results_df: DataFrame with algo-vs-BC results (columns renamed to match standard format)
        output_dir: Directory to save plots
    """
    colors = {'cql': '#1f77b4', 'ddqn': '#ff7f0e', 'bcq': '#2ca02c', 'nfq': '#d62728'}

    for split in results_df['split'].unique():
        df = results_df[results_df['split'] == split]
        algos = df['algorithm'].unique()
        x = np.arange(len(algos))
        bar_colors = [colors.get(a, 'gray') for a in algos]

        # Create figure with 2x3 subplots
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        fig.suptitle(f'RL Algorithms vs BC - {split.upper()} Split', fontsize=14, fontweight='bold')

        # 1. Action Match with BC (top-left)
        ax = axes[0, 0]
        vals = [df[df['algorithm'] == a]['action_match'].mean() * 100 for a in algos]
        ax.bar(x, vals, color=bar_colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in algos])
        ax.set_ylabel('Action Match (%)')
        ax.set_title('Agreement with BC')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)

        # 2. RRT Rate per State comparison (top-middle)
        ax = axes[0, 1]
        vals_algo = [df[df['algorithm'] == a]['rrt_rate_per_state_algo'].mean() * 100 for a in algos]
        bc_rate = df['rrt_rate_per_state_data'].iloc[0] * 100
        ax.bar(x, vals_algo, color=bar_colors, alpha=0.8)
        ax.axhline(y=bc_rate, color='purple', linestyle='--', label=f'BC: {bc_rate:.1f}%')
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in algos])
        ax.set_ylabel('RRT Rate (%)')
        ax.set_title('RRT Rate per State')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 3. RRT Rate per Episode comparison (top-right)
        ax = axes[0, 2]
        vals_algo = [df[df['algorithm'] == a]['rrt_rate_per_episode_algo'].mean() * 100 for a in algos]
        bc_rate = df['rrt_rate_per_episode_data'].iloc[0] * 100
        ax.bar(x, vals_algo, color=bar_colors, alpha=0.8)
        ax.axhline(y=bc_rate, color='purple', linestyle='--', label=f'BC: {bc_rate:.1f}%')
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in algos])
        ax.set_ylabel('RRT Rate (%)')
        ax.set_title('RRT Rate per Episode')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 4. RRT Timing vs BC (bottom-left) - stacked bar
        ax = axes[1, 0]
        timing_metrics = ['rrt_timing_earlier', 'rrt_timing_same', 'rrt_timing_later',
                         'rrt_timing_algo_only', 'rrt_timing_data_only', 'rrt_timing_neither']
        timing_labels = ['Earlier than BC', 'Same as BC', 'Later than BC', 'RL only', 'BC only', 'Neither']
        timing_colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728', '#7f7f7f']

        bottom = np.zeros(len(algos))
        for metric, label, color in zip(timing_metrics, timing_labels, timing_colors):
            vals = [df[df['algorithm'] == a][metric].mean() * 100 for a in algos]
            ax.bar(x, vals, bottom=bottom, label=label, color=color, alpha=0.8)
            bottom += np.array(vals)

        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in algos])
        ax.set_ylabel('Percentage (%)')
        ax.set_title('RRT Timing vs BC')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)

        # 5. Earlier + Same percentage (bottom-middle)
        ax = axes[1, 1]
        earlier_same = [(df[df['algorithm'] == a]['rrt_timing_earlier'].mean() +
                        df[df['algorithm'] == a]['rrt_timing_same'].mean()) * 100 for a in algos]
        ax.bar(x, earlier_same, color=bar_colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in algos])
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Earlier or Same as BC')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(earlier_same):
            ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)

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
                f"{row['rrt_timing_later']*100:.1f}%"
            ])
        table = ax.table(
            cellText=summary_data,
            colLabels=['Algo', 'Match', 'Earlier', 'Same', 'Later'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax.set_title('Summary vs BC', pad=20)

        plt.tight_layout()
        plt.savefig(output_dir / f'evaluation_algo_vs_bc_{split}.png', dpi=150)
        plt.close()
        print(f"  Saved: evaluation_algo_vs_bc_{split}.png")


if __name__ == "__main__":
    main()
