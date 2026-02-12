"""
Shared plotting functions for RL pipeline.

Contains:
- Training curve plots (TD error, ISV, action match)
- FQE comparison bar charts
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_curves(results: dict, mdps: list, output_dir: Path):
    """
    Create training curve comparison plots.

    Args:
        results: Dict {mdp_name: {'metrics': DataFrame}} with columns:
                 step, td_train, td_val, isv_val, action_match
        mdps: List of MDP names to plot
        output_dir: Output directory for saving plots
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, name in enumerate(mdps):
        if name not in results:
            continue
        df = results[name]['metrics']
        c = colors[i % len(colors)]

        # TD Error (train & val)
        axes[0, 0].plot(df['step'], df['td_train'], label=f'{name} train', color=c, lw=2)
        axes[0, 0].plot(df['step'], df['td_val'], label=f'{name} val', color=c, lw=2, ls='--', alpha=0.7)

        # ISV
        axes[0, 1].plot(df['step'], df['isv_val'], label=name, color=c, lw=2)

        # Action Match
        axes[1, 0].plot(df['step'], df['action_match'] * 100, label=name, color=c, lw=2)

    # Labels
    axes[0, 0].set(xlabel='Steps', ylabel='TD Error', title='TD Error (solid=train, dashed=val)')
    axes[0, 1].set(xlabel='Steps', ylabel='Value', title='Initial State Value (Validation)')
    axes[1, 0].set(xlabel='Steps', ylabel='%', title='Action Match with Clinician')

    for ax in axes.flat[:3]:
        ax.legend()
        ax.grid(alpha=0.3)

    # Summary bar chart
    valid = [m for m in mdps if m in results]
    x = np.arange(len(valid))
    isv_peaks = [results[m]['metrics']['isv_val'].max() for m in valid]
    action_final = [results[m]['metrics']['action_match'].iloc[-1] for m in valid]

    axes[1, 1].bar(x - 0.2, isv_peaks, 0.35, label='ISV Peak', color='steelblue')
    axes[1, 1].bar(x + 0.2, action_final, 0.35, label='Action Match', color='darkorange')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([n.upper() for n in valid])
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()


def plot_fqe_comparison(fqe_cql: dict, fqe_bc: dict, mc: dict, mdps: list, output_dir: Path):
    """
    Create FQE comparison bar chart.

    Args:
        fqe_cql: Dict {mdp: {'fqe_isv': float, 'ci_low': float, 'ci_high': float}}
        fqe_bc: Dict {mdp: {'fqe_isv': float, 'ci_low': float, 'ci_high': float}}
        mc: Dict {mdp: {'mean': float, 'std': float}}
        mdps: List of MDP names to plot
        output_dir: Output directory for saving plots
    """
    valid = [m for m in mdps if m in fqe_cql and m in fqe_bc]
    if not valid:
        return

    x = np.arange(len(valid))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    # Data
    mc_vals = [mc[m]['mean'] for m in valid]
    cql_vals = [fqe_cql[m]['fqe_isv'] for m in valid]
    cql_err = [
        [abs(fqe_cql[m]['fqe_isv'] - fqe_cql[m].get('ci_low', fqe_cql[m]['fqe_isv'])) for m in valid],
        [abs(fqe_cql[m].get('ci_high', fqe_cql[m]['fqe_isv']) - fqe_cql[m]['fqe_isv']) for m in valid]
    ]
    bc_vals = [fqe_bc[m]['fqe_isv'] for m in valid]
    bc_err = [
        [abs(fqe_bc[m]['fqe_isv'] - fqe_bc[m].get('ci_low', fqe_bc[m]['fqe_isv'])) for m in valid],
        [abs(fqe_bc[m].get('ci_high', fqe_bc[m]['fqe_isv']) - fqe_bc[m]['fqe_isv']) for m in valid]
    ]

    # Bars
    ax.bar(x - width, mc_vals, width, label='MC (Data)', color='steelblue')
    ax.bar(x, cql_vals, width, yerr=cql_err, capsize=4, label='FQE (CQL)', color='darkorange')
    ax.bar(x + width, bc_vals, width, yerr=bc_err, capsize=4, label='FQE (BC)', color='forestgreen')

    ax.set(xlabel='MDP', ylabel='Value', title='Policy Evaluation: MC vs FQE [95% CI]')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in valid])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fqe_comparison.png", dpi=150)
    plt.close()


def plot_evaluation_vs_behaviour_policy(results_df, output_dir: Path, title_prefix: str, filename_prefix: str):
    """
    Create evaluation comparison plots (algorithms vs behaviour policy).

    Args:
        results_df: DataFrame with evaluation results (columns: algorithm, split, action_match,
                    rrt_rate_per_state_algo, rrt_rate_per_state_data, rrt_timing_*, fqe_isv, etc.)
        output_dir: Directory to save plots
        title_prefix: Prefix for plot title (e.g., "Evaluation Results" or "External Validation: AUMC → MIMIC")
        filename_prefix: Prefix for output filename (e.g., "evaluation" or "external_validation")
    """
    colors = {'cql': '#1f77b4', 'ddqn': '#ff7f0e', 'bcq': '#2ca02c', 'nfq': '#d62728', 'bc': '#9467bd'}

    for split in results_df['split'].unique():
        df = results_df[results_df['split'] == split]
        algos = df['algorithm'].unique()
        x = np.arange(len(algos))
        bar_colors = [colors.get(a, 'gray') for a in algos]

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        fig.suptitle(f'{title_prefix} - {split.upper()} Split', fontsize=14, fontweight='bold')

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
            if 'mc_return_mean' in df.columns:
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
        plt.savefig(output_dir / f'{filename_prefix}_{split}.png', dpi=150)
        plt.close()
        print(f"  Saved: {filename_prefix}_{split}.png")


def plot_evaluation_algo_vs_bc(results_df, output_dir: Path, title_prefix: str, filename_prefix: str):
    """
    Create comparison plots for RL algorithms vs BC.

    Args:
        results_df: DataFrame with algo-vs-BC results (columns renamed to standard format:
                    algorithm, split, action_match, rrt_rate_per_state_algo/data, rrt_timing_*, etc.)
        output_dir: Directory to save plots
        title_prefix: Prefix for plot title (e.g., "RL Algorithms vs BC" or "RL vs BC: AUMC → MIMIC")
        filename_prefix: Prefix for output filename (e.g., "evaluation_algo_vs_bc")
    """
    colors = {'cql': '#1f77b4', 'ddqn': '#ff7f0e', 'bcq': '#2ca02c', 'nfq': '#d62728'}

    for split in results_df['split'].unique():
        df = results_df[results_df['split'] == split]
        algos = df['algorithm'].unique()
        x = np.arange(len(algos))
        bar_colors = [colors.get(a, 'gray') for a in algos]

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        fig.suptitle(f'{title_prefix} - {split.upper()} Split', fontsize=14, fontweight='bold')

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
        plt.savefig(output_dir / f'{filename_prefix}_{split}.png', dpi=150)
        plt.close()
        print(f"  Saved: {filename_prefix}_{split}.png")
