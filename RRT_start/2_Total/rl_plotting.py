"""
Shared plotting functions for RL pipeline.

Contains:
- Individual evaluation plots (FQE ISV, action match, RRT rates, timing, summary)
- Algo vs BC comparison plots
- Ablation training curve plots
- FQE comparison bar charts
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


########################################################################################
# INDIVIDUAL EVALUATION PLOTS (vs Behaviour Policy)
########################################################################################

def plot_fqe_isv(results_df, output_dir: Path, title_prefix: str = "", filename_prefix: str = ""):
    """
    FQE ISV bar chart with MC return reference line.
    When bootstrap data is available, uses fqe_isv_mean as bar height with CI error bars.
    """
    colors = {'cql': '#1f77b4', 'ddqn': '#ff7f0e', 'bcq': '#2ca02c', 'nfq': '#d62728', 'bc': '#9467bd'}

    if 'fqe_isv' not in results_df.columns:
        return

    algos = results_df['algorithm'].unique()
    x = np.arange(len(algos))
    bar_colors = [colors.get(a, 'gray') for a in algos]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Use fqe_isv_mean as bar height when bootstrap is available
    if 'fqe_isv_mean' in results_df.columns and results_df['fqe_isv_mean'].notna().any():
        vals = [results_df[results_df['algorithm'] == a]['fqe_isv_mean'].mean() for a in algos]
        ci_lo = [results_df[results_df['algorithm'] == a]['fqe_ci_low'].mean() for a in algos]
        ci_hi = [results_df[results_df['algorithm'] == a]['fqe_ci_high'].mean() for a in algos]
        err = [np.abs(np.array(vals) - np.array(ci_lo)), np.abs(np.array(ci_hi) - np.array(vals))]
        ax.bar(x, vals, yerr=err, capsize=5, color=bar_colors, alpha=0.8)
    else:
        vals = [results_df[results_df['algorithm'] == a]['fqe_isv'].mean() for a in algos]
        ax.bar(x, vals, color=bar_colors, alpha=0.8)

    if 'mc_return_mean' in results_df.columns:
        mc_return = results_df['mc_return_mean'].iloc[0]
        ax.axhline(y=mc_return, color='red', linestyle='--', label=f'MC Return: {mc_return:.3f}')
        ax.legend()

    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in algos])
    ax.set_ylabel('FQE ISV')
    ax.set_title(f'{title_prefix} - Policy Value (FQE)')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{filename_prefix}_fqe_isv.png', dpi=150)
    plt.close()
    print(f"  Saved: {filename_prefix}_fqe_isv.png")


def plot_action_match(results_df, output_dir: Path, title_prefix: str = "", filename_prefix: str = ""):
    """Action match percentage bar chart."""
    colors = {'cql': '#1f77b4', 'ddqn': '#ff7f0e', 'bcq': '#2ca02c', 'nfq': '#d62728', 'bc': '#9467bd'}

    algos = results_df['algorithm'].unique()
    x = np.arange(len(algos))
    bar_colors = [colors.get(a, 'gray') for a in algos]

    fig, ax = plt.subplots(figsize=(8, 6))

    vals = [results_df[results_df['algorithm'] == a]['action_match'].mean() * 100 for a in algos]
    ax.bar(x, vals, color=bar_colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in algos])
    ax.set_ylabel('Action Match (%)')
    ax.set_title(f'{title_prefix} - Clinician Agreement')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / f'{filename_prefix}_action_match.png', dpi=150)
    plt.close()
    print(f"  Saved: {filename_prefix}_action_match.png")


def plot_rrt_rate_per_state(results_df, output_dir: Path, title_prefix: str = "", filename_prefix: str = ""):
    """RRT rate per state bar chart with data reference line."""
    colors = {'cql': '#1f77b4', 'ddqn': '#ff7f0e', 'bcq': '#2ca02c', 'nfq': '#d62728', 'bc': '#9467bd'}

    algos = results_df['algorithm'].unique()
    x = np.arange(len(algos))
    bar_colors = [colors.get(a, 'gray') for a in algos]

    fig, ax = plt.subplots(figsize=(8, 6))

    vals_algo = [results_df[results_df['algorithm'] == a]['rrt_rate_per_state_algo'].mean() * 100 for a in algos]
    data_rate = results_df['rrt_rate_per_state_data'].iloc[0] * 100
    ax.bar(x, vals_algo, color=bar_colors, alpha=0.8)
    ax.axhline(y=data_rate, color='red', linestyle='--', label=f'Data: {data_rate:.1f}%')
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in algos])
    ax.set_ylabel('RRT Rate (%)')
    ax.set_title(f'{title_prefix} - RRT Rate per State')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{filename_prefix}_rrt_rate_per_state.png', dpi=150)
    plt.close()
    print(f"  Saved: {filename_prefix}_rrt_rate_per_state.png")


def plot_rrt_rate_per_episode(results_df, output_dir: Path, title_prefix: str = "", filename_prefix: str = ""):
    """RRT rate per episode bar chart with data reference line."""
    colors = {'cql': '#1f77b4', 'ddqn': '#ff7f0e', 'bcq': '#2ca02c', 'nfq': '#d62728', 'bc': '#9467bd'}

    algos = results_df['algorithm'].unique()
    x = np.arange(len(algos))
    bar_colors = [colors.get(a, 'gray') for a in algos]

    fig, ax = plt.subplots(figsize=(8, 6))

    vals_algo = [results_df[results_df['algorithm'] == a]['rrt_rate_per_episode_algo'].mean() * 100 for a in algos]
    data_rate = results_df['rrt_rate_per_episode_data'].iloc[0] * 100
    ax.bar(x, vals_algo, color=bar_colors, alpha=0.8)
    ax.axhline(y=data_rate, color='red', linestyle='--', label=f'Data: {data_rate:.1f}%')
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in algos])
    ax.set_ylabel('RRT Rate (%)')
    ax.set_title(f'{title_prefix} - RRT Rate per Episode')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{filename_prefix}_rrt_rate_per_episode.png', dpi=150)
    plt.close()
    print(f"  Saved: {filename_prefix}_rrt_rate_per_episode.png")


def plot_rrt_timing(results_df, output_dir: Path, title_prefix: str = "", filename_prefix: str = ""):
    """RRT timing stacked bar chart."""
    algos = results_df['algorithm'].unique()
    x = np.arange(len(algos))

    fig, ax = plt.subplots(figsize=(8, 6))

    timing_metrics = ['rrt_timing_earlier', 'rrt_timing_same', 'rrt_timing_later',
                     'rrt_timing_algo_only', 'rrt_timing_data_only', 'rrt_timing_neither']
    timing_labels = ['Earlier', 'Same', 'Later', 'Algo only', 'Data only', 'Neither']
    timing_colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728', '#7f7f7f']

    bottom = np.zeros(len(algos))
    for metric, label, color in zip(timing_metrics, timing_labels, timing_colors):
        vals = [results_df[results_df['algorithm'] == a][metric].mean() * 100 for a in algos]
        ax.bar(x, vals, bottom=bottom, label=label, color=color, alpha=0.8)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in algos])
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'{title_prefix} - RRT Timing vs Data')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{filename_prefix}_rrt_timing.png', dpi=150)
    plt.close()
    print(f"  Saved: {filename_prefix}_rrt_timing.png")


def plot_summary_table(results_df, output_dir: Path, title_prefix: str = "", filename_prefix: str = ""):
    """Summary table as image."""
    algos = results_df['algorithm'].unique()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')

    summary_data = []
    for a in algos:
        row = results_df[results_df['algorithm'] == a].iloc[0]
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
    ax.set_title(f'{title_prefix} - Summary', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / f'{filename_prefix}_summary.png', dpi=150)
    plt.close()
    print(f"  Saved: {filename_prefix}_summary.png")


def plot_all_evaluation_vs_behaviour_policy(results_df, output_dir: Path, title_prefix: str = "", filename_prefix: str = ""):
    """Convenience wrapper: creates all 6 individual evaluation plots."""
    plot_fqe_isv(results_df, output_dir, title_prefix, filename_prefix)
    plot_action_match(results_df, output_dir, title_prefix, filename_prefix)
    plot_rrt_rate_per_state(results_df, output_dir, title_prefix, filename_prefix)
    plot_rrt_rate_per_episode(results_df, output_dir, title_prefix, filename_prefix)
    plot_rrt_timing(results_df, output_dir, title_prefix, filename_prefix)
    plot_summary_table(results_df, output_dir, title_prefix, filename_prefix)


########################################################################################
# ALGO VS BC COMPARISON PLOT
########################################################################################

def plot_evaluation_algo_vs_bc(results_df, output_dir: Path, title_prefix: str, filename_prefix: str):
    """
    Create comparison plots for RL algorithms vs BC (2x3 grid).

    Args:
        results_df: DataFrame with algo-vs-BC results (columns renamed to standard format:
                    algorithm, action_match, rrt_rate_per_state_algo/data, rrt_timing_*, etc.)
        output_dir: Directory to save plots
        title_prefix: Prefix for plot title
        filename_prefix: Prefix for output filename
    """
    colors = {'cql': '#1f77b4', 'ddqn': '#ff7f0e', 'bcq': '#2ca02c', 'nfq': '#d62728'}

    algos = results_df['algorithm'].unique()
    x = np.arange(len(algos))
    bar_colors = [colors.get(a, 'gray') for a in algos]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    fig.suptitle(f'{title_prefix}', fontsize=14, fontweight='bold')

    # 1. Action Match with BC (top-left)
    ax = axes[0, 0]
    vals = [results_df[results_df['algorithm'] == a]['action_match'].mean() * 100 for a in algos]
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
    vals_algo = [results_df[results_df['algorithm'] == a]['rrt_rate_per_state_algo'].mean() * 100 for a in algos]
    bc_rate = results_df['rrt_rate_per_state_data'].iloc[0] * 100
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
    vals_algo = [results_df[results_df['algorithm'] == a]['rrt_rate_per_episode_algo'].mean() * 100 for a in algos]
    bc_rate = results_df['rrt_rate_per_episode_data'].iloc[0] * 100
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
        vals = [results_df[results_df['algorithm'] == a][metric].mean() * 100 for a in algos]
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
    earlier_same = [(results_df[results_df['algorithm'] == a]['rrt_timing_earlier'].mean() +
                    results_df[results_df['algorithm'] == a]['rrt_timing_same'].mean()) * 100 for a in algos]
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
        row = results_df[results_df['algorithm'] == a].iloc[0]
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
    plt.savefig(output_dir / f'{filename_prefix}.png', dpi=150)
    plt.close()
    print(f"  Saved: {filename_prefix}.png")


########################################################################################
# ABLATION TRAINING CURVE PLOTS
########################################################################################

def plot_ablation_training_curves(results: dict, comparisons: dict, output_dir: Path):
    """
    Per comparison group from config: 3 separate plots (td_train, td_val, isv_train).

    Args:
        results: Dict {mdp_name: {'metrics': DataFrame with step, td_train, td_val, isv_train, ...}}
        comparisons: Dict from config['ablation']['comparisons']
        output_dir: Output directory for saving plots
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    metrics_to_plot = [
        ('td_train', 'TD Error (Train)'),
        ('td_val', 'TD Error (Validation)'),
        ('isv_train', 'Initial State Value (Train)'),
    ]

    for group_key, group_cfg in comparisons.items():
        mdps = [m for m in group_cfg['mdps'] if m in results]
        if not mdps:
            continue
        group_name = group_cfg['name']

        for metric_col, metric_label in metrics_to_plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            for i, name in enumerate(mdps):
                df = results[name]['metrics']
                ax.plot(df['step'], df[metric_col],
                        label=name.upper(), color=colors[i % len(colors)], lw=2)
            ax.set(xlabel='Steps', ylabel=metric_label,
                   title=f'{group_name}: {metric_label}')
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / f"{group_key}_{metric_col}.png", dpi=150)
            plt.close()
            print(f"  Saved: {group_key}_{metric_col}.png")


########################################################################################
# ABLATION FQE COMPARISON PLOTS
########################################################################################

def plot_fqe_comparison(fqe_cql: dict, fqe_bc: dict, mc: dict, mdps: list, output_dir: Path):
    """
    Create FQE comparison bar chart (CQL vs BC vs MC).
    When bootstrap data is available, uses mean_isv as bar height.
    """
    valid = [m for m in mdps if m in fqe_cql and m in fqe_bc]
    if not valid:
        return

    x = np.arange(len(valid))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    # MC data
    mc_vals = [mc[m]['mean'] for m in valid]

    # CQL bars - use mean_isv when bootstrap available
    cql_vals = [fqe_cql[m].get('mean_isv', fqe_cql[m]['fqe_isv']) for m in valid]
    cql_err = None
    if 'ci_low' in fqe_cql[valid[0]]:
        cql_err = [
            [abs(cql_vals[i] - fqe_cql[m]['ci_low']) for i, m in enumerate(valid)],
            [abs(fqe_cql[m]['ci_high'] - cql_vals[i]) for i, m in enumerate(valid)]
        ]

    # BC bars - use mean_isv when bootstrap available
    bc_vals = [fqe_bc[m].get('mean_isv', fqe_bc[m]['fqe_isv']) for m in valid]
    bc_err = None
    if 'ci_low' in fqe_bc[valid[0]]:
        bc_err = [
            [abs(bc_vals[i] - fqe_bc[m]['ci_low']) for i, m in enumerate(valid)],
            [abs(fqe_bc[m]['ci_high'] - bc_vals[i]) for i, m in enumerate(valid)]
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


def plot_fqe_vs_mc(fqe_cql: dict, mc: dict, mdps: list, output_dir: Path):
    """
    FQE ISV (CQL) vs MC return bar chart — without BC.
    When bootstrap data is available, uses mean_isv as bar height.
    """
    valid = [m for m in mdps if m in fqe_cql and m in mc]
    if not valid:
        return

    x = np.arange(len(valid))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    mc_vals = [mc[m]['mean'] for m in valid]

    # Use mean_isv when bootstrap available
    cql_vals = [fqe_cql[m].get('mean_isv', fqe_cql[m]['fqe_isv']) for m in valid]

    cql_err = None
    if 'ci_low' in fqe_cql[valid[0]]:
        cql_err = [
            [abs(cql_vals[i] - fqe_cql[m]['ci_low']) for i, m in enumerate(valid)],
            [abs(fqe_cql[m]['ci_high'] - cql_vals[i]) for i, m in enumerate(valid)],
        ]

    ax.bar(x - width / 2, mc_vals, width, label='MC Return', color='steelblue')
    ax.bar(x + width / 2, cql_vals, width, yerr=cql_err, capsize=4,
           label='FQE ISV (CQL)', color='darkorange')

    ax.set(xlabel='MDP', ylabel='Value',
           title='Policy Evaluation: CQL FQE ISV vs MC Return (Validation)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in valid])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fqe_vs_mc.png", dpi=150)
    plt.close()
    print(f"  Saved: fqe_vs_mc.png")


########################################################################################
# ABLATION INDIVIDUAL TRAINING CURVE PLOTS
########################################################################################

def plot_ablation_td_error(results: dict, mdps: list, output_dir: Path):
    """Separate TD error plot (train & val) for ablation."""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(mdps):
        if name not in results:
            continue
        df = results[name]['metrics']
        c = colors[i % len(colors)]
        ax.plot(df['step'], df['td_train'], label=f'{name} train', color=c, lw=2)
        ax.plot(df['step'], df['td_val'], label=f'{name} val', color=c, lw=2, ls='--', alpha=0.7)
    ax.set(xlabel='Steps', ylabel='TD Error', title='TD Error (solid=train, dashed=val)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_td_error.png", dpi=150)
    plt.close()
    print(f"  Saved: training_td_error.png")


def plot_ablation_isv(results: dict, mdps: list, output_dir: Path):
    """Separate ISV plot (train & val) for ablation."""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(mdps):
        if name not in results:
            continue
        df = results[name]['metrics']
        c = colors[i % len(colors)]
        ax.plot(df['step'], df['isv_train'], label=f'{name} train', color=c, lw=2)
        ax.plot(df['step'], df['isv_val'], label=f'{name} val', color=c, lw=2, ls='--', alpha=0.7)
    ax.set(xlabel='Steps', ylabel='Value', title='Initial State Value (solid=train, dashed=val)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_isv.png", dpi=150)
    plt.close()
    print(f"  Saved: training_isv.png")


def plot_ablation_action_match(results: dict, mdps: list, output_dir: Path):
    """Separate action match training curve for ablation."""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(mdps):
        if name not in results:
            continue
        df = results[name]['metrics']
        ax.plot(df['step'], df['action_match'] * 100, label=name, color=colors[i % len(colors)], lw=2)
    ax.set(xlabel='Steps', ylabel='%', title='Action Match with Clinician')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_action_match.png", dpi=150)
    plt.close()
    print(f"  Saved: training_action_match.png")


def plot_ablation_summary_bar(results: dict, mdps: list, output_dir: Path):
    """Separate summary bar chart (ISV peak + action match) for ablation."""
    valid = [m for m in mdps if m in results]
    if not valid:
        return
    x = np.arange(len(valid))
    isv_peaks = [results[m]['metrics']['isv_val'].max() for m in valid]
    action_final = [results[m]['metrics']['action_match'].iloc[-1] for m in valid]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 0.2, isv_peaks, 0.35, label='ISV Peak', color='steelblue')
    ax.bar(x + 0.2, action_final, 0.35, label='Action Match', color='darkorange')
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in valid])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_title('Summary: ISV Peak & Action Match')
    plt.tight_layout()
    plt.savefig(output_dir / "training_summary.png", dpi=150)
    plt.close()
    print(f"  Saved: training_summary.png")
