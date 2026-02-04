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
        fqe_cql: Dict {mdp: {'mean': float, 'ci_low': float, 'ci_high': float}}
        fqe_bc: Dict {mdp: {'mean': float, 'ci_low': float, 'ci_high': float}}
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
    cql_vals = [fqe_cql[m]['mean'] for m in valid]
    cql_err = [
        [abs(fqe_cql[m]['mean'] - fqe_cql[m].get('ci_low', fqe_cql[m]['mean'])) for m in valid],
        [abs(fqe_cql[m].get('ci_high', fqe_cql[m]['mean']) - fqe_cql[m]['mean']) for m in valid]
    ]
    bc_vals = [fqe_bc[m]['mean'] for m in valid]
    bc_err = [
        [abs(fqe_bc[m]['mean'] - fqe_bc[m].get('ci_low', fqe_bc[m]['mean'])) for m in valid],
        [abs(fqe_bc[m].get('ci_high', fqe_bc[m]['mean']) - fqe_bc[m]['mean']) for m in valid]
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
