"""
Explainability Analysis — Compare RL recommendations vs behaviour policy.

For a chosen algorithm and database, produces:
  1. Frequency plot: hours_since_t0 at first RRT start (RL vs behaviour)
  2. Comparison table: mean ± std of each feature at RRT start (RL vs behaviour)

Note: observations are standardized (z-scores). The comparison between
RL and behaviour policy remains valid under scaling.
"""

import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from utils import load_config, get_data_paths, load_mdp, load_model, save_config_snapshot

# =============================================================================
# MAIN
# =============================================================================

def main():
    config = load_config()
    all_paths = get_data_paths(config)

    for db_key in ['aumc', 'mimic']:
        if config['evaluation']['databases'][db_key]:
            if all_paths[db_key]['mdp_dir'].exists():
                run_explainability_for_db(db_paths=all_paths[db_key], config=config)
            else:
                print(f"Skipping {db_key}: MDP directory not found")


# =============================================================================
# DATA HELPERS
# =============================================================================

def get_feature_names(mdp_dir, mdp_name):
    """Load feature names from MDP config."""
    mdp_config = joblib.load(mdp_dir / f"{mdp_name}_config.joblib")
    return mdp_config['state_cols']


def find_first_rrt_behaviour(episode):
    """First timestep where the behaviour policy started RRT (action=1).

    Returns index or None if RRT was never started.
    """
    actions = episode.actions
    rrt_mask = (actions == 1)
    if not rrt_mask.any():
        return None
    return int(np.argmax(rrt_mask))


def find_first_rrt_rl(episode, model):
    """First timestep where the RL model recommends starting RRT.

    Returns index or None if model never recommends RRT.
    """
    predictions = model.predict(episode.observations)
    rrt_mask = (predictions == 1)
    if not rrt_mask.any():
        return None
    return int(np.argmax(rrt_mask))


# =============================================================================
# COLLECT RRT START DATA
# =============================================================================

def collect_rrt_starts(episodes, model, feature_names):
    """For each episode, collect feature vectors at first RRT start.

    Returns:
        rl_df: DataFrame of features at RL-recommended RRT start
        behav_df: DataFrame of features at behaviour policy RRT start
        stats: dict with counts (n_episodes, n_rl_rrt, n_behav_rrt)
    """
    rl_rows = []
    behav_rows = []

    for ep in episodes:
        t_behav = find_first_rrt_behaviour(ep)
        if t_behav is not None:
            behav_rows.append(ep.observations[t_behav])

        t_rl = find_first_rrt_rl(ep, model)
        if t_rl is not None:
            rl_rows.append(ep.observations[t_rl])

    rl_df = pd.DataFrame(rl_rows, columns=feature_names)
    behav_df = pd.DataFrame(behav_rows, columns=feature_names)

    stats = {
        'n_episodes': len(episodes),
        'n_rl_rrt': len(rl_rows),
        'n_behav_rrt': len(behav_rows),
    }
    return rl_df, behav_df, stats


# =============================================================================
# ANALYSIS 1: TIMING COMPARISON
# =============================================================================

def plot_timing_histogram(rl_df, behav_df, time_col, output_dir, algo_name):
    """Frequency plot of RRT start times: RL vs behaviour policy."""
    rl_times = rl_df[time_col].values
    behav_times = behav_df[time_col].values

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.histogram_bin_edges(np.concatenate([rl_times, behav_times]), bins=30)

    ax.hist(behav_times, bins=bins, alpha=0.6, label=f'Behaviour policy (n={len(behav_times)})',
            color='steelblue', edgecolor='white')
    ax.hist(rl_times, bins=bins, alpha=0.6, label=f'{algo_name.upper()} policy (n={len(rl_times)})',
            color='darkorange', edgecolor='white')

    ax.set_xlabel(f'{time_col} (standardized)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'RRT Start Timing — {algo_name.upper()} vs Behaviour Policy')
    ax.legend()
    plt.tight_layout()

    path = output_dir / f"timing_{algo_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path.name}")


# =============================================================================
# ANALYSIS 2: VARIABLE COMPARISON TABLE
# =============================================================================

def build_comparison_table(rl_df, behav_df):
    """Mean ± std per feature at RRT start, for RL vs behaviour.

    Returns DataFrame with columns:
        feature, rl_mean, rl_std, behav_mean, behav_std
    """
    features = rl_df.columns.tolist()

    rows = []
    for feat in features:
        rows.append({
            'feature': feat,
            'rl_mean': rl_df[feat].mean(),
            'rl_std': rl_df[feat].std(),
            'behav_mean': behav_df[feat].mean(),
            'behav_std': behav_df[feat].std(),
        })

    return pd.DataFrame(rows)


def save_comparison_table(table_df, output_dir, algo_name):
    """Save the comparison table as CSV."""
    path = output_dir / f"variable_comparison_{algo_name}.csv"
    table_df.to_csv(path, index=False, float_format='%.4f')
    print(f"  Saved: {path.name}")


def print_comparison_table(table_df, algo_name):
    """Print the comparison table to console."""
    print(f"\n  {'Feature':<30s}  {'RL mean':>9s} ({'std':>7s})  {'Behav mean':>11s} ({'std':>7s})")
    print(f"  {'-'*80}")
    for _, row in table_df.iterrows():
        print(f"  {row['feature']:<30s}  {row['rl_mean']:>9.3f} ({row['rl_std']:>7.3f})  "
              f"{row['behav_mean']:>11.3f} ({row['behav_std']:>7.3f})")


def plot_comparison_table(table_df, output_dir, algo_name, top_n=20):
    """Bar chart comparing RL vs behaviour mean values at RRT start."""
    df = table_df.head(top_n).copy()
    df = df.iloc[::-1]  # Reverse so top feature is at top of plot

    y = np.arange(len(df))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.4)))

    ax.barh(y - height/2, df['behav_mean'], height, xerr=df['behav_std'],
            label='Behaviour policy', color='steelblue', alpha=0.7, capsize=2)
    ax.barh(y + height/2, df['rl_mean'], height, xerr=df['rl_std'],
            label=f'{algo_name.upper()} policy', color='darkorange', alpha=0.7, capsize=2)

    ax.set_yticks(y)
    ax.set_yticklabels(df['feature'], fontsize=8)
    ax.set_xlabel('Feature value (standardized)')
    ax.set_title(f'Patient State at RRT Start — {algo_name.upper()} vs Behaviour')
    ax.legend()
    plt.tight_layout()

    path = output_dir / f"variable_comparison_{algo_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path.name}")


# =============================================================================
# PER-ALGORITHM PIPELINE
# =============================================================================

def run_explainability_for_algo(model, algo_name, episodes, feature_names,
                                output_dir, time_col):
    """Full explainability pipeline for one algorithm."""
    print(f"\n--- {algo_name.upper()} ---")

    # Collect RRT start data
    rl_df, behav_df, stats = collect_rrt_starts(episodes, model, feature_names)
    print(f"  Episodes: {stats['n_episodes']}")
    print(f"  RL starts RRT:        {stats['n_rl_rrt']} ({stats['n_rl_rrt']/stats['n_episodes']:.1%})")
    print(f"  Behaviour starts RRT: {stats['n_behav_rrt']} ({stats['n_behav_rrt']/stats['n_episodes']:.1%})")

    if rl_df.empty or behav_df.empty:
        print(f"  Skipping: not enough RRT starts for comparison")
        return

    # Analysis 1: Timing histogram
    if time_col in feature_names:
        plot_timing_histogram(rl_df, behav_df, time_col, output_dir, algo_name)
    else:
        print(f"  Warning: '{time_col}' not in features, skipping timing plot")

    # Analysis 2: Variable comparison
    table_df = build_comparison_table(rl_df, behav_df)
    print_comparison_table(table_df, algo_name)
    save_comparison_table(table_df, output_dir, algo_name)
    plot_comparison_table(table_df, output_dir, algo_name)


# =============================================================================
# DATABASE-LEVEL RUNNER
# =============================================================================

def run_explainability_for_db(db_paths, config):
    """Run explainability analysis for one database."""
    db_name = db_paths['name']
    mdp_dir = db_paths['mdp_dir']
    hpo_dir = db_paths['reward_dir'] / "HPO_results"
    bc_dir = db_paths['reward_dir'] / "BC_results"
    output_dir = db_paths['reward_dir'] / "Explainability_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(output_dir)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    mdp_name = config['evaluation']['mdps'][0]
    bc_train_split = config['evaluation']['bc']['train_split']
    time_col = 'hours_since_t0'

    print(f"\n{'='*60}")
    print(f"EXPLAINABILITY ANALYSIS — {db_name}")
    print(f"{'='*60}")

    # Load feature names
    feature_names = get_feature_names(mdp_dir, mdp_name)

    # Load test episodes
    dataset_test = load_mdp(db_paths=db_paths, mdp_name=mdp_name, split='test')
    episodes = dataset_test.episodes
    print(f"  MDP: {mdp_name} | Split: test | Episodes: {len(episodes)}")
    print(f"  Features: {len(feature_names)}")

    # Run for each enabled algorithm
    algos = config['evaluation']['algorithms']

    for algo_name in ['cql', 'ddqn', 'bcq', 'nfq']:
        if not algos.get(algo_name):
            continue
        model_path = hpo_dir / f"best_{algo_name}_model.d3"
        model = load_model(model_path=model_path, device=device)
        if model is None:
            print(f"\n  Skipping {algo_name}: model not found")
            continue
        run_explainability_for_algo(model, algo_name, episodes, feature_names,
                                    output_dir, time_col)

    # BC model
    if algos.get('bc'):
        bc_path = bc_dir / f"bc_{mdp_name}_{bc_train_split}.d3"
        bc_model = load_model(model_path=bc_path, device=device)
        if bc_model is None:
            print(f"\n  Skipping BC: model not found")
        else:
            run_explainability_for_algo(bc_model, 'bc', episodes, feature_names,
                                        output_dir, time_col)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
