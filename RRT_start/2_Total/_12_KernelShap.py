"""
KernelSHAP Analysis — Explain CQL treatment decisions.

Computes SHAP values on the action advantage: Q(s, RRT) - Q(s, no RRT).
A positive SHAP value means the feature pushes toward starting RRT.

References:
    - Lundberg & Lee (2017), "A Unified Approach to Interpreting Model Predictions"
    - Beechey et al. (2023), "Explaining Reinforcement Learning with Shapley Values" (ICML)
"""

import numpy as np
import pandas as pd
import joblib
import shap
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from utils import load_config, get_data_paths, load_mdp, load_model, save_config_snapshot


# =============================================================================
# DATA HELPERS
# =============================================================================

def get_feature_names(mdp_dir, mdp_name):
    """Load feature names from MDP config."""
    mdp_config = joblib.load(mdp_dir / f"{mdp_name}_config.joblib")
    return mdp_config['state_cols']


def get_observations(dataset):
    """Stack all episode observations into a single array."""
    return np.concatenate([ep.observations for ep in dataset.episodes])


def get_actions(dataset):
    """Stack all episode actions into a single array."""
    return np.concatenate([ep.actions for ep in dataset.episodes])


def sample_background(observations, n_background, seed):
    """Sample background data for KernelSHAP."""
    rng = np.random.default_rng(seed)
    n_total = len(observations)
    if n_total <= n_background:
        return observations
    idx = rng.choice(n_total, size=n_background, replace=False)
    return observations[idx]


def sample_explain(observations, n_explain, seed):
    """Sample observations to explain."""
    rng = np.random.default_rng(seed)
    n_total = len(observations)
    if n_total <= n_explain:
        return observations, np.arange(n_total)
    idx = rng.choice(n_total, size=n_explain, replace=False)
    return observations[idx], idx


# =============================================================================
# PREDICTION FUNCTION
# =============================================================================

def make_advantage_fn(model):
    """
    Create a function: observations -> action advantage.

    Returns Q(s, RRT=1) - Q(s, RRT=0) for each row.
    Positive = model favors RRT.
    """
    def advantage_fn(X):
        X = np.asarray(X, dtype=np.float32)
        n = len(X)
        q_no_rrt = model.predict_value(X, np.zeros(n, dtype=np.int32))
        q_rrt = model.predict_value(X, np.ones(n, dtype=np.int32))
        return q_rrt - q_no_rrt
    return advantage_fn


# =============================================================================
# SHAP COMPUTATION
# =============================================================================

def compute_shap_values(model, background, X_explain, feature_names, n_kmeans):
    """Run KernelSHAP on action advantage."""
    advantage_fn = make_advantage_fn(model)

    background_summary = shap.kmeans(background, min(n_kmeans, len(background)))

    explainer = shap.KernelExplainer(advantage_fn, background_summary)

    shap_values = explainer.shap_values(X_explain, silent=True)

    return shap.Explanation(
        values=shap_values,
        base_values=np.full(len(X_explain), explainer.expected_value),
        data=X_explain,
        feature_names=feature_names,
    )


# =============================================================================
# PLOTS
# =============================================================================

def plot_summary(shap_explanation, output_dir, algo_name):
    """Beeswarm: which features drive RRT decisions globally."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_explanation.values,
        shap_explanation.data,
        feature_names=shap_explanation.feature_names,
        show=False,
        max_display=20,
    )
    plt.title(f"SHAP Summary — {algo_name.upper()} Action Advantage")
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_summary_{algo_name}.png", dpi=150)
    plt.close()
    print(f"  Saved: shap_summary_{algo_name}.png")


def plot_bar(shap_explanation, output_dir, algo_name):
    """Bar chart: mean |SHAP| per feature."""
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_explanation, max_display=20, show=False)
    plt.title(f"Mean |SHAP| — {algo_name.upper()}")
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_bar_{algo_name}.png", dpi=150)
    plt.close()
    print(f"  Saved: shap_bar_{algo_name}.png")


def plot_dependence(shap_explanation, feature_name, output_dir, algo_name):
    """Dependence plot for a single feature."""
    feature_names = shap_explanation.feature_names
    if feature_name not in feature_names:
        print(f"  Warning: {feature_name} not in features, skipping dependence plot")
        return

    idx = feature_names.index(feature_name)
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(
        idx,
        shap_explanation.values,
        shap_explanation.data,
        feature_names=feature_names,
        show=False,
    )
    plt.title(f"{feature_name} — {algo_name.upper()}")
    plt.tight_layout()
    safe_name = feature_name.replace("/", "_")
    plt.savefig(output_dir / f"shap_dep_{algo_name}_{safe_name}.png", dpi=150)
    plt.close()
    print(f"  Saved: shap_dep_{algo_name}_{safe_name}.png")


def plot_waterfall_example(shap_explanation, example_idx, output_dir, algo_name):
    """Waterfall plot for one patient-timestep."""
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(shap_explanation[example_idx], max_display=15, show=False)
    plt.title(f"Example #{example_idx} — {algo_name.upper()}")
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_waterfall_{algo_name}_ex{example_idx}.png", dpi=150)
    plt.close()
    print(f"  Saved: shap_waterfall_{algo_name}_ex{example_idx}.png")


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_shap_csv(shap_explanation, output_dir, algo_name):
    """Save mean |SHAP| per feature to CSV."""
    mean_abs = np.mean(np.abs(shap_explanation.values), axis=0)
    df = pd.DataFrame({
        'feature': shap_explanation.feature_names,
        'mean_abs_shap': mean_abs,
    }).sort_values('mean_abs_shap', ascending=False)
    path = output_dir / f"shap_importance_{algo_name}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved: shap_importance_{algo_name}.csv")
    return df


# =============================================================================
# PER-ALGORITHM PIPELINE
# =============================================================================

def run_shap_for_algo(model, algo_name, obs_train, obs_test, feature_names,
                      output_dir, seed, n_background, n_explain, n_kmeans):
    """Full SHAP pipeline for one algorithm."""
    print(f"\n--- {algo_name.upper()} ---")

    # 1. Sample data
    background = sample_background(obs_train, n_background, seed)
    X_explain, explain_idx = sample_explain(obs_test, n_explain, seed)
    print(f"  Background: {len(background)} | Explain: {len(X_explain)}")

    # 2. Compute SHAP
    print(f"  Computing KernelSHAP...")
    explanation = compute_shap_values(model, background, X_explain, feature_names, n_kmeans)

    # 3. Save CSV
    importance_df = save_shap_csv(explanation, output_dir, algo_name)
    print(f"\n  Top 10 features:")
    for _, row in importance_df.head(10).iterrows():
        print(f"    {row['feature']:30s}  |SHAP| = {row['mean_abs_shap']:.4f}")

    # 4. Summary + bar plots
    plot_summary(explanation, output_dir, algo_name)
    plot_bar(explanation, output_dir, algo_name)

    # 5. Dependence plots for all features
    for feat in feature_names:
        plot_dependence(explanation, feat, output_dir, algo_name)

    # 6. Waterfall for first example
    plot_waterfall_example(explanation, 0, output_dir, algo_name)

    # 7. Save raw SHAP values
    np.save(output_dir / f"shap_values_{algo_name}.npy", explanation.values)
    print(f"  Saved: shap_values_{algo_name}.npy")

    return explanation


# =============================================================================
# DATABASE-LEVEL RUNNER
# =============================================================================

def run_shap_for_db(db_paths, config):
    """Run SHAP analysis for one database."""
    db_name = db_paths['name']
    mdp_dir = db_paths['mdp_dir']
    hpo_dir = db_paths['reward_dir'] / "HPO_results"
    bc_dir = db_paths['reward_dir'] / "BC_results"
    output_dir = db_paths['reward_dir'] / "SHAP_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(output_dir)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = config['processing']['random_state']

    # SHAP settings from config
    shap_cfg = config['kernelshap']
    n_background = shap_cfg['n_background']
    n_explain = shap_cfg['n_explain']
    n_kmeans = shap_cfg['n_kmeans']

    mdp_name = config['evaluation']['mdps'][0]  # Use first MDP (typically mdp1)
    bc_train_split = config['evaluation']['bc']['train_split']

    print(f"\n{'='*60}")
    print(f"KERNEL SHAP ANALYSIS — {db_name}")
    print(f"{'='*60}")
    print(f"  MDP: {mdp_name} | Device: {device} | Seed: {seed}")
    print(f"  Background: {n_background} | Explain: {n_explain}")

    # Load feature names
    feature_names = get_feature_names(mdp_dir, mdp_name)
    print(f"  Features: {len(feature_names)}")

    # Load data
    dataset_train = load_mdp(db_paths=db_paths, mdp_name=mdp_name, split='train')
    dataset_test = load_mdp(db_paths=db_paths, mdp_name=mdp_name, split='test')

    obs_train = get_observations(dataset_train)
    obs_test = get_observations(dataset_test)
    print(f"  Train obs: {obs_train.shape} | Test obs: {obs_test.shape}")

    # Run SHAP for each enabled algorithm
    algos = config['evaluation']['algorithms']

    for algo_name in ['cql', 'ddqn', 'bcq', 'nfq']:
        if not algos.get(algo_name):
            continue
        model_path = hpo_dir / f"best_{algo_name}_model.d3"
        model = load_model(model_path=model_path, device=device)
        if model is None:
            print(f"\n  Skipping {algo_name}: model not found at {model_path}")
            continue
        run_shap_for_algo(
            model, algo_name, obs_train, obs_test, feature_names,
            output_dir, seed, n_background, n_explain, n_kmeans,
        )

    # BC model
    if algos.get('bc'):
        bc_path = bc_dir / f"bc_{mdp_name}_{bc_train_split}.d3"
        bc_model = load_model(model_path=bc_path, device=device)
        if bc_model is None:
            print(f"\n  Skipping BC: model not found at {bc_path}")
        else:
            run_shap_for_algo(
                bc_model, 'bc', obs_train, obs_test, feature_names,
                output_dir, seed, n_background, n_explain, n_kmeans,
            )

    print(f"\n{'='*60}")
    print(f"SHAP results saved to: {output_dir}")
    print(f"{'='*60}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = load_config()
    all_paths = get_data_paths(config)

    for db_key in ['aumc', 'mimic']:
        if config['evaluation']['databases'][db_key]:
            if all_paths[db_key]['mdp_dir'].exists():
                run_shap_for_db(db_paths=all_paths[db_key], config=config)
            else:
                print(f"Skipping {db_key}: MDP directory not found")


if __name__ == "__main__":
    main()
