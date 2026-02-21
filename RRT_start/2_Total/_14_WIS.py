"""
Weighted Importance Sampling (WIS) — Off-policy evaluation.

Estimates the expected return of a trained RL policy using data
collected under the behaviour policy (clinician decisions).

Method:
  1. Fit behaviour policy π_b(a|s) via logistic regression on training data
  2. Compute target policy π_e(a|s) via softmax over Q-values (temperature τ)
  3. Per episode: importance ratio ρ = Π_t π_e(a_t|s_t) / π_b(a_t|s_t)
  4. WIS = Σ (ρ_i · G_i) / Σ ρ_i   (self-normalized, lower variance)
     OIS = (1/N) · Σ (ρ_i · G_i)   (ordinary, unbiased but high variance)

Effective sample size n_eff = (Σ ρ)² / Σ ρ² indicates how many
episodes effectively contribute. Low n_eff means high variance.

Reference: Precup et al. (2000), Thomas & Brunskill (2016)
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from utils import load_config, get_data_paths, load_mdp, load_model, save_config_snapshot
from rl_utils import compute_mc_return
from run_metadata import get_run_metadata, print_run_header, save_run_config, add_metadata_to_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = load_config()
    all_paths = get_data_paths(config)

    if config['wis']['databases']['aumc'] == True:
        print("AUMCdb is enabled for WIS")
        if all_paths['aumc']['mdp_dir'].exists():
            run_wis_for_db(db_paths=all_paths['aumc'], all_paths=all_paths, config=config)
        else:
            print("Skipping AUMCdb: MDP directory not found")

    if config['wis']['databases']['mimic'] == True:
        print("MIMIC is enabled for WIS")
        if all_paths['mimic']['mdp_dir'].exists():
            run_wis_for_db(db_paths=all_paths['mimic'], all_paths=all_paths, config=config)
        else:
            print("Skipping MIMIC: MDP directory not found")


# =============================================================================
# WIS PIPELINE
# =============================================================================

def run_wis_for_db(db_paths, all_paths, config):
    """Run WIS evaluation for one database."""
    db_name = db_paths['name']
    wis_cfg = config['wis']

    # Metadata & header
    metadata = get_run_metadata(config, db_name)
    print_run_header(metadata, title="WEIGHTED IMPORTANCE SAMPLING")

    rl_source = wis_cfg['rl_source_database']
    rl_model_source = wis_cfg['rl_model_source']
    mdp_name = wis_cfg['mdp']
    gamma = wis_cfg['gamma']
    temperature = wis_cfg['temperature']

    print(f"\nWIS CONFIG:")
    print(f"  RL source: {rl_source} ({rl_model_source}) | MDP: {mdp_name}")
    print(f"  Gamma: {gamma} | Temperature: {temperature}")
    print(f"  Splits: {wis_cfg['splits']}")
    print("=" * 80)

    # Output
    output_dir = db_paths['reward_dir'] / "WIS_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(output_dir)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Model directory (from source database)
    source_paths = all_paths[rl_source]
    if rl_model_source == 'hpo':
        model_dir = source_paths['reward_dir'] / "HPO_results"
    else:
        model_dir = source_paths['reward_dir'] / "Ablation_results"

    # Fit behaviour policy π_b(a|s) on training data
    train_ds = load_mdp(db_paths=db_paths, mdp_name=mdp_name, split='train')
    print(f"\nFitting behaviour policy on {len(train_ds.episodes)} training episodes...")
    behaviour_lr = fit_behaviour_policy(train_ds.episodes)

    all_results = []

    for split in wis_cfg['splits']:
        dataset = load_mdp(db_paths=db_paths, mdp_name=mdp_name, split=split)
        episodes = dataset.episodes

        mc_mean, mc_std = compute_mc_return(episodes=episodes, gamma=gamma)
        print(f"\n--- {split.upper()} ({len(episodes)} episodes) ---")
        print(f"  MC Return (behaviour): {mc_mean:.4f} (±{mc_std:.4f})")

        for algo_name in ['cql', 'ddqn', 'bcq', 'nfq']:
            if not wis_cfg['algorithms'][algo_name]:
                continue

            if rl_model_source == 'hpo':
                model_path = model_dir / f"best_{algo_name}_model.d3"
            else:
                model_path = model_dir / f"{mdp_name}_{algo_name}.d3"

            model = load_model(model_path=model_path, device=device)
            if model is None:
                print(f"  {algo_name.upper()}: model not found at {model_path}")
                continue

            # Pre-compute importance weights and returns for all episodes
            rho_arr, G_arr = precompute_weights(
                episodes, model, behaviour_lr, gamma, temperature)

            # WIS estimate
            wis_est, ois_est, n_eff = compute_wis(rho_arr, G_arr)
            print(f"  {algo_name.upper()}: WIS={wis_est:.4f} | OIS={ois_est:.4f} | "
                  f"n_eff={n_eff:.1f}/{len(episodes)}")

            result = {
                'algorithm': algo_name,
                'split': split,
                'wis': wis_est,
                'ois': ois_est,
                'mc_return_mean': mc_mean,
                'mc_return_std': mc_std,
                'n_episodes': len(episodes),
                'n_effective': n_eff,
                'temperature': temperature,
            }

            # Bootstrap CI
            if wis_cfg['bootstrap']['enabled']:
                ci_low, ci_high = bootstrap_wis_ci(
                    rho_arr, G_arr,
                    n_iterations=wis_cfg['bootstrap']['n_iterations'],
                    confidence_level=wis_cfg['bootstrap']['confidence_level'])
                result['wis_ci_low'] = ci_low
                result['wis_ci_high'] = ci_high
                print(f"    Bootstrap CI: [{ci_low:.4f}, {ci_high:.4f}]")

            all_results.append(result)

    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
        print(results_df.to_string(index=False))

        if wis_cfg['output']['save_metrics']:
            results_df = add_metadata_to_df(results_df, metadata)
            results_df.to_csv(output_dir / "wis_results.csv", index=False)

        if wis_cfg['output']['save_plots']:
            plot_wis_comparison(results_df, output_dir)

    save_run_config(output_dir, metadata, eval_config={
        'rl_source': rl_source,
        'rl_model_source': rl_model_source,
        'mdp': mdp_name,
        'gamma': gamma,
        'temperature': temperature,
        'splits': wis_cfg['splits'],
    })

    run_id = metadata['run_id']
    print(f"\nResults saved to: {output_dir}")
    print(f"Run config saved to: run_{run_id}_config.json")


# =============================================================================
# BEHAVIOUR POLICY
# =============================================================================

def fit_behaviour_policy(episodes):
    """
    Fit logistic regression to estimate π_b(a|s) from training data.

    Simple approach: use all (observation, action) pairs from the
    training episodes to train a classifier. predict_proba gives
    calibrated action probabilities per state.
    """
    obs_list, act_list = [], []
    for ep in episodes:
        obs_list.append(ep.observations)
        act_list.append(ep.actions.flatten())

    X = np.concatenate(obs_list).astype(np.float32)
    y = np.concatenate(act_list).astype(int)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y)

    print(f"  Behaviour policy accuracy (train): {lr.score(X, y):.3f}")
    return lr


def get_behaviour_probs(behaviour_lr, observations):
    """Get π_b(a|s) for each observation. Returns (n, 2) array."""
    return behaviour_lr.predict_proba(observations.astype(np.float32))


# =============================================================================
# TARGET POLICY
# =============================================================================

def get_target_probs(rl_model, observations, temperature=1.0):
    """
    Get π_e(a|s) via softmax over Q-values.

    Temperature controls how peaked the distribution is:
      τ → 0: deterministic (greedy)
      τ = 1: standard softmax
      τ → ∞: uniform random
    Returns (n, 2) array of probabilities.
    """
    n = len(observations)
    obs = observations.astype(np.float32)
    q0 = rl_model.predict_value(obs, np.zeros(n, dtype=np.int32))
    q1 = rl_model.predict_value(obs, np.ones(n, dtype=np.int32))

    logits = np.stack([q0, q1], axis=1) / temperature

    # Stable softmax (subtract max to prevent overflow)
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


# =============================================================================
# WIS COMPUTATION
# =============================================================================

def precompute_weights(episodes, rl_model, behaviour_lr, gamma, temperature):
    """
    Pre-compute importance ratios ρ and discounted returns G per episode.

    For each episode i with T timesteps:
      ρ_i = Π_{t=0}^{T} π_e(a_t|s_t) / π_b(a_t|s_t)
      G_i = Σ_{t=0}^{T} γ^t · r_t

    Returns: (rho_array, G_array) both shape (n_episodes,)
    """
    rho_list = []
    G_list = []

    for ep in episodes:
        obs = ep.observations.astype(np.float32)
        actions = ep.actions.flatten().astype(int)
        rewards = ep.rewards.flatten()
        T = len(actions)

        # Probabilities for the actions actually taken
        pi_e = get_target_probs(rl_model, obs, temperature)
        pi_b = get_behaviour_probs(behaviour_lr, obs)

        pi_e_a = pi_e[np.arange(T), actions]
        pi_b_a = np.clip(pi_b[np.arange(T), actions], 1e-6, 1.0)

        # Cumulative importance ratio (log-space for numerical stability)
        log_rho = np.sum(np.log(pi_e_a) - np.log(pi_b_a))
        rho = np.exp(np.clip(log_rho, -20, 20))

        # Discounted return
        gammas = gamma ** np.arange(len(rewards))
        G = np.sum(gammas * rewards)

        rho_list.append(rho)
        G_list.append(G)

    return np.array(rho_list), np.array(G_list)


def compute_wis(rho_arr, G_arr):
    """
    Compute WIS, OIS, and effective sample size.

    WIS (weighted/self-normalized): lower variance, slight bias
    OIS (ordinary): unbiased but often very high variance
    n_eff: effective sample size — if much lower than N, estimates are unreliable
    """
    # Ordinary IS
    ois = np.mean(rho_arr * G_arr)

    # Weighted IS (self-normalized)
    rho_sum = rho_arr.sum()
    wis = np.sum(rho_arr * G_arr) / rho_sum if rho_sum > 0 else np.nan

    # Effective sample size: (Σρ)² / Σρ²
    rho_sq_sum = np.sum(rho_arr ** 2)
    n_eff = (rho_sum ** 2 / rho_sq_sum) if rho_sq_sum > 0 else 0.0

    return wis, ois, n_eff


def bootstrap_wis_ci(rho_arr, G_arr, n_iterations, confidence_level):
    """
    Bootstrap confidence interval for WIS.

    Resamples episodes (with replacement) and recomputes WIS
    to get a distribution of estimates.
    """
    rng = np.random.default_rng(42)
    n = len(rho_arr)
    estimates = np.zeros(n_iterations)

    for i in range(n_iterations):
        idx = rng.choice(n, size=n, replace=True)
        rho_sum = rho_arr[idx].sum()
        if rho_sum > 0:
            estimates[i] = np.sum(rho_arr[idx] * G_arr[idx]) / rho_sum
        else:
            estimates[i] = np.nan

    alpha = 1 - confidence_level
    ci_low = np.nanpercentile(estimates, 100 * alpha / 2)
    ci_high = np.nanpercentile(estimates, 100 * (1 - alpha / 2))

    return ci_low, ci_high


# =============================================================================
# PLOT
# =============================================================================

def plot_wis_comparison(results_df, output_dir):
    """Bar chart: WIS estimates vs MC return per algorithm."""
    for split in results_df['split'].unique():
        df = results_df[results_df['split'] == split]
        algos = df['algorithm'].values
        x = np.arange(len(algos))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, df['wis'].values, width, label='WIS', color='darkorange')
        ax.bar(x + width/2, df['mc_return_mean'].values, width, label='MC Return', color='steelblue')

        # Bootstrap CI error bars
        if 'wis_ci_low' in df.columns:
            err_low = df['wis'].values - df['wis_ci_low'].values
            err_high = df['wis_ci_high'].values - df['wis'].values
            ax.errorbar(x - width/2, df['wis'].values,
                        yerr=[err_low, err_high],
                        fmt='none', color='black', capsize=5)

        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in algos])
        ax.set_ylabel('Estimated Value')
        ax.set_title(f'WIS vs MC Return — {split.upper()} Split')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"wis_comparison_{split}.png", dpi=150)
        plt.close()
        print(f"  Saved: wis_comparison_{split}.png")


# =============================================================================
if __name__ == "__main__":
    main()
