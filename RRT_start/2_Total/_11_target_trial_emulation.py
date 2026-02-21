"""
Target Trial Emulation using Clone-Censor-Reweight (IPCW).

Emulates a randomized trial comparing two treatment strategies:
  - Arm 1: RL policy (e.g., CQL trained on source database)
  - Arm 2: Observed data (standard of care — actual treatment decisions)

Each patient is "cloned" into both arms. The RL arm is censored when
actual behaviour deviates from the RL recommendation, and reweighted
via stabilized IPCW to correct for informative censoring.
The data arm uses the actual outcomes without censoring or reweighting.

Outcome: Weighted Kaplan-Meier survival curves over a configurable horizon.

Reference: Hernan & Robins (2016), "Using Big Data to Emulate a Target Trial"
"""

import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
from lifelines import KaplanMeierFitter

warnings.filterwarnings('ignore', module='lifelines')

from utils import load_config, get_data_paths, load_model, save_config_snapshot
from _4_mdp_preparation import get_feature_cols
from run_metadata import get_run_metadata, print_run_header, save_run_config, add_metadata_to_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = load_config()
    all_paths = get_data_paths(config)
    run_tte(all_paths, config)


# =============================================================================
# TTE PIPELINE
# =============================================================================

def run_tte(all_paths, config):

    # Target database:
    db_paths = all_paths[config['target_trial_emulation']['database']]

    # TTE settings
    rl_source = config['target_trial_emulation']['rl_source_database']
    rl_model_source = config['target_trial_emulation']['rl_model_source']
    rl_dir = "HPO_results" if rl_model_source == 'hpo' else "Ablation_results"
    mdp_name = config['target_trial_emulation']['mdp']
    horizon = config['target_trial_emulation']['horizon_days']

    # Metadata & header
    metadata = get_run_metadata(config, db_paths['name'])
    print_run_header(metadata, title="TARGET TRIAL EMULATION")

    print(f"\nTTE CONFIG:")
    print(f"  RL source: {rl_source} ({rl_model_source}) | MDP: {mdp_name}")
    print(f"  Target database: {config['target_trial_emulation']['database']}")
    print(f"  Horizon: {horizon} days")
    print("=" * 80)

    # Output of results
    output_dir = all_paths[rl_source]['reward_dir'] / "TTE_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(output_dir)

    # 1. Load scaled data
    df_orig = pd.read_parquet(db_paths['scaled_path'])
    df_orig = df_orig.sort_values(['visit_occurrence_id', 'hours_since_t0']).reset_index(drop=True)
    print(f"Loaded {len(df_orig)} rows, {df_orig['visit_occurrence_id'].nunique()} patients")

    # Feature columns (same logic as MDP preparation)
    base_cols, _, _ = get_feature_cols(df_orig, config)
    feature_cols = sorted(base_cols)
    obs = df_orig[feature_cols].values.astype(np.float32)
    print(f"Features: {len(feature_cols)}")

    # Recover raw hours (needed for survival data)
    df_orig = add_raw_hours(df_orig, db_paths)

    # IPCW settings
    baseline_vars = config["target_trial_emulation"]['ipcw']['baseline_vars']
    time_varying_vars = [c for c in feature_cols if c not in baseline_vars]

    # 2. Loop over enabled RL algorithms
    enabled_algos = [name for name, enabled in config["target_trial_emulation"]['algorithms'].items() if enabled]
    print(f"\nAlgorithms: {', '.join(a.upper() for a in enabled_algos)}")

    for algo_name in enabled_algos:
        print(f"\n{'='*60}")
        print(f"  TTE: {algo_name.upper()} vs Observed")
        print(f"{'='*60}")

        # Load RL model
        if rl_model_source == 'hpo':
            rl_path = all_paths[rl_source]['reward_dir'] / rl_dir / f"best_{algo_name}_model.d3"
        else:
            rl_path = all_paths[rl_source]['reward_dir'] / rl_dir / f"{mdp_name}_{algo_name}.d3"
        rl_model = load_model(rl_path)

        # Check if model exists
        if rl_model is None:
            print(f"  Skipping {algo_name}: model not found at {rl_path}")
            continue

        # Copy df --> avoid potential leakage or dubble use
        df = df_orig.copy()

        # Predict RL actions
        df['action_rl'] = rl_model.predict(obs)

        # Action statistics
        print_action_stats(df, algo_name)

        # Censor RL arm (data arm needs no censoring)
        df = add_censoring(df, 'action_rl', 'C_rl')

        n_patients = df['visit_occurrence_id'].nunique()
        n_adherent_rl = df.groupby('visit_occurrence_id')['C_rl'].max().eq(0).sum()
        print(f"  Adherent to {algo_name.upper()}: {n_adherent_rl}/{n_patients}")

        # IPCW (only for RL arm)
        print("  Computing IPCW weights...")
        ipcw_rl_col = compute_ipcw(df, 'C_rl', algo_name, baseline_vars, time_varying_vars, config)

        # Patient-level survival
        patient = build_patient_survival(df, horizon, ipcw_rl_col)

        # Survival at horizon
        surv_data = km_survival_at_horizon(patient, 'time_data', 'event_data', 'weight_data', horizon)
        surv_rl = km_survival_at_horizon(patient, 'time_rl', 'event_rl', 'weight_rl', horizon)
        print(f"\n  {horizon}-day survival:")
        print(f"    Observed:           {surv_data:.4f}")
        print(f"    {algo_name.upper()} (IPCW-weighted): {surv_rl:.4f}")
        print(f"    Difference (RL-Data): {surv_rl - surv_data:+.4f}")

        # Bootstrap survival difference
        boot_cfg = config["target_trial_emulation"]['bootstrap']
        if boot_cfg['enabled']:
            boot_results = bootstrap_survival_difference(patient, horizon, boot_cfg, algo_name)
            boot_results.to_csv(output_dir / f"tte_{algo_name}_bootstrap.csv", index=False)

        # Kaplan-Meier
        plot_weighted_km(patient, horizon, output_dir, algo_name)

        # Save per-algorithm results
        patient.to_csv(output_dir / f"tte_{algo_name}_patient_survival.csv", index=False)

    # Save run config
    save_run_config(output_dir, metadata, training_config={
        'algorithms': enabled_algos,
        'rl_source': rl_source,
        'horizon_days': horizon,
    })
    print(f"\nAll results saved to {output_dir}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_action_stats(df, algo_name):
    """Print action frequency and episode-level RRT agreement statistics."""

    # Per-state RRT rates
    rrt_state_algo = (df['action_rl'] == 1).mean()
    rrt_state_data = (df['action_rrt'] == 1).mean()

    # Per-episode: does RRT start at least once?
    ep = df.groupby('visit_occurrence_id')
    algo_starts = ep['action_rl'].max()   # 1 if RRT started in episode
    data_starts = ep['action_rrt'].max()

    rrt_ep_algo = algo_starts.mean()
    rrt_ep_data = data_starts.mean()

    # Agreement: of episodes where data starts RRT, how many does algo also start?
    data_rrt_eps = data_starts[data_starts == 1]
    algo_rrt_eps = algo_starts[algo_starts == 1]

    if len(data_rrt_eps) > 0:
        algo_given_data = algo_starts.reindex(data_rrt_eps.index).mean()
    else:
        algo_given_data = float('nan')

    if len(algo_rrt_eps) > 0:
        data_given_algo = data_starts.reindex(algo_rrt_eps.index).mean()
    else:
        data_given_algo = float('nan')

    print(f"  RRT per state  — {algo_name.upper()}: {rrt_state_algo:.1%} | Data: {rrt_state_data:.1%}")
    print(f"  RRT per episode — {algo_name.upper()}: {rrt_ep_algo:.1%} | Data: {rrt_ep_data:.1%}")
    print(f"  Data starts RRT → {algo_name.upper()} also starts: {algo_given_data:.1%}")
    print(f"  {algo_name.upper()} starts RRT → Data also starts: {data_given_algo:.1%}")


def add_censoring(df, action_col, censor_col):
    """Flag deviation from arm. cummax() = once censored, always censored."""
    df[censor_col] = (df['action_rrt'] != df[action_col]).astype(int)
    df[censor_col] = df.groupby('visit_occurrence_id')[censor_col].cummax()
    return df


def add_raw_hours(df, db_paths):
    """Recover raw hours_since_t0 by inverting the standard scaling."""
    scaler_info = joblib.load(db_paths['scaler_path'])
    scaler = scaler_info['scaler']
    scaled_cols = scaler_info['columns_to_scale']
    hours_idx = scaled_cols.index('hours_since_t0')
    df['raw_hours'] = df['hours_since_t0'] * scaler.scale_[hours_idx] + scaler.mean_[hours_idx]
    return df


def compute_ipcw(df, censor_col, label, baseline_vars, time_varying_vars, config):
    """
    Stabilized IPCW weights for one arm.

    Per timestep, fits two logistic regressions:
      - Numerator:   P(uncensored | baseline)           [stabilization]
      - Denominator: P(uncensored | baseline + time-varying)
    Cumulative product over time = final weight.
    """
    max_iter = config["target_trial_emulation"]['ipcw']['max_iter']
    clip_lo = config["target_trial_emulation"]['ipcw']['weight_clip_lower']
    clip_hi = config["target_trial_emulation"]['ipcw']['weight_clip_upper']
    cap_q = config["target_trial_emulation"]['ipcw']['cumulative_cap_quantile']

    # Lagged censoring: was the patient already censored at previous step?
    prev_col = f'{censor_col}_prev'
    df[prev_col] = df.groupby('visit_occurrence_id')[censor_col].shift(1).fillna(0).astype(int)

    # At-risk = not yet censored
    at_risk = df[df[prev_col] == 0].copy()
    at_risk['stayed'] = (at_risk[censor_col] == 0).astype(int)

    print(f"  {label}: {len(at_risk)} at-risk obs, {(at_risk['stayed'] == 0).sum()} censored")

    at_risk['sw_t'] = 1.0
    for t in sorted(at_risk['hours_since_t0'].unique()):
        mask = at_risk['hours_since_t0'] == t
        sub = at_risk.loc[mask]

        if len(sub) < 10 or sub['stayed'].nunique() < 2:
            continue

        # Numerator: P(uncensored | baseline)
        lr_num = LogisticRegression(max_iter=max_iter)
        lr_num.fit(sub[baseline_vars], sub['stayed'])
        p_num = lr_num.predict_proba(sub[baseline_vars])[:, 1]

        # Denominator: P(uncensored | baseline + time-varying)
        lr_den = LogisticRegression(max_iter=max_iter)
        lr_den.fit(sub[baseline_vars + time_varying_vars], sub['stayed'])
        p_den = lr_den.predict_proba(sub[baseline_vars + time_varying_vars])[:, 1]

        at_risk.loc[mask, 'sw_t'] = np.clip(p_num / np.maximum(p_den, clip_lo), clip_lo, clip_hi)

    # Cumulative product, capped
    at_risk['ipcw'] = at_risk.groupby('visit_occurrence_id')['sw_t'].cumprod()
    at_risk['ipcw'] = at_risk['ipcw'].clip(upper=at_risk['ipcw'].quantile(cap_q))

    # Map back, forward-fill
    ipcw_col = f'ipcw_{label}'
    df[ipcw_col] = np.nan
    df.loc[at_risk.index, ipcw_col] = at_risk['ipcw']
    df[ipcw_col] = df.groupby('visit_occurrence_id')[ipcw_col].ffill().fillna(1.0)

    uncens = df[df[censor_col] == 0]
    print(f"    Weights: mean={uncens[ipcw_col].mean():.3f}, "
          f"median={uncens[ipcw_col].median():.3f}, max={uncens[ipcw_col].max():.3f}")
    return ipcw_col


def build_patient_survival(df, horizon_days, ipcw_rl_col):
    """
    Patient-level time-to-event for both arms.

    Data arm: actual outcomes — no censoring, weight = 1.
    RL arm:   censored when treatment deviates, reweighted by IPCW.
    """
    patient = df.groupby('visit_occurrence_id').agg(
        t0=('t0', 'first'), death_dt=('death_dt', 'first'),
    ).reset_index()

    patient['t0'] = pd.to_datetime(patient['t0'])
    patient['death_dt'] = pd.to_datetime(patient['death_dt'])
    patient['death_days'] = (patient['death_dt'] - patient['t0']).dt.total_seconds() / 86400
    died_within = patient['death_days'].notna() & (patient['death_days'] <= horizon_days)
    print(f"\nDeaths within {horizon_days} days: {died_within.sum()}/{len(patient)}")

    # --- Data arm: actual outcomes, no censoring, weight = 1 ---
    patient['time_data'] = float(horizon_days)
    patient['event_data'] = 0
    patient.loc[died_within, 'time_data'] = patient.loc[died_within, 'death_days']
    patient.loc[died_within, 'event_data'] = 1
    patient['time_data'] = patient['time_data'].clip(lower=0.01)
    patient['weight_data'] = 1.0

    n_events_data = patient['event_data'].sum()
    print(f"  DATA arm: {n_events_data} events, no censoring, weight=1")

    # --- RL arm: censored + IPCW ---
    patient['time_rl'] = float(horizon_days)
    patient['event_rl'] = 0
    patient.loc[died_within, 'time_rl'] = patient.loc[died_within, 'death_days']
    patient.loc[died_within, 'event_rl'] = 1

    # Censor if non-adherence occurs before event
    first_censor = df[df['C_rl'] == 1].groupby('visit_occurrence_id')['raw_hours'].first()
    patient['censor_days_rl'] = first_censor.reindex(patient['visit_occurrence_id']).values / 24

    has_censor = patient['censor_days_rl'].notna()
    censored_before = has_censor & (patient['censor_days_rl'] < patient['time_rl'])
    patient.loc[censored_before, 'time_rl'] = patient.loc[censored_before, 'censor_days_rl']
    patient.loc[censored_before, 'event_rl'] = 0

    # IPCW weight = last uncensored observation's weight
    last_w = df[df['C_rl'] == 0].groupby('visit_occurrence_id')[ipcw_rl_col].last()
    patient['weight_rl'] = last_w.reindex(patient['visit_occurrence_id']).fillna(1.0).values

    patient['time_rl'] = patient['time_rl'].clip(lower=0.01)

    n_events_rl = patient['event_rl'].sum()
    n_cens_rl = patient['censor_days_rl'].notna().sum()
    print(f"  RL arm:   {n_events_rl} events, {n_cens_rl} censored, "
          f"median weight={patient['weight_rl'].median():.3f}")

    return patient


def km_survival_at_horizon(patient, time_col, event_col, weight_col, horizon_days):
    """Compute KM survival probability at the horizon using weighted KM."""
    kmf = KaplanMeierFitter()
    kmf.fit(patient[time_col], patient[event_col], weights=patient[weight_col])
    # Evaluate survival at horizon (use last value if horizon exceeds timeline)
    idx = kmf.survival_function_.index
    at_horizon = idx[idx <= horizon_days].max()
    return float(kmf.survival_function_.loc[at_horizon].values[0])


def bootstrap_survival_difference(patient, horizon_days, boot_cfg, algo_name):
    """Bootstrap the survival difference (RL - Data) at the horizon.

    Resamples patients with replacement, computes weighted KM survival
    for both arms, and records the difference per iteration.
    """
    n_iter = boot_cfg['n_iterations']
    CI = boot_cfg['CI']
    seed = boot_cfg['seed']
    z = norm.ppf(1 - (1 - CI) / 2)

    rng = np.random.default_rng(seed)
    n_patients = len(patient)
    diffs = np.zeros(n_iter)

    print(f"\n  Bootstrapping survival difference ({n_iter} iterations)...")
    for i in range(n_iter):
        sample = patient.iloc[rng.choice(n_patients, size=n_patients, replace=True)]
        surv_data = km_survival_at_horizon(sample, 'time_data', 'event_data', 'weight_data', horizon_days)
        surv_rl = km_survival_at_horizon(sample, 'time_rl', 'event_rl', 'weight_rl', horizon_days)
        diffs[i] = surv_rl - surv_data

    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    lower = mean_diff - z * std_diff
    upper = mean_diff + z * std_diff

    print(f"  {algo_name.upper()} - Observed survival difference at {horizon_days}d:")
    print(f"    Mean: {mean_diff:+.4f}  ({CI:.0%} CI: [{lower:+.4f}, {upper:+.4f}])")

    return pd.DataFrame({
        'algo': [algo_name],
        'mean_diff': [mean_diff],
        'std_diff': [std_diff],
        'ci_lower': [lower],
        'ci_upper': [upper],
        'n_bootstrap': [n_iter],
        'CI': [CI],
    })


def plot_weighted_km(patient, horizon_days, output_dir, algo_name):
    """Weighted Kaplan-Meier survival curves: RL policy vs observed data."""
    fig, ax = plt.subplots(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    n = len(patient)

    # Data arm (observed, weight=1)
    kmf.fit(patient['time_data'], patient['event_data'], weights=patient['weight_data'],
            label=f'Observed (n={n})')
    kmf.plot(ax=ax, ci_show=False)

    # RL arm (IPCW-weighted)
    kmf.fit(patient['time_rl'], patient['event_rl'], weights=patient['weight_rl'],
            label=f'{algo_name.upper()} policy (n={n})')
    kmf.plot(ax=ax, ci_show=False)

    ax.set_xlabel('Days since AKI onset')
    ax.set_ylabel('Survival probability')
    ax.set_title(f'Target Trial Emulation: {algo_name.upper()} vs Observed ({horizon_days}-day follow-up)')
    ax.set_xlim(0, horizon_days)
    ax.legend()
    plt.tight_layout()

    fname = f"tte_survival_{algo_name}.png"
    fig.savefig(output_dir / fname, dpi=150)
    plt.close(fig)
    print(f"Saved survival plot to {output_dir / fname}")


# =============================================================================
if __name__ == "__main__":
    main()
