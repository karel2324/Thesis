"""
Shared training and evaluation functions for RL pipeline.

Contains:
- Monte Carlo return computation
- FQE (Fitted Q-Evaluation) functions
- Bootstrap confidence intervals
- Action frequency metrics
- Algorithm evaluation (compute_metrics, evaluate_algo)
- Training functions (train_cql, train_bc)
"""

import numpy as np
import d3rlpy
from d3rlpy.ope import DiscreteFQE, FQEConfig
from d3rlpy.metrics import TDErrorEvaluator, InitialStateValueEstimationEvaluator, DiscreteActionMatchEvaluator
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.algos import DiscreteCQLConfig, DiscreteBCConfig
import pandas as pd
from utils import episodes_to_mdp


def compute_mc_return(episodes, gamma=0.99):
    """
    Compute mean discounted return across episodes.

    Args:
        episodes: List of episode objects with .rewards attribute
        gamma: Discount factor

    Returns:
        Tuple of (mean_return, std_return)
    """
    returns = []
    for ep in episodes:
        G = 0
        for r in reversed(ep.rewards):
            G = r + gamma * G
        returns.append(G)
    return np.mean(returns), np.std(returns)


def function_fqe(algo, dataset_train, dataset_val, fqe_config, n_steps_per_epoch, n_epochs, device, seed, return_config=False):
    """
    Run single FQE evaluation.

    Args:
        algo: Trained RL algorithm
        dataset: Dataset to evaluate on
        fqe_config: FQEConfig object
        n_steps: Number of training steps
        device: 'cuda:0' or 'cpu'
        seed: Random seed
        return_config: If True, also return FQE config dict

    Returns:
        Initial state value estimate (or tuple with config if return_config=True)
    """
    d3rlpy.seed(seed)

    fqe = DiscreteFQE(algo=algo, config=fqe_config, device=device)
    fqe.fit(
        dataset_train,
        n_steps=n_steps_per_epoch * n_epochs,
        n_steps_per_epoch=n_steps_per_epoch,
        show_progress=True,
        evaluators={
            'td_train': TDErrorEvaluator(dataset_train.episodes),
            'td_val': TDErrorEvaluator(dataset_val.episodes),
            'isv_val': InitialStateValueEstimationEvaluator(dataset_val.episodes)}
    )

    isv_evaluator = InitialStateValueEstimationEvaluator(dataset_val.episodes)
    isv = isv_evaluator(fqe, dataset_val)

    if return_config:
        eval_config = {
            'fqe_n_steps': n_steps_per_epoch * n_epochs,
            'fqe_learning_rate': fqe_config.learning_rate,
            'fqe_gamma': fqe_config.gamma,
            'n_episodes_eval': len(dataset_val.episodes),
        }
        return isv, eval_config

    return isv


def bootstrap_fqe(algo, dataset_train, dataset_val, fqe_config, n_bootstrap, n_steps, device, seed, CI=0.95):
    """
    FQE with bootstrap confidence intervals.

    Args:
        algo: Trained RL algorithm
        dataset: Dataset to evaluate on
        fqe_config: FQEConfig object
        n_bootstrap: Number of bootstrap samples
        n_steps: FQE training steps per bootstrap
        device: 'cuda:0' or 'cpu'
        seed: Random seed
        CI: Confidence interval level (default 0.95)

    Returns:
        Tuple of (mean_value, ci_lower, ci_upper)
        Bootstrap is for uncertainty of model. So states are bootstrapped for FQE,
        but bootstrapped mean ISV is of the bootstrapped FQE-function on the original states. 
    """
    episodes_train = dataset_train.episodes
    boot_vals = []

    for b in range(n_bootstrap):
        d3rlpy.seed(seed + b)

        # Create bootstrapped dataset
        idx = np.random.choice(len(episodes_train), size=len(episodes_train), replace=True)
        bootstrap_dataset = episodes_to_mdp([episodes_train[j] for j in idx])

        # Fit FQE
        fqe = DiscreteFQE(algo=algo, config=fqe_config, device=device)
        fqe.fit(
            bootstrap_dataset,
            n_steps=n_steps,
            n_steps_per_epoch=n_steps,
            show_progress=True
            )

        # Evaluate
        isv_evaluator = InitialStateValueEstimationEvaluator(dataset_val.episodes)
        boot_vals.append(isv_evaluator(fqe, dataset_val))

    # Calculate confidence interval
    alpha = 1 - CI
    lower_ci = np.percentile(boot_vals, 100 * alpha / 2)
    upper_ci = np.percentile(boot_vals, 100 * (1 - alpha / 2))
    mean_val = np.mean(boot_vals)

    return mean_val, lower_ci, upper_ci

def uncertainty_fqe(algo, dataset_train, dataset_val, fqe_config, n_bootstrap, n_steps, device, seed, CI=0.95):
    """
    FQE with bootstrap confidence intervals.

    Args:
        algo: Trained RL algorithm
        dataset: Dataset to evaluate on
        fqe_config: FQEConfig object
        n_bootstrap: Number of bootstrap samples
        n_steps: FQE training steps per bootstrap
        device: 'cuda:0' or 'cpu'
        seed: Random seed
        CI: Confidence interval level (default 0.95)

    Returns:
        Tuple of (mean_value, ci_lower, ci_upper)
        Bootstrap is for uncertainty of model. So states are bootstrapped for FQE,
        but bootstrapped mean ISV is of the bootstrapped FQE-function on the original states. 
    """
    boot_vals = []

    for b in range(n_bootstrap):
        d3rlpy.seed(seed + b)

        # Fit FQE
        fqe = DiscreteFQE(algo=algo, config=fqe_config, device=device)
        fqe.fit(
            dataset_train,
            n_steps=n_steps,
            n_steps_per_epoch=n_steps,
            show_progress=True
            )

        # Evaluate
        isv_evaluator = InitialStateValueEstimationEvaluator(dataset_val.episodes)
        boot_vals.append(isv_evaluator(fqe, dataset_val))

    # Calculate confidence interval
    alpha = 1 - CI
    lower_ci = np.percentile(boot_vals, 100 * alpha / 2)
    upper_ci = np.percentile(boot_vals, 100 * (1 - alpha / 2))
    mean_val = np.mean(boot_vals)

    return mean_val, lower_ci, upper_ci


def get_action_frequency_per_episode(algo, dataset):
    """
    Get frequency of action=1 (treat) across episodes.

    Args:
        algo: Trained RL algorithm
        dataset: Dataset to evaluate on

    Returns:
        Fraction of episodes where action=1 is predicted at least once
    """
    count = 0
    for ep in dataset.episodes:
        actions = algo.predict(ep.observations)
        if np.any(actions == 1):
            count += 1
    return count / len(dataset.episodes)


def compute_metrics_vs_behaviour_policy(algo, dataset):
    """
    Compute action match, RRT rates, and timing metrics.

    Args:
        algo: Trained RL algorithm
        dataset: MDPDataset to evaluate on

    Returns:
        Dictionary with all metrics
    """
    obs = np.concatenate([ep.observations for ep in dataset.episodes])
    actions = np.concatenate([ep.actions for ep in dataset.episodes])
    preds = algo.predict(obs)

    # Per-state metrics
    action_match = (preds == actions).mean()
    rrt_rate_per_state_algo = (preds == 1).mean()
    rrt_rate_per_state_data = (actions == 1).mean()

    # Per-episode metrics
    rrt_rate_per_episode_algo = get_action_frequency_per_episode(algo=algo, dataset=dataset)
    rrt_rate_per_episode_data = sum(1 for ep in dataset.episodes if np.any(ep.actions == 1)) / len(dataset.episodes)

    # Timing metrics: when does algo start RRT vs data?
    earlier, same, later, algo_only, data_only, neither = 0, 0, 0, 0, 0, 0

    for ep in dataset.episodes:
        ep_preds = algo.predict(ep.observations)

        # Find first RRT index (-1 if none)
        algo_first = np.where(ep_preds == 1)[0]
        data_first = np.where(ep.actions == 1)[0]

        algo_starts = algo_first[0] if len(algo_first) > 0 else -1
        data_starts = data_first[0] if len(data_first) > 0 else -1

        if algo_starts == -1 and data_starts == -1:
            neither += 1
        elif algo_starts == -1:
            data_only += 1
        elif data_starts == -1:
            algo_only += 1
        elif algo_starts < data_starts:
            earlier += 1
        elif algo_starts == data_starts:
            same += 1
        else:
            later += 1

    n_episodes = len(dataset.episodes)

    return {
        'action_match': action_match,
        'rrt_rate_per_state_algo': rrt_rate_per_state_algo,
        'rrt_rate_per_state_data': rrt_rate_per_state_data,
        'rrt_rate_per_episode_algo': rrt_rate_per_episode_algo,
        'rrt_rate_per_episode_data': rrt_rate_per_episode_data,
        'rrt_timing_earlier': earlier / n_episodes,
        'rrt_timing_same': same / n_episodes,
        'rrt_timing_later': later / n_episodes,
        'rrt_timing_algo_only': algo_only / n_episodes,
        'rrt_timing_data_only': data_only / n_episodes,
        'rrt_timing_neither': neither / n_episodes,
    }


def compute_metrics_algo_vs_algo(algo1, algo2, dataset):
    """
    Compute action match, RRT rates, and timing metrics between two algorithms.

    Args:
        algo1: First trained RL algorithm
        algo2: Second trained RL algorithm
        dataset: MDPDataset to evaluate on

    Returns:
        Dictionary with all metrics comparing algo1 vs algo2
    """
    obs = np.concatenate([ep.observations for ep in dataset.episodes])
    preds1 = algo1.predict(obs)
    preds2 = algo2.predict(obs)

    # Per-state metrics
    action_match = (preds1 == preds2).mean()
    rrt_rate_per_state_algo1 = (preds1 == 1).mean()
    rrt_rate_per_state_algo2 = (preds2 == 1).mean()

    # Per-episode metrics
    rrt_rate_per_episode_algo1 = get_action_frequency_per_episode(algo=algo1, dataset=dataset)
    rrt_rate_per_episode_algo2 = get_action_frequency_per_episode(algo=algo2, dataset=dataset)

    # Timing metrics: when does algo1 start RRT vs algo2?
    earlier, same, later, algo1_only, algo2_only, neither = 0, 0, 0, 0, 0, 0

    for ep in dataset.episodes:
        ep_preds1 = algo1.predict(ep.observations)
        ep_preds2 = algo2.predict(ep.observations)

        # Find first RRT index (-1 if none)
        algo1_first = np.where(ep_preds1 == 1)[0]
        algo2_first = np.where(ep_preds2 == 1)[0]

        algo1_starts = algo1_first[0] if len(algo1_first) > 0 else -1
        algo2_starts = algo2_first[0] if len(algo2_first) > 0 else -1

        if algo1_starts == -1 and algo2_starts == -1:
            neither += 1
        elif algo1_starts == -1:
            algo2_only += 1
        elif algo2_starts == -1:
            algo1_only += 1
        elif algo1_starts < algo2_starts:
            earlier += 1
        elif algo1_starts == algo2_starts:
            same += 1
        else:
            later += 1

    n_episodes = len(dataset.episodes)

    return {
        'action_match': action_match,
        'rrt_rate_per_state_algo1': rrt_rate_per_state_algo1,
        'rrt_rate_per_state_algo2': rrt_rate_per_state_algo2,
        'rrt_rate_per_episode_algo1': rrt_rate_per_episode_algo1,
        'rrt_rate_per_episode_algo2': rrt_rate_per_episode_algo2,
        'rrt_timing_earlier': earlier / n_episodes,
        'rrt_timing_same': same / n_episodes,
        'rrt_timing_later': later / n_episodes,
        'rrt_timing_algo1_only': algo1_only / n_episodes,
        'rrt_timing_algo2_only': algo2_only / n_episodes,
        'rrt_timing_neither': neither / n_episodes,
    }


def evaluate_algo(algo, algo_name, dataset_train, dataset_val, device, seed, mc_mean, mc_std,
                  fqe_n_steps_per_epoch, fqe_n_epochs,
                  fqe_enabled=True, fqe_learning_rate=0.0001, 
                  fqe_bootstrap_enabled=False, fqe_bootstrap_n_bootstrap=10,
                  fqe_bootstrap_n_steps=10000, fqe_bootstrap_confidence_level=0.95,
                  gamma=0.99):
    """
    Evaluate a single algorithm on a dataset.

    Args:
        algo: Trained RL algorithm
        algo_name: Name of the algorithm (e.g., 'cql', 'bc')
        dataset: MDPDataset to evaluate on
        device: Device for FQE ('cuda:0' or 'cpu')
        seed: Random seed
        mc_mean: Monte Carlo return mean from data
        mc_std: Monte Carlo return std from data
        fqe_enabled: Whether to run FQE evaluation
        fqe_learning_rate: Learning rate for FQE
        fqe_n_steps: Number of FQE training steps
        fqe_bootstrap_enabled: Whether to compute bootstrap CI
        fqe_bootstrap_n_bootstrap: Number of bootstrap samples
        fqe_bootstrap_n_steps: FQE steps per bootstrap
        fqe_bootstrap_confidence_level: Confidence level for CI
        gamma: Discount factor

    Returns:
        Dictionary with all evaluation results
    """
    d3rlpy.seed(seed)

    # Basic metrics
    metrics = compute_metrics_vs_behaviour_policy(algo=algo, dataset=dataset_val)

    result = {
        'algorithm': algo_name,
        'mc_return_mean': mc_mean,
        'mc_return_std': mc_std,
        'n_episodes_eval': len(dataset_val.episodes),
        **metrics
    }

    print(f"    {algo_name.upper()}: Match={metrics['action_match']:.1%}, "
          f"RRT_state={metrics['rrt_rate_per_state_algo']:.1%}, "
          f"Earlier={metrics['rrt_timing_earlier']:.1%}, "
          f"Same={metrics['rrt_timing_same']:.1%}", end='')

    # FQE evaluation
    if fqe_enabled:
        fqe_config = FQEConfig(
            learning_rate=fqe_learning_rate,
            gamma=gamma
        )

        # Always compute FQE ISV from function_fqe
        fqe_isv = function_fqe(
            algo=algo,
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            fqe_config=fqe_config,
            n_steps_per_epoch=fqe_n_steps_per_epoch,
            n_epochs=fqe_n_epochs,
            device=device,
            seed=seed
        )
        result['fqe_isv'] = fqe_isv
        result['fqe_n_steps'] = fqe_n_steps_per_epoch * fqe_n_epochs
        result['fqe_learning_rate'] = fqe_learning_rate
        result['fqe_gamma'] = gamma

        # Bootstrap only for confidence intervals
        if fqe_bootstrap_enabled:
            fqe_isv_mean, ci_lo, ci_hi = uncertainty_fqe(
                algo=algo,
                dataset_train=dataset_train,
                dataset_val=dataset_val,
                fqe_config=fqe_config,
                n_bootstrap=fqe_bootstrap_n_bootstrap,
                n_steps=fqe_bootstrap_n_steps,
                device=device,
                seed=seed,
                CI=fqe_bootstrap_confidence_level
            )
            result['fqe_isv_mean'] = fqe_isv_mean
            result['fqe_ci_low'] = ci_lo
            result['fqe_ci_high'] = ci_hi
            print(f", FQE={fqe_isv:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
        else:
            print(f", FQE={fqe_isv:.4f}")
    else:
        print()

    return result


def train_cql(train_ds, val_ds, alpha, learning_rate, batch_size, gamma, n_critics, hidden_units, n_steps, n_steps_per_epoch, device, save_interval, name):
    """Train CQL and collect metrics.

    Returns:
        Tuple of (model, metrics_df, training_config)
    """

    # Configuration of CQL
    cql = DiscreteCQLConfig(
        alpha=alpha,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        n_critics=n_critics,
        encoder_factory=VectorEncoderFactory(hidden_units=hidden_units),
    ).create(device=device)

    # Store training config for traceability
    training_config = {
        'algorithm': 'cql',
        'alpha': alpha,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'gamma': gamma,
        'n_critics': n_critics,
        'hidden_units': hidden_units,
        'n_steps': n_steps,
        'n_steps_per_epoch': n_steps_per_epoch,
    }

    evaluators = {
        'td_train': TDErrorEvaluator(episodes=train_ds.episodes),
        'td_val': TDErrorEvaluator(episodes=val_ds.episodes),
        'isv_val': InitialStateValueEstimationEvaluator(episodes=val_ds.episodes),
        'action_match': DiscreteActionMatchEvaluator(episodes=val_ds.episodes),
    }

    metrics = []

    for epoch, m in cql.fit(train_ds,
                            n_steps=n_steps,
                            n_steps_per_epoch=n_steps_per_epoch,
                            evaluators=evaluators,
                            show_progress=True,
                            save_interval=save_interval,
                            experiment_name=f"cql_{name}"):

        m['epoch'], m['step'] = epoch, epoch * n_steps_per_epoch
        metrics.append(m)
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: TD={m['td_val']:.4f}, ISV={m['isv_val']:.4f}, Match={m['action_match']:.1%}")

    return cql, pd.DataFrame(metrics), training_config

def train_bc(train_ds, val_ds, learning_rate, batch_size, beta, hidden_units, n_steps, n_steps_per_epoch, device, name):
    """Train Behavior Cloning and collect metrics.

    Returns:
        Tuple of (model, metrics_df, training_config)
    """

    bc = DiscreteBCConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        beta=beta,
        encoder_factory=VectorEncoderFactory(hidden_units=hidden_units),
    ).create(device=device)

    # Store training config for traceability
    training_config = {
        'algorithm': 'bc',
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'beta': beta,
        'hidden_units': hidden_units,
        'n_steps': n_steps,
        'n_steps_per_epoch': n_steps_per_epoch,
    }

    metrics = []
    for epoch, m in bc.fit(train_ds,
                           n_steps=n_steps,
                           n_steps_per_epoch=n_steps_per_epoch,
                           experiment_name=f"bc_{name}",
                           show_progress=True):

        m['epoch'], m['step'] = epoch, epoch * n_steps_per_epoch
        metrics.append(m)

    return bc, pd.DataFrame(metrics), training_config