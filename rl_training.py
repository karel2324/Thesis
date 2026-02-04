"""
Shared training and evaluation functions for RL pipeline.

Contains:
- Monte Carlo return computation
- FQE (Fitted Q-Evaluation) functions
- Bootstrap confidence intervals
- Action frequency metrics
"""

import numpy as np
import d3rlpy
from d3rlpy.ope import DiscreteFQE, FQEConfig
from d3rlpy.metrics import InitialStateValueEstimationEvaluator
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


def function_fqe(algo, dataset, fqe_config, n_steps, device, seed):
    """
    Run single FQE evaluation.

    Args:
        algo: Trained RL algorithm
        dataset: Dataset to evaluate on
        fqe_config: FQEConfig object
        n_steps: Number of training steps
        device: 'cuda:0' or 'cpu'
        seed: Random seed

    Returns:
        Initial state value estimate
    """
    d3rlpy.seed(seed)

    fqe = DiscreteFQE(algo=algo, config=fqe_config, device=device)
    fqe.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps,
        show_progress=True,
        evaluators={'isv': InitialStateValueEstimationEvaluator(dataset.episodes)}
    )

    isv_evaluator = InitialStateValueEstimationEvaluator(dataset.episodes)
    return isv_evaluator(fqe, dataset)


def bootstrap_fqe(algo, dataset, fqe_config, n_bootstrap, n_steps, device, seed, CI=0.95):
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
    """
    episodes = dataset.episodes
    boot_vals = []

    for b in range(n_bootstrap):
        d3rlpy.seed(seed + b)

        # Create bootstrapped dataset
        idx = np.random.choice(len(episodes), size=len(episodes), replace=True)
        bootstrap_dataset = episodes_to_mdp([episodes[j] for j in idx])

        # Fit FQE
        fqe = DiscreteFQE(algo=algo, config=fqe_config, device=device)
        fqe.fit(
            bootstrap_dataset,
            n_steps=n_steps,
            n_steps_per_epoch=n_steps,
            show_progress=True,
            evaluators={'isv': InitialStateValueEstimationEvaluator(bootstrap_dataset.episodes)}
        )

        # Evaluate
        isv_evaluator = InitialStateValueEstimationEvaluator(bootstrap_dataset.episodes)
        boot_vals.append(isv_evaluator(fqe, bootstrap_dataset))

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

def train_cql(train_ds, val_ds, alpha, learning_rate, batch_size, gamma, n_critics, hidden_units, n_steps, n_steps_per_epoch, device, save_interval, name):
    """Train CQL and collect metrics."""

    # Configuration of CQL
    cql = DiscreteCQLConfig(
        alpha=alpha,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        n_critics=n_critics,
        encoder_factory=VectorEncoderFactory(hidden_units=hidden_units),
    ).create(device=device)


    evaluators = {
        'td_train': TDErrorEvaluator(episodes=train_ds.episodes),
        'td_val': TDErrorEvaluator(episodes=val_ds.episodes),
        'isv_val': InitialStateValueEstimationEvaluator(episodes=val_ds.episodes),
        'action_match': DiscreteActionMatchEvaluator(episodes=val_ds.episodes),
    }

    n_steps = n_steps   
    n_steps_per_epoch = n_steps_per_epoch   
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

    return cql, pd.DataFrame(metrics)

def train_bc(train_ds, val_ds, learning_rate, batch_size, beta, hidden_units, n_steps, n_steps_per_epoch, device, name):
    """Train Behavior Cloning and collect metrics."""

    bc = DiscreteBCConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        beta=beta,
        encoder_factory=VectorEncoderFactory(hidden_units=hidden_units),
    ).create(device=device)

    metrics = []
    for epoch, m in bc.fit(train_ds,
                           n_steps=n_steps,
                           n_steps_per_epoch=n_steps_per_epoch,
                           experiment_name=f"bc_{name}",
                           show_progress=True):

        m['epoch'], m['step'] = epoch, epoch * n_steps_per_epoch
        metrics.append(m)

    return bc, pd.DataFrame(metrics)