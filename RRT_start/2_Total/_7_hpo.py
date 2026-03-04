"""
Hyperparameter Optimization (HPO) - Grid search for CQL, DDQN, BCQ, NFQ
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import joblib
import torch
import d3rlpy
from d3rlpy.algos import DiscreteCQLConfig, DoubleDQNConfig, DiscreteBCQConfig, NFQConfig
from d3rlpy.ope import DiscreteFQE, FQEConfig
from d3rlpy.metrics import TDErrorEvaluator, InitialStateValueEstimationEvaluator, DiscreteActionMatchEvaluator
from d3rlpy.models.encoders import VectorEncoderFactory
from utils import load_mdp, episodes_to_mdp, save_config_snapshot, start_logging
from rl_utils import function_fqe, bootstrap_fqe, compute_metrics_vs_behaviour_policy, compute_mc_return, uncertainty_fqe
from run_metadata import get_run_metadata, print_run_header, save_run_config, add_metadata_to_df


def main():
    """Main entry point for HPO study."""
    from utils import load_config, get_data_paths

    config = load_config()
    all_paths = get_data_paths(config)

    # Check whether to run AUMCdb
    if config['hpo']['databases']['aumc'] == True:
        print("AUMCdb is enabled for HPO study")
        if all_paths['aumc']['mdp_dir'].exists():
            run_hpo_for_db(all_paths['aumc'], config)
        else:
            print(f"Skipping AUMCdb: MDP directory not found")

    # Check whether to run MIMIC
    if config['hpo']['databases']['mimic'] == True:
        print("MIMIC is enabled for HPO study")
        if all_paths['mimic']['mdp_dir'].exists():
            run_hpo_for_db(all_paths['mimic'], config)
        else:
            print(f"Skipping MIMIC: MDP directory not found")


def run_hpo_for_db(db_paths: dict, config: dict):
    """
    Run HPO grid search for one database.

    Args:
        db_paths: Dictionary with database paths
        config: Full configuration dictionary
    """

    ########################################################################################
    # 1. SOURCES
    ## 1.0 MDP en database
    db_name = db_paths['name']
    db_key = 'aumc' if 'aumc' in db_name.lower() else 'mimic'
    mdp_name = config['hpo']['mdp']

    ## 1.1. Datasets
    train_fit_split = config['hpo']['train_fit_split']
    fqe_fit_split = config['hpo']['fqe_fit_split']
    fqe_isv_split = config['hpo']['fqe_isv_split']

    train_ds = load_mdp(db_paths, mdp_name, train_fit_split)
    fqe_fit_ds = load_mdp(db_paths, mdp_name, fqe_fit_split)
    fqe_isv_ds = load_mdp(db_paths, mdp_name, fqe_isv_split)

    mdp_config = joblib.load(db_paths['mdp_dir'] / f"{mdp_name}_config.joblib")

    ## 1.2. Metadata
    metadata = get_run_metadata(config, db_name)
    print_run_header(metadata, title="HPO RUN")

    ########################################################################################
    # 2. OUTPUT
    output_dir = db_paths['reward_dir'] / "HPO_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = start_logging(output_dir, name="hpo")
    save_config_snapshot(output_dir)

    ########################################################################################
    # 3. SETTINGS
    # 3.1. General settings
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = config['processing']['random_state']

    # 3.2. Training settings (per database)
    training_cfg = config['hpo'][f'training_{db_key}']
    gamma_cfg = training_cfg['gamma']
    hidden_units_cfg = training_cfg['hidden_units']

    ########################################################################################
    # Print CHECKs

    print(f"\nHPO CONFIG:")
    print(f"  Database: {db_name.upper()} | MDP: {mdp_name} ({mdp_config['n_states']} features)")
    print(f"  Splits: train={train_fit_split} | FQE fit={fqe_fit_split} | FQE ISV={fqe_isv_split}")
    print(f"  Training: gamma={gamma_cfg} | hidden_units={hidden_units_cfg}")
    print(f"  Train episodes: {len(train_ds.episodes)}")
    print("=" * 80)

    ########################################################################################
    # 4. EVALUATORS
    evaluators = {
        'td_train': TDErrorEvaluator(episodes=train_ds.episodes),
        'td_val': TDErrorEvaluator(episodes=fqe_isv_ds.episodes),
        'isv_train': InitialStateValueEstimationEvaluator(episodes=train_ds.episodes),
        'isv_val': InitialStateValueEstimationEvaluator(episodes=fqe_isv_ds.episodes),
        'action_match_train': DiscreteActionMatchEvaluator(episodes=train_ds.episodes),
        'action_match_val': DiscreteActionMatchEvaluator(episodes=fqe_isv_ds.episodes),
    }

    ########################################################################################
    # 5. MC RETURN
    mc_return_mean, mc_return_std = compute_mc_return(fqe_isv_ds.episodes, gamma=gamma_cfg)
    print(f"MC Return (val): {mc_return_mean:.4f} (±{mc_return_std:.4f})")

    ########################################################################################
    # 6. TRAINING (grid search per algorithm)

    all_results = {}

    ## 6.1. CQL
    if config['hpo']['algorithms']['cql']:
        grid = config['hpo']['algorithm_grids']['cql']
        if grid:
            print(f"\nTraining CQL for {db_name}")

            keys = list(grid.keys())
            vals = [v if isinstance(v, list) else [v] for v in grid.values()]
            combos = [dict(zip(keys, c)) for c in product(*vals)]
            print(f"\n--- CQL ({len(combos)} configs) ---")

            results_CQL = []
            for i, hp in enumerate(combos):

                name = '_'.join(f"{k}={v}" for k, v in hp.items())
                print(f"  [{i+1}/{len(combos)}] {name}", end='')

                d3rlpy.seed(seed)

                cql_hpo = DiscreteCQLConfig(
                    alpha=hp.get('alpha'),
                    learning_rate=hp.get('learning_rate'),
                    batch_size=hp.get('batch_size'),
                    n_critics=hp.get('n_critics'),
                    target_update_interval=hp.get('n_steps')//10,
                    gamma=gamma_cfg,
                    encoder_factory=VectorEncoderFactory(hidden_units=hidden_units_cfg)
                ).create(device=device)

                cql_hpo.fit(train_ds,
                        n_steps=hp.get('n_steps'),
                        n_steps_per_epoch=hp.get('n_steps')//20,
                        evaluators=evaluators,
                        show_progress=True,
                        experiment_name=f"cql_{name}")

                # Results
                isv_val_hpo = evaluators['isv_val'](cql_hpo, fqe_isv_ds)
                print(f" ISV_val={isv_val_hpo:.4f}", end='')

                metrics = compute_metrics_vs_behaviour_policy(cql_hpo, fqe_isv_ds)
                print(f" rrt_rate_per_state_algo={metrics['rrt_rate_per_state_algo']:.1%}", end='')
                print(f" rrt_rate_per_state_data={metrics['rrt_rate_per_state_data']:.1%}", end='')

                td_val_hpo = evaluators['td_val'](cql_hpo, fqe_isv_ds)
                print(f" TD_val={td_val_hpo:.4f}")

                results_CQL.append({
                    'model': cql_hpo,
                    'config': name,
                    'isv_val': isv_val_hpo,
                    'td_val': td_val_hpo,
                    'mc_return_mean': mc_return_mean,
                    'mc_return_std': mc_return_std,
                    'rrt_rate_per_state_algo': metrics['rrt_rate_per_state_algo'],
                    'rrt_rate_per_state_data': metrics['rrt_rate_per_state_data'],
                    'rrt_rate_per_episode_algo': metrics['rrt_rate_per_episode_algo'],
                    'rrt_rate_per_episode_data': metrics['rrt_rate_per_episode_data'],
                })

            all_results['cql'] = sorted(results_CQL, key=lambda x: x['isv_val'], reverse=True)
        else:
            print(f"Skipping CQL: no grid hyperparameters defined for CQL")
    else:
        print("Skipping CQL: disabled in hpo.algorithms")

    ## 6.2. DDQN
    if config['hpo']['algorithms']['ddqn']:
        grid = config['hpo']['algorithm_grids']['ddqn']
        if grid:
            print(f"\nTraining DDQN for {db_name}")

            keys = list(grid.keys())
            vals = [v if isinstance(v, list) else [v] for v in grid.values()]
            combos = [dict(zip(keys, c)) for c in product(*vals)]
            print(f"\n--- DDQN ({len(combos)} configs) ---")

            results_DDQN = []
            for i, hp in enumerate(combos):

                name = '_'.join(f"{k}={v}" for k, v in hp.items())
                print(f"  [{i+1}/{len(combos)}] {name}", end='')

                d3rlpy.seed(seed)

                ddqn_hpo = DoubleDQNConfig(
                    target_update_interval=hp.get('n_steps')//10,
                    learning_rate=hp.get('learning_rate'),
                    batch_size=hp.get('batch_size'),
                    n_critics=hp.get('n_critics'),
                    gamma=gamma_cfg,
                    encoder_factory=VectorEncoderFactory(hidden_units=hidden_units_cfg)
                ).create(device=device)

                ddqn_hpo.fit(train_ds,
                        n_steps=hp.get('n_steps'),
                        n_steps_per_epoch=hp.get('n_steps')//10,
                        evaluators=evaluators,
                        show_progress=True,
                        experiment_name=f"ddqn_{name}")

                # Results
                isv_val_hpo = evaluators['isv_val'](ddqn_hpo, fqe_isv_ds)
                print(f" ISV_val={isv_val_hpo:.4f}", end='')

                metrics = compute_metrics_vs_behaviour_policy(ddqn_hpo, fqe_isv_ds)
                print(f" rrt_rate_per_state_algo={metrics['rrt_rate_per_state_algo']:.1%}", end='')
                print(f" rrt_rate_per_state_data={metrics['rrt_rate_per_state_data']:.1%}", end='')

                td_val_hpo = evaluators['td_val'](ddqn_hpo, fqe_isv_ds)
                print(f" TD_val={td_val_hpo:.4f}")

                results_DDQN.append({
                    'model': ddqn_hpo,
                    'config': name,
                    'isv_val': isv_val_hpo,
                    'td_val': td_val_hpo,
                    'mc_return_mean': mc_return_mean,
                    'mc_return_std': mc_return_std,
                    'rrt_rate_per_state_algo': metrics['rrt_rate_per_state_algo'],
                    'rrt_rate_per_state_data': metrics['rrt_rate_per_state_data'],
                    'rrt_rate_per_episode_algo': metrics['rrt_rate_per_episode_algo'],
                    'rrt_rate_per_episode_data': metrics['rrt_rate_per_episode_data'],
                })

            all_results['ddqn'] = sorted(results_DDQN, key=lambda x: x['isv_val'], reverse=True)
        else:
            print(f"Skipping DDQN: no grid hyperparameters defined for DDQN")
    else:
        print("Skipping DDQN: disabled in hpo.algorithms")

    ## 6.3. BCQ
    if config['hpo']['algorithms']['bcq']:
        grid = config['hpo']['algorithm_grids']['bcq']
        if grid:
            print(f"\nTraining BCQ for {db_name}")

            keys = list(grid.keys())
            vals = [v if isinstance(v, list) else [v] for v in grid.values()]
            combos = [dict(zip(keys, c)) for c in product(*vals)]
            print(f"\n--- BCQ ({len(combos)} configs) ---")

            results_BCQ = []
            for i, hp in enumerate(combos):

                name = '_'.join(f"{k}={v}" for k, v in hp.items())
                print(f"  [{i+1}/{len(combos)}] {name}", end='')

                d3rlpy.seed(seed)

                bcq_hpo = DiscreteBCQConfig(
                    action_flexibility=hp.get('action_flexibility'),
                    beta=hp.get('beta'),
                    learning_rate=hp.get('learning_rate'),
                    batch_size=hp.get('batch_size'),
                    gamma=gamma_cfg,
                    encoder_factory=VectorEncoderFactory(hidden_units=hidden_units_cfg)
                ).create(device=device)

                bcq_hpo.fit(train_ds,
                        n_steps=hp.get('n_steps'),
                        n_steps_per_epoch=hp.get('n_steps')//20,
                        evaluators=evaluators,
                        show_progress=True,
                        experiment_name=f"bcq_{name}")

                # Results
                isv_val_hpo = evaluators['isv_val'](bcq_hpo, fqe_isv_ds)
                print(f" ISV_val={isv_val_hpo:.4f}", end='')

                metrics = compute_metrics_vs_behaviour_policy(bcq_hpo, fqe_isv_ds)
                print(f" rrt_rate_per_state_algo={metrics['rrt_rate_per_state_algo']:.1%}", end='')
                print(f" rrt_rate_per_state_data={metrics['rrt_rate_per_state_data']:.1%}", end='')

                td_val_hpo = evaluators['td_val'](bcq_hpo, fqe_isv_ds)
                print(f" TD_val={td_val_hpo:.4f}")

                results_BCQ.append({
                    'model': bcq_hpo,
                    'config': name,
                    'isv_val': isv_val_hpo,
                    'td_val': td_val_hpo,
                    'mc_return_mean': mc_return_mean,
                    'mc_return_std': mc_return_std,
                    'rrt_rate_per_state_algo': metrics['rrt_rate_per_state_algo'],
                    'rrt_rate_per_state_data': metrics['rrt_rate_per_state_data'],
                    'rrt_rate_per_episode_algo': metrics['rrt_rate_per_episode_algo'],
                    'rrt_rate_per_episode_data': metrics['rrt_rate_per_episode_data'],
                })

            all_results['bcq'] = sorted(results_BCQ, key=lambda x: x['isv_val'], reverse=True)
        else:
            print(f"Skipping BCQ: no grid hyperparameters defined for BCQ")
    else:
        print("Skipping BCQ: disabled in hpo.algorithms")

    ## 6.4. NFQ
    if config['hpo']['algorithms']['nfq']:
        grid = config['hpo']['algorithm_grids']['nfq']
        if grid:
            print(f"\nTraining NFQ for {db_name}")

            keys = list(grid.keys())
            vals = [v if isinstance(v, list) else [v] for v in grid.values()]
            combos = [dict(zip(keys, c)) for c in product(*vals)]
            print(f"\n--- NFQ ({len(combos)} configs) ---")

            results_NFQ = []
            for i, hp in enumerate(combos):

                name = '_'.join(f"{k}={v}" for k, v in hp.items())
                print(f"  [{i+1}/{len(combos)}] {name}", end='')

                d3rlpy.seed(seed)

                nfq_hpo = NFQConfig(
                    learning_rate=hp.get('learning_rate'),
                    batch_size=hp.get('batch_size'),
                    n_critics=hp.get('n_critics'),
                    gamma=gamma_cfg,
                    encoder_factory=VectorEncoderFactory(hidden_units=hidden_units_cfg)
                ).create(device=device)

                nfq_hpo.fit(train_ds,
                        n_steps=hp.get('n_steps'),
                        n_steps_per_epoch=hp.get('n_steps')//20,
                        evaluators=evaluators,
                        show_progress=True,
                        experiment_name=f"nfq_{name}")

                # Results
                isv_val_hpo = evaluators['isv_val'](nfq_hpo, fqe_isv_ds)
                print(f" ISV_val={isv_val_hpo:.4f}", end='')

                metrics = compute_metrics_vs_behaviour_policy(nfq_hpo, fqe_isv_ds)
                print(f" rrt_rate_per_state_algo={metrics['rrt_rate_per_state_algo']:.1%}", end='')
                print(f" rrt_rate_per_state_data={metrics['rrt_rate_per_state_data']:.1%}", end='')

                td_val_hpo = evaluators['td_val'](nfq_hpo, fqe_isv_ds)
                print(f" TD_val={td_val_hpo:.4f}")

                results_NFQ.append({
                    'model': nfq_hpo,
                    'config': name,
                    'isv_val': isv_val_hpo,
                    'td_val': td_val_hpo,
                    'mc_return_mean': mc_return_mean,
                    'mc_return_std': mc_return_std,
                    'rrt_rate_per_state_algo': metrics['rrt_rate_per_state_algo'],
                    'rrt_rate_per_state_data': metrics['rrt_rate_per_state_data'],
                    'rrt_rate_per_episode_algo': metrics['rrt_rate_per_episode_algo'],
                    'rrt_rate_per_episode_data': metrics['rrt_rate_per_episode_data'],
                })

            all_results['nfq'] = sorted(results_NFQ, key=lambda x: x['isv_val'], reverse=True)
        else:
            print(f"Skipping NFQ: no grid hyperparameters defined for NFQ")
    else:
        print("Skipping NFQ: disabled in hpo.algorithms")

    ########################################################################################
    # 7. SAVE TRAINING RESULTS
    print(f"\n--- Saving results to {output_dir} ---")

    training_config = {
        'gamma': gamma_cfg,
        'hidden_units': hidden_units_cfg,
    }

    for algo_name, results in all_results.items():
        # Save models
        for r in results:
            r['model'].save(str(output_dir / f"{algo_name}_{r['config']}.d3"))

        # Save training results CSV with metadata
        df = pd.DataFrame([{k: v for k, v in r.items() if k != 'model'} for r in results])
        df['gamma'] = gamma_cfg
        df['hidden_units'] = str(hidden_units_cfg)
        df = add_metadata_to_df(df, metadata)
        df.to_csv(output_dir / f"{algo_name}_results.csv", index=False)
        print(f"  {algo_name}: {len(results)} models saved")

    ########################################################################################
    # 8. FQE COMPARISON
    if not all_results:
        print("No models trained, skipping FQE comparison")
        return all_results

    all_fqe, summary_df = compare_performance_algorithms(all_results, fqe_fit_ds, fqe_isv_ds, config, device, seed, db_key)

    # Save FQE comparison results
    summary_df = add_metadata_to_df(summary_df, metadata)
    summary_df.to_csv(output_dir / "fqe_comparison.csv", index=False)

    # Save all FQE details (without model objects)
    all_fqe_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'model'} for r in all_fqe])
    all_fqe_df = add_metadata_to_df(all_fqe_df, metadata)
    all_fqe_df.to_csv(output_dir / "all_fqe_results.csv", index=False)

    # Save best model per algorithm (highest FQE ISV)
    print("\n--- Saving best models ---")
    for algo_name in all_results.keys():
        algo_fqe = [r for r in all_fqe if r['algorithm'] == algo_name]
        if algo_fqe:
            best = max(algo_fqe, key=lambda x: x['fqe_isv'])
            best_model = best['model']
            best_model.save(str(output_dir / f"best_{algo_name}_model.d3"))
            print(f"  {algo_name}: saved best model (FQE ISV={best['fqe_isv']:.4f}, config={best['config']})")

    ########################################################################################
    # 9. SAVE CONFIGURATION
    fqe_config = {
        'fqe_n_steps': config['hpo']['fqe']['n_steps'],
        'fqe_n_epochs': config['hpo']['fqe']['n_epochs'],
        'fqe_learning_rate': config['hpo']['fqe']['learning_rate'],
    }
    save_run_config(output_dir, metadata, training_config=training_config, eval_config=fqe_config)

    print(f"\nFQE results saved to {output_dir / 'fqe_comparison.csv'} and {output_dir / 'all_fqe_results.csv'}")
    run_id = metadata['run_id']
    print(f"Run config saved to {output_dir / f'run_{run_id}_config.json'}")
    logger.close()

    return all_results, all_fqe_df, summary_df


########################################################################################
# FQE COMPARISON FUNCTION
def compare_performance_algorithms(all_results: dict, fqe_fit_ds, fqe_isv_ds, config: dict, device: str, seed: int, db_key: str):
    """FQE evaluation for all models, bootstrap CI for top N."""

    ########################################################################################
    # 1. SETTINGS
    # 1.1. FQE
    n_steps_cfg = config['hpo']['fqe']['n_steps']
    n_epochs_cfg = config['hpo']['fqe']['n_epochs']
    learning_rate_cfg = config['hpo']['fqe']['learning_rate']
    gamma_cfg = config['hpo'][f'training_{db_key}']['gamma']

    # 1.2. FQE bootstrap
    fqe_bootstrap_enabled = config['hpo']['fqe']['bootstrap']['enabled']
    fqe_bootstrap_top = config['hpo']['fqe']['bootstrap']['top_n_per_algorithm']
    n_bootstrap_cfg = config['hpo']['fqe']['bootstrap']['n_bootstrap']
    n_steps_bootstrap_cfg = config['hpo']['fqe']['bootstrap']['n_steps']
    ci_bootstrap = config['hpo']['fqe']['bootstrap']['confidence_level']
    n_epochs_bootstrap_cfg = config['hpo']['fqe']['bootstrap']['n_epochs']

    # FQE config object
    fqe_config = FQEConfig(
        learning_rate=learning_rate_cfg,
        gamma=gamma_cfg)

    ########################################################################################
    # 2. FQE FOR ALL MODELS
    all_fqe = []

    for algo, results in all_results.items():
        print(f"\n--- FQE {algo} ({len(results)} models) ---")

        for i, r in enumerate(results):
            fqe_isv = function_fqe(algo=r['model'],
                                   dataset_train=fqe_fit_ds,
                                   dataset_val=fqe_isv_ds,
                                   fqe_config=fqe_config,
                                   n_steps=n_steps_cfg,
                                   n_epochs=n_epochs_cfg,
                                   device=device,
                                   seed=seed)

            all_fqe.append({**r, 'algorithm': algo, 'fqe_isv': fqe_isv})
            print(f"  [{i+1}/{len(results)}] {r['config']}: FQE={fqe_isv:.4f}")

    ########################################################################################
    # 3. SORT & BOOTSTRAP TOP N
    all_fqe = sorted(all_fqe, key=lambda x: x['fqe_isv'], reverse=True)

    if fqe_bootstrap_enabled:
        print(f"\n--- Bootstrap TOP {fqe_bootstrap_top} ({n_bootstrap_cfg}x) ---")

        for i, r in enumerate(all_fqe[:fqe_bootstrap_top]):
            r['fqe_isv_mean'], r['ci_low'], r['ci_high'] = uncertainty_fqe(
                        algo=r['model'],
                        dataset_train=fqe_fit_ds,
                        dataset_val=fqe_isv_ds,
                        fqe_config=fqe_config,
                        n_bootstrap=n_bootstrap_cfg,
                        n_steps=n_steps_bootstrap_cfg,
                        n_epochs=n_epochs_bootstrap_cfg,
                        device=device,
                        seed=seed,
                        CI=ci_bootstrap)

            print(f"  [{i+1}] {r['algorithm'].upper()} | {r['config']}: {r['fqe_isv']:.4f} [{r['fqe_isv_mean']:.4f},{r['ci_low']:.4f}, {r['ci_high']:.4f}]")

    ########################################################################################
    # 4. SUMMARY
    summary_df = pd.DataFrame([{
        'rank': i+1,
        'algorithm': r['algorithm'],
        'config': r['config'],
        'fqe_isv': r['fqe_isv'],
        'fqe_isv_mean': r.get('fqe_isv_mean', np.nan),
        'fqe_isv_ci_low': r.get('ci_low', np.nan),
        'fqe_isv_ci_high': r.get('ci_high', np.nan),
        'isv_val': r['isv_val'],
        'td_val': r['td_val'],
        'mc_return_mean': r['mc_return_mean'],
        'mc_return_std': r['mc_return_std'],
        'rrt_rate_per_state_algo': r['rrt_rate_per_state_algo'],
        'rrt_rate_per_state_data': r['rrt_rate_per_state_data'],
        'rrt_rate_per_episode_algo': r['rrt_rate_per_episode_algo'],
        'rrt_rate_per_episode_data': r['rrt_rate_per_episode_data'],
    } for i, r in enumerate(all_fqe)])

    print(f"\n--- TOP 10 ---\n{summary_df.head(10).to_string(index=False)}")

    return all_fqe, summary_df


if __name__ == "__main__":
    main()
