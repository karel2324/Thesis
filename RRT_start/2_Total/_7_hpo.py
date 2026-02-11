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
from utils import load_mdp, episodes_to_mdp
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


############################################################################################################
# FUNCTION FOR RUNNING FULL HYPERPARAMETER OPIMIZATION (GRID SEARCH)
def run_hpo_for_db(db_paths: dict, config: dict):
    """Run HPO trainings for one database."""

    # 1. RETRIEVE ALL CONFIGURATIONS
    ## Derive db_key from db_paths['name'] (e.g., "AUMCdb" -> "aumc", "MIMIC" -> "mimic")
    db_name = db_paths['name']
    db_key = 'aumc' if 'aumc' in db_name.lower() else 'mimic'

    ## Generate run metadata for traceability
    metadata = get_run_metadata(config, db_name)
    print_run_header(metadata, title="HPO RUN")

    ## Standard configuration
    training_cfg=config['hpo'][f'training_{db_key}']

    ## Training settings (different per dataset)
    gamma_cfg = training_cfg['gamma']
    hidden_units_cfg = training_cfg['hidden_units']
    n_steps_cfg = training_cfg['n_steps']
    n_per_epoch_cfg = training_cfg['n_steps_per_epoch']

    ## Seed
    seed = config['processing']['random_state']
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"\nTRAINING CONFIG:")
    print(f"  n_steps: {n_steps_cfg} | n_steps_per_epoch: {n_per_epoch_cfg}")
    print(f"  gamma: {gamma_cfg} | hidden_units: {hidden_units_cfg}")
    print("=" * 80)

    # 2. INPUT & OUTPUT
    ## Output
    output_dir = db_paths['reward_dir'] / "HPO_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    ## Load data
    mdp_name = config['hpo']['mdp']
    train_ds = load_mdp(db_paths, mdp_name, 'train')
    val_ds = load_mdp(db_paths, mdp_name, 'val')
    mdp_config = joblib.load(db_paths['mdp_dir'] / f"{mdp_name}_config.joblib")
    print(f"MDP: {mdp_name} ({mdp_config['n_states']} features), {len(train_ds.episodes)} train episodes")

    ## Compute MC return (based on validation data - behaviour policy)
    gamma_cfg = training_cfg['gamma']
    mc_return_mean, mc_return_std = compute_mc_return(val_ds.episodes, gamma=gamma_cfg)
    print(f"MC Return (val): {mc_return_mean:.4f} (±{mc_return_std:.4f})")

    # 3. START TRAINING
    ## Evaluators
    evaluators = {
        'td_val': TDErrorEvaluator(episodes=val_ds.episodes),
        'isv_val': InitialStateValueEstimationEvaluator(episodes=val_ds.episodes),
        'action_match': DiscreteActionMatchEvaluator(episodes=val_ds.episodes),
    }

    ## initilaize all results
    all_results = {}

    ############################################################################################################

    # Train all configurations
    ## 1.  Conservative Q-learning:
    if config['hpo']['algorithms']['cql']:
        
         # Find grid per algorithm, and skip algorithm with no hyperparameters
        grid = config['hpo']['algorithm_grids']['cql']
        if grid:
            print(f"Training CQL for {db_name}")

            # Generate all combinations of hyperparameters
            ## Retrieve the names of the hyperparameters
            keys=list(grid.keys()) 

            ## Makes list of list for values for hyperparameters
            vals=[v if isinstance(v, list) 
                  else [v] 
                  for v in grid.values()] 
            
            ## Dictionary for all 
            combos=[dict(zip(keys, c)) 
                    for c in product(*vals)]

            print(f"\n--- CQL ({len(combos)} configs) ---")

            # Initialize results CQL
            results_CQL = []

            # Loop over all combinations of hyperparameters
            for i, hp in enumerate(combos):

                # 1. Training algorithm: 
                # Make name for each hyperparameter combo
                name = '_'.join(f"{k}={v}" for k, v in hp.items()) 
                print(f"  [{i+1}/{len(combos)}] {name}", end='') 

                d3rlpy.seed(seed)

                # Set configuration of a CQL algorithm with a combination of HPs
                cql_hpo=DiscreteCQLConfig(
                    # Hyperparameters to be tuned
                    alpha=hp.get('alpha'),
                    learning_rate=hp.get('learning_rate'),
                    batch_size=hp.get('batch_size'),
                    n_critics=hp.get('n_critics'),

                    # Chosen parameters in start of function
                    gamma=gamma_cfg,
                    encoder_factory=VectorEncoderFactory(hidden_units=hidden_units_cfg)
                ).create(device=device)

                # Fit CQL algorithm on training dataset
                cql_hpo.fit(train_ds, 
                        n_steps=n_steps_cfg, # Chosen parameter in start of function
                        n_steps_per_epoch=n_per_epoch_cfg, # Chosen parameter in start of function
                        evaluators=evaluators, # Chosen at start of function
                        show_progress=True, 
                        experiment_name=f"cql_{name}")
                
                # 2. Results
                ## Mean initial state value on validation set
                isv_val_hpo = evaluators['isv_val'](cql_hpo, val_ds)
                print(f" ISV_val={isv_val_hpo:.4f}", end='')

                ## Compute metrics vs behaviour policy
                metrics = compute_metrics_vs_behaviour_policy(cql_hpo, val_ds)
                print(f" rrt_rate_per_state_algo={metrics['rrt_rate_per_state_algo']:.1%}", end='')
                print(f" rrt_rate_per_state_data={metrics['rrt_rate_per_state_data']:.1%}", end='')

                ## TD-error validation
                td_val_hpo = evaluators['td_val'](cql_hpo, val_ds)
                print(f" TD_val={td_val_hpo:.4f}")

                # Append all results to results_CQL
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
            
            # Append results_CQL to all results
            all_results['cql'] = sorted(results_CQL, key=lambda x: x['isv_val'], reverse=True)

        else:
            print(f"Skipping CQL: no grid hyperparameters defined for CQL")
    else:
        print("Skipping CQL: disabled in hpo.algorithms")

    ## 2.  DDQN:
    if config['hpo']['algorithms']['ddqn']:
        
         # Find grid per algorithm, and skip algorithm with no hyperparameters
        grid = config['hpo']['algorithm_grids']['ddqn']
        if grid:
            print("Training ddqn")

            # Generate all combinations of hyperparameters
            keys=list(grid.keys())
            vals=[v if isinstance(v, list) else [v] for v in grid.values()]
            combos=[dict(zip(keys, c)) for c in product(*vals)]
            print(f"\n--- DDQN ({len(combos)} configs) ---")

            # Initialize results DDQN
            results_DDQN = []
            for i, hp in enumerate(combos):

                # 1. Training algorithm: 

                # Make name for each hyperparameter combo
                name = '_'.join(f"{k}={v}" for k, v in hp.items()) 
                print(f"  [{i+1}/{len(combos)}] {name}", end='') 

                d3rlpy.seed(seed)

                # Set configuration of a DDQN algorithm with a combination of HPs
                ddqn_hpo=DoubleDQNConfig(
                    # Hyperparameters to be tuned
                    target_update_interval=hp.get('target_update_interval'),
                    learning_rate=hp.get('learning_rate'),
                    batch_size=hp.get('batch_size'),
                    n_critics=hp.get('n_critics'),

                    # Chosen parameters in start of function
                    gamma=gamma_cfg,
                    encoder_factory=VectorEncoderFactory(hidden_units=hidden_units_cfg)
                ).create(device=device)

                # Fit DDQN algorithm on training dataset
                ddqn_hpo.fit(train_ds, 
                        n_steps=n_steps_cfg, # Chosen parameter in start of function
                        n_steps_per_epoch=n_per_epoch_cfg, # Chosen parameter in start of function
                        evaluators=evaluators, 
                        show_progress=True, 
                        experiment_name=f"ddqn_{name}")
                
                # Results
                ## Mean initial state value on validation set
                isv_val_hpo = evaluators['isv_val'](ddqn_hpo, val_ds)
                print(f" ISV_val={isv_val_hpo:.4f}", end='')

                ## Compute metrics vs behaviour policy
                metrics = compute_metrics_vs_behaviour_policy(ddqn_hpo, val_ds)
                print(f" rrt_rate_per_state_algo={metrics['rrt_rate_per_state_algo']:.1%}", end='')
                print(f" rrt_rate_per_state_data={metrics['rrt_rate_per_state_data']:.1%}", end='')

                ## TD-error validation
                td_val_hpo = evaluators['td_val'](ddqn_hpo, val_ds)
                print(f" TD_val={td_val_hpo:.4f}")

                # Append all results
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
            
            # Append results_DDQN to all results
            all_results['ddqn'] = sorted(results_DDQN, key=lambda x: x['isv_val'], reverse=True)

        else:
            print(f"Skipping DDQN: no grid hyperparameters defined for DDQN")
    else:
        print("Skipping DDQN: disabled in hpo.algorithms")

    ## 3.  BCQ:
    if config['hpo']['algorithms']['bcq']:
        
         # Find grid per algorithm, and skip algorithm with no hyperparameters
        grid = config['hpo']['algorithm_grids']['bcq']
        if grid:
            print("Training bcq")

            # Generate all combinations of hyperparameters
            keys=list(grid.keys())
            vals=[v if isinstance(v, list) else [v] for v in grid.values()]
            combos=[dict(zip(keys, c)) for c in product(*vals)]
            print(f"\n--- BCQ ({len(combos)} configs) ---")

            # Initialize results BCQ
            results_BCQ = []
            for i, hp in enumerate(combos):

                # 1. Training algorithm: 
                # Make name for each hyperparameter combo
                name = '_'.join(f"{k}={v}" for k, v in hp.items()) 
                print(f"  [{i+1}/{len(combos)}] {name}", end='') 

                d3rlpy.seed(seed)

                # Set configuration of a BCQ algorithm with a combination of HPs
                bcq_hpo=DiscreteBCQConfig(
                    # Hyperparameters to be tuned
                    action_flexibility=hp.get('action_flexibility'),
                    beta=hp.get('beta'),
                    learning_rate=hp.get('learning_rate'),
                    batch_size=hp.get('batch_size'),

                    # Chosen parameters in start of function
                    gamma=gamma_cfg,
                    encoder_factory=VectorEncoderFactory(hidden_units=hidden_units_cfg)
                ).create(device=device)

                # Fit BCQ algorithm on training dataset
                bcq_hpo.fit(train_ds, 
                        n_steps=n_steps_cfg, # Chosen parameter in start of function
                        n_steps_per_epoch=n_per_epoch_cfg, # Chosen parameter in start of function
                        evaluators=evaluators, 
                        show_progress=True, 
                        experiment_name=f"bcq_{name}")
                
                # Results
                ## Mean initial state value on validation set
                isv_val_hpo = evaluators['isv_val'](bcq_hpo, val_ds)
                print(f" ISV_val={isv_val_hpo:.4f}", end='')

                ## Compute metrics vs behaviour policy
                metrics = compute_metrics_vs_behaviour_policy(bcq_hpo, val_ds)
                print(f" rrt_rate_per_state_algo={metrics['rrt_rate_per_state_algo']:.1%}", end='')
                print(f" rrt_rate_per_state_data={metrics['rrt_rate_per_state_data']:.1%}", end='')

                ## TD-error validation
                td_val_hpo = evaluators['td_val'](bcq_hpo, val_ds)
                print(f" TD_val={td_val_hpo:.4f}")

                # Append all results
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

            # Append results_BCQ to all results
            all_results['bcq'] = sorted(results_BCQ, key=lambda x: x['isv_val'], reverse=True)
            
        else:
            print(f"Skipping BCQ: no grid hyperparameters defined for BCQ")
    else:
        print("Skipping BCQ: disabled in hpo.algorithms")

    ## 4. NFQ  :
    if config['hpo']['algorithms']['nfq']:
        
         # Find grid per algorithm, and skip algorithm with no hyperparameters
        grid = config['hpo']['algorithm_grids']['nfq']
        if grid:
            print("Training nfq")

            # Generate all combinations of hyperparameters
            keys=list(grid.keys())
            vals=[v if isinstance(v, list) else [v] for v in grid.values()]
            combos=[dict(zip(keys, c)) for c in product(*vals)]
            print(f"\n--- NFQ ({len(combos)} configs) ---")

            # Initialize results NFQ
            results_NFQ = []
            for i, hp in enumerate(combos):

                # 1. Training algorithm: 
                # Make name for each hyperparameter combo
                name = '_'.join(f"{k}={v}" for k, v in hp.items()) 
                print(f"  [{i+1}/{len(combos)}] {name}", end='') 

                d3rlpy.seed(seed)

                # Set configuration of a NFQ algorithm with a combination of HPs
                nfq_hpo=NFQConfig(
                    # Hyperparameters to be tuned
                    learning_rate=hp.get('learning_rate'),
                    batch_size=hp.get('batch_size'),
                    n_critics=hp.get('n_critics'),

                    # Chosen parameters in start of function
                    gamma=gamma_cfg,
                    encoder_factory=VectorEncoderFactory(hidden_units=hidden_units_cfg)
                ).create(device=device)

                # Fit NFQ algorithm on training dataset
                nfq_hpo.fit(train_ds, 
                        n_steps=n_steps_cfg, # Chosen parameter in start of function
                        n_steps_per_epoch=n_per_epoch_cfg, # Chosen parameter in start of function
                        evaluators=evaluators, 
                        show_progress=True, 
                        experiment_name=f"nfq_{name}")
                
                # Results
                ## Mean initial state value on validation set
                isv_val_hpo = evaluators['isv_val'](nfq_hpo, val_ds)
                print(f" ISV_val={isv_val_hpo:.4f}", end='')

                ## Compute metrics vs behaviour policy
                metrics = compute_metrics_vs_behaviour_policy(nfq_hpo, val_ds)
                print(f" rrt_rate_per_state_algo={metrics['rrt_rate_per_state_algo']:.1%}", end='')
                print(f" rrt_rate_per_state_data={metrics['rrt_rate_per_state_data']:.1%}", end='')

                ## TD-error validation
                td_val_hpo = evaluators['td_val'](nfq_hpo, val_ds)
                print(f" TD_val={td_val_hpo:.4f}")

                # Append all results
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
            
            # Append results_NFQ to all results
            all_results['nfq'] = sorted(results_NFQ, key=lambda x: x['isv_val'], reverse=True)
             
        else:
            print(f"Skipping NFQ: no grid hyperparameters defined for NFQ")
    else:
        print("Skipping NFQ: disabled in hpo.algorithms")

    # SAVE RESULTS AND ALGORTIHMS
    print(f"\n--- Saving results to {output_dir} ---")

    # Store training config for traceability
    training_config = {
        'n_steps': n_steps_cfg,
        'n_steps_per_epoch': n_per_epoch_cfg,
        'gamma': gamma_cfg,
        'hidden_units': hidden_units_cfg,
    }

    for algo_name, results in all_results.items():
        # 1. Save models
        for r in results:
            r['model'].save(str(output_dir / f"{algo_name}_{r['config']}.d3"))

        # 2. Save training results CSV with metadata
        df = pd.DataFrame([{k: v for k, v in r.items() if k != 'model'} for r in results])

        # Add training config columns
        df['n_steps_train'] = n_steps_cfg
        df['gamma'] = gamma_cfg
        df['hidden_units'] = str(hidden_units_cfg)

        # Add metadata columns
        df = add_metadata_to_df(df, metadata)

        df.to_csv(output_dir / f"{algo_name}_results.csv", index=False)
        print(f"  {algo_name}: {len(results)} models saved")

    # RUN FQE FOR ALL ALGORITHMS

    # If there are no results, then stop
    if not all_results:
        print("No models trained, skipping FQE comparison")
        return all_results
    
    # Use compare_performance_algorithm
    all_fqe, summary_df = compare_performance_algorithms(all_results, val_ds, train_ds, config, device, seed, db_key)

    # Add metadata to FQE results
    summary_df = add_metadata_to_df(summary_df, metadata)
    summary_df.to_csv(output_dir / "fqe_comparison.csv", index=False)

    # Save all_fqe details (without model objects) with metadata
    all_fqe_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'model'} for r in all_fqe])
    all_fqe_df = add_metadata_to_df(all_fqe_df, metadata)
    all_fqe_df.to_csv(output_dir / "all_fqe_results.csv", index=False)

    # Save best model per algorithm (highest FQE ISV)
    print("\n--- Saving best models ---")
    for algo_name in all_results.keys():
        # Filter FQE results for this algorithm
        algo_fqe = [r for r in all_fqe if r['algorithm'] == algo_name]
        if algo_fqe:
            # Find best by FQE ISV
            best = max(algo_fqe, key=lambda x: x['fqe_isv'])
            best_model = best['model']
            best_model.save(str(output_dir / f"best_{algo_name}_model.d3"))
            print(f"  {algo_name}: saved best model (FQE ISV={best['fqe_isv']:.4f}, config={best['config']})")

    # Save run configuration JSON for full traceability
    fqe_config = {
        'fqe_n_steps': config['hpo']['fqe']['n_steps'],
        'fqe_learning_rate': config['hpo']['fqe']['learning_rate'],
    }
    save_run_config(output_dir, metadata, training_config=training_config, eval_config=fqe_config)

    # End of function
    print(f"FQE results saved to {output_dir / 'fqe_comparison.csv'} and {output_dir / 'all_fqe_results.csv'}")
    run_id = metadata['run_id']
    print(f"Run config saved to {output_dir / f'run_{run_id}_config.json'}")

    return all_results, all_fqe_df, summary_df


############################################################################################################
# FUNCTION FOR PERFORMANCE COMPARANCE BETWEEN ALGORTIHMS WITH FQE AND POSSIBLY BOOTSTRAP
def compare_performance_algorithms(all_results: dict, val_ds, train_ds, config: dict, device: str, seed: int, db_key: str):
    """FQE evaluation for all models, bootstrap CI for top N."""

    # 1. RETRIEVE ALL CONFIGURATIONS
    ## For FQE
    n_steps_per_epoch_cfg = config['hpo']['fqe']['n_steps_per_epoch']
    n_epochs_cfg = config['hpo']['fqe']['n_epochs']
    learning_rate_cfg = config['hpo']['fqe']['learning_rate']
    gamma_cfg = config['hpo'][f'training_{db_key}']['gamma']

    ## For bootstrapping
    fqe_bootstrap_enabled=config['hpo']['fqe']['bootstrap']['enabled'] # Should we bootstrap? 
    fqe_bootstrap_top=config['hpo']['fqe']['bootstrap']['top_n_per_algorithm'] # Top algorithms for which you want bootstrap
    n_bootstrap_cfg = config['hpo']['fqe']['bootstrap']['n_bootstrap'] # Retrieve number of bootstraps for FQE
    n_steps_bootstrap_cfg = config['hpo']['fqe']['bootstrap']['n_steps'] # N_steps for bootstrap
    ci_bootstrap = config['hpo']['fqe']['bootstrap']['confidence_level'] # Retrieve confidence level for bootstrap

    # 2. SET UP DATABASE AND FQE
    ## Set up database
    episodes_val=val_ds.episodes # Retrieve the episodes 
    
    ## Set up FQE
    fqe_config = FQEConfig( # Set up configuration for FQE
        learning_rate=learning_rate_cfg, 
        gamma=gamma_cfg)
    
    ############################################################################################################
    # 3. FQE FOR ALL MODELS 

    ## Initialize all results
    all_fqe = []

    ## Loop over all_results (output of previous algorithm)
    for algo, results in all_results.items(): # Loop over the broad algorithms (CQL / DDQN / BCQ / NFQ)
        print(f"\n--- FQE {algo} ({len(results)} models) ---")

        for i, r in enumerate(results):

            fqe_isv = function_fqe(algo=r['model'], 
                                   dataset_train = train_ds,
                                   dataset_val = val_ds, 
                                   fqe_config = fqe_config, 
                                   n_steps_per_epoch= n_steps_per_epoch_cfg, 
                                   n_epochs = n_epochs_cfg,
                                   device = device, 
                                   seed = seed) 
            
            all_fqe.append({**r, 'algorithm': algo, 'fqe_isv': fqe_isv})
            print(f"  [{i+1}/{len(results)}] {r['config']}: FQE={fqe_isv:.4f}")

    ############################################################################################################
    # 2. SORT BY FQE ISV AND BOOTSTRAP TOP N

    # Sort from highest to lowest (to find best performing algorithms)
    all_fqe = sorted(all_fqe, key=lambda x: x['fqe_isv'], reverse=True)

    ## Only bootstrap if enabled:
    if fqe_bootstrap_enabled:
        print(f"\n--- Bootstrap TOP {fqe_bootstrap_top} ({n_bootstrap_cfg}x) ---")

        for i, r in enumerate(all_fqe[:fqe_bootstrap_top]):
            r['fqe_isv_mean'], r['ci_low'], r['ci_high'] = uncertainty_fqe(
                        algo=r['model'], # Use bootstrap function
                        dataset_train=train_ds,
                        dataset_val=val_ds, 
                        fqe_config=fqe_config, 
                        n_bootstrap = n_bootstrap_cfg, # Times for bootstrapping
                        n_steps = n_steps_bootstrap_cfg, # Total steps 
                        device= device, 
                        seed=seed, 
                        CI = ci_bootstrap) # Confidence interval
            
            print(f"  [{i+1}] {r['algorithm'].upper()} | {r['config']}: {r['fqe_isv']:.4f} [{r['fqe_isv_mean']:.4f},{r['ci_low']:.4f}, {r['ci_high']:.4f}]")

    # 3. Summary (always create, CI columns will be NaN if bootstrap disabled)
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
