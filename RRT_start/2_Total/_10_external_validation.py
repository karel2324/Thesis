"""
External Validation - Evaluate models across databases.

Tests models trained on source database (e.g., AUMCdb) on target database (e.g., MIMIC).
Uses shared evaluation utilities from rl_utils.
"""

import pandas as pd
import torch

from utils import load_config, get_data_paths, load_mdp, load_model, save_config_snapshot, start_logging
from rl_utils import evaluate_algo, compute_mc_return, compute_metrics_algo_vs_algo
from rl_plotting import (plot_fqe_isv, plot_action_match, plot_rrt_rate_per_state,
                         plot_rrt_rate_per_episode, plot_rrt_timing, plot_summary_table,
                         plot_evaluation_algo_vs_bc)
from run_metadata import get_run_metadata, print_run_header, save_run_config, add_metadata_to_df


def main():
    """Main entry point for external validation."""
    config = load_config()
    all_paths = get_data_paths(config)

    model_source_db = config['external_validation']['model_source_database']
    fqe_fit_db = config['external_validation']['fqe_fit_database']
    fqe_target_db = config['external_validation']['fqe_target_database']

    # Validate databases exist
    if not all_paths[model_source_db]['mdp_dir'].exists():
        print(f"Source database {model_source_db} MDP directory not found")
        return
    
    if not all_paths[fqe_fit_db]['mdp_dir'].exists():
        print(f"Fit database {fqe_fit_db} MDP directory not found")
        return

    if not all_paths[fqe_target_db]['mdp_dir'].exists():
        print(f"Target database {fqe_target_db} MDP directory not found")
        return

    run_external_validation(config=config, all_paths=all_paths)


def run_external_validation(config: dict, all_paths: dict):
    """
    Run cross-database validation.

    Args:
        config: Full configuration dictionary
        all_paths: Dictionary with paths for all databases
    """
    ######################################################################################## 
    # 1. SOURCES
    ## 1.0 MDP 
    mdp_name = config['external_validation']['mdp']

    ## 1.1. Model source
    model_source_db = config['external_validation']['model_source_database']
    model_source_paths = all_paths[model_source_db]

    # Model directories (from source database)
    model_source = config['external_validation']['model_source']
    if model_source == 'hpo':
        model_dir = model_source_paths['reward_dir'] / "HPO_results"
    else:
        model_dir = model_source_paths['reward_dir'] / "Ablation_results"
    bc_dir = model_source_paths['reward_dir'] / "BC_results"

    ## 1.2. FQE fit source
    fqe_fit_db = config['external_validation']['fqe_fit_database']
    fqe_fit_paths = all_paths[fqe_fit_db]
    fqe_fit_split = config['external_validation']['fqe_fit_split']
    dataset_train = load_mdp(db_paths=fqe_fit_paths, mdp_name=mdp_name, split=config['external_validation']['fqe_fit_split'])

    ## 1.3. FQE target source
    fqe_target_db = config['external_validation']['fqe_target_database']
    fqe_target_paths = all_paths[fqe_target_db]
    fqe_target_split = config['external_validation']['fqe_target_split']
    dataset_val = load_mdp(db_paths=fqe_target_paths, mdp_name=mdp_name, split=config['external_validation']['fqe_target_split'])

    ## 1.4. Metadata
    metadata = get_run_metadata(config, f"{model_source_db}_to_{fqe_target_db}")
    print_run_header(metadata, title="EXTERNAL VALIDATION")

    ######################################################################################## 
    # 2. OUTPUT
    # Output directory (in source database)
    output_dir = model_source_paths['reward_dir'] / "External_validation_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save current configuration file 
    logger = start_logging(output_dir, name="external_val")
    save_config_snapshot(output_dir)

    ######################################################################################## 
    # 3. SETTINGS
    # 3.1. General settings
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = config['processing']['random_state']
    gamma = 0.99

    # 3.2. FQE
    fqe_enabled = config['external_validation']['fqe']['enabled']
    fqe_learning_rate = config['external_validation']['fqe']['learning_rate']
    fqe_n_steps = config['external_validation']['fqe']['n_steps']
    fqe_n_epochs = config['external_validation']['fqe']['n_epochs']

    # 3.3. FQE bootstrap
    fqe_bootstrap_enabled = config['external_validation']['fqe']['bootstrap']['enabled']
    fqe_bootstrap_n_bootstraps = config['external_validation']['fqe']['bootstrap']['n_bootstrap']
    fqe_bootstrap_CI = config['external_validation']['fqe']['bootstrap']['confidence_level']
    fqe_bootstrap_n_steps = config['external_validation']['fqe']['bootstrap']['n_steps']
    fqe_bootstrap_n_epochs = config['external_validation']['fqe']['bootstrap']['n_epochs']
    
    ######################################################################################## 
    # Print CHECKs

    print(f"\nVALIDATION CONFIG:")
    print(f"  Source: {model_source_db.upper()} | Target: {fqe_target_db.upper()}")
    print(f"  FQE Fit: {fqe_fit_split} | FQE target split: {fqe_target_split} | MDPs: {mdp_name}")
    print(f"  FQE: enabled={config['external_validation']['fqe']['enabled']}, n_steps={config['external_validation']['fqe']['n_steps']}, n_epochs={config['external_validation']['fqe']['n_epochs']}")

    print(f"\nMODEL SOURCES ({model_source.upper()}):")
    print(f"  RL models: {model_dir}")
    print(f"  BC model: {bc_dir} (train_split={config['external_validation']['bc']['train_split']})")
    print("=" * 80)

    print(f"\n--- {mdp_name.upper()} ---")

    ######################################################################################## 
    # 4. EVALUATION

    # 4.1. Initialize results
    all_results = []

    # 4.2. Monte Carlo return from target data
    mc_mean, mc_std = compute_mc_return(episodes=dataset_val.episodes, gamma=gamma)
    print(f"\n  TARGET ({len(dataset_val.episodes)} episodes) - MC Return: {mc_mean:.4f} (±{mc_std:.4f})")

    # 4.3 Evaluate HPO models (CQL, DDQN, BCQ, NFQ) trained on SOURCE
    for algo_name in ['cql', 'ddqn', 'bcq', 'nfq']:
        if config['external_validation']['algorithms'][algo_name]:
            if model_source == 'hpo':
                model_path = model_dir / f"best_{algo_name}_model.d3"
            else:
                model_path = model_dir / f"{algo_name}_{mdp_name}_model.d3"

            model = load_model(model_path=model_path, device=device)
            if model is not None:
                result = evaluate_algo(
                    algo=model,
                    algo_name=algo_name,
                    dataset_val=dataset_val,
                    dataset_train=dataset_train,
                    device=device,
                    seed=seed,
                    mc_mean=mc_mean,
                    mc_std=mc_std,
                    fqe_enabled=fqe_enabled,
                    fqe_learning_rate=fqe_learning_rate,
                    fqe_n_steps=fqe_n_steps,
                    fqe_n_epochs=fqe_n_epochs,
                    fqe_bootstrap_enabled=fqe_bootstrap_enabled,
                    fqe_bootstrap_n_bootstrap=fqe_bootstrap_n_bootstraps,
                    fqe_bootstrap_n_steps=fqe_bootstrap_n_steps,
                    fqe_bootstrap_n_epochs=fqe_bootstrap_n_epochs,
                    fqe_bootstrap_confidence_level=fqe_bootstrap_CI,
                    gamma=gamma
                    )
                
                result['source'] = model_source_db
                result['fqe_fit'] = fqe_fit_db
                result['target'] = fqe_target_db
                result['mdp'] = mdp_name
                result['fqe_fit_split'] = fqe_fit_split
                result['fqe_isv_split'] = fqe_target_split
                all_results.append(result)
            else:
                print(f"    {algo_name.upper()}: Model not found at {model_path}")

    # 4.4. Evaluate BC model trained on SOURCE
    if config['external_validation']['algorithms']['bc']:
        train_split = config['external_validation']['bc']['train_split']
        model_path = bc_dir / f"bc_{mdp_name}_{train_split}.d3"
        model = load_model(model_path=model_path, device=device)
        if model is not None:
            result = evaluate_algo(
                algo=model,
                algo_name='bc',
                dataset_val=dataset_val,
                dataset_train=dataset_train,
                device=device,
                seed=seed,
                mc_mean=mc_mean,
                mc_std=mc_std,
                fqe_enabled=fqe_enabled,
                fqe_learning_rate=fqe_learning_rate,
                fqe_n_steps=fqe_n_steps,
                fqe_n_epochs=fqe_n_epochs,
                fqe_bootstrap_enabled=fqe_bootstrap_enabled,
                fqe_bootstrap_n_bootstrap=fqe_bootstrap_n_bootstraps,
                fqe_bootstrap_n_steps=fqe_bootstrap_n_steps,
                fqe_bootstrap_n_epochs=fqe_bootstrap_n_epochs,
                fqe_bootstrap_confidence_level=fqe_bootstrap_CI,
                gamma=gamma
            )
            result['source'] = model_source_db
            result['fqe_fit'] = fqe_fit_db
            result['target'] = fqe_target_db
            result['mdp'] = mdp_name
            result['fqe_fit_split'] = fqe_fit_split
            result['fqe_isv_split'] = fqe_target_split
            all_results.append(result)
        else:
            print(f"    BC: Model not found at {model_path}")

    ## 4.5 SAVE RESULTS
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(f"\n{'='*60}\nSUMMARY (vs Behaviour Policy)\n{'='*60}")
        print(results_df.to_string(index=False))

        if config['external_validation']['output']['save_metrics']:
            # Add metadata columns
            results_df = add_metadata_to_df(results_df, metadata)
            results_df.to_csv(output_dir / "external_validation_results.csv", index=False)

        if config['external_validation']['output']['save_plots']:
            ext_title = f"External Validation: {model_source_db.upper()} → {fqe_target_db.upper()}"
            plot_fqe_isv(results_df, output_dir, ext_title, "external_validation")
            plot_action_match(results_df, output_dir, ext_title, "external_validation")
            plot_rrt_rate_per_state(results_df, output_dir, ext_title, "external_validation")
            plot_rrt_rate_per_episode(results_df, output_dir, ext_title, "external_validation")
            plot_rrt_timing(results_df, output_dir, ext_title, "external_validation")
            plot_summary_table(results_df, output_dir, ext_title, "external_validation")

    ## 4.6 COMPARE ALGORITHM VS BEHAVIOR CLONING
    if config['external_validation']['algorithms']['bc']:
        print(f"\n{'='*60}\nALGO VS BC COMPARISON\n{'='*60}")

        algo_vs_bc_results = []

        print(f"\n--- {mdp_name.upper()} ---")

        # Load BC model once per MDP (from SOURCE database)
        train_split = config['external_validation']['bc']['train_split']
        bc_path = bc_dir / f"bc_{mdp_name}_{train_split}.d3"
        bc_model = load_model(model_path=bc_path, device=device)

        if bc_model is None:
            print(f"  BC model not found, skipping algo-vs-BC comparison")
        
        else:
            # Load TARGET dataset
            dataset_val = load_mdp(db_paths=fqe_target_paths, mdp_name=mdp_name, split=fqe_target_split)
            print(f"\n  {fqe_target_split.upper()} ({len(dataset_val.episodes)} episodes)")

            # Compare each RL algo vs BC
            for algo_name in ['cql', 'ddqn', 'bcq', 'nfq']:
                if config['external_validation']['algorithms'][algo_name]:
                    if model_source == 'hpo':
                        model_path = model_dir / f"best_{algo_name}_model.d3"
                    else:
                        model_path = model_dir / f"{algo_name}_{mdp_name}_model.d3"

                    rl_model = load_model(model_path=model_path, device=device)

                    if rl_model is not None:
                        metrics = compute_metrics_algo_vs_algo(
                            algo1=rl_model,
                            algo2=bc_model,
                            dataset=dataset_val
                        )
                        metrics['algo1'] = algo_name
                        metrics['algo2'] = 'bc'
                        metrics['source'] = model_source_db
                        metrics['target'] = fqe_target_db
                        metrics['mdp'] = mdp_name
                        metrics['fqe_target_split'] = fqe_target_split
                        metrics['fqe_fit_split'] = fqe_fit_split
                        algo_vs_bc_results.append(metrics)

                        print(f"    {algo_name.upper()} vs BC: "
                                f"Match={metrics['action_match']:.1%}, "
                                f"Earlier={metrics['rrt_timing_earlier']:.1%}, "
                                f"Same={metrics['rrt_timing_same']:.1%}")

            # Save algo-vs-BC results
            if algo_vs_bc_results:
                algo_vs_bc_df = pd.DataFrame(algo_vs_bc_results)
                print(f"\n{'='*60}\nSUMMARY (RL Algos vs BC)\n{'='*60}")
                print(algo_vs_bc_df.to_string(index=False))

                if config['external_validation']['output']['save_metrics']:
                    # Add metadata columns
                    algo_vs_bc_df = add_metadata_to_df(algo_vs_bc_df, metadata)
                    algo_vs_bc_df.to_csv(output_dir / "external_validation_algo_vs_bc.csv", index=False)

                if config['external_validation']['output']['save_plots']:
                    # Rename columns to match plot function format
                    plot_df = algo_vs_bc_df.rename(columns={
                        'algo1': 'algorithm',
                        'rrt_rate_per_state_algo1': 'rrt_rate_per_state_algo',
                        'rrt_rate_per_state_algo2': 'rrt_rate_per_state_data',
                        'rrt_rate_per_episode_algo1': 'rrt_rate_per_episode_algo',
                        'rrt_rate_per_episode_algo2': 'rrt_rate_per_episode_data',
                        'rrt_timing_algo1_only': 'rrt_timing_algo_only',
                        'rrt_timing_algo2_only': 'rrt_timing_data_only',
                    })
                    plot_evaluation_algo_vs_bc(
                        results_df=plot_df,
                        output_dir=output_dir,
                        title_prefix=f"RL vs BC: {model_source_db.upper()} → {fqe_target_db.upper()}",
                        filename_prefix="external_validation_algo_vs_bc"
                    )

    ######################################################################################## 
    ##  5. SAVE CONFIGURATION
    eval_config = {
        'source_database': model_source_db,
        'fqe_fit_database': fqe_fit_db,
        'target_database': fqe_target_db,
        'fqe_enabled': fqe_enabled,
        'fqe_bootstrap_enabled': fqe_bootstrap_enabled,
        'fqe_n_steps': fqe_n_steps,
        'fqe_n_epochs': fqe_n_epochs,
        'fqe_learning_rate': fqe_learning_rate,
        'fqe_fit_split': fqe_fit_split,
        'fqe_target_split': fqe_target_split,
        'mdp': mdp_name,
    }
    save_run_config(output_dir, metadata, eval_config=eval_config)

    run_id = metadata['run_id']
    print(f"\nResults saved to: {output_dir}")
    print(f"Run config saved to: run_{run_id}_config.json")
    logger.close()


if __name__ == "__main__":
    main()
