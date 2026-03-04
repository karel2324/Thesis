"""
Evaluation Pipeline - Compare trained models on test data.

Compares HPO best models (CQL, DDQN, BCQ, NFQ) with BC baseline
using FQE policy evaluation and basic metrics.
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
    """Main entry point for evaluation."""
    config = load_config()
    all_paths = get_data_paths(config)

    if config['evaluation']['databases']['aumc'] == True:
        print("AUMCdb is enabled for evaluation")
        if all_paths['aumc']['mdp_dir'].exists():
            run_evaluation_for_db(db_paths=all_paths['aumc'], config=config)
        else:
            print(f"Skipping AUMCdb: MDP directory not found")

    if config['evaluation']['databases']['mimic'] == True:
        print("MIMIC is enabled for evaluation")
        if all_paths['mimic']['mdp_dir'].exists():
            run_evaluation_for_db(db_paths=all_paths['mimic'], config=config)
        else:
            print(f"Skipping MIMIC: MDP directory not found")


def run_evaluation_for_db(db_paths: dict, config: dict):
    """
    Run evaluation for one database.

    Args:
        db_paths: Dictionary with database paths
        config: Full configuration dictionary
    """

    ########################################################################################
    # 1. SOURCES
    ## 1.0 MDP en database
    db_name = db_paths['name']
    mdp_name = config['evaluation']['mdp']

    ## 1.1. Datasets
    fqe_fit_split = config['evaluation']['fqe_fit_split']
    fqe_isv_split = config['evaluation']['fqe_isv_split']
    dataset_train = load_mdp(db_paths=db_paths, mdp_name=mdp_name, split=fqe_fit_split)
    dataset_val = load_mdp(db_paths=db_paths, mdp_name=mdp_name, split=fqe_isv_split)

    ## 1.2. Model directories
    hpo_dir = db_paths['reward_dir'] / "HPO_results"
    bc_dir = db_paths['reward_dir'] / "BC_results"

    ## 1.3. Metadata
    metadata = get_run_metadata(config, db_name)
    print_run_header(metadata, title="EVALUATION")

    ########################################################################################
    # 2. OUTPUT
    output_dir = db_paths['reward_dir'] / "Evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = start_logging(output_dir, name="evaluation")
    save_config_snapshot(output_dir)

    ########################################################################################
    # 3. SETTINGS
    # 3.1. General settings
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = config['processing']['random_state']
    gamma = 0.99

    # 3.2. FQE
    fqe_enabled = config['evaluation']['fqe']['enabled']
    fqe_learning_rate = config['evaluation']['fqe']['learning_rate']
    fqe_n_steps = config['evaluation']['fqe']['n_steps']
    fqe_n_epochs = config['evaluation']['fqe']['n_epochs']

    # 3.3. FQE bootstrap
    fqe_bootstrap_enabled = config['evaluation']['fqe']['bootstrap']['enabled']
    fqe_bootstrap_n_bootstraps = config['evaluation']['fqe']['bootstrap']['n_bootstrap']
    fqe_bootstrap_CI = config['evaluation']['fqe']['bootstrap']['confidence_level']
    fqe_bootstrap_n_steps = config['evaluation']['fqe']['bootstrap']['n_steps']
    fqe_bootstrap_n_epochs = config['evaluation']['fqe']['bootstrap']['n_epochs']

    ########################################################################################
    # Print CHECKs

    print(f"\nEVALUATION CONFIG:")
    print(f"  Database: {db_name.upper()}")
    print(f"  FQE Fit: {fqe_fit_split} | FQE ISV: {fqe_isv_split} | MDP: {mdp_name}")
    print(f"  FQE: enabled={fqe_enabled}, n_steps={fqe_n_steps}, n_epochs={fqe_n_epochs}")
    print(f"\nMODEL SOURCES:")
    print(f"  RL models: {hpo_dir}")
    print(f"  BC model: {bc_dir} (train_split={config['evaluation']['bc']['train_split']})")
    print("=" * 80)

    print(f"\n--- {mdp_name.upper()} ---")

    ########################################################################################
    # 4. EVALUATION

    # 4.1. Initialize results
    all_results = []

    # 4.2. Monte Carlo return from target data
    mc_mean, mc_std = compute_mc_return(episodes=dataset_val.episodes, gamma=gamma)
    print(f"\n  {fqe_isv_split.upper()} ({len(dataset_val.episodes)} episodes) - MC Return: {mc_mean:.4f} (±{mc_std:.4f})")

    # 4.3. Evaluate HPO models (CQL, DDQN, BCQ, NFQ)
    for algo_name in ['cql', 'ddqn', 'bcq', 'nfq']:
        if config['evaluation']['algorithms'][algo_name]:
            model_path = hpo_dir / f"best_{algo_name}_model.d3"
            model = load_model(model_path=model_path, device=device)
            if model is not None:
                result = evaluate_algo(
                    algo=model,
                    algo_name=algo_name,
                    dataset_train=dataset_train,
                    dataset_val=dataset_val,
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
                result['mdp'] = mdp_name
                result['fqe_fit_split'] = fqe_fit_split
                result['fqe_isv_split'] = fqe_isv_split
                all_results.append(result)
            else:
                print(f"    {algo_name.upper()}: Model not found at {model_path}")

    # 4.4. Evaluate BC model
    if config['evaluation']['algorithms']['bc']:
        bc_train_split = config['evaluation']['bc']['train_split']
        model_path = bc_dir / f"bc_{mdp_name}_{bc_train_split}.d3"
        model = load_model(model_path=model_path, device=device)
        if model is not None:
            result = evaluate_algo(
                algo=model,
                algo_name='bc',
                dataset_train=dataset_train,
                dataset_val=dataset_val,
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
            result['mdp'] = mdp_name
            result['fqe_fit_split'] = fqe_fit_split
            result['fqe_isv_split'] = fqe_isv_split
            all_results.append(result)
        else:
            print(f"    BC: Model not found at {model_path}")

    ## 4.5. SAVE RESULTS
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(f"\n{'='*60}\nSUMMARY (vs Behaviour Policy)\n{'='*60}")
        print(results_df.to_string(index=False))

        if config['evaluation']['output']['save_metrics']:
            # Add metadata columns
            results_df = add_metadata_to_df(results_df, metadata)
            results_df.to_csv(output_dir / "evaluation_results.csv", index=False)

        if config['evaluation']['output']['save_plots']:
            plot_fqe_isv(results_df, output_dir, "Evaluation Results", "evaluation")
            plot_action_match(results_df, output_dir, "Evaluation Results", "evaluation")
            plot_rrt_rate_per_state(results_df, output_dir, "Evaluation Results", "evaluation")
            plot_rrt_rate_per_episode(results_df, output_dir, "Evaluation Results", "evaluation")
            plot_rrt_timing(results_df, output_dir, "Evaluation Results", "evaluation")
            plot_summary_table(results_df, output_dir, "Evaluation Results", "evaluation")

    ## 4.6. COMPARE ALGORITHM VS BEHAVIOR CLONING
    if config['evaluation']['algorithms']['bc']:
        print(f"\n{'='*60}\nALGO VS BC COMPARISON\n{'='*60}")

        algo_vs_bc_results = []

        print(f"\n--- {mdp_name.upper()} ---")

        # Load BC model once per MDP
        bc_train_split = config['evaluation']['bc']['train_split']
        bc_path = bc_dir / f"bc_{mdp_name}_{bc_train_split}.d3"
        bc_model = load_model(model_path=bc_path, device=device)

        if bc_model is None:
            print(f"  BC model not found, skipping algo-vs-BC comparison")

        else:
            print(f"\n  {fqe_isv_split.upper()} ({len(dataset_val.episodes)} episodes)")

            # Compare each RL algo vs BC
            for algo_name in ['cql', 'ddqn', 'bcq', 'nfq']:
                if config['evaluation']['algorithms'][algo_name]:
                    model_path = hpo_dir / f"best_{algo_name}_model.d3"
                    rl_model = load_model(model_path=model_path, device=device)

                    if rl_model is not None:
                        metrics = compute_metrics_algo_vs_algo(
                            algo1=rl_model,
                            algo2=bc_model,
                            dataset=dataset_val
                        )
                        metrics['algo1'] = algo_name
                        metrics['algo2'] = 'bc'
                        metrics['mdp'] = mdp_name
                        metrics['fqe_fit_split'] = fqe_fit_split
                        metrics['fqe_isv_split'] = fqe_isv_split
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

                if config['evaluation']['output']['save_metrics']:
                    # Add metadata columns
                    algo_vs_bc_df = add_metadata_to_df(algo_vs_bc_df, metadata)
                    algo_vs_bc_df.to_csv(output_dir / "evaluation_algo_vs_bc.csv", index=False)

                if config['evaluation']['output']['save_plots']:
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
                        title_prefix="RL Algorithms vs BC",
                        filename_prefix="evaluation_algo_vs_bc"
                    )

    ########################################################################################
    ##  5. SAVE CONFIGURATION
    eval_config = {
        'fqe_enabled': fqe_enabled,
        'fqe_bootstrap_enabled': fqe_bootstrap_enabled,
        'fqe_n_steps': fqe_n_steps,
        'fqe_n_epochs': fqe_n_epochs,
        'fqe_learning_rate': fqe_learning_rate,
        'fqe_fit_split': fqe_fit_split,
        'fqe_isv_split': fqe_isv_split,
        'mdp': mdp_name,
    }
    save_run_config(output_dir, metadata, eval_config=eval_config)

    run_id = metadata['run_id']
    print(f"\nResults saved to: {output_dir}")
    print(f"Run config saved to: run_{run_id}_config.json")
    logger.close()


if __name__ == "__main__":
    main()
