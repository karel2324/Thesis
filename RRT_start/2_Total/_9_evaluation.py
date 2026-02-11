"""
Evaluation Pipeline - Compare trained models on test data.

Compares HPO best models (CQL, DDQN, BCQ, NFQ) with BC baseline
using FQE policy evaluation and basic metrics.
"""

import pandas as pd
import torch

from utils import load_config, get_data_paths, load_mdp, load_model
from rl_utils import evaluate_algo, compute_mc_return, compute_metrics_algo_vs_algo
from rl_plotting import plot_evaluation_vs_behaviour_policy, plot_evaluation_algo_vs_bc
from run_metadata import get_run_metadata, print_run_header, save_run_config, add_metadata_to_df


def main():
    """Main entry point for evaluation."""
    config = load_config()
    all_paths = get_data_paths(config)

    for db_key in ['aumc', 'mimic']:
        if config['evaluation']['databases'][db_key]:
            if all_paths[db_key]['mdp_dir'].exists():
                run_evaluation_for_db(db_paths=all_paths[db_key], config=config)
            else:
                print(f"Skipping {db_key}: MDP directory not found")


def run_evaluation_for_db(db_paths: dict, config: dict):
    """
    Run evaluation for one database.

    Args:
        db_paths: Dictionary with database paths
        config: Full configuration dictionary
    """
    db_name = db_paths['name']

    # Generate run metadata for traceability
    metadata = get_run_metadata(config, db_name)
    print_run_header(metadata, title="EVALUATION")

    # Print evaluation config
    eval_cfg = config['evaluation']
    print(f"\nEVALUATION CONFIG:")
    print(f"  splits: {eval_cfg['splits']} | MDPs: {eval_cfg['mdps']}")
    print(f"  FQE: enabled={eval_cfg['fqe']['enabled']}, n_steps_per_epoch={eval_cfg['fqe']['n_steps_per_epoch']}, n_epochs={eval_cfg['fqe']['n_epochs']}")
    print("=" * 80)

    # Setup paths
    hpo_dir = db_paths['reward_dir'] / "HPO_results"
    bc_dir = db_paths['reward_dir'] / "BC_results"
    output_dir = db_paths['reward_dir'] / "Evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Settings
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = config['processing']['random_state']
    gamma = 0.99
    bc_train_split = config['evaluation']['bc']['train_split']  # First split from list

    all_results = []

    for mdp_name in config['evaluation']['mdps']:
        print(f"\n--- {mdp_name.upper()} ---")

        for split in config['evaluation']['splits']:
            dataset_val = load_mdp(db_paths=db_paths, mdp_name=mdp_name, split=split)
            dataset_train = load_mdp(db_paths=db_paths, mdp_name=mdp_name, split='train')

            # Monte Carlo return from data (behavior policy)
            mc_mean, mc_std = compute_mc_return(episodes=dataset_val.episodes, gamma=gamma)
            print(f"\n  {split.upper()} ({len(dataset_val.episodes)} episodes) - MC Return: {mc_mean:.4f} (±{mc_std:.4f})")

            # Evaluate HPO models (CQL, DDQN, BCQ, NFQ)
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
                            fqe_enabled=config['evaluation']['fqe']['enabled'],
                            fqe_learning_rate=config['evaluation']['fqe']['learning_rate'],
                            fqe_n_steps_per_epoch=config['evaluation']['fqe']['n_steps_per_epoch'],
                            fqe_n_epochs=config['evaluation']['fqe']['n_epochs'],
                            fqe_bootstrap_enabled=config['evaluation']['fqe']['bootstrap']['enabled'],
                            fqe_bootstrap_n_bootstrap=config['evaluation']['fqe']['bootstrap']['n_bootstrap'],
                            fqe_bootstrap_n_steps=config['evaluation']['fqe']['bootstrap']['n_steps'],
                            fqe_bootstrap_confidence_level=config['evaluation']['fqe']['bootstrap']['confidence_level'],
                            gamma=gamma)
                        result['mdp'] = mdp_name
                        result['split'] = split
                        all_results.append(result)

            # Evaluate BC model
            if config['evaluation']['algorithms']['bc']:
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
                        fqe_enabled=config['evaluation']['fqe']['enabled'],
                        fqe_learning_rate=config['evaluation']['fqe']['learning_rate'],
                        fqe_n_steps_per_epoch=config['evaluation']['fqe']['n_steps_per_epoch'],
                        fqe_n_epochs=config['evaluation']['fqe']['n_epochs'],
                        fqe_bootstrap_enabled=config['evaluation']['fqe']['bootstrap']['enabled'],
                        fqe_bootstrap_n_bootstrap=config['evaluation']['fqe']['bootstrap']['n_bootstrap'],
                        fqe_bootstrap_n_steps=config['evaluation']['fqe']['bootstrap']['n_steps'],
                        fqe_bootstrap_confidence_level=config['evaluation']['fqe']['bootstrap']['confidence_level'],
                        gamma=gamma
                    )
                    result['mdp'] = mdp_name
                    result['split'] = split
                    all_results.append(result)

    # Save results vs behaviour policy
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(f"\n{'='*60}\nSUMMARY (vs Behaviour Policy)\n{'='*60}")
        print(results_df.to_string(index=False))

        if config['evaluation']['output']['save_metrics']:
            # Add metadata columns
            results_df = add_metadata_to_df(results_df, metadata)
            results_df.to_csv(output_dir / "evaluation_results.csv", index=False)

        if config['evaluation']['output']['save_plots']:
            plot_evaluation_vs_behaviour_policy(
                results_df=results_df,
                output_dir=output_dir,
                title_prefix="Evaluation Results",
                filename_prefix="evaluation"
            )

    # =========================================================================
    # ALGO VS ALGO COMPARISON (each RL algo vs BC)
    # =========================================================================
    if config['evaluation']['algorithms']['bc']:
        print(f"\n{'='*60}\nALGO VS BC COMPARISON\n{'='*60}")

        algo_vs_bc_results = []

        for mdp_name in config['evaluation']['mdps']:
            print(f"\n--- {mdp_name.upper()} ---")

            # Load BC model once per MDP
            bc_path = bc_dir / f"bc_{mdp_name}_{bc_train_split}.d3"
            bc_model = load_model(model_path=bc_path, device=device)

            if bc_model is None:
                print(f"  BC model not found, skipping algo-vs-BC comparison")
                continue

            for split in config['evaluation']['splits']:
                dataset = load_mdp(db_paths=db_paths, mdp_name=mdp_name, split=split)
                print(f"\n  {split.upper()} ({len(dataset.episodes)} episodes)")

                # Compare each RL algo vs BC
                for algo_name in ['cql', 'ddqn', 'bcq', 'nfq']:
                    if config['evaluation']['algorithms'][algo_name]:
                        model_path = hpo_dir / f"best_{algo_name}_model.d3"
                        rl_model = load_model(model_path=model_path, device=device)

                        if rl_model is not None:
                            metrics = compute_metrics_algo_vs_algo(
                                algo1=rl_model,
                                algo2=bc_model,
                                dataset=dataset
                            )
                            metrics['algo1'] = algo_name
                            metrics['algo2'] = 'bc'
                            metrics['mdp'] = mdp_name
                            metrics['split'] = split
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

    # Save run configuration JSON
    eval_config = {
        'fqe_enabled': config['evaluation']['fqe']['enabled'],
        'fqe_n_steps': (config['evaluation']['fqe']['n_steps_per_epoch'])*(config['evaluation']['fqe']['n_epochs']),
        'fqe_learning_rate': config['evaluation']['fqe']['learning_rate'],
        'splits': config['evaluation']['splits'],
        'mdps': config['evaluation']['mdps'],
    }
    save_run_config(output_dir, metadata, eval_config=eval_config)

    run_id = metadata['run_id']
    print(f"\nResults saved to: {output_dir}")
    print(f"Run config saved to: run_{run_id}_config.json")


if __name__ == "__main__":
    main()
