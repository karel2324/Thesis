"""
External Validation - Evaluate models across databases.

Tests models trained on source database (e.g., AUMCdb) on target database (e.g., MIMIC).
Uses shared evaluation utilities from rl_utils.
"""

import pandas as pd
import torch

from utils import load_config, get_data_paths, load_mdp, load_model, save_config_snapshot
from rl_utils import evaluate_algo, compute_mc_return, compute_metrics_algo_vs_algo
from rl_plotting import plot_evaluation_vs_behaviour_policy, plot_evaluation_algo_vs_bc
from run_metadata import get_run_metadata, print_run_header, save_run_config, add_metadata_to_df


def main():
    """Main entry point for external validation."""
    config = load_config()
    all_paths = get_data_paths(config)

    source_db = config['external_validation']['source_database']
    target_db = config['external_validation']['target_database']

    # Validate databases exist
    if not all_paths[source_db]['mdp_dir'].exists():
        print(f"Source database {source_db} MDP directory not found")
        return
    if not all_paths[target_db]['mdp_dir'].exists():
        print(f"Target database {target_db} MDP directory not found")
        return

    run_external_validation(config=config, all_paths=all_paths)


def run_external_validation(config: dict, all_paths: dict):
    """
    Run cross-database validation.

    Args:
        config: Full configuration dictionary
        all_paths: Dictionary with paths for all databases
    """

    source_db = config['external_validation']['source_database']
    target_db = config['external_validation']['target_database']
    source_paths = all_paths[source_db]
    target_paths = all_paths[target_db]

    # Generate run metadata for traceability (using target db as context)
    metadata = get_run_metadata(config, f"{source_db}_to_{target_db}")
    print_run_header(metadata, title="EXTERNAL VALIDATION")

    print(f"\nVALIDATION CONFIG:")
    print(f"  Source: {source_db.upper()} | Target: {target_db.upper()}")
    print(f"  splits: {config['external_validation']['splits']} | MDPs: {config['external_validation']['mdps']}")
    print(f"  FQE: enabled={config['external_validation']['fqe']['enabled']}, n_steps={config['external_validation']['fqe']['n_steps']}, n_epochs={config['external_validation']['fqe']['n_epochs']}")

    # Settings
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = config['processing']['random_state']
    gamma = 0.99

    # Model directories (from source database)
    model_source = config['external_validation']['model_source']
    if model_source == 'hpo':
        model_dir = source_paths['reward_dir'] / "HPO_results"
    else:
        model_dir = source_paths['reward_dir'] / "Ablation_results"
    bc_dir = source_paths['reward_dir'] / "BC_results"

    print(f"\nMODEL SOURCES ({model_source.upper()}):")
    print(f"  RL models: {model_dir}")
    print(f"  BC model: {bc_dir} (train_split={config['external_validation']['bc']['train_split']})")
    print("=" * 80)

    # Output directory (in target database)
    output_dir = target_paths['reward_dir'] / "External_validation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(output_dir)

    all_results = []

    for mdp_name in config['external_validation']['mdps']:
        print(f"\n--- {mdp_name.upper()} ---")

        for split in config['external_validation']['splits']:
            # Load TARGET dataset (this is the key difference from _9_evaluation)
            dataset_val = load_mdp(db_paths=target_paths, mdp_name=mdp_name, split=split)
            dataset_train = load_mdp(db_paths=source_paths, mdp_name=mdp_name, split=config['external_validation']['split_source'])

            # Monte Carlo return from target data
            mc_mean, mc_std = compute_mc_return(episodes=dataset_val.episodes, gamma=gamma)
            print(f"\n  {split.upper()} ({len(dataset_val.episodes)} episodes) - MC Return: {mc_mean:.4f} (±{mc_std:.4f})")

            # Evaluate HPO models (CQL, DDQN, BCQ, NFQ) trained on SOURCE
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
                            fqe_enabled=config['external_validation']['fqe']['enabled'],
                            fqe_learning_rate=config['external_validation']['fqe']['learning_rate'],
                            fqe_n_steps=config['external_validation']['fqe']['n_steps'],
                            fqe_n_epochs=config['external_validation']['fqe']['n_epochs'],
                            fqe_bootstrap_enabled=config['external_validation']['fqe']['bootstrap']['enabled'],
                            fqe_bootstrap_n_bootstrap=config['external_validation']['fqe']['bootstrap']['n_bootstrap'],
                            fqe_bootstrap_n_steps=config['external_validation']['fqe']['bootstrap']['n_steps'],
                            fqe_bootstrap_confidence_level=config['external_validation']['fqe']['bootstrap']['confidence_level'],
                            gamma=gamma
                        )
                        result['source'] = source_db
                        result['target'] = target_db
                        result['mdp'] = mdp_name
                        result['split'] = split
                        all_results.append(result)
                    else:
                        print(f"    {algo_name.upper()}: Model not found at {model_path}")

            # Evaluate BC model trained on SOURCE
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
                        fqe_enabled=config['external_validation']['fqe']['enabled'],
                        fqe_learning_rate=config['external_validation']['fqe']['learning_rate'],
                        fqe_n_steps=config['external_validation']['fqe']['n_steps'],
                        fqe_n_epochs=config['external_validation']['fqe']['n_epochs'],
                        fqe_bootstrap_enabled=config['external_validation']['fqe']['bootstrap']['enabled'],
                        fqe_bootstrap_n_bootstrap=config['external_validation']['fqe']['bootstrap']['n_bootstrap'],
                        fqe_bootstrap_n_steps=config['external_validation']['fqe']['bootstrap']['n_steps'],
                        fqe_bootstrap_confidence_level=config['external_validation']['fqe']['bootstrap']['confidence_level'],
                        gamma=gamma
                    )
                    result['source'] = source_db
                    result['target'] = target_db
                    result['mdp'] = mdp_name
                    result['split'] = split
                    all_results.append(result)
                else:
                    print(f"    BC: Model not found at {model_path}")

    # Save results vs behaviour policy
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(f"\n{'='*60}\nSUMMARY (vs Behaviour Policy)\n{'='*60}")
        print(results_df.to_string(index=False))

        if config['external_validation']['output']['save_metrics']:
            # Add metadata columns
            results_df = add_metadata_to_df(results_df, metadata)
            results_df.to_csv(output_dir / "external_validation_results.csv", index=False)

        if config['external_validation']['output']['save_plots']:
            plot_evaluation_vs_behaviour_policy(
                results_df=results_df,
                output_dir=output_dir,
                title_prefix=f"External Validation: {source_db.upper()} → {target_db.upper()}",
                filename_prefix="external_validation"
            )

    # =========================================================================
    # ALGO VS ALGO COMPARISON (each RL algo vs BC)
    # =========================================================================
    if config['external_validation']['algorithms']['bc']:
        print(f"\n{'='*60}\nALGO VS BC COMPARISON\n{'='*60}")

        algo_vs_bc_results = []

        for mdp_name in config['external_validation']['mdps']:
            print(f"\n--- {mdp_name.upper()} ---")

            # Load BC model once per MDP (from SOURCE database)
            train_split = config['external_validation']['bc']['train_split']
            bc_path = bc_dir / f"bc_{mdp_name}_{train_split}.d3"
            bc_model = load_model(model_path=bc_path, device=device)

            if bc_model is None:
                print(f"  BC model not found, skipping algo-vs-BC comparison")
                continue

            for split in config['external_validation']['splits']:
                # Load TARGET dataset
                dataset_val = load_mdp(db_paths=target_paths, mdp_name=mdp_name, split=split)
                print(f"\n  {split.upper()} ({len(dataset_val.episodes)} episodes)")

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
                            metrics['source'] = source_db
                            metrics['target'] = target_db
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
                    title_prefix=f"RL vs BC: {source_db.upper()} → {target_db.upper()}",
                    filename_prefix="external_validation_algo_vs_bc"
                )

    # Save run configuration JSON
    eval_config = {
        'source_database': source_db,
        'target_database': target_db,
        'fqe_enabled': config['external_validation']['fqe']['enabled'],
        'fqe_n_steps': config['external_validation']['fqe']['n_steps'],
        'fqe_n_epochs': config['external_validation']['fqe']['n_epochs'],
        'fqe_learning_rate': config['external_validation']['fqe']['learning_rate'],
        'splits': config['external_validation']['splits'],
        'mdps': config['external_validation']['mdps'],
    }
    save_run_config(output_dir, metadata, eval_config=eval_config)

    run_id = metadata['run_id']
    print(f"\nResults saved to: {output_dir}")
    print(f"Run config saved to: run_{run_id}_config.json")


if __name__ == "__main__":
    main()
