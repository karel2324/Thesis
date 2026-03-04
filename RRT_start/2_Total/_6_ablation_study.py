"""
Ablation Study - Compare MDP configurations with CQL and BC.

Settings from config.yaml ablation section.
"""

import numpy as np
import pandas as pd
import joblib
import torch
import d3rlpy
from d3rlpy.ope import FQEConfig

from utils import load_mdp, save_config_snapshot, start_logging
from rl_utils import compute_mc_return, train_cql, train_bc, function_fqe, uncertainty_fqe, compute_metrics_vs_behaviour_policy
from rl_plotting import (plot_fqe_comparison, plot_ablation_training_curves, plot_fqe_vs_mc,
                         plot_ablation_td_error, plot_ablation_isv, plot_ablation_action_match,
                         plot_ablation_summary_bar)
from run_metadata import get_run_metadata, print_run_header, save_run_config, add_metadata_to_df


def main():
    """Main entry point for ablation study."""
    from utils import load_config, get_data_paths

    config = load_config()
    all_paths = get_data_paths(config)

    if config['ablation']['databases']['aumc'] == True:
        print("AUMCdb is enabled for ablation study")
        if all_paths['aumc']['mdp_dir'].exists():
            run_ablation_for_db(all_paths['aumc'], config)
        else:
            print(f"Skipping AUMCdb: MDP directory not found")

    if config['ablation']['databases']['mimic'] == True:
        print("MIMIC is enabled for ablation study")
        if all_paths['mimic']['mdp_dir'].exists():
            run_ablation_for_db(all_paths['mimic'], config)
        else:
            print(f"Skipping MIMIC: MDP directory not found")


def run_ablation_for_db(db_paths: dict, config: dict):
    """
    Run ablation study for one database.

    Args:
        db_paths: Dictionary with database paths
        config: Full configuration dictionary
    """

    ########################################################################################
    # 1. SOURCES
    ## 1.0 Database
    db_name = db_paths['name']
    mdp_dir = db_paths['mdp_dir']

    ## 1.1 MDPs to run
    mdps_to_run = config['ablation']['mdps_to_run']

    ## 1.2 Metadata
    metadata = get_run_metadata(config, db_name)
    print_run_header(metadata, title="ABLATION STUDY")

    ########################################################################################
    # 2. OUTPUT
    output_dir = db_paths['reward_dir'] / "Ablation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = start_logging(output_dir, name="ablation")
    save_config_snapshot(output_dir)

    ########################################################################################
    # 3. SETTINGS
    # 3.1. General settings
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = config['processing']['random_state']

    # 3.2. CQL hyperparameters
    algo_cfg = config['ablation']['algorithm']['hyperparameters']
    gamma = algo_cfg['gamma']

    # 3.3. BC hyperparameters
    bc_enabled = config['ablation']['bc_baseline']['enabled']
    bc_cfg = config['ablation']['bc_baseline']['hyperparameters'] if bc_enabled else None

    # 3.4. FQE
    fqe_enabled = config['ablation']['fqe']['enabled']
    fqe_learning_rate = config['ablation']['fqe']['learning_rate']
    fqe_n_steps = config['ablation']['fqe']['n_steps']
    fqe_n_epochs = config['ablation']['fqe']['n_epochs']

    # 3.5. FQE bootstrap
    fqe_bootstrap_enabled = config['ablation']['fqe']['bootstrap']['enabled']
    fqe_bootstrap_n_bootstrap = config['ablation']['fqe']['bootstrap']['n_bootstrap']
    fqe_bootstrap_n_steps = config['ablation']['fqe']['bootstrap']['n_steps']
    fqe_bootstrap_n_epochs = config['ablation']['fqe']['bootstrap']['n_epochs']
    fqe_bootstrap_CI = config['ablation']['fqe']['bootstrap']['confidence_level']

    ########################################################################################
    # Print CHECKs

    print(f"\nABLATION CONFIG:")
    print(f"  Database: {db_name.upper()} | MDPs: {mdps_to_run}")
    print(f"  CQL: n_steps={algo_cfg['n_steps']} | batch_size={algo_cfg['batch_size']} | alpha={algo_cfg['alpha']}")
    print(f"  CQL: lr={algo_cfg['learning_rate']} | gamma={gamma} | hidden_units={algo_cfg['hidden_units']}")
    print(f"  BC: enabled={bc_enabled}")
    print(f"  FQE: enabled={fqe_enabled}, n_steps={fqe_n_steps}, n_epochs={fqe_n_epochs}")
    print(f"  FQE bootstrap: enabled={fqe_bootstrap_enabled}")
    print("=" * 80)

    ########################################################################################
    # 4. TRAINING LOOP (per MDP)

    all_results = {}

    for name_mdp in mdps_to_run:

        # 4.0. Check if MDP exists
        if not (mdp_dir / f"{name_mdp}_train.h5").exists():
            print(f"\n  {name_mdp}: SKIPPED (not found)")
            continue

        print(f"\n{'='*60}")
        print(f"  MDP: {name_mdp.upper()}")
        print(f"{'='*60}")

        # 4.1. Load datasets & config
        datasets = {
            'train': load_mdp(db_paths, name_mdp, 'train'),
            'val': load_mdp(db_paths, name_mdp, 'val'),
            'test': load_mdp(db_paths, name_mdp, 'test'),
        }
        mdp_config = joblib.load(mdp_dir / f"{name_mdp}_config.joblib")
        print(f"  {mdp_config['n_states']} features, {len(datasets['train'].episodes)} episodes")

        result = {'n_features': mdp_config['n_states']}

        # 4.2. Train CQL
        print(f"\n  Training CQL...")
        d3rlpy.seed(seed)

        cql, metrics, _ = train_cql(
            train_ds=datasets['train'],
            val_ds=datasets['val'],
            alpha=algo_cfg['alpha'],
            learning_rate=algo_cfg['learning_rate'],
            batch_size=algo_cfg['batch_size'],
            gamma=gamma,
            n_critics=algo_cfg['n_critics'],
            hidden_units=algo_cfg['hidden_units'],
            n_steps=algo_cfg['n_steps'],
            n_epochs=algo_cfg['n_epochs'],
            target_update_interval=algo_cfg['target_update_interval'],
            device=device,
            save_interval=5,
            name=name_mdp)

        metrics_bp = compute_metrics_vs_behaviour_policy(cql, datasets['val'])
        for k, v in metrics_bp.items():
            print(f"    {k}: {v:.3f}")

        result['metrics'] = metrics
        result['cql_model'] = cql

        metrics_with_meta = add_metadata_to_df(metrics.copy(), metadata)
        metrics_with_meta.to_csv(output_dir / f"{name_mdp}_metrics.csv", index=False)
        cql.save(str(output_dir / f"{name_mdp}_cql.d3"))

        # 4.3. Train BC (baseline)
        result['bc_model'] = None
        if bc_enabled:
            print(f"\n  Training BC (baseline)...")
            d3rlpy.seed(seed)

            bc, _, _ = train_bc(
                train_ds=datasets['train'],
                val_ds=datasets['val'],
                learning_rate=bc_cfg['learning_rate'],
                batch_size=bc_cfg['batch_size'],
                beta=bc_cfg['beta'],
                hidden_units=bc_cfg['hidden_units'],
                n_steps=bc_cfg['n_steps'],
                n_steps_per_epoch=bc_cfg['n_steps'],
                device=device,
                name=name_mdp)

            result['bc_model'] = bc
            bc.save(str(output_dir / f"{name_mdp}_bc.d3"))

        # 4.4. Monte Carlo Returns
        mc_mean, mc_std = compute_mc_return(episodes=datasets['val'].episodes, gamma=gamma)
        result['mc_mean'] = mc_mean
        result['mc_std'] = mc_std
        print(f"\n  MC Return: {mc_mean:.4f} +/- {mc_std:.4f}")

        # 4.5. FQE
        result['fqe_cql'] = {}
        result['fqe_bc'] = {}

        if fqe_enabled:
            fqe_cfg = FQEConfig(learning_rate=fqe_learning_rate, gamma=gamma)

            # FQE for CQL
            fqe_isv = function_fqe(
                algo=cql,
                dataset_train=datasets['train'],
                dataset_val=datasets['val'],
                fqe_config=fqe_cfg,
                n_steps=fqe_n_steps,
                n_epochs=fqe_n_epochs,
                seed=seed,
                device=device)
            result['fqe_cql'] = {'fqe_isv': fqe_isv}
            print(f"  FQE (CQL): {fqe_isv:.4f}")

            if fqe_bootstrap_enabled:
                m_isv, ci_lo, ci_hi = uncertainty_fqe(
                    algo=cql,
                    dataset_train=datasets['train'],
                    dataset_val=datasets['val'],
                    fqe_config=fqe_cfg,
                    n_bootstrap=fqe_bootstrap_n_bootstrap,
                    n_steps=fqe_bootstrap_n_steps,
                    n_epochs=fqe_bootstrap_n_epochs,
                    device=device,
                    seed=seed,
                    CI=fqe_bootstrap_CI)
                result['fqe_cql'].update({'mean_isv': m_isv, 'ci_low': ci_lo, 'ci_high': ci_hi})
                print(f"  FQE (CQL) bootstrap: {m_isv:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

            # FQE for BC
            if result['bc_model'] is not None:
                fqe_isv_bc = function_fqe(
                    algo=result['bc_model'],
                    dataset_train=datasets['train'],
                    dataset_val=datasets['val'],
                    fqe_config=fqe_cfg,
                    n_steps=fqe_n_steps,
                    n_epochs=fqe_n_epochs,
                    seed=seed,
                    device=device)
                result['fqe_bc'] = {'fqe_isv': fqe_isv_bc}
                print(f"  FQE (BC): {fqe_isv_bc:.4f}")

                if fqe_bootstrap_enabled:
                    m_isv, ci_lo, ci_hi = uncertainty_fqe(
                        algo=result['bc_model'],
                        dataset_train=datasets['train'],
                        dataset_val=datasets['val'],
                        fqe_config=fqe_cfg,
                        n_bootstrap=fqe_bootstrap_n_bootstrap,
                        n_steps=fqe_bootstrap_n_steps,
                        n_epochs=fqe_bootstrap_n_epochs,
                        device=device,
                        seed=seed,
                        CI=fqe_bootstrap_CI)
                    result['fqe_bc'].update({'mean_isv': m_isv, 'ci_low': ci_lo, 'ci_high': ci_hi})
                    print(f"  FQE (BC) bootstrap: {m_isv:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

        all_results[name_mdp] = result

    ########################################################################################
    # 5. PLOTS
    mdp_names = list(all_results.keys())

    print("\n--- Creating plots ---")

    ## 5.1. Training curves (separate plots)
    plot_ablation_td_error(all_results, mdp_names, output_dir)
    plot_ablation_isv(all_results, mdp_names, output_dir)
    plot_ablation_action_match(all_results, mdp_names, output_dir)
    plot_ablation_summary_bar(all_results, mdp_names, output_dir)

    ## 5.2. Per-comparison training curves
    comparisons = config['ablation'].get('comparisons', {})
    if comparisons:
        plot_ablation_training_curves(all_results, comparisons, output_dir)

    ## 5.3. FQE comparison plots
    fqe_cql = {n: r['fqe_cql'] for n, r in all_results.items() if r['fqe_cql']}
    fqe_bc = {n: r['fqe_bc'] for n, r in all_results.items() if r['fqe_bc']}
    mc = {n: {'mean': r['mc_mean'], 'std': r['mc_std']} for n, r in all_results.items()}

    if fqe_cql and fqe_bc:
        plot_fqe_comparison(fqe_cql, fqe_bc, mc, mdp_names, output_dir)

    if fqe_cql:
        plot_fqe_vs_mc(fqe_cql, mc, mdp_names, output_dir)

    ########################################################################################
    # 6. SUMMARY
    summary = [{
        'MDP': name,
        'n_features': r['n_features'],
        'TD_val_final': r['metrics']['td_val'].iloc[-1],
        'ISV_end': r['metrics']['isv_val'].iloc[-1],
        'ISV_peak': r['metrics']['isv_val'].max(),
        'Action_match': r['metrics']['action_match'].iloc[-1],
        'MC_return': r['mc_mean'],
        'FQE_CQL': r['fqe_cql'].get('fqe_isv', np.nan),
        'FQE_BC': r['fqe_bc'].get('fqe_isv', np.nan),
    } for name, r in all_results.items()]

    summary_df = pd.DataFrame(summary)
    print(f"\n{summary_df.to_string(index=False)}")

    summary_df = add_metadata_to_df(summary_df, metadata)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    ########################################################################################
    # 7. SAVE CONFIGURATION
    training_config = {
        'algorithm': 'cql',
        **algo_cfg
    }
    fqe_config_save = {
        'fqe_enabled': fqe_enabled,
        'fqe_n_steps': fqe_n_steps if fqe_enabled else None,
        'fqe_n_epochs': fqe_n_epochs if fqe_enabled else None,
        'fqe_learning_rate': fqe_learning_rate if fqe_enabled else None,
        'fqe_bootstrap_enabled': fqe_bootstrap_enabled,
    }
    save_run_config(output_dir, metadata, training_config=training_config, eval_config=fqe_config_save)

    run_id = metadata['run_id']
    print(f"\nResults saved to: {output_dir}")
    print(f"Run config saved to: run_{run_id}_config.json")
    logger.close()


if __name__ == "__main__":
    main()
