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

from utils import load_mdp, save_config_snapshot
from rl_utils import compute_mc_return, train_cql, train_bc, function_fqe, uncertainty_fqe, compute_metrics_vs_behaviour_policy
from rl_plotting import plot_training_curves, plot_fqe_comparison
from run_metadata import get_run_metadata, print_run_header, save_run_config, add_metadata_to_df

def main():
    """Main entry point for ablation study."""
    from utils import load_config, get_data_paths

    config = load_config()
    all_paths = get_data_paths(config) # This includes configuration for reward 

    # Check whether to run AUMCdb
    if config['ablation']['databases']['aumc'] == True:
        print("AUMCdb is enabled for ablation study")
        if all_paths['aumc']['mdp_dir'].exists():
            run_ablation_for_db(all_paths['aumc'], config)
        else:
            print(f"Skipping AUMCdb: MDP directory not found")
    
    # Check whether to run MIMIC
    if config['ablation']['databases']['mimic'] == True:     
        print("MIMIC is enabled for ablation study")   
        if all_paths['mimic']['mdp_dir'].exists():
            run_ablation_for_db(all_paths['mimic'], config)
        else:
            print(f"Skipping MIMIC: MDP directory not found")


def run_ablation_for_db(db_paths: dict, config: dict):
    """Run ablation study for one database."""

    # OUTPUT
    output_dir = db_paths['reward_dir'] / "Ablation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(output_dir)

    # INPUT
    mdp_dir = db_paths['mdp_dir']

    # CONFIGURATION
    db_name = db_paths['name']
    ablation_cfg = config['ablation']
    seed = config['processing']['random_state']
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Generate run metadata for traceability
    metadata = get_run_metadata(config, db_name)
    print_run_header(metadata, title="ABLATION STUDY")

    # Print training config
    algo_cfg = ablation_cfg['algorithm']['hyperparameters']
    print(f"\nTRAINING CONFIG (CQL):")
    print(f"  n_steps: {algo_cfg['n_steps']} | batch_size: {algo_cfg['batch_size']}")
    print(f"  alpha: {algo_cfg['alpha']} | lr: {algo_cfg['learning_rate']} | gamma: {algo_cfg['gamma']}")
    print("=" * 80)

    # =========================================================================
    # MAIN LOOP: per MDP — load, train CQL, train BC, MC, FQE
    # =========================================================================
    mdps_to_run = ablation_cfg['mdps_to_run']
    gamma = algo_cfg['gamma']
    all_results = {}

    for name_mdp in mdps_to_run:

        # --- Check if MDP exists ---
        if not (mdp_dir / f"{name_mdp}_train.h5").exists():
            print(f"\n  {name_mdp}: SKIPPED (not found)")
            continue

        print(f"\n{'='*60}")
        print(f"  MDP: {name_mdp.upper()}")
        print(f"{'='*60}")

        # --- Load datasets & config ---
        datasets = {
            'train': load_mdp(db_paths, name_mdp, 'train'),
            'val': load_mdp(db_paths, name_mdp, 'val'),
            'test': load_mdp(db_paths, name_mdp, 'test'),
        }
        mdp_config = joblib.load(mdp_dir / f"{name_mdp}_config.joblib")
        print(f"  {mdp_config['n_states']} features, {len(datasets['train'].episodes)} episodes")

        result = {'n_features': mdp_config['n_states']}

        # --- Train CQL ---
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

        # --- Train BC (baseline) ---
        result['bc_model'] = None
        if config['ablation']['bc_baseline']['enabled']:
            print(f"\n  Training BC (baseline)...")
            d3rlpy.seed(seed)
            bc_hp = config['ablation']['bc_baseline']['hyperparameters']

            bc, _, _ = train_bc(
                train_ds=datasets['train'],
                val_ds=datasets['val'],
                learning_rate=bc_hp['learning_rate'],
                batch_size=bc_hp['batch_size'],
                beta=bc_hp['beta'],
                hidden_units=bc_hp['hidden_units'],
                n_steps=bc_hp['n_steps'],
                n_steps_per_epoch=bc_hp['n_steps'],
                device=device,
                name=name_mdp)

            result['bc_model'] = bc
            bc.save(str(output_dir / f"{name_mdp}_bc.d3"))

        # --- Monte Carlo Returns ---
        mc_mean, mc_std = compute_mc_return(episodes=datasets['val'].episodes, gamma=gamma)
        result['mc_mean'] = mc_mean
        result['mc_std'] = mc_std
        print(f"\n  MC Return: {mc_mean:.4f} +/- {mc_std:.4f}")

        # --- FQE ---
        result['fqe_cql'] = {}
        result['fqe_bc'] = {}

        if config['ablation']['fqe']['enabled']:
            fqe_cfg = FQEConfig(
                learning_rate=config['ablation']['fqe']['learning_rate'],
                gamma=gamma)

            # FQE for CQL
            fqe_isv = function_fqe(
                algo=cql,
                dataset_train=datasets['train'],
                dataset_val=datasets['val'],
                fqe_config=fqe_cfg,
                n_steps=config['ablation']['fqe']['n_steps'],
                n_epochs=config['ablation']['fqe']['n_epochs'],
                seed=seed,
                device=device)
            result['fqe_cql'] = {'fqe_isv': fqe_isv}
            print(f"  FQE (CQL): {fqe_isv:.4f}")

            if config['ablation']['fqe']['bootstrap']['enabled']:
                m_isv, ci_lo, ci_hi = uncertainty_fqe(
                    algo=cql,
                    dataset_train=datasets['train'],
                    dataset_val=datasets['val'],
                    fqe_config=fqe_cfg,
                    n_bootstrap=config['ablation']['fqe']['bootstrap']['n_bootstrap'],
                    n_steps=config['ablation']['fqe']['bootstrap']['n_steps'],
                    device=device,
                    seed=seed,
                    CI=config['ablation']['fqe']['bootstrap']['confidence_level'])
                result['fqe_cql'].update({'mean_isv': m_isv, 'ci_low': ci_lo, 'ci_high': ci_hi})
                print(f"  FQE (CQL) bootstrap: {m_isv:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

            # FQE for BC
            if result['bc_model'] is not None:
                fqe_isv_bc = function_fqe(
                    algo=result['bc_model'],
                    dataset_train=datasets['train'],
                    dataset_val=datasets['val'],
                    fqe_config=fqe_cfg,
                    n_steps=config['ablation']['fqe']['n_steps'],
                    n_epochs=config['ablation']['fqe']['n_epochs'],
                    seed=seed,
                    device=device)
                result['fqe_bc'] = {'fqe_isv': fqe_isv_bc}
                print(f"  FQE (BC): {fqe_isv_bc:.4f}")

                if config['ablation']['fqe']['bootstrap']['enabled']:
                    m_isv, ci_lo, ci_hi = uncertainty_fqe(
                        algo=result['bc_model'],
                        dataset_train=datasets['train'],
                        dataset_val=datasets['val'],
                        fqe_config=fqe_cfg,
                        n_bootstrap=config['ablation']['fqe']['bootstrap']['n_bootstrap'],
                        n_steps=config['ablation']['fqe']['bootstrap']['n_steps'],
                        device=device,
                        seed=seed,
                        CI=config['ablation']['fqe']['bootstrap']['confidence_level'])
                    result['fqe_bc'].update({'mean_isv': m_isv, 'ci_low': ci_lo, 'ci_high': ci_hi})
                    print(f"  FQE (BC) bootstrap: {m_isv:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

        all_results[name_mdp] = result

    # =========================================================================
    # PLOTS & SUMMARY
    # =========================================================================
    mdp_names = list(all_results.keys())

    print("\n--- Creating plots ---")
    plot_training_curves(all_results, mdp_names, output_dir)

    fqe_cql = {n: r['fqe_cql'] for n, r in all_results.items() if r['fqe_cql']}
    fqe_bc = {n: r['fqe_bc'] for n, r in all_results.items() if r['fqe_bc']}
    mc = {n: {'mean': r['mc_mean'], 'std': r['mc_std']} for n, r in all_results.items()}

    if fqe_cql and fqe_bc:
        plot_fqe_comparison(fqe_cql, fqe_bc, mc, mdp_names, output_dir)

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

    # Add metadata and save
    summary_df = add_metadata_to_df(summary_df, metadata)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    # Save run configuration JSON
    training_config = {
        'algorithm': 'cql',
        **config['ablation']['algorithm']['hyperparameters']
    }
    fqe_config = {
        'fqe_n_steps': config['ablation']['fqe']['n_steps'] if config['ablation']['fqe']['enabled'] else None,
        'fqe_learning_rate': config['ablation']['fqe']['learning_rate'] if config['ablation']['fqe']['enabled'] else None,
    }
    save_run_config(output_dir, metadata, training_config=training_config, eval_config=fqe_config)

    run_id = metadata['run_id']
    print(f"\nResults saved to: {output_dir}")
    print(f"Run config saved to: run_{run_id}_config.json")

if __name__ == "__main__":
    main()
