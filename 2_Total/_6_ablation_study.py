"""
Ablation Study - Compare MDP configurations with CQL and BC.

Settings from config.yaml ablation section.
"""

import numpy as np
import pandas as pd
import joblib
import torch
import d3rlpy
from d3rlpy.algos import DiscreteCQLConfig, DiscreteBCConfig
from d3rlpy.ope import FQEConfig
from d3rlpy.metrics import TDErrorEvaluator, InitialStateValueEstimationEvaluator, DiscreteActionMatchEvaluator
from d3rlpy.models.encoders import VectorEncoderFactory

from utils import load_mdp
from rl_training import compute_mc_return, bootstrap_fqe, train_cql, train_bc, function_fqe
from rl_plotting import plot_training_curves, plot_fqe_comparison

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

    # INPUT
    mdp_dir = db_paths['mdp_dir']

    # CONFIGURATION
    db_name = db_paths['name']
    ablation_cfg = config['ablation']
    seed = config['processing']['random_state']
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*60}\nABLATION STUDY: {db_name}\n{'='*60}")
    print(f"Device: {device}")

    # =========================================================================
    # 1. LOAD DATASETS
    # =========================================================================
    print("\n--- Loading datasets ---")
    mdps_to_run = ablation_cfg['mdps_to_run'] # Which mdps to run

    # Initialize datasets and MDP_configurations
    DATASETS, MDP_CONFIGS = {}, {}

    # Loop over the mdp's which should be evaluated
    for name in mdps_to_run:
        # Check
        if not (mdp_dir / f"{name}_train.h5").exists():
            print(f"  {name}: SKIPPED (not found)")
            continue

        # Load different dataset
        DATASETS[name] = {
            'train': load_mdp(db_paths, name, 'train'),
            'val': load_mdp(db_paths, name, 'val'),
            'test': load_mdp(db_paths, name, 'test'),
        }

        # Load the configuration
        MDP_CONFIGS[name] = joblib.load(mdp_dir / f"{name}_config.joblib")

        # End check 
        print(f"  {name}: {MDP_CONFIGS[name]['n_states']} features, {len(DATASETS[name]['train'].episodes)} episodes")

    # =========================================================================
    # 2. TRAIN CQL
    # =========================================================================
    RESULTS, CQL_MODELS = {}, {}

    print(f"\n--- Training CQL ---")

    # Loop over MDPs to run, and find the right database (train/val/test)
    for name_mdp, datasets in DATASETS.items():

        d3rlpy.seed(seed)

        print(f"\n{name_mdp.upper()}:")

        cql, metrics = train_cql(
                train_ds = datasets['train'], 
                val_ds = datasets['val'], 
                alpha = config['ablation']['algorithm']['hyperparameters']['alpha'], 
                learning_rate = config['ablation']['algorithm']['hyperparameters']['learning_rate'], 
                batch_size  = config['ablation']['algorithm']['hyperparameters']['batch_size'], 
                gamma  = config['ablation']['algorithm']['hyperparameters']['gamma'], 
                n_critics  = config['ablation']['algorithm']['hyperparameters']['n_critics'], 
                hidden_units  = config['ablation']['algorithm']['hyperparameters']['hidden_units'], 
                n_steps  = config['ablation']['algorithm']['hyperparameters']['n_steps'], 
                n_steps_per_epoch  = config['ablation']['algorithm']['hyperparameters']['n_steps_per_epoch'], 
                device = device,
                save_interval  = 10, # At which epochs to save the algorithm
                name = name_mdp)

        CQL_MODELS[name_mdp] = cql
        RESULTS[name_mdp] = {'metrics': metrics}
        metrics.to_csv(output_dir / f"{name_mdp}_metrics.csv", index=False)
        cql.save(str(output_dir / f"{name_mdp}_cql.d3"))

    # =========================================================================
    # 3. TRAIN BC (baseline)
    # =========================================================================
    
    # Initialize the behavior cloning models (as baseline compare)
    BC_MODELS = {}

    # Only run if enabled
    if config['ablation']['bc_baseline']['enabled']:
        print(f"\n--- Training BC (baseline) ---")
        for name, datasets in DATASETS.items():

            d3rlpy.seed(seed)

            print(f"\n{name.upper()}:")

            bc, metrics =    train_bc(train_ds = datasets['train'],
                                       val_ds = datasets['val'],
                                       learning_rate = config['ablation']['bc_baseline']['hyperparameters']['learning_rate'], 
                                       batch_size = config['ablation']['bc_baseline']['hyperparameters']['batch_size'], 
                                       beta = config['ablation']['bc_baseline']['hyperparameters']['beta'], 
                                       hidden_units = config['ablation']['bc_baseline']['hyperparameters']['hidden_units'], 
                                       n_steps = config['ablation']['bc_baseline']['hyperparameters']['n_steps'], 
                                       n_steps_per_epoch = config['ablation']['bc_baseline']['hyperparameters']['n_steps'], 
                                       device = device , 
                                       name = name)

            BC_MODELS[name] = bc
            bc.save(str(output_dir / f"{name}_bc.d3"))

    # =========================================================================
    # 4. MONTE CARLO RETURNS
    # =========================================================================
    print(f"\n--- Monte Carlo Returns ---")
    MC = {}
    for name, datasets in DATASETS.items():

        mean, std = compute_mc_return(
            episodes = datasets['val'].episodes,
            gamma = config['ablation']['algorithm']['hyperparameters']['gamma']
        )
        MC[name] = {'mean': mean, 'std': std}
        print(f"  {name}: {mean:.4f} +/- {std:.4f}")

    # =========================================================================
    # 5. FQE WITH BOOTSTRAP
    # =========================================================================

    FQE_CQL, FQE_BC = {}, {}

    if config['ablation']['fqe']['enabled']:
        print("Running FQE")

        # Set up configuration for FQE
        fqe_config = FQEConfig(
            learning_rate=config['ablation']['fqe']['learning_rate'],
            gamma=config['ablation']['algorithm']['hyperparameters']['gamma'])
        
        # FOR CONSERVATIVE Q-LEARNING
        # Loop over different CQL-models, just normal FQE
        for name in CQL_MODELS:
            m_isv = function_fqe(
                algo = CQL_MODELS[name],
                dataset = DATASETS[name]['val'],
                fqe_config = fqe_config,
                n_steps = config['ablation']['fqe']['n_steps'],
                seed = seed,
                device = device)
            FQE_CQL[name] = {'mean': m_isv}
            print(f"  {name}: {m_isv:.4f}")
        
        # Only if you want to bootstrap
        if config['ablation']['fqe']['bootstrap']['enabled']:
            n_bootstrap = config['ablation']['fqe']['bootstrap']['n_bootstrap']
            print(f"\n--- FQE (CQL) with {n_bootstrap}x bootstrap ---")

            for name in CQL_MODELS:
                _, ci_lo, ci_hi = bootstrap_fqe(
                    algo=CQL_MODELS[name],
                    dataset=DATASETS[name]['val'],
                    fqe_config=fqe_config,
                    n_bootstrap=config['ablation']['fqe']['bootstrap']['n_bootstrap'],
                    n_steps=config['ablation']['fqe']['bootstrap']['n_steps'],
                    device=device,
                    seed=seed,
                    CI=config['ablation']['fqe']['bootstrap']['confidence_level']
                )

                FQE_CQL[name].update({'ci_low': ci_lo, 'ci_high': ci_hi})
                print(f"  {name}: {m_isv:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

        # FOR BEHAVIOR CLONING (BASELINE COMPARE)
        if BC_MODELS:
            print("FQE for behavior cloning (baseline)")

            # Loop over different BC-models, just normal FQE
            for name in BC_MODELS:
                m_isv = function_fqe(
                    algo = BC_MODELS[name],
                    dataset = DATASETS[name]['val'],
                    fqe_config = fqe_config,
                    n_steps = config['ablation']['fqe']['n_steps'],
                    seed = seed,
                    device = device)
                FQE_BC[name] = {'mean': m_isv}
                print(f"  {name}: {m_isv:.4f}")
            
            # Only if you want to bootstrap
            if config['ablation']['fqe']['bootstrap']['enabled']:
                n_bootstrap = config['ablation']['fqe']['bootstrap']['n_bootstrap']
                print(f"\n--- FQE (BC) with {n_bootstrap}x bootstrap ---")

                for name in BC_MODELS:
                    _, ci_lo, ci_hi = bootstrap_fqe(
                        algo=BC_MODELS[name],
                        dataset=DATASETS[name]['val'],
                        fqe_config=fqe_config,
                        n_bootstrap=config['ablation']['fqe']['bootstrap']['n_bootstrap'],
                        n_steps=config['ablation']['fqe']['bootstrap']['n_steps'],
                        device=device,
                        seed=seed,
                        CI=config['ablation']['fqe']['bootstrap']['confidence_level']
                    )

                    FQE_BC[name].update({'ci_low': ci_lo, 'ci_high': ci_hi})
                    print(f"  {name}: {m_isv:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
    # =========================================================================
    # 6. PLOTS & SUMMARY
    # =========================================================================
    print("\n--- Creating plots ---")
    plot_training_curves(RESULTS, list(DATASETS.keys()), output_dir)
    if FQE_CQL and FQE_BC:
        plot_fqe_comparison(FQE_CQL, FQE_BC, MC, list(DATASETS.keys()), output_dir)

    summary = [{
        'MDP': name,
        'n_features': MDP_CONFIGS[name]['n_states'],
        'TD_val_final': RESULTS[name]['metrics']['td_val'].iloc[-1],
        'ISV_peak': RESULTS[name]['metrics']['isv_val'].max(),
        'Action_match': RESULTS[name]['metrics']['action_match'].iloc[-1],
        'MC_return': MC[name]['mean'],
        'FQE_CQL': FQE_CQL.get(name, {}).get('mean', np.nan),
        'FQE_BC': FQE_BC.get(name, {}).get('mean', np.nan),
    } for name in DATASETS]

    summary_df = pd.DataFrame(summary)
    print(f"\n{summary_df.to_string(index=False)}")
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()
