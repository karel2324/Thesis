"""
Behavior Cloning (BC) Pipeline

Trains Behavior Cloning models on selected MDPs.
BC learns to imitate the clinician's behavior policy directly.

Settings controlled via config.yaml behavior_cloning section.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import torch
import d3rlpy

from utils import load_config, get_data_paths, load_mdp
from rl_utils import train_bc, get_action_frequency_per_episode


def main():
    """Main entry point for behavior cloning."""
    from utils import load_config, get_data_paths

    config = load_config()
    all_paths = get_data_paths(config)

    # Check whether to run AUMCdb
    if config['behavior_cloning']['databases']['aumc'] == True:
        print("AUMCdb is enabled for behavior cloning")
        if all_paths['aumc']['mdp_dir'].exists():
            run_bc_for_db(all_paths['aumc'], config)
        else:
            print(f"Skipping AUMCdb: MDP directory not found")
    
    # Check whether to run MIMIC
    if config['behavior_cloning']['databases']['mimic'] == True:     
        print("MIMIC is enabled for behavior cloning")   
        if all_paths['mimic']['mdp_dir'].exists():
            run_bc_for_db(all_paths['mimic'], config)
        else:
            print(f"Skipping MIMIC: MDP directory not found")


############################################################################################################
# MAIN FUNCTION

def run_bc_for_db(db_paths: dict, config: dict):
    """Run Behavior Cloning for one database."""

    # 1. RETRIEVE ALL CONFIGURATIONS
    db_name = db_paths['name']
    db_key = 'aumc' if 'aumc' in db_name.lower() else 'mimic'
    bc_cfg = config.get('behavior_cloning', {})

    # Check database selection
    db_selection = bc_cfg.get('database', {})
    if not db_selection.get(db_key, True):
        print(f"Skipping {db_name}: disabled in behavior_cloning.database config")
        return

    print(f"\n{'='*60}")
    print(f"BEHAVIOR CLONING: {db_name}")
    print(f"{'='*60}")

    # 2. INPUT & OUTPUT
    mdp_dir = db_paths['mdp_dir']
    output_dir = db_paths['reward_dir'] / "BC_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = config.get('processing', {}).get('random_state', 42)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 3. GET SETTINGS FROM CONFIG
    mdps_to_train = bc_cfg.get('mdps', ['mdp1'])
    train_split = bc_cfg.get('train_split', 'train')
    hp = bc_cfg.get('hyperparameters', {})
    output_cfg = bc_cfg.get('output', {})

    ## Hyperparameters
    learning_rate = hp.get('learning_rate', 1e-4)
    batch_size = hp.get('batch_size', 256)
    hidden_units = hp.get('hidden_units', [256, 256])
    beta = hp.get('beta', 0.5)
    n_steps = hp.get('n_steps', 10000)
    n_steps_per_epoch = hp.get('n_steps_per_epoch', 2000)

    print(f"\nSettings:")
    print(f"  MDPs: {mdps_to_train}")
    print(f"  Train split: {train_split}")
    print(f"\nHyperparameters:")
    print(f"  learning_rate: {learning_rate}")
    print(f"  batch_size: {batch_size}")
    print(f"  hidden_units: {hidden_units}")
    print(f"  beta: {beta}")
    print(f"  n_steps: {n_steps}")

    ############################################################################################################
    # 4. TRAIN BC FOR EACH MDP

    results = []
    models = {}

    for mdp_name in mdps_to_train:
        print(f"\n{'='*50}")
        print(f"Training BC on {mdp_name.upper()} ({train_split})")
        print(f"{'='*50}")

        # Check if data exists
        train_path = mdp_dir / f"{mdp_name}_{train_split}.h5"
        if not train_path.exists():
            print(f"  Skipping {mdp_name}: data not found at {train_path}")
            continue

        # Load data
        train_ds = load_mdp(db_paths, mdp_name, train_split)
        val_ds = load_mdp(db_paths, mdp_name, 'val') if (mdp_dir / f"{mdp_name}_val.h5").exists() else train_ds
        mdp_config = joblib.load(mdp_dir / f"{mdp_name}_config.joblib")

        print(f"  Training data: {len(train_ds.episodes)} episodes")
        print(f"  Features: {mdp_config['n_states']}")

        d3rlpy.seed(seed)

        # Train BC using rl_training.train_bc
        bc, metrics_df = train_bc(
            train_ds=train_ds,
            val_ds=val_ds,
            learning_rate=learning_rate,
            batch_size=batch_size,
            beta=beta,
            hidden_units=hidden_units,
            n_steps=n_steps,
            n_steps_per_epoch=n_steps_per_epoch,
            device=device,
            name=mdp_name
        )

        # Final metrics
        final = metrics_df.iloc[-1].to_dict() if len(metrics_df) > 0 else {}

        # Action distribution on validation
        val_obs = np.concatenate([ep.observations for ep in val_ds.episodes])
        val_actions = np.concatenate([ep.actions for ep in val_ds.episodes])
        bc_pred = bc.predict(val_obs)
        bc_rrt_rate = (bc_pred == 1).mean()
        data_rrt_rate = (val_actions == 1).mean()

        # Action match
        action_match = (bc_pred == val_actions).mean()

        # Episode-level RRT frequency
        episode_rrt_freq = get_action_frequency_per_episode(bc, val_ds)

        print(f"\n  Final metrics:")
        print(f"    Loss: {final.get('loss', 'N/A'):.4f}" if 'loss' in final else "    Loss: N/A")
        print(f"    Action Match (val): {action_match:.1%}")
        print(f"    RRT Rate: BC={bc_rrt_rate:.1%}, Data={data_rrt_rate:.1%}")
        print(f"    Episode RRT Frequency: {episode_rrt_freq:.1%}")

        # Store results
        result = {
            'mdp': mdp_name,
            'train_split': train_split,
            'n_features': mdp_config['n_states'],
            'loss': final.get('loss', np.nan),
            'action_match': action_match,
            'rrt_rate': bc_rrt_rate,
            'data_rrt_rate': data_rrt_rate,
            'episode_rrt_freq': episode_rrt_freq,
        }

        results.append(result)
        models[mdp_name] = bc

        # Save model
        if output_cfg.get('save_models', True):
            bc.save(str(output_dir / f"bc_{mdp_name}_model.d3"))
            print(f"  Saved: bc_{mdp_name}_model.d3")

    ############################################################################################################
    # 5. CREATE SUMMARY

    if results:
        results_df = pd.DataFrame(results)

        if output_cfg.get('save_metrics', True):
            results_df.to_csv(output_dir / "bc_results.csv", index=False)

        print(f"\n{'='*60}")
        print("BC SUMMARY")
        print(f"{'='*60}")
        print(results_df.to_string(index=False))

        if output_cfg.get('save_plots', True) and len(results) > 1:
            create_bc_plots(results_df, output_dir)

    print(f"\n{'='*60}")
    print(f"BC complete! Results saved to: {output_dir}")
    print(f"{'='*60}")


############################################################################################################
# PLOTTING FUNCTION

def create_bc_plots(results_df: pd.DataFrame, output_dir: Path):
    """Create BC comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    mdps = results_df['mdp'].tolist()
    x = np.arange(len(mdps))

    # Plot 1: Action Match
    ax1 = axes[0]
    values = results_df['action_match'].values * 100
    ax1.bar(x, values, color='steelblue', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in mdps])
    ax1.set_ylabel('Action Match (%)')
    ax1.set_title('BC Action Match with Clinicians')
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(values):
        ax1.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)

    # Plot 2: RRT Rate comparison
    ax2 = axes[1]
    width = 0.35
    bc_rates = results_df['rrt_rate'].values * 100
    data_rates = results_df['data_rrt_rate'].values * 100
    ax2.bar(x - width/2, bc_rates, width, label='BC Policy', color='steelblue', alpha=0.8)
    ax2.bar(x + width/2, data_rates, width, label='Clinician', color='darkorange', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.upper() for m in mdps])
    ax2.set_ylabel('RRT Rate (%)')
    ax2.set_title('RRT Initiation Rate: BC vs Clinicians')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'bc_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: bc_comparison.png")


############################################################################################################
# MAIN

if __name__ == "__main__":
    main()
