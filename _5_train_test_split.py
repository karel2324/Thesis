"""
Train/Test Split - Split and save MDPDatasets
"""

import numpy as np
from sklearn.model_selection import train_test_split
from utils import load_mdp, episodes_to_mdp

def main():
    """Main entry point for train/test split."""
    from utils import load_config, get_data_paths

    config = load_config()
    all_paths = get_data_paths(config)

    # Check whether to run AUMCdb
    if config['databases']['aumc'] == True:
        if all_paths['aumc']['mdp_dir'].exists():
            run_split_for_db(all_paths['aumc'], config)
        else:
            print(f"Skipping {all_paths['aumc']['name']}: run _4_mdp_preparation.py first")
    
    # Check whether to run MIMIC
    if config['databases']['mimic'] == True:
        if all_paths['mimic']['mdp_dir'].exists():
            run_split_for_db(all_paths['mimic'], config)
        else:
            print(f"Skipping {all_paths['mimic']['name']}: run _4_mdp_preparation.py first")


def run_split_for_db(db_paths: dict, config: dict):
    """Split MDPDatasets into train/val/test and save."""

    # OUTPUT
    output_dir = db_paths['mdp_dir']
    output_dir.mkdir(parents=True, exist_ok=True)  # Defensive mkdir

    # INPUT
    # -- See code below

    # CONFIGURATION
    ## General configurations 
    db_name = db_paths['name'] # Name of database processed
    mdp_cfg = config['mdp']

    print(f"\n{'='*50}")
    print(f"TRAIN/TEST/VAL SPLIT: {db_name}")
    print(f"{'='*50}")

    ## Ratios and seeds
    train_ratio = config['processing']['train_ratio']
    val_ratio = config['processing']['val_ratio']
    test_ratio = config['processing']['test_ratio']
    
    # CREATE SPLITS
    for mdp_name in mdp_cfg['configurations'].keys(): # Find all different MDPs
        # Retrieve specific input
        full_path = output_dir / f"{mdp_name}_full.h5" 
        if not full_path.exists():
            print(f"  {mdp_name}: SKIPPED (not found)")
            continue
        
        # Load dataset
        dataset = load_mdp(db_paths, mdp_name, 'full')
        n_episodes = len(dataset.episodes)

        # Split indices
        seed = 42 # Same seed for every MDP --> to be easier to compare
        indices = np.arange(n_episodes)
        train_idx, temp_idx = train_test_split(indices, train_size=train_ratio, random_state=seed) # First split train - test/val
        val_idx, test_idx = train_test_split(temp_idx, train_size=(val_ratio / (test_ratio + val_ratio)), random_state=seed) # Then split val - test

        # Check for overlapping indices
        if set(train_idx).intersection(val_idx) or set(train_idx).intersection(test_idx) or set(val_idx).intersection(test_idx):
            print(f"  {mdp_name}: SKIPPED (overlapping indices)")
            continue

        # Save different MDPs to same directory
        episodes_to_mdp([dataset.episodes[i] for i in train_idx]).dump(str(output_dir / f"{mdp_name}_train.h5"))
        episodes_to_mdp([dataset.episodes[i] for i in val_idx]).dump(str(output_dir / f"{mdp_name}_val.h5"))
        episodes_to_mdp([dataset.episodes[i] for i in test_idx]).dump(str(output_dir / f"{mdp_name}_test.h5"))

        # Print summary
        print(f"  {mdp_name.upper()}: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
