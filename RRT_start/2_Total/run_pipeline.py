"""
Run Complete RRT Pipeline for Renal Replacement Therapy Decision Support
Settings controlled via config.yaml
"""
import sys
from utils import load_config

# Import main() from each pipeline step
from _0_variable_inspection import main as variable_inspection
from _1_imputation import main as imputation
from _2_reward_calculation import main as reward_calculation
from _3_scaling import main as scaling
from _4_mdp_preparation import main as mdp_preparation
from _5_train_test_split import main as train_test_split
from _6_ablation_study import main as ablation_study
from _7_hpo import main as hpo_grid_search
from _7_hpo_optuna import main as hpo_optuna
from _8_behavior_cloning import main as behavior_cloning
from _9_evaluation import main as evaluation
from _10_external_validation import main as external_validation


def main():
    print("Running the Reinforcement Learning Pipeline for RRT Decision Support!\n")

    config = load_config()

    # =========================================================================
    # CONFIGURATION OVERVIEW
    # =========================================================================
    print("=" * 60)
    print("CONFIGURATION OVERVIEW")
    print("=" * 60)

    # Databases
    print("\nDATABASES:")
    db_status_aumc = "[x]" if config['databases']['aumc'] else "[ ]"
    db_status_mimic = "[x]" if config['databases']['mimic'] else "[ ]"
    print(f"  {db_status_aumc} AUMCdb")
    print(f"  {db_status_mimic} MIMIC-IV")

    # Reward settings
    print("\nREWARD SETTINGS:")
    reward = config['reward']
    print(f"  Horizon days:       {reward['horizon_days']}")
    print(f"  Discount factor:    {reward['discount']}")
    print(f"  Mortality penalty:  {reward['mortality_penalty']}")
    print(f"  Max SOFA:           {reward['max_sofa']}")
    print(f"  Intermediate scale: {reward['intermediate_reward_scale']}")

    # Steps to run
    print("\nSTEPS TO RUN:")
    steps = config['steps']
    step_names = [
        ('inspection', 'Step 0: Variable Inspection'),
        ('imputation', 'Step 1: Imputation'),
        ('reward_calculation', 'Step 2: Reward Calculation'),
        ('scaling', 'Step 3: Scaling'),
        ('mdp_preparation', 'Step 4: MDP Preparation'),
        ('train_test_split', 'Step 5: Train/Test Split'),
        ('ablation_study', 'Step 6: Ablation Study'),
        ('hpo', 'Step 7a: HPO Grid Search'),
        ('hpo_optuna', 'Step 7b: HPO Optuna'),
        ('behavior_cloning', 'Step 8: Behavior Cloning'),
        ('evaluation', 'Step 9: Evaluation'),
        ('external_validation', 'Step 10: External Validation'),
    ]

    for key, name in step_names:
        status = "[x]" if steps.get(key, False) else "[ ]"
        print(f"  {status} {name}")

    print("\n" + "=" * 60 + "\n")

    # =========================================================================
    # DATA PREPARATION PHASE
    # =========================================================================
    print("=" * 60)
    print("DATA PREPARATION PHASE")
    print("=" * 60)

    if config['steps']['inspection']:
        print("\nStep 0: Variable Inspection...")
        print("Generating inspection reports for raw and imputed data...")
        variable_inspection()

    if config['steps']['imputation']:
        print("\nStep 1: Imputation...")
        print("Applying clinical imputations and MICE...")
        imputation()

    if config['steps']['reward_calculation']:
        print("\nStep 2: Reward Calculation...")
        print("Computing terminal and intermediate rewards...")
        reward_calculation()

    if config['steps']['scaling']:
        print("\nStep 3: Scaling...")
        print("Standardizing continuous variables...")
        scaling()

    # =========================================================================
    # MDP CONSTRUCTION PHASE
    # =========================================================================
    print("\n" + "=" * 60)
    print("MDP CONSTRUCTION PHASE")
    print("=" * 60)

    if config['steps']['mdp_preparation']:
        print("\nStep 4: MDP Preparation...")
        print("Building MDPDatasets for each configuration...")
        mdp_preparation()

    if config['steps']['train_test_split']:
        print("\nStep 5: Train/Test Split...")
        print("Splitting episodes into train/validation/test sets...")
        train_test_split()

    # =========================================================================
    # MODEL TRAINING PHASE
    # =========================================================================
    print("\n" + "=" * 60)
    print("MODEL TRAINING PHASE")
    print("=" * 60)

    if config['steps']['ablation_study']:
        print("\nStep 6: Ablation Study...")
        print("Comparing MDP configurations with CQL and BC...")
        ablation_study()

    if config['steps']['hpo']:
        print("\nStep 7a: Hyperparameter Optimization (Grid Search)...")
        print("Training CQL, DDQN, BCQ, NFQ with different hyperparameters...")
        hpo_grid_search()

    if config['steps']['hpo_optuna']:
        print("\nStep 7b: Hyperparameter Optimization (Optuna)...")
        print("Bayesian optimization with TPE sampler...")
        hpo_optuna()

    if config['steps']['behavior_cloning']:
        print("\nStep 8: Behavior Cloning...")
        print("Training BC baseline for comparison...")
        behavior_cloning()

    # =========================================================================
    # EVALUATION PHASE
    # =========================================================================
    print("\n" + "=" * 60)
    print("EVALUATION PHASE")
    print("=" * 60)

    if config['steps']['evaluation']:
        print("\nStep 9: Model Evaluation...")
        print("Evaluating best models with FQE and bootstrap CI...")
        evaluation()

    if config['steps']['external_validation']:
        print("\nStep 10: External Validation...")
        print("Testing models on external database...")
        external_validation()

    # =========================================================================
    # DONE
    # =========================================================================
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
