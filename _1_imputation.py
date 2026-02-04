"""
Imputation Pipeline (Config-Driven)

Performs for each database based on config.yaml settings:
1. Clinical imputations (impute_method: clinical_value)
2. Age column imputation (_age_h, _hours)
3. MICE for remaining missing values (impute_method: mice)
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import joblib
import gc

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

def main():
    """Main entry point for imputation."""
    from utils import load_config, get_data_paths
    
    config = load_config()
    all_paths = get_data_paths(config)

    for db_key, db_paths in all_paths.items():
        if db_paths['raw_path'].exists():
            run_imputation_for_db(db_paths, config)
        else:
            print(f"Skipping {db_paths['name']}: raw data not found")

# Clinical imputations
def apply_clinical_imputations(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply clinical imputations based on config.yaml."""
    print("\nApplying clinical imputations (from config)...")
    variables = config.get('variables', {}) # Retrieve the variable name

    for var_name, var_config in variables.items(): # Inspect whether variable is in dataframe
        if var_name not in df.columns: # If not in df, then pass over variable
            continue
        
        # Retrieve the method
        method = var_config.get('impute_method', 'none')

        # Find variables for which the imputation method is 'clinical value'
        if method == 'clinical_value':
            value = var_config.get('impute_value') # Find the value to impute
            if value is not None:
                n = df[var_name].isnull().sum() # Find number of missing variables
                if n > 0: 
                    df[var_name] = df[var_name].fillna(value) # Fill missing variables with clinical value
                    print(f"  {var_name}: {n:,} missing -> {value}") # Print outcome

        # Specially for relative increase (MIMIC problem)
        if var_name in ['creat_rel_inc_48h', 'urea_rel_inc_48h']:
            if method == 'clinical_value':
                value= var_config.get('impute_value')
                if value is not None:
                    n = (df[var_name]<1).sum()
                    if n > 0:
                        df[var_name] = df[var_name].clip(lower=1) # All values below 1 --> 1, since is relative increase to lowest value last 48 hours
                        print(f"  {var_name}: {n:,} below 1 -> {value}")
    return df


def apply_age_column_imputations(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply imputations for age columns (_age_h, _hours)."""
    # Retrieve the value to impute
    age_value = config.get('processing', {}).get('age_columns_impute_value')

    # Retrieve the columns for the time since measurements
    age_cols = [c for c in df.columns if c.endswith('_age_h') or c.endswith('_hours')]

    print(f"\nApplying age column imputations -> {age_value}:")

    # Impute value into columns
    for col in age_cols:
        n = df[col].isnull().sum()
        if n > 0:
            df[col] = df[col].fillna(age_value)
            print(f"  {col}: {n:,} missing -> {age_value}")

    return df

def get_mice_columns(df: pd.DataFrame, config: dict) -> list:
    """Identify columns for MICE based on config (impute_method: mice)."""
    variables = config.get('variables', {})

    mice_cols = []
    for var_name, var_config in variables.items():
        if var_name in df.columns:
            if var_config.get('impute_method') == 'mice':
                if df[var_name].isnull().any():
                    mice_cols.append(var_name)

    return mice_cols


def get_mice_predictors(df: pd.DataFrame, config: dict) -> tuple:
    """
    Get predictor columns for MICE based on mice_predictor: true in config.

    Only variables with mice_predictor: true AND missing values are used as predictors.
    This is because MICE works iteratively - variables predict each other.
    Variables without missing values don't need to be in the MICE loop.
    """
    variables = config.get('variables', {})

    predictors = []

    for var_name, var_config in variables.items():
        if var_name not in df.columns:
            continue

        if var_config.get('mice_predictor', False):
            has_missing = df[var_name].isnull().any()

            # Only include if variable has missing values (MICE imputes these iteratively)
            if not has_missing:
                predictors.append(var_name)

    return predictors


def apply_mice(df: pd.DataFrame, mice_cols: list, predictors: list, config: dict) -> tuple:
    # Filter to columns that still have missing values
    mice_cols = [c for c in mice_cols if c in df.columns and df[c].isnull().any()]

    if not mice_cols:
        print("\nNo remaining missing values for MICE")
        return df, None, [], [], {}

    random_state = config.get('processing', {}).get('random_state', 42)

    mice = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=10,
        random_state=random_state,
        verbose=2,
        imputation_order='ascending'
    )

    all_cols = mice_cols + [p for p in predictors if p not in mice_cols]

    print(f"\nMICE imputation:")
    print(f"  Columns to impute: {len(mice_cols)}")
    print(f"  Predictor columns: {len(predictors)}")

    # Show which columns are being imputed
    for col in mice_cols[:100]: 
        n = df[col].isnull().sum()
        print(f"    {col}: {n:,} missing")

    imputed = mice.fit_transform(df[all_cols])
    df[mice_cols] = imputed[:, :len(mice_cols)]

    print("  MICE complete!")
    return df, mice, mice_cols, predictors, {}


def run_imputation_for_db(db_paths: dict, config: dict):
    db_name = db_paths['name']
    random_state = config.get('processing', {}).get('random_state', 42)

    print(f"\n{'='*60}")
    print(f"IMPUTATION: {db_name}")

    # Load data
    input_path = db_paths['raw_path']
    output_path = db_paths['imputed_path']
    imputer_path = db_paths['imputer_path']

    print(f"Loading: {input_path}")
    table = pq.read_table(input_path)
    df = table.to_pandas(ignore_metadata=True)
    print(f"Shape: {df.shape}, Visits: {df['visit_occurrence_id'].nunique()}")

    # Convert object columns to numeric where appropriate
    exclude_cols = {'gender', 'terminal_event', 'inclusion_applied', 'weight_source_person'}
    for col in df.columns:
        if df[col].dtype == 'object' and col not in exclude_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Show initial missing value summary
    print(f"\nInitial missing values:")
    missing_summary = df.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0].sort_values(ascending=False)
    print(f"  Columns with missing: {len(missing_cols)}")
    print(f"  Total missing values: {missing_cols.sum():,}")

    # Step 1: Clinical imputations (impute_method: clinical_value)
    df = apply_clinical_imputations(df, config)

    # Step 2: Age column imputations (_age_h, _hours)
    df = apply_age_column_imputations(df, config)

    # Create helper columns for MICE
    if 'hours_since_t0' not in df.columns:
        df['hours_since_t0'] = (
            pd.to_datetime(df['grid_ts']) - pd.to_datetime(df['t0'])
        ).dt.total_seconds() / 3600

    if 'gender_encoded' not in df.columns:
        df['gender_encoded'] = df['gender'].map({'M': 1, 'F': 0})


    # Step 3: MICE imputation (impute_method: mice)
    mice_cols = get_mice_columns(df, config)
    predictors = get_mice_predictors(df, config)
    
    # Step 4: Add hours since t0 and gender_encode:
    if df['gender_encoded'].isna().any():
        mice_cols.append('gender_encoded')
    else:
        predictors.append('gender_encoded')
    
    if 'gender' in mice_cols:
        mice_cols.remove('gender')
    if 'gender' in predictors:
        predictors.remove('gender')
    
    predictors.append('hours_since_t0')
    
    df, mice_imputer, final_cols, final_preds, _ = apply_mice(
        df, mice_cols, predictors, config
    )

    # Verify no missing values remain in imputed columns
    print(f"\nVerification:")
    remaining_missing = df[final_cols].isnull().sum().sum() if final_cols else 0
    print(f"  Remaining missing in MICE cols: {remaining_missing}")

    # Save imputed data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving: {output_path}")
    df.to_parquet(output_path, index=False)
    print(f"Saved: {df.shape} ({output_path.stat().st_size / 1024**2:.1f} MB)")

    # Save imputer for later use
    imputer_data = {
        'imputer': mice_imputer,
        'impute_cols': final_cols,
        'extra_predictors': final_preds,
        'random_state': random_state,
    }
    joblib.dump(imputer_data, imputer_path)
    print(f"Imputer saved: {imputer_path}")

    # Summary
    print(f"\n{'='*60}")
    print("IMPUTATION SUMMARY")
    print(f"{'='*60}")
    variables = config.get('variables', {})

    clinical_count = sum(1 for v in variables.values() if v.get('impute_method') == 'clinical_value')
    mice_count = sum(1 for v in variables.values() if v.get('impute_method') == 'mice')
    logistic_count = sum(1 for v in variables.values() if v.get('impute_method') == 'logistic')
    predictor_count = sum(1 for v in variables.values() if v.get('mice_predictor', False))

    print(f"  Clinical imputations (config): {clinical_count}")
    print(f"  Logistic imputations (config): {logistic_count}")
    print(f"  MICE imputations (config): {mice_count}")
    print(f"  MICE predictors (config): {predictor_count}")
    print(f"  Actual MICE columns used: {len(final_cols)}")
    print(f"{'='*60}")

    gc.collect()

if __name__ == "__main__":
    main()
