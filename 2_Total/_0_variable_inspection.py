"""
Variable Inspection Pipeline

Generates inspection reports for both databases:
1. Raw inspection: Missing percentages, distributions before imputation
2. Post-imputation inspection: Distributions after imputation

Outputs HTML reports and CSV summaries.
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    """Main entry point for variable inspection."""
    from utils import load_config, get_data_paths

    config = load_config()
    all_paths = get_data_paths(config)

    for db_key, db_paths in all_paths.items():
        run_inspection_for_db(db_paths, config)

def inspect_raw(df: pd.DataFrame, db_name: str, output_dir: Path) -> pd.DataFrame:
    """Inspect raw data before imputation."""
    print(f"\n  Raw inspection for {db_name}...")

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Calculate statistics
    stats = []
    for col in numeric_cols:
        n_total = len(df)
        n_missing = df[col].isnull().sum()
        n_valid = n_total - n_missing

        stats.append({
            'column': col,
            'n_total': n_total,
            'n_valid': n_valid,
            'n_missing': n_missing,
            'missing_pct': round(n_missing / n_total * 100, 2),
            'mean': df[col].mean() if n_valid > 0 else np.nan,
            'std': df[col].std() if n_valid > 0 else np.nan,
            'min': df[col].min() if n_valid > 0 else np.nan,
            'p25': df[col].quantile(0.25) if n_valid > 0 else np.nan,
            'median': df[col].median() if n_valid > 0 else np.nan,
            'p75': df[col].quantile(0.75) if n_valid > 0 else np.nan,
            'max': df[col].max() if n_valid > 0 else np.nan,
        })

    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.sort_values('missing_pct', ascending=False)

    # Save CSV
    csv_path = output_dir / f"{db_name.lower()}_raw_inspection.csv"
    stats_df.to_csv(csv_path, index=False)
    print(f"    Saved: {csv_path.name}")

    # Create missing heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Top 30 columns by missing %
    top_missing = stats_df.head(30)

    ax1 = axes[0]
    bars = ax1.barh(range(len(top_missing)), top_missing['missing_pct'], color='coral')
    ax1.set_yticks(range(len(top_missing)))
    ax1.set_yticklabels(top_missing['column'], fontsize=8)
    ax1.set_xlabel('Missing %')
    ax1.set_title(f'{db_name} - Top 30 Missing Variables (Raw)')
    ax1.invert_yaxis()
    ax1.axvline(x=50, color='red', linestyle='--', alpha=0.5)

    # Add percentage labels
    for i, (pct, bar) in enumerate(zip(top_missing['missing_pct'], bars)):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=7)

    # Summary stats
    ax2 = axes[1]
    ax2.axis('off')
    summary_text = f"""
    {db_name} Raw Data Summary
    ========================

    Total rows: {len(df):,}
    Total columns: {len(df.columns)}
    Numeric columns: {len(numeric_cols)}

    Missing Statistics:
    - Columns with >50% missing: {(stats_df['missing_pct'] > 50).sum()}
    - Columns with >25% missing: {(stats_df['missing_pct'] > 25).sum()}
    - Columns with >10% missing: {(stats_df['missing_pct'] > 10).sum()}
    - Columns with 0% missing: {(stats_df['missing_pct'] == 0).sum()}

    Top 10 Most Missing:
    """
    for _, row in stats_df.head(10).iterrows():
        summary_text += f"\n    {row['column']}: {row['missing_pct']:.1f}%"

    ax2.text(0.1, 0.95, summary_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    fig_path = output_dir / f"{db_name.lower()}_raw_missing.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fig_path.name}")

    return stats_df


def inspect_imputed(df: pd.DataFrame, db_name: str, output_dir: Path,
                    raw_stats: pd.DataFrame = None) -> pd.DataFrame:
    """Inspect data after imputation."""
    print(f"\n  Post-imputation inspection for {db_name}...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Calculate statistics
    stats = []
    for col in numeric_cols:
        n_total = len(df)
        n_missing = df[col].isnull().sum()
        n_valid = n_total - n_missing

        stats.append({
            'column': col,
            'n_total': n_total,
            'n_valid': n_valid,
            'n_missing': n_missing,
            'missing_pct': round(n_missing / n_total * 100, 2),
            'mean': df[col].mean() if n_valid > 0 else np.nan,
            'std': df[col].std() if n_valid > 0 else np.nan,
            'min': df[col].min() if n_valid > 0 else np.nan,
            'p25': df[col].quantile(0.25) if n_valid > 0 else np.nan,
            'median': df[col].median() if n_valid > 0 else np.nan,
            'p75': df[col].quantile(0.75) if n_valid > 0 else np.nan,
            'max': df[col].max() if n_valid > 0 else np.nan,
        })

    stats_df = pd.DataFrame(stats)

    # Save CSV
    csv_path = output_dir / f"{db_name.lower()}_imputed_inspection.csv"
    stats_df.to_csv(csv_path, index=False)
    print(f"    Saved: {csv_path.name}")

    # Create comparison plot if raw stats available
    if raw_stats is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))

        # Merge raw and imputed stats
        comparison = raw_stats[['column', 'missing_pct']].merge(
            stats_df[['column', 'missing_pct']],
            on='column',
            suffixes=('_raw', '_imputed')
        )
        comparison['imputed'] = comparison['missing_pct_raw'] - comparison['missing_pct_imputed']
        comparison = comparison.sort_values('missing_pct_raw', ascending=False).head(30)

        # Before/After comparison
        ax1 = axes[0]
        x = range(len(comparison))
        width = 0.35
        ax1.barh([i - width/2 for i in x], comparison['missing_pct_raw'], width,
                label='Raw', color='coral', alpha=0.8)
        ax1.barh([i + width/2 for i in x], comparison['missing_pct_imputed'], width,
                label='Imputed', color='steelblue', alpha=0.8)
        ax1.set_yticks(x)
        ax1.set_yticklabels(comparison['column'], fontsize=8)
        ax1.set_xlabel('Missing %')
        ax1.set_title(f'{db_name} - Missing % Before vs After Imputation')
        ax1.legend()
        ax1.invert_yaxis()

        # Summary
        ax2 = axes[1]
        ax2.axis('off')

        total_missing_raw = raw_stats['n_missing'].sum()
        total_missing_imp = stats_df['n_missing'].sum()

        summary_text = f"""
    {db_name} Imputation Summary
    ============================

    Total rows: {len(df):,}

    Missing Values:
    - Before imputation: {total_missing_raw:,}
    - After imputation: {total_missing_imp:,}
    - Imputed: {total_missing_raw - total_missing_imp:,}

    Columns with remaining missing:
    - >0% missing: {(stats_df['missing_pct'] > 0).sum()}

    Imputation Methods Applied:
    - Clinical: fio2, GCS, SOFA, UO, creat/urea ratios
    - Age columns: 48h max lookback
    - Gender: Logistic regression
    - Remaining: MICE (BayesianRidge)
        """

        ax2.text(0.1, 0.95, summary_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        fig_path = output_dir / f"{db_name.lower()}_imputation_comparison.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {fig_path.name}")

    return stats_df


def run_inspection_for_db(db_paths: dict, config: dict, inspect_raw_data: bool = True,
                          inspect_imputed_data: bool = True):
    """Run variable inspection for one database."""
    db_name = db_paths['name']

    print(f"\n{'='*50}")
    print(f"VARIABLE INSPECTION: {db_name}")
    print(f"{'='*50}")

    # Create output directory
    output_dir = db_paths['pre_reward_dir'] / "Inspection"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_stats = None

    # Raw inspection
    if inspect_raw_data and db_paths['raw_path'].exists():
        print(f"\nLoading raw data: {db_paths['raw_path']}")
        table = pq.read_table(db_paths['raw_path'])
        df_raw = table.to_pandas(ignore_metadata=True)
        print(f"Shape: {df_raw.shape}")

        raw_stats = inspect_raw(df_raw, db_name, output_dir)
        del df_raw
    elif inspect_raw_data:
        print(f"  Raw data not found: {db_paths['raw_path']}")

    # Imputed inspection
    if inspect_imputed_data and db_paths['imputed_path'].exists():
        print(f"\nLoading imputed data: {db_paths['imputed_path']}")
        df_imp = pd.read_parquet(db_paths['imputed_path'])
        print(f"Shape: {df_imp.shape}")

        inspect_imputed(df_imp, db_name, output_dir, raw_stats)
        del df_imp
    elif inspect_imputed_data:
        print(f"  Imputed data not found: {db_paths['imputed_path']}")

    print(f"\nInspection reports saved to: {output_dir}")

if __name__ == "__main__":
    main()
