#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_snr_cqi_mcs_comparison(tables_dict, save_path=None):
    """
    Plot multiple SNR-CQI/MCS lookup tables for comparison
    
    Args:
        tables_dict: Dictionary with format {'table_name': dataframe}
                    Each dataframe should have 'SNR' column and either 'CQI' or 'MCS' column
        save_path: Optional path to save the figure
    """
    
    if not tables_dict:
        print("No tables provided!")
        return
    
    print(f"Processing {len(tables_dict)} tables...")
    for name, df in tables_dict.items():
        print(f"  {name}: {df.shape} - Columns: {df.columns.tolist()}")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Colors for different tables - extended palette for 5+ tables
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # 1. Line plot comparison (top left)
    ax1 = axes[0, 0]
    for i, (name, df) in enumerate(tables_dict.items()):
        # Determine if it's CQI or MCS column (check both upper and lower case)
        if 'CQI' in df.columns:
            y_col = 'CQI'
            x_col = 'SNR'
        elif 'cqi' in df.columns:
            y_col = 'cqi'
            x_col = 'snr' if 'snr' in df.columns else 'SNR'
        elif 'MCS' in df.columns:
            y_col = 'MCS'
            x_col = 'SNR'
        elif 'mcs' in df.columns:
            y_col = 'mcs'
            x_col = 'snr' if 'snr' in df.columns else 'SNR'
        else:
            continue
            
        if y_col in df.columns and x_col in df.columns:
            # Sort by SNR for proper line plotting
            df_sorted = df.sort_values(x_col)
            ax1.plot(df_sorted[x_col], df_sorted[y_col], 
                    marker='o', markersize=4, linewidth=2, 
                    color=colors[i % len(colors)], label=name, alpha=0.8)
    
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('CQI/MCS Index')
    ax1.set_title('SNR vs CQI/MCS Mapping Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.7)
    
    # 2. Step function comparison (top right)
    ax2 = axes[0, 1]
    for i, (name, df) in enumerate(tables_dict.items()):
        # Determine column names (check both cases)
        if 'CQI' in df.columns:
            y_col, x_col = 'CQI', 'SNR'
        elif 'cqi' in df.columns:
            y_col = 'cqi'
            x_col = 'snr' if 'snr' in df.columns else 'SNR'
        elif 'MCS' in df.columns:
            y_col, x_col = 'MCS', 'SNR'
        elif 'mcs' in df.columns:
            y_col = 'mcs'
            x_col = 'snr' if 'snr' in df.columns else 'SNR'
        else:
            continue
            
        if y_col in df.columns and x_col in df.columns:
            df_sorted = df.sort_values(x_col)
            ax2.step(df_sorted[x_col], df_sorted[y_col], 
                    where='post', linewidth=2, 
                    color=colors[i % len(colors)], label=name, alpha=0.8)
    
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('CQI/MCS Index')
    ax2.set_title('Step Function Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.7)
    
    # 3. Histogram comparison (bottom left)
    ax3 = axes[1, 0]
    for i, (name, df) in enumerate(tables_dict.items()):
        # Determine column names (check both cases)
        if 'CQI' in df.columns:
            y_col = 'CQI'
        elif 'cqi' in df.columns:
            y_col = 'cqi'
        elif 'MCS' in df.columns:
            y_col = 'MCS'
        elif 'mcs' in df.columns:
            y_col = 'mcs'
        else:
            continue
            
        if y_col in df.columns:
            ax3.hist(df[y_col], bins=range(int(df[y_col].min()), int(df[y_col].max())+2), 
                    alpha=0.6, label=name, color=colors[i % len(colors)])
    
    ax3.set_xlabel('CQI/MCS Index')
    ax3.set_ylabel('Frequency')
    ax3.set_title('CQI/MCS Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.7)
    
    # 4. Scatter plot with trend lines (bottom right)
    ax4 = axes[1, 1]
    for i, (name, df) in enumerate(tables_dict.items()):
        # Determine column names (check both cases)
        if 'CQI' in df.columns:
            y_col, x_col = 'CQI', 'SNR'
        elif 'cqi' in df.columns:
            y_col = 'cqi'
            x_col = 'snr' if 'snr' in df.columns else 'SNR'
        elif 'MCS' in df.columns:
            y_col, x_col = 'MCS', 'SNR'
        elif 'mcs' in df.columns:
            y_col = 'mcs'
            x_col = 'snr' if 'snr' in df.columns else 'SNR'
        else:
            continue
            
        if y_col in df.columns and x_col in df.columns:
            ax4.scatter(df[x_col], df[y_col], 
                       alpha=0.6, s=20, color=colors[i % len(colors)], label=name)
            
            # Add trend line
            z = np.polyfit(df[x_col], df[y_col], 1)
            p = np.poly1d(z)
            ax4.plot(df[x_col], p(df[x_col]), "--", 
                    color=colors[i % len(colors)], alpha=0.8)
    
    ax4.set_xlabel('SNR (dB)')
    ax4.set_ylabel('CQI/MCS Index')
    ax4.set_title('Scatter Plot with Trend Lines')
    ax4.legend()
    ax4.grid(True, alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig

def analyze_table_differences(tables_dict):
    """
    Analyze and print statistics about the differences between tables
    """
    print("=== SNR-CQI/MCS Table Analysis ===\n")
    
    for name, df in tables_dict.items():
        # Determine column names (check both cases)
        if 'CQI' in df.columns:
            y_col, x_col = 'CQI', 'SNR'
        elif 'cqi' in df.columns:
            y_col = 'cqi'
            x_col = 'snr' if 'snr' in df.columns else 'SNR'
        elif 'MCS' in df.columns:
            y_col, x_col = 'MCS', 'SNR'
        elif 'mcs' in df.columns:
            y_col = 'mcs'
            x_col = 'snr' if 'snr' in df.columns else 'SNR'
        else:
            continue
            
        if y_col in df.columns and x_col in df.columns:
            print(f"{name}:")
            print(f"  SNR range: {df[x_col].min():.1f} to {df[x_col].max():.1f} dB")
            print(f"  {y_col.upper()} range: {df[y_col].min()} to {df[y_col].max()}")
            print(f"  Data points: {len(df)}")
            print(f"  Average {y_col.upper()} per SNR bin:")
            
            # Group by SNR bins and show average CQI/MCS
            df['SNR_bin'] = pd.cut(df[x_col], bins=10, precision=1)
            grouped = df.groupby('SNR_bin')[y_col].mean()
            for bin_range, avg_val in grouped.items():
                print(f"    {bin_range}: {avg_val:.2f}")
            print()

def create_difference_heatmap(table1, table2, table1_name='Table 1', table2_name='Table 2'):
    """
    Create a heatmap showing the difference between two CQI/MCS tables
    """
    # Determine column names for both tables (check both cases)
    def get_columns(df):
        if 'CQI' in df.columns:
            return 'CQI', 'SNR'
        elif 'cqi' in df.columns:
            return 'cqi', ('snr' if 'snr' in df.columns else 'SNR')
        elif 'MCS' in df.columns:
            return 'MCS', 'SNR'
        elif 'mcs' in df.columns:
            return 'mcs', ('snr' if 'snr' in df.columns else 'SNR')
        return None, None
    
    y_col1, x_col1 = get_columns(table1)
    y_col2, x_col2 = get_columns(table2)
    
    if (y_col1 and x_col1 and y_col1 in table1.columns and x_col1 in table1.columns and
        y_col2 and x_col2 and y_col2 in table2.columns and x_col2 in table2.columns):
        
        # Merge tables on SNR values (normalize column names for merging)
        table1_copy = table1[[x_col1, y_col1]].copy()
        table2_copy = table2[[x_col2, y_col2]].copy()
        
        # Standardize column names for merging
        table1_copy.columns = ['SNR', 'VALUE']
        table2_copy.columns = ['SNR', 'VALUE']
        
        merged = pd.merge(table1_copy, table2_copy, on='SNR', suffixes=('_1', '_2'), how='outer')
        
        # Calculate difference
        merged['difference'] = merged['VALUE_1'] - merged['VALUE_2']
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a scatter plot colored by difference
        scatter = ax.scatter(merged['SNR'], merged['VALUE_1'], 
                           c=merged['difference'], cmap='RdBu_r', 
                           alpha=0.7, s=30)
        
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel(f'{y_col1.upper()} ({table1_name})')
        ax.set_title(f'{y_col1.upper()} Difference: {table1_name} - {table2_name}')
        ax.grid(True, alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Difference')
        
        plt.show()
        return merged
    else:
        print(f"Could not create comparison - incompatible column structures")
        return None

# Example usage
if __name__ == "__main__":
    # Load all SNR-CQI LUT files from ML/data directory
    ml_data_dir = Path('../ML/data')
    
    # Load all 5 LUT files
    tables = {}
    
    # Load each LUT file
    lut_files = [
        ('snr_cqi_lut_dense.csv', 'Dense LUT'),
        ('snr_cqi_lut_dense_lower.csv', 'Dense Lower LUT'), 
        ('snr_cqi_lut_empirical.csv', 'Empirical LUT'),
        ('snr_cqi_lut_empirical_tput.csv', 'Empirical Throughput LUT'),
        ('snr_cqi_lut_t95_q30_lowest.csv', 'T95 Q30 Lowest LUT')
    ]
    
    print("Loading SNR-CQI LUT files...")
    for filename, display_name in lut_files:
        file_path = ml_data_dir / filename
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                tables[display_name] = df
                print(f"  ✓ Loaded {display_name}: {df.shape}")
                print(f"    Columns: {df.columns.tolist()}")
                
                # Show basic statistics for each table
                if 'SNR' in df.columns:
                    snr_col = 'SNR'
                elif 'snr' in df.columns:
                    snr_col = 'snr'
                    df = df.rename(columns={'snr': 'SNR'})  # Standardize column name
                else:
                    print(f"    Warning: No SNR column found in {display_name}")
                    continue
                    
                # Find CQI or similar column
                if 'CQI' in df.columns:
                    value_col = 'CQI'
                elif 'cqi' in df.columns:
                    df = df.rename(columns={'cqi': 'CQI'})
                    value_col = 'CQI'
                elif 'MCS' in df.columns:
                    value_col = 'MCS'
                elif 'mcs' in df.columns:
                    df = df.rename(columns={'mcs': 'MCS'})
                    value_col = 'MCS'
                else:
                    print(f"    Warning: No CQI/MCS column found in {display_name}")
                    # Try to find any numeric column that might be the value
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) > 1:  # SNR + another numeric column
                        value_col = [col for col in numeric_cols if col != 'SNR'][0]
                        print(f"    Using column '{value_col}' as value column")
                    else:
                        continue
                
                print(f"    SNR range: {df['SNR'].min():.1f} to {df['SNR'].max():.1f} dB")
                print(f"    {value_col} range: {df[value_col].min()} to {df[value_col].max()}")
                
            except Exception as e:
                print(f"  ✗ Failed to load {display_name}: {str(e)}")
        else:
            print(f"  ✗ File not found: {file_path}")
    
    print(f"\nSuccessfully loaded {len(tables)} LUT tables")
    
    if not tables:
        print("No valid tables loaded. Please check file paths and formats.")
        exit(1)
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    plot_snr_cqi_mcs_comparison(tables, save_path='snr_cqi_lut_comparison.png')
    
    # Analyze differences between tables
    print("\nAnalyzing LUT characteristics...")
    analyze_table_differences(tables)
    
    # Create pairwise difference analysis for key comparisons
    print("\nCreating pairwise difference analysis...")
    if len(tables) >= 2:
        table_names = list(tables.keys())
        
        # Compare empirical vs dense
        if 'Empirical LUT' in tables and 'Dense LUT' in tables:
            print("\n--- Empirical vs Dense LUT Comparison ---")
            create_difference_heatmap(tables['Empirical LUT'], tables['Dense LUT'], 
                                    'Empirical LUT', 'Dense LUT')
        
        # Compare empirical vs empirical throughput
        if 'Empirical LUT' in tables and 'Empirical Throughput LUT' in tables:
            print("\n--- Empirical vs Empirical Throughput LUT Comparison ---")
            create_difference_heatmap(tables['Empirical LUT'], tables['Empirical Throughput LUT'], 
                                    'Empirical LUT', 'Empirical Throughput LUT')
    
    print(f"\nAnalysis complete!")
    print("Plots saved as: snr_cqi_lut_comparison.png")
    print("Individual difference plots shown above.")

    # Additional summary statistics
    print("\n=== SUMMARY COMPARISON ===")
    for name, df in tables.items():
        value_col = 'CQI' if 'CQI' in df.columns else ('MCS' if 'MCS' in df.columns else df.select_dtypes(include=[np.number]).columns.tolist()[-1])
        if 'SNR' in df.columns and value_col in df.columns:
            correlation = df['SNR'].corr(df[value_col])
            print(f"{name}:")
            print(f"  SNR-{value_col} correlation: {correlation:.3f}")
            print(f"  Avg {value_col} per SNR dB: {df[value_col].sum() / (df['SNR'].max() - df['SNR'].min()):.3f}")
            
            # Find optimal SNR ranges
            high_cqi_snr = df[df[value_col] >= df[value_col].quantile(0.8)]['SNR']
            if len(high_cqi_snr) > 0:
                print(f"  SNR range for top 20% {value_col}: {high_cqi_snr.min():.1f} to {high_cqi_snr.max():.1f} dB")
            print()
