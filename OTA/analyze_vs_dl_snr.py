#!/usr/bin/env python3
"""
5G NTN Link Adaptation Analysis: Curves vs DL SNR
Generates performance curves plotted against DL SNR instead of time

Usage:
    python3 analyze_vs_dl_snr.py [csv_files...]
    python3 analyze_vs_dl_snr.py bler01/*.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import glob
import os
from pathlib import Path

def load_and_process_csv(csv_file):
    """Load CSV file and prepare data for SNR-based analysis."""
    try:
        df = pd.read_csv(csv_file)
        print(f"📖 Loaded {len(df)} entries from {os.path.basename(csv_file)}")
        
        # Filter out entries without DL SINR data
        df_filtered = df[df['dl_sinr'].notna()].copy()
        print(f"   - {len(df_filtered)} entries with DL SINR data")
        
        return df_filtered
    except Exception as e:
        print(f"❌ Error loading {csv_file}: {e}")
        return None

def create_snr_bins(df, bin_width=1.0):
    """Create SNR bins for aggregating data."""
    if df.empty or 'dl_sinr' not in df.columns:
        return None
    
    min_snr = df['dl_sinr'].min()
    max_snr = df['dl_sinr'].max()
    
    # Create bins
    bins = np.arange(np.floor(min_snr), np.ceil(max_snr) + bin_width, bin_width)
    df['snr_bin'] = pd.cut(df['dl_sinr'], bins=bins, include_lowest=True)
    df['snr_bin_center'] = df['snr_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan)
    
    return df

def calculate_bler_per_snr(df):
    """Calculate BLER per SNR bin."""
    if df.empty:
        return pd.DataFrame()
    
    # Group by SNR bin and calculate BLER
    snr_stats = df.groupby('snr_bin_center').agg({
        'ack_status': ['count', lambda x: (x == 0).sum()],  # total, NACK count
        'mcs': 'mean',
        'tbs': 'mean',
        'dl_rsrp': 'mean',
        'dl_rsrq': 'mean',
        'ul_sinr': 'mean'
    }).round(4)
    
    # Flatten column names
    snr_stats.columns = ['total_transmissions', 'nack_count', 'avg_mcs', 'avg_tbs', 'avg_dl_rsrp', 'avg_dl_rsrq', 'avg_ul_sinr']
    
    # Calculate BLER percentage
    snr_stats['bler_percent'] = (snr_stats['nack_count'] / snr_stats['total_transmissions']) * 100
    
    # Reset index to make snr_bin_center a regular column
    snr_stats = snr_stats.reset_index()
    
    return snr_stats

def calculate_throughput_per_snr(df):
    """Calculate average throughput per SNR bin."""
    if df.empty:
        return pd.DataFrame()
    
    # Check if we have the new throughput columns from updated processor
    if 'throughput_mbps_per_sec' in df.columns:
        # Use the per-second aggregated throughput (more meaningful than per-transmission)
        throughput_col = 'throughput_mbps_per_sec'
        trueput_col = 'trueput_mbps_per_sec'
        print("📊 Using per-second aggregated throughput metrics (Mbps)")
    elif 'instantaneous_throughput_mbps' in df.columns:
        # Use the proper HARQ-aware throughput calculations
        throughput_col = 'instantaneous_throughput_mbps'
        trueput_col = 'instantaneous_trueput_mbps'
        print("📊 Using HARQ-aware instantaneous throughput metrics (Mbps)")
    elif 'throughput_kbps_per_sec' in df.columns:
        # Legacy kbps columns, convert to Mbps
        throughput_col = 'throughput_kbps_per_sec'
        trueput_col = 'trueput_kbps_per_sec'
        print("📊 Using per-second aggregated throughput metrics (legacy kbps, will convert to Mbps)")
    elif 'instantaneous_throughput_kbps' in df.columns:
        # Legacy kbps columns, convert to Mbps
        throughput_col = 'instantaneous_throughput_kbps'
        trueput_col = 'instantaneous_trueput_kbps'
        print("📊 Using HARQ-aware instantaneous throughput metrics (legacy kbps, will convert to Mbps)")
    else:
        # Fallback to legacy calculation with proper kbps (÷1024)
        df['bitrate_kbps'] = df['tbs'] * 8 / 1024  # True kbps calculation
        throughput_col = 'bitrate_kbps'
        trueput_col = None
        print("📊 Using legacy bitrate calculation (kbps, will convert to Mbps)")
    
    # Aggregate throughput statistics
    agg_dict = {
        throughput_col: ['mean', 'std', 'count'],
        'mcs': 'mean',
        'tbs': 'mean'
    }
    
    # Add trueput if available
    if trueput_col and trueput_col in df.columns:
        agg_dict[trueput_col] = ['mean', 'std']
    
    throughput_stats = df.groupby('snr_bin_center').agg(agg_dict).round(4)
    
    # Determine if we need to convert from kbps to Mbps
    is_legacy_kbps = ('kbps' in throughput_col)
    throughput_divisor = 1000 if is_legacy_kbps else 1  # Convert kbps to Mbps if needed
    
    # Flatten column names
    if trueput_col and trueput_col in df.columns:
        if is_legacy_kbps:
            throughput_stats.columns = ['avg_throughput_kbps', 'std_throughput_kbps', 'sample_count', 
                                       'avg_mcs', 'avg_tbs', 'avg_trueput_kbps', 'std_trueput_kbps']
            # Convert to Mbps
            throughput_stats['avg_throughput_mbps'] = throughput_stats['avg_throughput_kbps'] / throughput_divisor
            throughput_stats['std_throughput_mbps'] = throughput_stats['std_throughput_kbps'] / throughput_divisor
            throughput_stats['avg_trueput_mbps'] = throughput_stats['avg_trueput_kbps'] / throughput_divisor
            throughput_stats['std_trueput_mbps'] = throughput_stats['std_trueput_kbps'] / throughput_divisor
        else:
            # Already in Mbps
            throughput_stats.columns = ['avg_throughput_mbps', 'std_throughput_mbps', 'sample_count', 
                                       'avg_mcs', 'avg_tbs', 'avg_trueput_mbps', 'std_trueput_mbps']
    else:
        if is_legacy_kbps:
            throughput_stats.columns = ['avg_throughput_kbps', 'std_throughput_kbps', 'sample_count', 
                                       'avg_mcs', 'avg_tbs']
            # Convert to Mbps
            throughput_stats['avg_throughput_mbps'] = throughput_stats['avg_throughput_kbps'] / throughput_divisor
            throughput_stats['std_throughput_mbps'] = throughput_stats['std_throughput_kbps'] / throughput_divisor
        else:
            # Already in Mbps
            throughput_stats.columns = ['avg_throughput_mbps', 'std_throughput_mbps', 'sample_count', 
                                       'avg_mcs', 'avg_tbs']
    throughput_stats = throughput_stats.reset_index()
    
    return throughput_stats

def plot_performance_vs_snr(data_dict, output_prefix):
    """Create comprehensive plots of performance metrics vs DL SNR."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    # Plot 1: BLER vs DL SNR
    for i, (filename, data) in enumerate(data_dict.items()):
        if 'bler_stats' in data and not data['bler_stats'].empty:
            bler_data = data['bler_stats']
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            ax1.plot(bler_data['snr_bin_center'].values, bler_data['bler_percent'].values, 
                    color=color, marker=marker, linestyle='-', label=filename, linewidth=2, markersize=6)
    
    ax1.set_xlabel('DL SNR (dB)')
    ax1.set_ylabel('BLER (%)')
    ax1.set_title('Block Error Rate vs DL SNR')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(bottom=0)
    
    # Plot 2: Average MCS vs DL SNR
    for i, (filename, data) in enumerate(data_dict.items()):
        if 'bler_stats' in data and not data['bler_stats'].empty:
            bler_data = data['bler_stats']
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            ax2.plot(bler_data['snr_bin_center'].values, bler_data['avg_mcs'].values, 
                    color=color, marker=marker, linestyle='-', label=filename, linewidth=2, markersize=6)
    
    ax2.set_xlabel('DL SNR (dB)')
    ax2.set_ylabel('Average MCS')
    ax2.set_title('Average MCS vs DL SNR')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Average Throughput vs DL SNR
    for i, (filename, data) in enumerate(data_dict.items()):
        if 'throughput_stats' in data and not data['throughput_stats'].empty:
            throughput_data = data['throughput_stats']
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            # Use Mbps for better readability
            if 'avg_throughput_mbps' in throughput_data.columns:
                throughput_col = 'avg_throughput_mbps'
                y_label = 'Average Throughput (Mbps)'
            else:
                # Fallback to kbps columns
                throughput_col = 'avg_throughput_kbps' if 'avg_throughput_kbps' in throughput_data.columns else 'avg_bitrate_kbps'
                y_label = 'Average Throughput (kbps)'
            
            ax3.plot(throughput_data['snr_bin_center'].values, throughput_data[throughput_col].values, 
                    color=color, marker=marker, linestyle='-', label=filename, linewidth=2, markersize=6)
    
    ax3.set_xlabel('DL SNR (dB)')
    ax3.set_ylabel(y_label)
    ax3.set_title('Average Throughput vs DL SNR')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: SNR Distribution
    for i, (filename, data) in enumerate(data_dict.items()):
        if 'df' in data and not data['df'].empty:
            df = data['df']
            color = colors[i % len(colors)]
            
            ax4.hist(df['dl_sinr'].values, bins=30, alpha=0.6, label=filename, color=color, edgecolor='black')
    
    ax4.set_xlabel('DL SNR (dB)')
    ax4.set_ylabel('Count')
    ax4.set_title('DL SNR Distribution')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{output_prefix}_vs_dl_snr_analysis.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📈 Saved analysis plots to {plot_filename}")
    
    plt.show()

def print_summary_statistics(data_dict):
    """Print summary statistics for all datasets."""
    print("\n📊 SUMMARY STATISTICS:")
    print("=" * 80)
    
    for filename, data in data_dict.items():
        if 'df' in data and not data['df'].empty:
            df = data['df']
            print(f"\n📁 {filename}:")
            print(f"   Total transmissions: {len(df):,}")
            print(f"   DL SNR range: {df['dl_sinr'].min():.1f} to {df['dl_sinr'].max():.1f} dB")
            print(f"   Mean DL SNR: {df['dl_sinr'].mean():.1f} dB")
            print(f"   MCS range: {df['mcs'].min()} to {df['mcs'].max()}")
            
            # Overall BLER
            total_nacks = (df['ack_status'] == 0).sum()
            overall_bler = (total_nacks / len(df)) * 100
            print(f"   Overall BLER: {overall_bler:.2f}%")
            
            # ACK/NACK/DTX distribution
            ack_counts = df['ack_result'].value_counts()
            for result, count in ack_counts.items():
                percentage = (count / len(df)) * 100
                print(f"   {result}: {count:,} ({percentage:.1f}%)")

def main():
    """Main function to process CSV files and generate SNR-based analysis."""
    
    # Get CSV files to process
    if len(sys.argv) > 1:
        csv_files = sys.argv[1:]
    else:
        # Default: process all CSV files in bler01 folder
        csv_files = glob.glob("/bler01/*.csv")
        if not csv_files:
            print("No CSV files found in bler01/ folder")
            print("Usage: python3 analyze_vs_dl_snr.py [csv_files...]")
            return
    
    print(f"🔍 Processing {len(csv_files)} CSV files...")
    
    # Process all CSV files
    data_dict = {}
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file).replace('.csv', '')
        
        # Load data
        df = load_and_process_csv(csv_file)
        if df is None or df.empty:
            continue
        
        # Create SNR bins
        df_binned = create_snr_bins(df, bin_width=1.0)
        if df_binned is None:
            continue
        
        # Calculate statistics
        bler_stats = calculate_bler_per_snr(df_binned)
        throughput_stats = calculate_throughput_per_snr(df_binned)
        
        # Store results
        data_dict[filename] = {
            'df': df_binned,
            'bler_stats': bler_stats,
            'throughput_stats': throughput_stats
        }
    
    if not data_dict:
        print("❌ No valid data found in CSV files")
        return
    
    # Generate plots
    output_prefix = "bler01_combined" if "bler01" in csv_files[0] else "combined"
    plot_performance_vs_snr(data_dict, output_prefix)
    
    # Print summary statistics
    print_summary_statistics(data_dict)
    
    print(f"\n🎉 Analysis completed successfully!")

if __name__ == "__main__":
    main()
