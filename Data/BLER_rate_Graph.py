#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def add_rate_column(data):
    """
    Adds RATE (kbps) and cumulative throughput columns to a pandas DataFrame.
    
    RATE = TBS (bits) / 1000 if BLKErr == 0, else 0
    CUM_THROUGHPUT = cumulative sum of RATE (kbps)
    
    Args:
        data (pd.DataFrame): DataFrame with columns "TBS" and "BLKErr"
    
    Returns:
        pd.DataFrame: DataFrame with added columns "RATE" (kbps) and "CUM_THROUGHPUT" (kbps)
    """
    # Compute RATE and cumulative throughput
    TBS_bits = data['TBS']  # transport block size in bits
    BLKErr = data['BLKErr']  # block error indicator
    RATE_kbps = (TBS_bits * (BLKErr == 0)) / 1000  # kbps for successful transmission (BLKErr==0)
    cum_throughput_kbps = RATE_kbps.cumsum()  # cumulative throughput (kbps)
    
    # Add new columns to the DataFrame
    data['RATE'] = RATE_kbps
    data['CUM_THROUGHPUT'] = cum_throughput_kbps
    
    return data

def moving_average(data, window_size):
    """Simple moving average using pandas rolling window"""
    return data.rolling(window=window_size, min_periods=1).mean()

def group_by_snr_and_average(data, snr_col='SNR', round_digits=1):
    """
    Group data by SNR values and calculate mean BLER and RATE for each unique SNR
    to reduce fuzziness in the plot.
    
    Args:
        data (pd.DataFrame): DataFrame with SNR, BLER, and RATE columns
        snr_col (str): Name of the SNR column
        round_digits (int): Number of decimal places to round SNR values
    
    Returns:
        pd.DataFrame: DataFrame with unique SNR values and averaged BLER/RATE
    """
    # Round SNR values to reduce the number of unique values
    data_copy = data.copy()
    data_copy[snr_col + '_rounded'] = data_copy[snr_col].round(round_digits)
    
    # Group by rounded SNR and calculate means
    grouped = data_copy.groupby(snr_col + '_rounded').agg({
        'BLER': 'mean',
        'RATE': 'mean'
    }).reset_index()
    
    # Rename the rounded SNR column back to original name
    grouped = grouped.rename(columns={snr_col + '_rounded': snr_col})
    
    return grouped

# Load data
data_dir = Path('Data')  # Current directory is Data
b = pd.read_csv(data_dir / 'Archive' / 'Result_baseline.csv')
w20001 = pd.read_csv(data_dir / 'Case9_MCS_ThroughputCalulation_BLERw2000Tbler0.001_240608_153903.csv')
w2001 = pd.read_csv(data_dir / 'Case9_MCS_ThroughputCalulation_BLERw2000Tbler0.01_240603_133217.csv')
w2005 = pd.read_csv(data_dir / 'Case9_MCS_ThroughputCalulation_BLERw2000Tbler0.05_240604_085504.csv')
w2010 = pd.read_csv(data_dir / 'Case9_MCS_ThroughputCalulation_BLERw2000Tbler0.1_240605_171830.csv')
ml1 = pd.read_csv(data_dir / 'ML_Case9_MCS_ThroughputCalulation_snr_cqi_lut_empirical.csv')
# Additional ML datasets (skip ML_Emperical due to formatting issues)
ml_classifier = pd.read_csv(data_dir / 'ML_Classifier_Case9_MCS_ThroughputCalulation_.csv')
# ml_empirical = pd.read_csv(data_dir / 'ML_Emperical_Case9_MCS_ThroughputCalulation_BLERw2000Tbler0.001_250914_042207.csv')  # Skip due to formatting issues
ml_newton = pd.read_csv(data_dir / 'ML_newtwon_Case9_MCS_ThroughputCalulation_.csv')
ml_poly = pd.read_csv(data_dir / 'ML_poly_Case9_MCS_ThroughputCalulation_.csv')
# OTA datasets
ota_b01 = pd.read_csv(Path('OTA') / 'gnb-bler01.csv')
ota_b05 = pd.read_csv(Path('OTA') / 'gnb-bler05-r2.csv')
ota_b10 = pd.read_csv(Path('OTA') / 'gnb-bler10-r2.csv')
ota_no_olla = pd.read_csv(Path('OTA') / 'gnb-no-olla-r2.csv')
# Add rate columns to all datasets
b = add_rate_column(b)
w20001 = add_rate_column(w20001)
w2001 = add_rate_column(w2001)
w2005 = add_rate_column(w2005)
w2010 = add_rate_column(w2010)
ml1 = add_rate_column(ml1)
# Add rate columns to new ML datasets
ml_classifier = add_rate_column(ml_classifier)
# ml_empirical = add_rate_column(ml_empirical)  # Skip due to formatting issues
ml_newton = add_rate_column(ml_newton)
ml_poly = add_rate_column(ml_poly)
# ota_b01 = add_rate_column(ota_b01)

# Group by SNR and average to reduce fuzziness
b_avg = group_by_snr_and_average(b)
w20001_avg = group_by_snr_and_average(w20001)
w2001_avg = group_by_snr_and_average(w2001)
w2005_avg = group_by_snr_and_average(w2005)
w2010_avg = group_by_snr_and_average(w2010)
ml1_avg = group_by_snr_and_average(ml1)
# Average new ML datasets
ml_classifier_avg = group_by_snr_and_average(ml_classifier)
# ml_empirical_avg = group_by_snr_and_average(ml_empirical)  # Skip due to formatting issues
ml_newton_avg = group_by_snr_and_average(ml_newton)
ml_poly_avg = group_by_snr_and_average(ml_poly)

# For OTA data, use different SNR column name
ota_b01_grouped = ota_b01.copy()
ota_b01_grouped['SNR_rounded'] = ota_b01_grouped['dl_sinr'].round(1)
ota_b01_avg = ota_b01_grouped.groupby('SNR_rounded').agg({
    'bler': 'mean',
    'throughput_mbps_per_sec': 'mean'
}).reset_index()
ota_b01_avg = ota_b01_avg.rename(columns={'SNR_rounded': 'SNR', 'bler': 'BLER', 'throughput_mbps_per_sec': 'RATE'})

# Process other OTA datasets
ota_b05_grouped = ota_b05.copy()
ota_b05_grouped['SNR_rounded'] = ota_b05_grouped['dl_sinr'].round(1)
ota_b05_avg = ota_b05_grouped.groupby('SNR_rounded').agg({
    'bler': 'mean',
    'throughput_mbps_per_sec': 'mean'
}).reset_index()
ota_b05_avg = ota_b05_avg.rename(columns={'SNR_rounded': 'SNR', 'bler': 'BLER', 'throughput_mbps_per_sec': 'RATE'})

ota_b10_grouped = ota_b10.copy()
ota_b10_grouped['SNR_rounded'] = ota_b10_grouped['dl_sinr'].round(1)
ota_b10_avg = ota_b10_grouped.groupby('SNR_rounded').agg({
    'bler': 'mean',
    'throughput_mbps_per_sec': 'mean'
}).reset_index()
ota_b10_avg = ota_b10_avg.rename(columns={'SNR_rounded': 'SNR', 'bler': 'BLER', 'throughput_mbps_per_sec': 'RATE'})

ota_no_olla_grouped = ota_no_olla.copy()
ota_no_olla_grouped['SNR_rounded'] = ota_no_olla_grouped['dl_sinr'].round(1)
ota_no_olla_avg = ota_no_olla_grouped.groupby('SNR_rounded').agg({
    'bler': 'mean',
    'throughput_mbps_per_sec': 'mean'
}).reset_index()
ota_no_olla_avg = ota_no_olla_avg.rename(columns={'SNR_rounded': 'SNR', 'bler': 'BLER', 'throughput_mbps_per_sec': 'RATE'})

# Sort all datasets by SNR for proper line plotting
b_avg = b_avg.sort_values('SNR')
w20001_avg = w20001_avg.sort_values('SNR')
w2001_avg = w2001_avg.sort_values('SNR')
w2005_avg = w2005_avg.sort_values('SNR')
w2010_avg = w2010_avg.sort_values('SNR')
ml1_avg = ml1_avg.sort_values('SNR')
# Sort new ML datasets
ml_classifier_avg = ml_classifier_avg.sort_values('SNR')
# ml_empirical_avg = ml_empirical_avg.sort_values('SNR')  # Skip due to formatting issues
ml_newton_avg = ml_newton_avg.sort_values('SNR')
ml_poly_avg = ml_poly_avg.sort_values('SNR')
# Sort OTA datasets
ota_b01_avg = ota_b01_avg.sort_values('SNR')
ota_b05_avg = ota_b05_avg.sort_values('SNR')
ota_b10_avg = ota_b10_avg.sort_values('SNR')
ota_no_olla_avg = ota_no_olla_avg.sort_values('SNR')

def plot_rate_comparison(use_elevation_angle=False):
    """
    Plot rate comparison for all algorithms with option to use elevation angle or SNR as x-axis.
    
    Args:
        use_elevation_angle (bool): If True, plot against 'eleAnge' column. If False, plot against 'SNR'.
    """
    # Determine x-axis column and prepare data for elevation angle plotting if needed
    if use_elevation_angle:
        x_col = 'eleAnge'
        xlabel = "Elevation Angle (degrees)"
        xlim = None  # Auto-scale for elevation angle
        # Group by elevation angle instead of SNR for simulation datasets
        b_plot = group_by_snr_and_average(b, snr_col='eleAnge')
        w20001_plot = group_by_snr_and_average(w20001, snr_col='eleAnge')
        w2001_plot = group_by_snr_and_average(w2001, snr_col='eleAnge')
        w2005_plot = group_by_snr_and_average(w2005, snr_col='eleAnge')
        w2010_plot = group_by_snr_and_average(w2010, snr_col='eleAnge')
        ml1_plot = group_by_snr_and_average(ml1, snr_col='eleAnge')
        ml_classifier_plot = group_by_snr_and_average(ml_classifier, snr_col='eleAnge')
        # ml_empirical_plot = group_by_snr_and_average(ml_empirical, snr_col='eleAnge')  # Skip due to formatting issues
        ml_newton_plot = group_by_snr_and_average(ml_newton, snr_col='eleAnge')
        ml_poly_plot = group_by_snr_and_average(ml_poly, snr_col='eleAnge')
    else:
        x_col = 'SNR'
        xlabel = "SNR (dB)"
        xlim = [-8, 7]
        # Use already processed SNR data
        b_plot = b_avg
        w20001_plot = w20001_avg
        w2001_plot = w2001_avg
        w2005_plot = w2005_avg
        w2010_plot = w2010_avg
        ml1_plot = ml1_avg
        ml_classifier_plot = ml_classifier_avg
        # ml_empirical_plot = ml_empirical_avg  # Skip due to formatting issues
        ml_newton_plot = ml_newton_avg
        ml_poly_plot = ml_poly_avg
    
    # Define colors and markers for all datasets
    colors = [
        [0, 0.4470, 0.7410],        # blue
        [0.8500, 0.3250, 0.0980],   # orange  
        [0.4660, 0.6740, 0.1880],   # green
        [0.4940, 0.1840, 0.5560],   # purple
        [0.6350, 0.0780, 0.1840],   # red
        [0, 0, 0],                  # black
        [0.3010, 0.7450, 0.9330],   # cyan
        [0.8350, 0.3780, 0.5560],   # magenta
        [0.9290, 0.6940, 0.1250],   # yellow
        [0.7, 0.7, 0.7],           # light gray
        'darkgreen',                # dark green
        'navy',                     # navy blue
        'darkred',                  # dark red
        'chocolate'                 # brown
    ]
    
    markers = ['o', 's', '^', 'D', 'v', '*', 'x', '+', '1', '2', '3', '4', 'p', 'h']
    
    # Create figure with single axis (RATE only)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort datasets by x-axis for proper line plotting
    b_plot = b_plot.sort_values(x_col)
    w20001_plot = w20001_plot.sort_values(x_col)
    w2001_plot = w2001_plot.sort_values(x_col)
    w2005_plot = w2005_plot.sort_values(x_col)
    w2010_plot = w2010_plot.sort_values(x_col)
    ml1_plot = ml1_plot.sort_values(x_col)
    ml_classifier_plot = ml_classifier_plot.sort_values(x_col)
    # ml_empirical_plot = ml_empirical_plot.sort_values(x_col)  # Skip due to formatting issues
    ml_newton_plot = ml_newton_plot.sort_values(x_col)
    ml_poly_plot = ml_poly_plot.sort_values(x_col)
    
    # RATE plotting - Using line plots with markers for clarity
    ax.plot(b_plot[x_col], b_plot['RATE'], color=colors[0], marker=markers[0], markersize=4, linewidth=2, alpha=0.8, label='Baseline')
    ax.plot(w20001_plot[x_col], w20001_plot['RATE'], color=colors[1], marker=markers[1], markersize=4, linewidth=2, alpha=0.8, label='BLER_Target = 0.1%')
    ax.plot(w2001_plot[x_col], w2001_plot['RATE'], color=colors[2], marker=markers[2], markersize=4, linewidth=2, alpha=0.8, label='BLER_Target = 1%')
    ax.plot(w2005_plot[x_col], w2005_plot['RATE'], color=colors[3], marker=markers[3], markersize=4, linewidth=2, alpha=0.8, label='BLER_Target = 5%')
    ax.plot(w2010_plot[x_col], w2010_plot['RATE'], color=colors[4], marker=markers[4], markersize=4, linewidth=2, alpha=0.8, label='BLER_Target = 10%')
    ax.plot(ml1_plot[x_col], ml1_plot['RATE'], color=colors[5], marker=markers[5], markersize=6, linewidth=2, alpha=0.8, label='ML Empirical SNR-CQI')
    ax.plot(ml_classifier_plot[x_col], ml_classifier_plot['RATE'], color=colors[6], marker=markers[6], markersize=5, linewidth=2, alpha=0.8, label='ML Classifier')
    # ax.plot(ml_empirical_plot[x_col], ml_empirical_plot['RATE'], color=colors[7], marker=markers[7], markersize=5, linewidth=2, alpha=0.8, label='ML Empirical BLER')  # Skip due to formatting issues
    ax.plot(ml_newton_plot[x_col], ml_newton_plot['RATE'], color=colors[7], marker=markers[7], markersize=5, linewidth=2, alpha=0.8, label='ML Newton')
    ax.plot(ml_poly_plot[x_col], ml_poly_plot['RATE'], color=colors[8], marker=markers[8], markersize=5, linewidth=2, alpha=0.8, label='ML Polynomial')
    
    # Add OTA datasets (only when using SNR, as OTA data uses SNR)
    # if not use_elevation_angle:
    #     ax.plot(ota_b01_avg['SNR'], ota_b01_avg['RATE']/2, color=colors[9], marker=markers[9], markersize=5, linewidth=2, alpha=0.8, label='OTA BLER 1%')
    #     ax.plot(ota_b05_avg['SNR'], ota_b05_avg['RATE']/2, color=colors[10], marker=markers[10], markersize=5, linewidth=2, alpha=0.8, label='OTA BLER 5%')
    #     ax.plot(ota_b10_avg['SNR'], ota_b10_avg['RATE']/2, color=colors[11], marker=markers[11], markersize=5, linewidth=2, alpha=0.8, label='OTA BLER 10%')
    #     ax.plot(ota_no_olla_avg['SNR'], ota_no_olla_avg['RATE']/2, color=colors[12], marker=markers[12], markersize=5, linewidth=2, alpha=0.8, label='OTA No OLLA')
    
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_ylabel("Avg RATE (Mbps)", color='k')
    ax.set_xlabel(xlabel)
    
    # Add grid with thinner lines
    ax.grid(True, linewidth=0.5, alpha=0.7)
    
    # Use the built-in legend from the plot labels
    ax.legend(loc='upper left', title='Link Adaptation Algorithms', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.show()

# Define colors (matching MATLAB defaults) - keeping for backward compatibility
blue = [0, 0.4470, 0.7410]
orange = [0.8500, 0.3250, 0.0980]
green = [0.4660, 0.6740, 0.1880]
purple = [0.4940, 0.1840, 0.5560]
red = [0.6350, 0.0780, 0.1840]
# Additional colors for new datasets
cyan = [0.3010, 0.7450, 0.9330]
magenta = [0.8350, 0.3780, 0.5560]
yellow = [0.9290, 0.6940, 0.1250]
gray = [0.5, 0.5, 0.5]

# Plot using SNR (default behavior)
plot_rate_comparison(use_elevation_angle=False)

# Uncomment the line below to plot using elevation angle instead
# plot_rate_comparison(use_elevation_angle=True)
