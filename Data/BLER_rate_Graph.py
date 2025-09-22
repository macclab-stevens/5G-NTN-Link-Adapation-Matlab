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
data_dir = Path('Data')
b = pd.read_csv(data_dir / 'Archive' / 'Result_baseline.csv')
w20001 = pd.read_csv(data_dir / 'Case9_MCS_ThroughputCalulation_BLERw2000Tbler0.001_240608_153903.csv')
w2001 = pd.read_csv(data_dir / 'Case9_MCS_ThroughputCalulation_BLERw2000Tbler0.01_240603_133217.csv')
w2005 = pd.read_csv(data_dir / 'Case9_MCS_ThroughputCalulation_BLERw2000Tbler0.05_240604_085504.csv')
w2010 = pd.read_csv(data_dir / 'Case9_MCS_ThroughputCalulation_BLERw2000Tbler0.1_240605_171830.csv')
ml1 = pd.read_csv(data_dir / 'ML_Case9_MCS_ThroughputCalulation_snr_cqi_lut_empirical.csv')
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
# ota_b01 = add_rate_column(ota_b01)

# Group by SNR and average to reduce fuzziness
b_avg = group_by_snr_and_average(b)
w20001_avg = group_by_snr_and_average(w20001)
w2001_avg = group_by_snr_and_average(w2001)
w2005_avg = group_by_snr_and_average(w2005)
w2010_avg = group_by_snr_and_average(w2010)
ml1_avg = group_by_snr_and_average(ml1)

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
ota_b01_avg = ota_b01_avg.sort_values('SNR')
ota_b05_avg = ota_b05_avg.sort_values('SNR')
ota_b10_avg = ota_b10_avg.sort_values('SNR')
ota_no_olla_avg = ota_no_olla_avg.sort_values('SNR')

# Define colors (matching MATLAB defaults)
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

# Create figure with single axis (RATE only)
fig, ax = plt.subplots(figsize=(10, 8))

# RATE plotting - Using line plots with markers for clarity
ax.plot(b_avg['SNR'], b_avg['RATE'], color=blue, marker='o', markersize=4, linewidth=2, alpha=0.8, label='Baseline')
ax.plot(w20001_avg['SNR'], w20001_avg['RATE'], color=orange, marker='s', markersize=4, linewidth=2, alpha=0.8, label='BLER_Target = 0.1%')
ax.plot(w2001_avg['SNR'], w2001_avg['RATE'], color=green, marker='^', markersize=4, linewidth=2, alpha=0.8, label='BLER_Target = 1%')
ax.plot(w2005_avg['SNR'], w2005_avg['RATE'], color=purple, marker='D', markersize=4, linewidth=2, alpha=0.8, label='BLER_Target = 5%')
ax.plot(w2010_avg['SNR'], w2010_avg['RATE'], color=red, marker='v', markersize=4, linewidth=2, alpha=0.8, label='BLER_Target = 10%')
ax.plot(ml1_avg['SNR'], ml1_avg['RATE'], color='black', marker='*', markersize=6, linewidth=2, alpha=0.8, label='ML Empirical')
ax.plot(ota_b01_avg['SNR'], ota_b01_avg['RATE']/2, color='brown', marker='x', markersize=5, linewidth=2, alpha=0.8, label='OTA BLER 1%')
ax.plot(ota_b05_avg['SNR'], ota_b05_avg['RATE']/2, color=cyan, marker='+', markersize=5, linewidth=2, alpha=0.8, label='OTA BLER 5%')
ax.plot(ota_b10_avg['SNR'], ota_b10_avg['RATE']/2, color=magenta, marker='1', markersize=5, linewidth=2, alpha=0.8, label='OTA BLER 10%')
ax.plot(ota_no_olla_avg['SNR'], ota_no_olla_avg['RATE']/2, color=gray, marker='2', markersize=5, linewidth=2, alpha=0.8, label='OTA No OLLA')

ax.set_xlim([-8, 7])
ax.set_ylabel("Avg RATE (Mbps)", color='k')
ax.set_xlabel("SNR (dB)")

# Add grid with thinner lines
ax.grid(True, linewidth=0.5, alpha=0.7)

# Use the built-in legend from the plot labels
ax.legend(loc='upper left', title='Link Adaptation Algorithms', frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.show()
