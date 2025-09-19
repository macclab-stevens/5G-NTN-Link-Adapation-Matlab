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

# Load data
data_dir = Path('Data')
b = pd.read_csv(data_dir / 'Archive' / 'Result_baseline.csv')
w20001 = pd.read_csv(data_dir / 'Case9_MCS_ThroughputCalulation_BLERw2000Tbler0.001_240608_153903.csv')
w2001 = pd.read_csv(data_dir / 'Case9_MCS_ThroughputCalulation_BLERw2000Tbler0.01_240603_133217.csv')
w2005 = pd.read_csv(data_dir / 'Case9_MCS_ThroughputCalulation_BLERw2000Tbler0.05_240604_085504.csv')
w2010 = pd.read_csv(data_dir / 'Case9_MCS_ThroughputCalulation_BLERw2000Tbler0.1_240605_171830.csv')
ml1 = pd.read_csv(data_dir / 'ML_Case9_MCS_ThroughputCalulation_snr_cqi_lut_empirical.csv')

# Add rate columns to all datasets
b = add_rate_column(b)
w20001 = add_rate_column(w20001)
w2001 = add_rate_column(w2001)
w2005 = add_rate_column(w2005)
w2010 = add_rate_column(w2010)
ml1 = add_rate_column(ml1)

# Define colors (matching MATLAB defaults)
blue = [0, 0.4470, 0.7410]
orange = [0.8500, 0.3250, 0.0980]
green = [0.4660, 0.6740, 0.1880]
purple = [0.4940, 0.1840, 0.5560]
red = [0.6350, 0.0780, 0.1840]

# Create figure with dual y-axis
fig, ax1 = plt.subplots(figsize=(10, 8))
ax2 = ax1.twinx()

# BLER plotting (left y-axis)
lag_bler = 1500

# Calculate moving averages for BLER
wavg200b = moving_average(b['BLER'], lag_bler)
wavg20001 = moving_average(w20001['BLER'], lag_bler)
wavg2001 = moving_average(w2001['BLER'], lag_bler)
wavg2005 = moving_average(w2005['BLER'], lag_bler)
wavg2010 = moving_average(w2010['BLER'], lag_bler)
wavgml1 = moving_average(ml1['BLER'], lag_bler)

# Scatter plots for BLER with smaller point size
ax1.scatter(b['eleAnge'], wavg200b, s=0.5, c=[blue], alpha=0.7)
ax1.scatter(w20001['eleAnge'], wavg20001, s=0.5, c=[orange], alpha=0.7)
ax1.scatter(w2001['eleAnge'], wavg2001, s=0.5, c=[green], alpha=0.7)
ax1.scatter(w2005['eleAnge'], wavg2005, s=0.5, c=[purple], alpha=0.7)
ax1.scatter(w2010['eleAnge'], wavg2010, s=0.5, c=[red], alpha=0.7)
ax1.scatter(ml1['eleAnge'], wavgml1, s=0.5, c='black', alpha=0.7)

ax1.set_ylim([0, 0.3])
ax1.set_xlim([0.5, 90])
ax1.set_ylabel("Avg BLER", color='k')
ax1.set_xlabel("Elevation Angle")

# RATE plotting (right y-axis)
lag_rate = 5000

# Calculate moving averages for RATE
ravg200b = moving_average(b['RATE'], lag_rate)
ravg20001 = moving_average(w20001['RATE'], lag_rate)
ravg2001 = moving_average(w2001['RATE'], lag_rate)
ravg2005 = moving_average(w2005['RATE'], lag_rate)
ravg2010 = moving_average(w2010['RATE'], lag_rate)
ravgml1 = moving_average(ml1['RATE'], lag_rate)

# Scatter plots for RATE with smaller point size
ax2.scatter(b['eleAnge'], ravg200b, s=0.5, c=[blue], alpha=0.7)
ax2.scatter(w20001['eleAnge'], ravg20001, s=0.5, c=[orange], alpha=0.7)
ax2.scatter(w2001['eleAnge'], ravg2001, s=0.5, c=[green], alpha=0.7)
ax2.scatter(w2005['eleAnge'], ravg2005, s=0.5, c=[purple], alpha=0.7)
ax2.scatter(w2010['eleAnge'], ravg2010, s=0.5, c=[red], alpha=0.7)
ax2.scatter(ml1['eleAnge'], ravgml1, s=0.5, c='black', alpha=0.7)

ax2.set_ylabel("Avg RATE (kbps)", color='k')

# Add grid with thinner lines
ax1.grid(True, linewidth=0.5)

# Legend
legend_labels = ['Baseline', 'BLER_Target = 0.1%', 'BLER_Target = 1%', 'BLER_Target = 5%', 'BLER_Target = 10%', 'ML Empirical']
legend_colors = [blue, orange, green, purple, red, 'black']

# Create custom legend handles with smaller marker size
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=6) 
                 for color in legend_colors]

ax1.legend(legend_handles, legend_labels, loc='upper left', title='BLER Window = 2000')

plt.tight_layout()
plt.show()
