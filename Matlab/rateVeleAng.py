#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_data():
    """Load all the CSV files"""
    data_dir = Path('Matlab')
    
    # Load the new ML data
    ml_data = pd.read_csv(data_dir / 'ML_Case9_MCS_ThroughputCalulation_snr_cqi_lut_empirical.csv')
    
    # Load existing window size data
    datasets = {}
    files = [
        ('baseline', 'Case9MCSThroughputCalulation240524212808.csv'),
        ('w10', 'Case9MCSThroughputCalulationBLERw10Tbler0.csv'),
        ('w50', 'Case9MCSThroughputCalulationBLERw50Tbler0.csv'),
        ('w100', 'Case9MCSThroughputCalulationBLERw100Tbler0.csv'),
        ('w200', 'Case9MCSThroughputCalulationBLERw200Tbler0.csv'),
        ('w1000', 'Case9MCSThroughputCalulationBLERw1000Tbler0.csv'),
        ('w2000', 'Case9MCSThroughputCalulationBLERw2000Tbler0.csv')
    ]
    
    for name, filename in files:
        try:
            datasets[name] = pd.read_csv(data_dir / filename)
        except FileNotFoundError:
            print(f"Warning: {filename} not found, skipping...")
    
    return ml_data, datasets

def moving_average(data, window):
    """Calculate moving average"""
    return pd.Series(data).rolling(window=window, min_periods=1).mean()

def plot_mcs_vs_elevation(ml_data, datasets):
    """Plot MCS vs Elevation Angle"""
    plt.figure(figsize=(12, 8))
    
    # Plot ML data
    plt.scatter(ml_data['eleAnge'], ml_data['MCS'], alpha=0.6, s=1, 
               label='ML Empirical', color='red')
    
    # Plot existing datasets if they have MCS column
    colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
    labels = ['Baseline', 'Window=10', 'Window=50', 'Window=100', 'Window=200', 'Window=1000', 'Window=2000']
    
    for i, (name, data) in enumerate(datasets.items()):
        if 'MCS' in data.columns:
            plt.scatter(data['eleAnge'], data['MCS'], alpha=0.6, s=1,
                       color=colors[i], label=labels[i])
    
    plt.xlabel('Elevation Angle (degrees)')
    plt.ylabel('MCS')
    plt.title('MCS vs Elevation Angle')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0.5, 90)
    plt.tight_layout()
    plt.show()

def plot_rate_vs_elevation_comparison(ml_data, datasets):
    """Plot Rate vs Elevation Angle - comparison with existing data"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Colors and labels
    colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
    labels = ['Baseline', 'Window=10', 'Window=50', 'Window=100', 'Window=200', 'Window=1000', 'Window=2000']
    
    # Top plot: BLER vs Elevation
    lag_bler = 2000
    for i, (name, data) in enumerate(datasets.items()):
        if 'BLER' in data.columns:
            bler_avg = moving_average(data['BLER'], lag_bler)
            ax1.scatter(data['eleAnge'], bler_avg, s=1, color=colors[i], 
                       label=labels[i], alpha=0.7)
    
    ax1.set_ylabel('Avg BLER', color='black')
    ax1.set_ylim(0, 0.3)
    ax1.set_xlim(0.5, 90)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.set_title('BLER vs Elevation Angle (Target BLER = 1%)')
    
    # Bottom plot: Rate vs Elevation (including ML data)
    ax2_twin = ax2.twinx()
    
    # Plot ML data rate
    if 'RATE' in ml_data.columns:
        rate_avg_ml = moving_average(ml_data['RATE'], 5000)
        ax2.scatter(ml_data['eleAnge'], rate_avg_ml, s=1, color='red', 
                   label='ML Empirical', alpha=0.7)
    
    # Plot existing datasets
    lag_rate = 5000
    for i, (name, data) in enumerate(datasets.items()):
        if 'RATE' in data.columns:
            rate_avg = moving_average(data['RATE'], lag_rate)
            ax2.scatter(data['eleAnge'], rate_avg, s=1, color=colors[i], 
                       label=labels[i], alpha=0.7)
    
    ax2.set_xlabel('Elevation Angle (degrees)')
    ax2.set_ylabel('Avg Rate (bps)', color='black')
    ax2.set_ylim(-5e6, 12e6)
    ax2.set_xlim(0.5, 90)
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    ax2.legend(lines1, labels1, loc='upper left')
    
    plt.tight_layout()
    plt.show()

def plot_ml_data_only(ml_data):
    """Plot just the ML data - MCS and Rate vs Elevation"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # MCS vs Elevation
    ax1.scatter(ml_data['eleAnge'], ml_data['MCS'], alpha=0.6, s=1, color='red')
    ax1.set_xlabel('Elevation Angle (degrees)')
    ax1.set_ylabel('MCS')
    ax1.set_title('ML Empirical: MCS vs Elevation Angle')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, 90)
    
    # Rate vs Elevation (with moving average)
    if 'RATE' in ml_data.columns:
        rate_avg = moving_average(ml_data['RATE'], 1000)
        ax2.scatter(ml_data['eleAnge'], rate_avg, alpha=0.6, s=1, color='blue')
        ax2.set_xlabel('Elevation Angle (degrees)')
        ax2.set_ylabel('Avg Rate (bps)')
        ax2.set_title('ML Empirical: Rate vs Elevation Angle')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0.5, 90)
    
    plt.tight_layout()
    plt.show()

def main():
    # Load all data
    ml_data, datasets = load_data()
    
    print(f"ML data shape: {ml_data.shape}")
    print(f"ML data columns: {ml_data.columns.tolist()}")
    print(f"Loaded {len(datasets)} existing datasets")
    
    # Plot ML data only
    print("\nPlotting ML data...")
    plot_ml_data_only(ml_data)
    
    # Plot MCS comparison
    print("\nPlotting MCS comparison...")
    plot_mcs_vs_elevation(ml_data, datasets)
    
    # Plot rate comparison (replicating MATLAB script)
    print("\nPlotting rate comparison...")
    plot_rate_vs_elevation_comparison(ml_data, datasets)

if __name__ == "__main__":
    main()