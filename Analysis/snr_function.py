"""
snr_function.py
Generates a curve fit for SNR as a function of elevation angle (eleAngle).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# TODO: Set the correct path to your data file containing SNR and eleAngle columns
DATA_PATH = 'Data/Case9_MCS_ThroughputCalulation_BLERw10Tbler0.01_240531_210830.csv'  # Update if needed

def model(ele_angle, a, b, c):
    """Example model: quadratic fit. Adjust as needed."""
    return a * ele_angle**2 + b * ele_angle + c

def fit_snr_vs_eleangle(data_path=DATA_PATH):
    """
    Fits SNR as a function of elevation angle from a CSV file.
    Returns: fit parameters (a, b, c) and the model function.
    Also prints the fitted function equation.
    """
    df = pd.read_csv(data_path)
    snr_col = next((col for col in df.columns if 'snr' in col.lower()), None)
    ele_col = next((col for col in df.columns if 'ele' in col.lower()), None)
    if snr_col is None or ele_col is None:
        raise ValueError('Could not find SNR or eleAngle columns in the data file.')
    x = df[ele_col].values
    y = df[snr_col].values
    popt, _ = curve_fit(model, x, y)
    print(f"Fitted SNR function: SNR = {popt[0]:.4f} * eleAngle^2 + {popt[1]:.4f} * eleAngle + {popt[2]:.4f}")
    return popt, lambda ele_angle: model(ele_angle, *popt)

def main():
    popt, snr_func = fit_snr_vs_eleangle(DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    snr_col = next((col for col in df.columns if 'snr' in col.lower()), None)
    ele_col = next((col for col in df.columns if 'ele' in col.lower()), None)
    x = df[ele_col].values
    y = df[snr_col].values
    y_fit = snr_func(x)
    plt.figure(figsize=(8,5))
    plt.scatter(x, y, label='Data', color='blue')
    plt.plot(x, y_fit, label='Fit', color='red')
    plt.xlabel('Elevation Angle')
    plt.ylabel('SNR')
    plt.title('SNR vs Elevation Angle with Curve Fit')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f'Fit parameters: a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.4f}')

if __name__ == '__main__':
    main()
