import pandas as pd
import numpy as np

def clean_snr_csv():
    # Read the CSV file
    df = pd.read_csv('/Users/ericforbes/Documents/GitHub/5G-NTN-Link-Adapation-Matlab/Matlab/snr_cqi_lut_dense.csv')
    
    # Round the SNR column to 1 decimal place
    df['snr'] = np.round(df['snr'], 1)
    
    # Save the cleaned data back to the file
    df.to_csv('/Users/ericforbes/Documents/GitHub/5G-NTN-Link-Adapation-Matlab/Matlab/snr_cqi_lut_dense_v2.csv', index=False)
    
    print("SNR values have been rounded to 1 decimal place")
    print(f"First few rows:")
    print(df.head(10))
    
    # Show unique SNR values to verify they're clean
    unique_snrs = sorted(df['snr'].unique())
    print(f"\nFirst 10 unique SNR values: {unique_snrs[:10]}")
    print(f"Last 10 unique SNR values: {unique_snrs[-10:]}")

if __name__ == "__main__":
    clean_snr_csv()