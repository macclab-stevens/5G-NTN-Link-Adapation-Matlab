#!/usr/bin/env python3
"""
Compare CQI Table 3 and MCS Table 2 from 3GPP TS 38.214
Generates plots comparing indices and spectral efficiencies
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV files
cqi_table = pd.read_csv('TS38.214Tab5.2.2.1-4-CQI-Table-3.csv')
mcs_table = pd.read_csv('TS38.214-Table-5.1.3.1-2-MCS-Table-2-PDSCH.csv')

# Create comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Spectral Efficiency comparison
ax1.plot(cqi_table['CQI_Index'].values, cqi_table['Efficiency'].values, 'bo-', label='CQI Table 3', linewidth=2, markersize=6)
ax1.plot(mcs_table['MCS_Index'].values, mcs_table['Spectral_Efficiency'].values, 'ro-', label='MCS Table 2', linewidth=2, markersize=6)
ax1.set_xlabel('Index')
ax1.set_ylabel('Spectral Efficiency (bits/symbol)')
ax1.set_title('Spectral Efficiency Comparison: CQI vs MCS Tables')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Modulation Order comparison
ax2.bar(cqi_table['CQI_Index'].values - 0.2, cqi_table['Modulation'].values, width=0.4, label='CQI Table 3', alpha=0.7)
ax2.bar(mcs_table['MCS_Index'].values + 0.2, mcs_table['Modulation_Order'].values, width=0.4, label='MCS Table 2', alpha=0.7)
ax2.set_xlabel('Index')
ax2.set_ylabel('Modulation Order (Qm)')
ax2.set_title('Modulation Order Comparison: CQI vs MCS Tables')
ax2.set_xticks(range(0, max(len(cqi_table), len(mcs_table))))
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Code Rate comparison
ax3.plot(cqi_table['CQI_Index'].values, cqi_table['Code_Rate_x1024'].values, 'bo-', label='CQI Table 3', linewidth=2, markersize=6)
ax3.plot(mcs_table['MCS_Index'].values, mcs_table['Target_Code_Rate_x1024'].values, 'ro-', label='MCS Table 2', linewidth=2, markersize=6)
ax3.set_xlabel('Index')
ax3.set_ylabel('Code Rate × 1024')
ax3.set_title('Code Rate Comparison: CQI vs MCS Tables')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Efficiency difference between tables (where indices overlap)
common_indices = min(len(cqi_table), len(mcs_table))
cqi_subset = cqi_table.iloc[:common_indices]
mcs_subset = mcs_table.iloc[:common_indices]

efficiency_diff = mcs_subset['Spectral_Efficiency'].values - cqi_subset['Efficiency'].values
ax4.bar(range(common_indices), efficiency_diff, alpha=0.7, color='green')
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax4.set_xlabel('Index (0-14)')
ax4.set_ylabel('Efficiency Difference (MCS - CQI)')
ax4.set_title('Spectral Efficiency Difference: MCS - CQI')
ax4.grid(True, alpha=0.3)

# Add text annotation for the difference plot
ax4.text(0.02, 0.98, f'Mean difference: {np.mean(efficiency_diff):.4f}', 
         transform=ax4.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('CQI_MCS_Tables_Comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n📊 TABLE COMPARISON SUMMARY:")
print(f"CQI Table 3:")
print(f"  - Indices: {cqi_table['CQI_Index'].min()} to {cqi_table['CQI_Index'].max()}")
print(f"  - Efficiency range: {cqi_table['Efficiency'].min():.4f} to {cqi_table['Efficiency'].max():.4f}")
print(f"  - Modulation orders: {sorted(cqi_table['Modulation'].unique())}")

print(f"\nMCS Table 2:")
print(f"  - Indices: {mcs_table['MCS_Index'].min()} to {mcs_table['MCS_Index'].max()}")
print(f"  - Efficiency range: {mcs_table['Spectral_Efficiency'].min():.4f} to {mcs_table['Spectral_Efficiency'].max():.4f}")
print(f"  - Modulation orders: {sorted(mcs_table['Modulation_Order'].unique())}")

print(f"\nOverlap Analysis (indices 0-14):")
print(f"  - Mean efficiency difference (MCS - CQI): {np.mean(efficiency_diff):.4f}")
print(f"  - Max efficiency difference: {np.max(efficiency_diff):.4f}")
print(f"  - Min efficiency difference: {np.min(efficiency_diff):.4f}")

print(f"\n📈 Plot saved as: CQI_MCS_Tables_Comparison.png")
