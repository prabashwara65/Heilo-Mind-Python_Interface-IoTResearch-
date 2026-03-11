import pandas as pd
import numpy as np
import os

# -------------------------------
# 1. LOAD CLEANED DATA
# -------------------------------
print("Loading source data...")
soc_df = pd.read_csv("data/battery SOC_cleaned.csv")
combined_df = pd.read_csv("data/combined_cleaned.csv")

# Ensure they are aligned (should already be from scrub_data.py)
min_rows = min(len(soc_df), len(combined_df))
data = combined_df.iloc[:min_rows].copy()
data['SOC'] = soc_df['Battery_SoC_%'].iloc[:min_rows].values

print(f"Source data has {len(data)} rows.")

# -------------------------------
# 2. GENERATE SYNTHETIC DATA
# -------------------------------
# We need to go from 1,000 to 50,000 rows.
# Technique: Interpolation with slight Gaussian noise.
TARGET_ROWS = 50000
print(f"Generating {TARGET_ROWS} synthetic rows...")

# Create an expanded index for interpolation
old_indices = np.linspace(0, len(data) - 1, len(data))
new_indices = np.linspace(0, len(data) - 1, TARGET_ROWS)

synthetic_data = pd.DataFrame()

# Loop through columns ensuring consistency with training features
cols_to_gen = ['Voltage_measured', 'Current_load', 'Time', 'SOC']

for col in cols_to_gen:
    if col not in data.columns:
        print(f"Skipping missing column: {col}")
        continue
        
    # Interpolate
    interp_values = np.interp(new_indices, old_indices, data[col])
    
    # Noise calculation: 0.1% of the standard deviation
    noise_std = data[col].std() * 0.005 # 0.5% noise for variability
    noise = np.random.normal(0, noise_std, TARGET_ROWS)
    
    # Don't add noise to 'Time' as much to keep it monotonically increasing
    if col == 'Time':
        synthetic_data[col] = interp_values
    elif col == 'SOC':
        # Smooth SOC to prevent sudden jumps
        synthetic_data[col] = interp_values + (noise * 0.05)
    else:
        synthetic_data[col] = interp_values + noise

# Bounds check
synthetic_data['SOC'] = synthetic_data['SOC'].clip(0, 100)
synthetic_data['Voltage_measured'] = synthetic_data['Voltage_measured'].clip(lower=0)

# Ensure 'Time' is monotonically increasing
synthetic_data['Time'] = np.sort(synthetic_data['Time'].values)

# -------------------------------
# 3. SAVE SYNTHETIC DATA
# -------------------------------
output_path = "data/synthetic_battery_data.csv"
synthetic_data.to_csv(output_path, index=False)
print(f"✅ Synthetic data saved to {output_path}")
print(f"New dataset shape: {synthetic_data.shape}")
print(f"Columns: {synthetic_data.columns.tolist()}")
