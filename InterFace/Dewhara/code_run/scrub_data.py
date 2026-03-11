import pandas as pd
import numpy as np

def clean_file(path, numeric_cols):
    print(f"Scrubbing {path}...")
    # Read file
    df = pd.read_csv(path)
    
    # Strip column names
    df.columns = df.columns.str.strip()
    
    # Filter out header repetitions (if any data row contains the column name)
    # Move only on numeric columns to avoid issues with strings
    for col in numeric_cols:
        if col in df.columns:
            # Drop rows where this column is not a number
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows that have NaN in any of the ESSENTIAL numeric columns
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    
    return df

# Scrub the files properly
soc_df = clean_file("data/battery SOC.csv", ["Battery_SoC_%"])
combined_df = clean_file("data/combined.csv", ["Voltage_measured", "Current_load", "Time"])

# Align the datasets
min_rows = min(len(soc_df), len(combined_df))
print(f"Aligning to {min_rows} rows...")
soc_df = soc_df.iloc[:min_rows]
combined_df = combined_df.iloc[:min_rows]

# Save
soc_df.to_csv("data/battery SOC_cleaned.csv", index=False)
combined_df.to_csv("data/combined_cleaned.csv", index=False)

print(f"\n✅ RE-SCRUBBED! SOC rows: {len(soc_df)}, Combined rows: {len(combined_df)}")
