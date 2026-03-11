import pandas as pd
import glob
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

PANEL_AREA = 6.5
EFFICIENCY = 0.18

def load_nasa_csv(path):
    # Find the correct header row dynamically
    with open(path, "r") as f:
        lines = f.readlines()

    header_row = None
    for i, line in enumerate(lines):
        if line.startswith("YEAR"):
            header_row = i
            break

    if header_row is None:
        raise ValueError("Header row not found in file: " + path)

    # Read CSV from the correct header row
    df = pd.read_csv(path, skiprows=header_row)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Select required columns
    df = df[
        ["ALLSKY_SFC_SW_DWN", "T2M", "RH2M"]
    ]

    # Rename
    df.columns = [
        "light_intensity",
        "temperature",
        "humidity"
    ]

    # Convert light intensity → energy (kWh)
    PANEL_AREA = 6.5
    EFFICIENCY = 0.18

    df["energy_kwh"] = (
        df["light_intensity"] * PANEL_AREA * EFFICIENCY
    ) / 1000

    return df

# Load all provinces
dfs = [load_nasa_csv(f) for f in glob.glob("data/*.csv")]
data = pd.concat(dfs, ignore_index=True)

features = ["light_intensity", "temperature", "humidity"]
target = "energy_kwh"

# Scale inputs and output
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

data[features] = scaler_X.fit_transform(data[features])
data[[target]] = scaler_y.fit_transform(data[[target]])

joblib.dump(scaler_X, "model/scaler_X.save")
joblib.dump(scaler_y, "model/scaler_y.save")


def create_sequences(data, past=24, future=24):
    X, y = [], []
    values = data.values

    for i in range(len(values) - past - future):
        X.append(values[i:i+past, :3])          # inputs
        y.append(values[i+past:i+past+future, 3])  # energy

    return np.array(X), np.array(y)


X, y = create_sequences(data)

np.save("X.npy", X)
np.save("y.npy", y)

print("X shape:", X.shape)  # (samples, 24, 3)
print("y shape:", y.shape)  # (samples, 24)

