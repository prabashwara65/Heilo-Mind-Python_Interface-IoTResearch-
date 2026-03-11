import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import joblib
import os

# ===============================
# Load Model + Scalers
# ===============================

model = tf.keras.models.load_model("model/solar_lstm.keras")
scaler_X = joblib.load("model/scaler_X.save")

print("Model + Scalers loaded ✅")


# ===============================
# Load ALL Regional CSV Files
# ===============================

required_cols = ["light_intensity", "temperature", "humidity"]

data_list = []
csv_files = glob.glob("data/*.csv")

for file in csv_files:
    try:
        print(f"Reading {os.path.basename(file)}")

        # Find header row automatically
        with open(file, "r") as f:
            lines = f.readlines()

        header_index = 0
        for i, line in enumerate(lines):
            if line.startswith("YEAR"):
                header_index = i
                break

        # Load CSV using detected header
        df = pd.read_csv(file, skiprows=header_index)

        # Rename columns to match model features
        df = df.rename(columns={
            "ALLSKY_SFC_SW_DWN": "light_intensity",
            "T2M": "temperature",
            "RH2M": "humidity"
        })

        # Check if required columns exist
        if not all(col in df.columns for col in required_cols):
            print(f"Skipping {file} due to missing columns")
            continue

        # Keep required columns
        df = df[required_cols]

        # Convert numeric
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna()

        data_list.append(df)

    except Exception as e:
        print(f"Skipping {file} due to error: {e}")

# Merge datasets
if len(data_list) == 0:
    raise Exception("No valid datasets found after cleaning!")

data = pd.concat(data_list, ignore_index=True)

print("Dataset loaded successfully ✅")
print("Total samples:", len(data))


# ===============================
# Preprocessing
# ===============================

features = data.values.astype("float32")

# Scale using training scaler
features_scaled = scaler_X.transform(features)


# ===============================
# Time Series Sequence Creation
# ===============================

def create_sequences(data, seq_length=24):
    sequences = []

    if len(data) < seq_length:
        raise Exception("Dataset too small for sequence creation")

    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])

    return np.array(sequences, dtype=np.float32)


sample_sequences = create_sequences(features_scaled)

# Randomly sample 100 windows (better calibration)
np.random.shuffle(sample_sequences)
sample_sequences = sample_sequences[:100]

print("Time-series calibration samples prepared ✅")


# ===============================
# Representative Dataset Generator
# ===============================

def representative_dataset():
    for i in range(len(sample_sequences)):
        yield {model.input_names[0]: sample_sequences[i:i+1]}


# ===============================
# TFLite Quantization Conversion
# ===============================

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Required for LSTM models
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

converter._experimental_lower_tensor_list_ops = False
converter.representative_dataset = representative_dataset

converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

print("Starting quantization conversion")

tflite_model = converter.convert()


# ===============================
# Save Quantized Model
# ===============================

os.makedirs("model", exist_ok=True)

with open("model/solar_lstm_quant.tflite", "wb") as f:
    f.write(tflite_model)

print("Quantized model saved successfully ✅")
print("Location: model/solar_lstm_quant.tflite")