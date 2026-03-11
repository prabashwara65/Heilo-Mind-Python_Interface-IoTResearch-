import numpy as np
import tensorflow as tf
import joblib
import time
import psutil
import os
import matplotlib.pyplot as plt

# ===============================
# Load Keras Model
# ===============================

model = tf.keras.models.load_model("model/solar_lstm.keras")

# ===============================
# Load Scalers
# ===============================

scaler_X = joblib.load("model/scaler_X.save")
scaler_y = joblib.load("model/scaler_y.save")

# ===============================
# Sample Input Data
# ===============================

recent_data = np.array([
    [0.00, 0.55, 0.85],
    [0.00, 0.54, 0.86],
    [0.00, 0.53, 0.87],
    [0.00, 0.52, 0.88],
    [0.00, 0.51, 0.89],
    [0.00, 0.50, 0.90],

    [0.15, 0.55, 0.85],
    [0.30, 0.60, 0.80],
    [0.50, 0.65, 0.75],
    [0.70, 0.70, 0.70],

    [0.90, 0.75, 0.65],
    [1.00, 0.78, 0.60],
    [0.95, 0.77, 0.62],

    [0.80, 0.74, 0.65],
    [0.60, 0.70, 0.70],
    [0.40, 0.65, 0.75],

    [0.20, 0.60, 0.80],
    [0.05, 0.58, 0.82],

    [0.00, 0.55, 0.85],
    [0.00, 0.54, 0.86],
    [0.00, 0.53, 0.87],
    [0.00, 0.52, 0.88],
    [0.00, 0.51, 0.89],
    [0.00, 0.50, 0.90],
])

recent_data = recent_data.reshape(1, 24, 3)

# Scale input
recent_data_scaled = scaler_X.transform(
    recent_data.reshape(24,3)
).reshape(1,24,3)

# ===============================
# Performance Measurement
# ===============================

process = psutil.Process(os.getpid())

memory_before = process.memory_info().rss / (1024*1024)

times = []

for _ in range(50):

    start = time.time()

    pred_norm = model.predict(
        recent_data_scaled,
        verbose=0
    )

    times.append((time.time() - start) * 1000)

inference_time = np.mean(times)

memory_after = process.memory_info().rss / (1024*1024)

cpu_usage = psutil.cpu_percent()
ram_usage = memory_after

model_size = os.path.getsize(
    "model/solar_lstm.keras"
) / (1024*1024)

print("\n===== Keras Model Performance =====")

print("Model Size:", model_size, "MB")
print("Inference Time:", inference_time, "ms")
print("CPU Usage:", cpu_usage, "%")
print("RAM Usage:", ram_usage, "MB")

# ===============================
# Prediction Output
# ===============================

energy_kwh = scaler_y.inverse_transform(
    pred_norm.reshape(-1,1)
).flatten()

irradiance = recent_data[0,:,0].flatten()

# Physics constraints
energy_kwh[irradiance == 0] = 0
energy_kwh[energy_kwh < 0] = 0

print("\nHour | Energy (kWh)")
print("------------------")

total_energy = 0

for i in range(24):

    total_energy += energy_kwh[i]

    print(f"{i+1:02d} | {energy_kwh[i]:.3f}")

print(f"\nTotal 24h Energy: {total_energy:.2f} kWh")

