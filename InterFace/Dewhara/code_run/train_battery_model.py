# -------------------------------
# Battery SOC, SoH, and Runtime Prediction for LED Load
# Fully Updated with TFLite Conversion
# -------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import joblib

# -------------------------------
# Step 1: Load Real and Synthetic Data
# -------------------------------
print("Loading datasets...")
# Load Real Cleaned Data
soc_real = pd.read_csv("data/battery SOC_cleaned.csv")
combined_real = pd.read_csv("data/combined_cleaned.csv")

# Load Synthetic Data
synthetic_df = pd.read_csv("data/synthetic_battery_data.csv")

# -------------------------------
# Step 2: Strip and Rename Columns
# -------------------------------
soc_real.columns = soc_real.columns.str.strip()
combined_real.columns = combined_real.columns.str.strip()
synthetic_df.columns = synthetic_df.columns.str.strip()

# Map columns for Real Data
soc_real = soc_real.rename(columns={"Battery_SoC_%": "SOC"})
combined_real = combined_real.rename(columns={
    "Voltage_measured": "Voltage",
    "Current_load": "Current_load",
    "Time": "Time"
})

# Map columns for Synthetic Data (Ensure match)
synthetic_df = synthetic_df.rename(columns={
    "Voltage_measured": "Voltage",
    "Current_load": "Current_load",
    "Time": "Time",
    "SOC": "SOC"
})

# -------------------------------
# Step 3: Combine Real into one DF
# -------------------------------
min_real = min(len(soc_real), len(combined_real))
real_df = pd.concat([
    combined_real[['Voltage', 'Current_load', 'Time']].iloc[:min_real].reset_index(drop=True),
    soc_real['SOC'].iloc[:min_real].reset_index(drop=True)
], axis=1)

# -------------------------------
# Step 4: Merge Real + Synthetic
# -------------------------------
print(f"Merging {len(real_df)} real rows with {len(synthetic_df)} synthetic rows...")
merged_df = pd.concat([real_df, synthetic_df], axis=0).reset_index(drop=True)
print(f"Total training samples: {len(merged_df)}")

# -------------------------------
# Step 5: Convert relevant columns to numeric safely
# -------------------------------
numeric_cols = ['Voltage', 'Current_load', 'Time', 'SOC', 'Capacity']
for col in numeric_cols:
    if col in merged_df.columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

merged_df = merged_df.dropna(subset=['Voltage','Current_load','Time','SOC'])

# -------------------------------
# Step 6: Preprocess features
# -------------------------------
features = merged_df[['Voltage', 'Current_load', 'Time']].values
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

target_soc = merged_df['SOC'].values

# -------------------------------
# Step 7: Create sequences for LSTM
# -------------------------------
def create_sequences(X, y, time_steps=20):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 20
X_seq, y_seq = create_sequences(features_scaled, target_soc, time_steps)

# -------------------------------
# Step 8: Split train/test sets
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=False
)

# -------------------------------
# Step 9: Build Enhanced LSTM Model
# -------------------------------
from tensorflow.keras.layers import Dropout

model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.1),
    LSTM(32, return_sequences=False),
    Dropout(0.1),
    Dense(16, activation='relu'),
    Dense(1)  # Output: SOC
])

# Use Huber loss for more robust training against noisy sensor data
model.compile(optimizer='adam', loss='huber', metrics=['mae'])

# -------------------------------
# Step 10: Robust Training with Callbacks
# -------------------------------
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

history = model.fit(
    X_train, y_train,
    epochs=200,  # More efficient with early stopping
    batch_size=32,
    validation_split=0.2,
    shuffle=False,
    callbacks=[early_stop, reduce_lr]
)

# -------------------------------
# Step 11: Evaluate Model
# -------------------------------
loss, mae = model.evaluate(X_test, y_test)
print("Test MAE:", mae)

# -------------------------------
# Step 12: SOC Predictions
# -------------------------------
y_pred_soc = model.predict(X_test)

# -------------------------------
# Step 13: Compute SoH from SOC
# -------------------------------
if 'Capacity' in merged_df.columns:
    max_capacity = merged_df['Capacity'].max()
    y_pred_soh = (y_pred_soc / max_capacity) * 100
else:
    y_pred_soh = y_pred_soc.copy()  # fallback

# -------------------------------
# Step 14: Estimate Battery Runtime (Accurate Integration)
# -------------------------------
cutoff_soc = 0  # SOC at which battery is considered empty
y_pred_runtime = []

for i in range(len(y_pred_soc)):
    remaining_soc = max(y_pred_soc[i] - cutoff_soc, 0)
    # Get original current load and time from input sequence
    original_seq = scaler.inverse_transform(X_test[i])  # Voltage, Current_load, Time
    current_load_seq = original_seq[:,1]
    time_seq = original_seq[:,2]

    runtime_sec = 0
    for j in range(len(current_load_seq)-1):
        if remaining_soc <= 0:
            break
        delta_t = time_seq[j+1] - time_seq[j]
        runtime_sec += delta_t * (remaining_soc / max(y_pred_soc[i], 1e-5))
    y_pred_runtime.append(runtime_sec)

y_pred_runtime = np.array(y_pred_runtime)

# -------------------------------
# Step 15: Plot Predictions
# -------------------------------
plt.figure(figsize=(12,5))
plt.plot(y_test[:200], label='Actual SOC')
plt.plot(y_pred_soc[:200], label='Predicted SOC')
plt.legend()
plt.title("Battery SOC Prediction for LED Load")
plt.show()

plt.figure(figsize=(12,5))
plt.plot(y_pred_soh[:200], label='Predicted SoH')
plt.legend()
plt.title("Battery SoH Prediction")
plt.show()

plt.figure(figsize=(12,5))
plt.plot(y_pred_runtime[:200], label='Predicted Runtime (s)')
plt.legend()
plt.title("Battery Runtime Prediction")
plt.show()

# -------------------------------
# Step 16: Save Model and Scaler
# -------------------------------
model.save("models/battery_soc_led_model")
joblib.dump(scaler, "models/scaler_led.pkl")
print("Keras model and scaler saved!")

# -------------------------------
# Step 17: Convert Keras Model to TFLite
# -------------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # optional quantization
tflite_model = converter.convert()

tflite_model_path = "models/battery_soc_led_model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved to {tflite_model_path}")
