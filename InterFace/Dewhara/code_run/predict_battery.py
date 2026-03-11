# =============================
# predict_battery.py (Keras 3 + Device Priority)
# =============================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# -------------------------------
# Import TFSMLayer for TF SavedModel
# -------------------------------
from keras.layers import TFSMLayer
from optimization.optimizer import BatteryOptimizer

# -------------------------------
# USER INPUT: LED COUNT
# -------------------------------
while True:
    try:
        led_count = int(input("Enter number of LEDs ON (1, 2, or 3): "))
        if led_count in [1, 2, 3]:
            break
        else:
            print("Please enter 1, 2, or 3.")
    except:
        print("Invalid input. Try again.")

LED_CURRENT_MAP = {1: 0.02, 2: 0.04, 3: 0.06}
selected_current = LED_CURRENT_MAP[led_count]

# -------------------------------
# CONSTANTS
# -------------------------------
BATTERY_CAPACITY_AH = 1.8  # 1800 mAh
TIME_STEPS = 20

# -------------------------------
# DEVICE PRIORITY SETUP
# -------------------------------
# Lower number = more critical
# load_weight: percentage of total current consumed by this device
devices = [
    {"name": "SmokeDetector", "priority": 1, "load_weight": 0.10},  # 10% load
    {"name": "LEDs",          "priority": 2, "load_weight": 0.30},  # 30% load
    {"name": "Fan",           "priority": 3, "load_weight": 0.20},  # 20% load
    {"name": "Heater",        "priority": 4, "load_weight": 0.40}   # 40% load
]

# Map device to GPIO pins (for future Raspberry Pi)
DEVICE_PINS = {"LEDs": 17, "Fan": 27, "Heater": 22, "SmokeDetector": 4}  # example pins

# -------------------------------
# LOAD MODEL & SCALER
# -------------------------------
import tensorflow as tf
model = tf.keras.models.load_model("models/battery_soc_model.keras")
scaler = joblib.load("models/scaler_led.pkl")

# -------------------------------
# LOAD DATA
# -------------------------------
soc_df = pd.read_csv("data/battery SOC.csv")
combined_df = pd.read_csv("data/combined.csv")

soc_df.columns = soc_df.columns.str.strip()
combined_df.columns = combined_df.columns.str.strip()

soc_df = soc_df.rename(columns={"Battery_SoC_%": "SOC"})
combined_df = combined_df.rename(columns={
    "Voltage_measured": "Voltage",
    "Current_load": "Current_load",
    "Time": "Time"
})

df = pd.concat(
    [combined_df.reset_index(drop=True), soc_df["SOC"].reset_index(drop=True)],
    axis=1
)

for col in ["Voltage", "Current_load", "Time", "SOC"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df.dropna(inplace=True)

# -------------------------------
# PREPARE FEATURES
# -------------------------------
features = df[["Voltage", "Current_load", "Time"]].values
features_scaled = scaler.transform(features)

# -------------------------------
# CREATE SEQUENCES
# -------------------------------
def create_sequences(X, time_steps=10):
    if len(X) <= time_steps:
        return np.array([])
    Xs = []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
    return np.array(Xs)

X_seq = create_sequences(features_scaled, TIME_STEPS)
if len(X_seq) == 0:
    raise ValueError(f"Not enough data to create sequences. Require > {TIME_STEPS} rows.")

# -------------------------------
# PREDICT SOC USING LOADED MODEL
# -------------------------------
print("Running battery SOC prediction...")
soc_pred_raw = model.predict(X_seq, batch_size=32).flatten()

# -------------------------------
# PHYSICS-BASED SOC, SOH, RUNTIME
# -------------------------------
time_values = df["Time"].values
# -------------------------------
# OPTIMIZER + SAFETY + DEVICE PRIORITY FUNCTIONS
# -------------------------------
optimizer = BatteryOptimizer()

def safety_validator(action, soc, soh, runtime_sec):
    if soc < 15 and action == "NORMAL_OPERATION":
        return "FORCE_SHED_LOAD"
    if soh < 70 and action == "NORMAL_OPERATION":
        return "LIMIT_CHARGE_RATE"
    if runtime_sec < 3600 and action == "NORMAL_OPERATION":
        return "DELAY_NON_ESSENTIAL_TASKS"
    return action

def apply_device_priority(soc):
    """
    Returns a list of device names that should be SHED (turned OFF) based on SOC.
    Tiered approach:
    - SOC < 50%: Shed Priority 4 (Heater)
    - SOC < 40%: Shed Priority 3 (Fan)
    - SOC < 30%: Shed Priority 2 (LEDs)
    - SOC < 20%: Keep only Priority 1 (Smoke Detector)
    """
    shed_list = []
    # Sort devices by priority (low priority first for shedding)
    devices_sorted = sorted(devices, key=lambda d: d["priority"], reverse=True)
    
    for device in devices_sorted:
        if device["priority"] == 1:
            continue # Never shed critical
            
        if device["priority"] == 4 and soc < 50:
            shed_list.append(device["name"])
        elif device["priority"] == 3 and soc < 40:
            shed_list.append(device["name"])
        elif device["priority"] == 2 and soc < 30:
            shed_list.append(device["name"])
        elif soc < 20: # Critical low
            shed_list.append(device["name"])
            
    return list(set(shed_list)) # unique list

# -------------------------------
# CORE SIMULATION: BASELINE VS OPTIMIZED
# -------------------------------
soc_baseline_list = []
soc_optimized_list = []
soh_list = []
runtime_base_list = []
runtime_opt_list = []
decisions = []
final_decisions = []

# Initial state
remaining_baseline_ah = (25 / 100) * BATTERY_CAPACITY_AH
remaining_optimized_ah = remaining_baseline_ah
initial_capacity = remaining_baseline_ah

print("\nRunning battery simulation (Baseline vs Optimized)...")

for i in range(len(soc_pred_raw) - 1):
    delta_t = max(time_values[i + 1] - time_values[i], 1)
    
    # 1. Baseline Simulation (Constant load)
    discharge_baseline = selected_current * (delta_t / 3600)
    remaining_baseline_ah = max(remaining_baseline_ah - discharge_baseline, 0)
    soc_baseline_val = (remaining_baseline_ah / BATTERY_CAPACITY_AH) * 100
    soc_baseline_list.append(soc_baseline_val)
    curr_runtime_base = (remaining_baseline_ah / selected_current) * 3600 if selected_current > 0 else 0
    
    # 2. Optimized Simulation (Reactive priority shedding)
    curr_soc_opt = (remaining_optimized_ah / BATTERY_CAPACITY_AH) * 100
    curr_soh_opt = (remaining_optimized_ah / initial_capacity) * 100 if initial_capacity > 0 else 0
    curr_runtime_opt = (remaining_optimized_ah / selected_current) * 3600 if selected_current > 0 else 0
    
    # Priority Based Shedding Decision
    devices_off = apply_device_priority(curr_soc_opt)
    
    # Calculate load reduction factor based on which devices are OFF
    factor = 1.0
    for dev in devices:
        if dev["name"] in devices_off:
            factor -= dev["load_weight"]
    factor = max(factor, 0.1) # Minimum 10% load for critical only
    
    discharge_optimized = (selected_current * factor) * (delta_t / 3600)
    remaining_optimized_ah = max(remaining_optimized_ah - discharge_optimized, 0)
    soc_opt_val = (remaining_optimized_ah / BATTERY_CAPACITY_AH) * 100
    
    # Get optimization label
    action = optimizer.decide_action(curr_soc_opt, curr_soh_opt, curr_runtime_opt)
    safe_action = safety_validator(action, curr_soc_opt, curr_soh_opt, curr_runtime_opt)
    final_text = f"Priority Shedding: {', '.join(devices_off)}" if devices_off else "NORMAL_OPERATION"
    
    # Log results
    soc_optimized_list.append(soc_opt_val)
    soh_list.append(curr_soh_opt)
    runtime_base_list.append(curr_runtime_base)
    runtime_opt_list.append(curr_runtime_opt)
    decisions.append(safe_action)
    final_decisions.append(final_text)

    # Stop if both paths empty
    if remaining_baseline_ah <= 0 and remaining_optimized_ah <= 0:
        break

soc_baseline = np.array(soc_baseline_list)
soc_optimized = np.array(soc_optimized_list)
soh = np.array(soh_list)
runtime_base = np.array(runtime_base_list)
runtime_opt = np.array(runtime_opt_list)

# Results are handled in the dual simulation above.

os.makedirs("results", exist_ok=True)
df_results = pd.DataFrame({
    "TimeStep": np.arange(len(soc_optimized)),
    "SOC_Baseline": soc_baseline,
    "SOC_Optimized": soc_optimized,
    "SoH": soh,
    "Runtime_Base_sec": runtime_base,
    "Runtime_Opt_sec": runtime_opt,
    "OptimizerDecision": decisions,
    "FinalDecision": final_decisions
})
df_results.to_csv("results/simulation_results.csv", index=False)
print("✅ Simulation results saved to results/simulation_results.csv")

if len(soc_optimized) > 0:
    print("\n========== FINAL BATTERY STATUS ==========")
    print(f"LEDs ON               : {led_count}")
    print(f"SOC (Baseline)        : {soc_baseline[-1]:.2f} %")
    print(f"SOC (Optimized)       : {soc_optimized[-1]:.2f} %")
    print(f"SOC Increase (Benefit): \033[92m{soc_optimized[-1] - soc_baseline[-1]:.2f} %\033[0m")
    print(f"Estimated SoH         : {soh[-1]:.2f} %")
    print(f"Runtime (Baseline)    : {runtime_base[-1] / 3600:.2f} hours")
    print(f"Runtime (Optimized)   : {runtime_opt[-1] / 3600:.2f} hours")
    print(f"Optimizer Decision    : {decisions[-1]}")
    print(f"Latest Decision       : {final_decisions[-1]}")
    print("==========================================\n")
else:
    print("⚠️ No SOC predictions available. Check your input data or TIME_STEPS.")