"""
Inference script with auto time features + manual entry for the rest.
Panel power is automatically calculated from voltage and current.
"""

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
import os
import datetime

# Suppress TensorFlow CPU instruction warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LOW_THRESHOLD = 0.2
HIGH_THRESHOLD = 0.6

def get_priority(prob):
    if prob > HIGH_THRESHOLD:
        return "HIGH (sync recommended)"
    elif prob >= LOW_THRESHOLD:
        return "MEDIUM (sync optional)"
    else:
        return "LOW (avoid sync)"

def load_model_and_scaler(model_path='models/sync_scheduler/', training_path='data/training/'):
    # Find model file
    model_file = os.path.join(model_path, 'sync_scheduler_final.keras')
    if not os.path.exists(model_file):
        model_file = os.path.join(model_path, 'best_model.keras')
    model = keras.models.load_model(model_file)
    print(f"✅ Model loaded from {model_file}")

    # Load scaler
    scaler_file = os.path.join(training_path, 'feature_scaler.pkl')
    scaler = joblib.load(scaler_file)
    print(f"✅ Scaler loaded from {scaler_file}")

    # Load feature names
    feature_names_file = os.path.join(training_path, 'feature_names.txt')
    if os.path.exists(feature_names_file):
        with open(feature_names_file, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        print(f"✅ Feature names loaded: {feature_names}")
    else:
        feature_names = None
        print("⚠️ feature_names.txt not found. Will use model input shape.")

    return model, scaler, feature_names

def get_current_time_features():
    now = datetime.datetime.now()
    hour = now.hour
    month = now.month
    is_daytime = 1 if 6 <= hour < 18 else 0
    return hour, month, is_daytime

def predict_single(model, scaler, feature_dict, feature_names):
    df = pd.DataFrame([feature_dict])[feature_names]
    X_scaled = scaler.transform(df)
    prob = model.predict(X_scaled, verbose=0)[0, 0]
    return prob, get_priority(prob)

def main():
    model, scaler, feature_names = load_model_and_scaler()
    if feature_names is None:
        # Fallback: full manual entry without auto‑fill
        n_features = model.input_shape[1]
        print(f"Model expects {n_features} features. Please enter them in order:")
        values = []
        for i in range(n_features):
            val = float(input(f"Feature {i+1}: "))
            values.append(val)
        sample = {f"f{i}": v for i, v in enumerate(values)}
        feature_names = list(sample.keys())
    else:
        # Auto‑fill time features
        hour, month, is_daytime = get_current_time_features()
        print(f"\n🕒 Current system time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   → Auto‑set: hour = {hour}, month = {month}, is_daytime = {is_daytime}")

        sample = {
            'hour': float(hour),
            'month': float(month),
            'is_daytime': float(is_daytime)
        }

        # Decide whether to ask for panel_power or compute it
        ask_for_panel_power = True
        if ('panel_voltage' in feature_names and 
            'panel_current' in feature_names and 
            'panel_power' in feature_names):
            # We can compute panel_power from voltage and current
            ask_for_panel_power = False

        # Build list of remaining features to ask
        remaining = [f for f in feature_names if f not in sample]
        if not ask_for_panel_power:
            # Remove panel_power from the list – we'll compute it later
            remaining = [f for f in remaining if f != 'panel_power']

        print(f"\nPlease enter the remaining {len(remaining)} features:")
        for feat in remaining:
            while True:
                try:
                    val = float(input(f"  {feat}: "))
                    sample[feat] = val
                    break
                except ValueError:
                    print("    Please enter a valid number.")

        # Automatically calculate panel_power if possible
        if ('panel_voltage' in sample and 'panel_current' in sample and 
            'panel_power' in feature_names):
            sample['panel_power'] = sample['panel_voltage'] * sample['panel_current']
            print(f"   → Auto‑calculated panel_power = {sample['panel_power']:.2f}")

    # Predict
    prob, priority = predict_single(model, scaler, sample, feature_names)

    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"Input features: {sample}")
    print(f"Predicted sync probability: {prob:.4f}")
    print(f"Priority: {priority}")
    print("="*60)

if __name__ == "__main__":
    main()