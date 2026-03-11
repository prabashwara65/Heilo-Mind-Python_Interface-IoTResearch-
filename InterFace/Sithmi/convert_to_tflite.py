import tensorflow as tf
import os

# ===============================
# Paths
# ===============================
keras_model_path = "model/solar_lstm.keras"   # your existing Keras model
tflite_model_path = "model/solar_lstm.tflite"  # output TFLite model

os.makedirs("model", exist_ok=True)

# ===============================
# Load Keras model
# ===============================
model = tf.keras.models.load_model(keras_model_path)

# ===============================
# Convert to TFLite (Flex-free)
# ===============================
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Apply optimizations (optional but recommended)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Only use built-in TFLite ops (no Flex)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()

# ===============================
# Save TFLite model
# ===============================
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print("✅ Flex-free TFLite model saved successfully!")