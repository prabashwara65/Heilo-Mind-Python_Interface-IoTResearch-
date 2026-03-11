import tensorflow as tf

# Load trained LSTM model
model = tf.keras.models.load_model("models/battery_soc_led_model")

# Create TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimization (VERY IMPORTANT for edge devices)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert model
tflite_model = converter.convert()

# Save TFLite model
with open("models/battery_soc_led_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved successfully!")
