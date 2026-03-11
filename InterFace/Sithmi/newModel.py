import tensorflow as tf

model = tf.keras.models.load_model("model/solar_lstm.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Only use TFLITE_BUILTINS (no Flex ops)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()

with open("model/solar_lstm_quant.tflite", "wb") as f:
    f.write(tflite_model)