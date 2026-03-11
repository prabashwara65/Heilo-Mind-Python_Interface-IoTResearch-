import os

model_path = "model/solar_lstm_quant.tflite"

size_mb = os.path.getsize(model_path) / (1024*1024)

print(f"Model size: {size_mb:.2f} MB")