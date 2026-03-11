import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os


# Load preprocessed data

X = np.load("X.npy")
y = np.load("y.npy")

print("Loaded X shape:", X.shape)
print("Loaded y shape:", y.shape)


# Build LSTM model

model = Sequential([
    LSTM(32, return_sequences=True, input_shape=(24, 3)),
    LSTM(16),
    Dense(24)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()


# Train model

model.fit(
    X, y,
    epochs=15,
    batch_size=32,
    validation_split=0.2
)


# Save model

os.makedirs("model", exist_ok=True)
model.save("model/solar_lstm.keras")

print("Model saved successfully!")

