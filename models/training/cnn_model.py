"""
models/training/cnn_model.py
Level 3: 1D CNN Architecture for Voice Authenticity
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape=(366,), learning_rate=1e-3):
    """
    Build a lightweight 1D CNN for voice authenticity detection.
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        layers.Reshape((input_shape[0], 1)),  # Reshape for Conv1D
        
        # Block 1: Low-level pattern detection
        layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Block 2: Mid-level pattern combination
        layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Block 3: High-level artifact detection
        layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.4),
        
        # Decision layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # 0=synthetic, 1=authentic
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model

if __name__ == '__main__':
    model = build_model()
    model.summary()