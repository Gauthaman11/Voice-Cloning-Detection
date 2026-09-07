"""
models/training/train.py
Level 3: Training Pipeline
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from cnn_model import build_model

def main():
    print("🚀 Starting Model Training Pipeline...\n")
    
    # 1. Load the dataset we built in Level 2
    print("📂 Loading dataset...")
    X = np.load('features_train.npy')
    y = np.load('labels_train.npy')
    print(f"   Loaded {X.shape[0]} samples with {X.shape[1]} features.")
    
    # 2. Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Test set:  {X_test.shape[0]} samples\n")
    
    # 3. Build the model
    print("🏗️  Building 1D CNN...")
    model = build_model(input_shape=(X_train.shape[1],))
    
    # 4. Setup callbacks to stop training if it stops improving
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint('voice_authenticity_model.h5', save_best_only=True)
    ]
    
    # 5. Train the model!
    print("🧠 Training AI (this might take a minute)...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=4, # Small batch size because we only have 10 dummy samples
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. Evaluate
    print("\n📊 Final Evaluation:")
    loss, accuracy, auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Accuracy: {accuracy * 100:.2f}%")
    print(f"   Test AUC:      {auc:.3f}")
    
    print("\n💾 Model saved as 'voice_authenticity_model.h5'")
    print("\n🎉 LEVEL 3 COMPLETE! Ready for Level 4 (Quantization).")

if __name__ == "__main__":
    main()