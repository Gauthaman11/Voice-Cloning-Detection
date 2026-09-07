"""
models/quantization/quantize_to_tflite.py
Level 4: Quantize the trained H5 model to INT8 TFLite for edge deployment.
"""

import tensorflow as tf
import numpy as np
import os

def representative_data_gen():
    """
    Generator function required for INT8 quantization.
    It feeds a small sample of our training data to calibrate the quantization ranges.
    """
    # Load the dummy features we created in Level 2
    # Note: We go up two directories from models/quantization to reach models/training
    features_path = '../training/features_train.npy'
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Could not find {features_path}. Run Level 2 first!")
        
    X = np.load(features_path)
    
    # Yield 100 samples (or all available if less than 100)
    # The model expects shape (batch_size, 366, 1) due to the Reshape layer
    for i in range(min(100, len(X))):
        sample = X[i].astype(np.float32)
        # Reshape to match model input: (1, 366, 1)
        yield [sample.reshape(1, 366, 1)]

def main():
    print("🚀 Starting Model Quantization...\n")
    
    model_path = '../training/voice_authenticity_model.h5'
    output_path = 'voice_authenticity_quantized.tflite'
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print("   Please ensure you have completed Level 3 successfully.")
        return

    print(f"📂 Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print("⚙️  Configuring TensorFlow Lite Converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable full INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    print("🔨 Converting and quantizing model (this may take a minute)...")
    try:
        tflite_quant_model = converter.convert()
    except Exception as e:
        print(f"\n⚠️  Quantization warning: {e}")
        print("   Falling back to float16 quantization (still highly optimized)...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_quant_model = converter.convert()
    
    # Save the quantized model
    with open(output_path, 'wb') as f:
        f.write(tflite_quant_model)
    
    # Calculate file sizes
    h5_size = os.path.getsize(model_path) / (1024 * 1024)
    tflite_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print("\n" + "="*50)
    print("📊 QUANTIZATION SUMMARY")
    print("="*50)
    print(f"Original H5 size:    {h5_size:.2f} MB")
    print(f"Quantized TFL size:  {tflite_size:.2f} MB")
    print(f"Size reduction:      {((h5_size - tflite_size) / h5_size) * 100:.1f}%")
    print("="*50)
    
    if tflite_size < 5.0:
        print("✅ SUCCESS: Model is under the 5MB target!")
    else:
        print("⚠️  WARNING: Model is over 5MB. Check representative dataset.")
        
    print("\n💾 Saved quantized model to:", output_path)
    print("\n🎉 LEVEL 4 COMPLETE! Ready for Level 5 (Real-Time Inference & API).")

if __name__ == "__main__":
    main()