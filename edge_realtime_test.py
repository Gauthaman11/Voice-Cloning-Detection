import sounddevice as sd
import numpy as np
import librosa
import pickle
import tensorflow as tf
import time

# 1. Paths to our Edge AI files
MODEL_PATH = 'models/quantization/voice_authenticity_quantized.tflite'
NORMALIZER_PATH = 'models/training/feature_normalizer.pkl'

print(" Initializing Mobile App Simulation (Edge AI)...")

# Load the 60KB TFLite Model (Exactly like Android/iOS)
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load Normalizer
with open(NORMALIZER_PATH, 'rb') as f:
    normalizer = pickle.load(f)

print("✅ TFLite Model Loaded. Accessing Microphone...\n")

# 2. The 5-Lens Feature Extraction
def extract_features(audio):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=128, n_fft=2048, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_cqt(y=audio, sr=16000)
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=16000)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=16000)[0]
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    onset_env = librosa.onset.onset_strength(y=audio, sr=16000)
    
    features = np.concatenate([
        np.mean(mel_spec_db, axis=1), np.std(mel_spec_db, axis=1),
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(mfcc_delta, axis=1), np.std(mfcc_delta, axis=1),
        np.mean(mfcc_delta2, axis=1), np.std(mfcc_delta2, axis=1),
        np.mean(chroma, axis=1), np.std(chroma, axis=1),
        np.array([np.mean(spec_centroid)]), np.array([np.std(spec_centroid)]),
        np.array([np.mean(spec_rolloff)]), np.array([np.std(spec_rolloff)]),
        np.array([np.mean(zcr)]), np.array([np.std(zcr)]),
        np.array([np.mean(onset_env)]), np.array([np.std(onset_env)])
    ])
    return (features - normalizer['mean']) / (normalizer['std'] + 1e-6)

# 3. Real-Time Loop (Simulating a background phone call)
sr = 16000
duration = 2 # Analyze every 2 seconds

while True:
    print("🎙️ Listening... (Speak for 2 seconds)")
    
    # Record exactly 2 seconds from the microphone
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait() # Wait for recording to finish
    
    print("⚙️ Processing 5 Lenses & TFLite Inference...")
    
    # Extract features and run TFLite
    features = extract_features(audio.flatten()).astype(np.float32)
    input_data = np.expand_dims(features, axis=0)
    
    # Handle INT8 Quantization
    if input_details[0]['dtype'] == np.int8:
        scale, zero_point = input_details[0]['quantization']
        input_data = (input_data / scale + zero_point).astype(np.int8)
        
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Dequantize and calculate Risk
    if output_details[0]['dtype'] == np.int8:
        scale, zero_point = output_details[0]['quantization']
        prob = float((output_data[0][0] - zero_point) * scale)
    else:
        prob = float(output_data[0][0])
        
    prob = max(0.0, min(1.0, prob))
    risk = int((1.0 - prob) * 100)
    
    if risk <= 40: rec = "ALLOW 🟢"
    elif risk <= 70: rec = "MFA 🟡"
    else: rec = "BLOCK 🔴"
        
    print(f"✅ Result: Risk Score {risk}/100 | Recommendation: {rec}\n")
    print("-" * 40)
