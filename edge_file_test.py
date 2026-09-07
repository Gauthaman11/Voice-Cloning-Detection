import numpy as np
import librosa
import pickle
import tensorflow as tf
import os

# 1. Paths to our Edge AI files
MODEL_PATH = 'models/quantization/voice_authenticity_quantized.tflite'
NORMALIZER_PATH = 'models/training/feature_normalizer.pkl'

print("🛡️ Initializing VoiceGuard Edge AI (File Upload Mode)...")

# Load the 60KB TFLite Model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load Normalizer
with open(NORMALIZER_PATH, 'rb') as f:
    normalizer = pickle.load(f)

print("✅ TFLite Model Loaded. Ready to analyze files!\n")

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

# 3. Analyze the Uploaded File
def analyze_file(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return

    print(f"📂 Processing: {file_path}")
    
    # Load audio at 16kHz
    audio, sr = librosa.load(file_path, sr=16000)
    
    # Ensure exactly 2 seconds (32000 samples)
    if len(audio) < 32000:
        audio = np.pad(audio, (0, 32000 - len(audio)), 'constant')
    else:
        audio = audio[:32000]
        
    # Extract 5-Lens Features
    features = extract_features(audio).astype(np.float32)
    input_data = np.expand_dims(features, axis=0)
    
    # Handle INT8 Quantization
    if input_details[0]['dtype'] == np.int8:
        scale, zero_point = input_details[0]['quantization']
        input_data = (input_data / scale + zero_point).astype(np.int8)
        
    # Run TFLite Model
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Dequantize output
    if output_details[0]['dtype'] == np.int8:
        scale, zero_point = output_details[0]['quantization']
        prob_authentic = float((output_data[0][0] - zero_point) * scale)
    else:
        prob_authentic = float(output_data[0][0])
        
    prob_authentic = max(0.0, min(1.0, prob_authentic))
    risk_score = int((1.0 - prob_authentic) * 100)
    
    if risk_score <= 40:
        rec = "ALLOW 🟢"
    elif risk_score <= 70:
        rec = "MFA "
    else:
        rec = "BLOCK 🔴"
        
    print(f"✅ Analysis Complete!")
    print(f"   Risk Score: {risk_score}/100")
    print(f"   Authentic Probability: {prob_authentic*100:.1f}%")
    print(f"   Recommendation: {rec}\n")
    print("-" * 40)

# 4. User Input Loop
while True:
    print("\n📥 Enter the path to your audio file (or type 'q' to quit):")
    print("(Press Enter to test the default dummy file)")
    
    user_input = input(">>> ").strip()
    
    if user_input.lower() == 'q':
        print(" Exiting VoiceGuard. Stay safe!")
        break
        
    # Default to dummy file if they just press Enter
    if user_input == "":
        file_to_analyze = "data/authentic/train/authentic_000.wav"
    else:
        file_to_analyze = user_input
        
    analyze_file(file_to_analyze)
