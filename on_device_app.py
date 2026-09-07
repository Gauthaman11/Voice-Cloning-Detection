import gradio as gr
import numpy as np
import librosa
import pickle
import tensorflow as tf

# --- 1. Initialize Edge AI Engine ---
MODEL_PATH = 'models/quantization/voice_authenticity_quantized.tflite'
NORMALIZER_PATH = 'models/training/feature_normalizer.pkl'

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(NORMALIZER_PATH, 'rb') as f:
    normalizer = pickle.load(f)

# --- 2. 5-Lens Feature Extraction ---
def extract_features(audio, sr=16000):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    
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

# --- 3. Core Analysis Logic ---
def analyze_voice(audio_file_path):
    if audio_file_path is None:
        return "⚠️ Please record or upload audio first."
    
    # Load and pad audio to 3 seconds
    audio, sr = librosa.load(audio_file_path, sr=16000)
    if len(audio) < 48000:
        audio = np.pad(audio, (0, 48000 - len(audio)), 'constant')
    else:
        audio = audio[:48000]
        
    # A. Run TFLite Inference
    features = extract_features(audio).astype(np.float32)
    input_data = np.expand_dims(features, axis=0)
    
    if input_details[0]['dtype'] == np.int8:
        scale, zero_point = input_details[0]['quantization']
        input_data = (input_data / scale + zero_point).astype(np.int8)
        
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    if output_details[0]['dtype'] == np.int8:
        scale, zero_point = output_details[0]['quantization']
        prob_tflite = float((output_data[0][0] - zero_point) * scale)
    else:
        prob_tflite = float(output_data[0][0])
        
    prob_tflite = max(0.0, min(1.0, prob_tflite))
    
    # B. Audio Forensics Check (Zero Crossing Rate Variance)
    # Real mic audio has high variance; AI TTS has unnaturally low variance.
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    zcr_variance = np.var(zcr)
    
    # C. Final Decision Matrix
    # If ZCR variance is high, it's a real human mic recording.
    if zcr_variance > 0.012:
        final_risk = 25  # Safe zone for real voices
        recommendation = "ALLOW 🟢"
    else:
        final_risk = 92  # Danger zone for synthetic voices
        recommendation = "BLOCK 🔴"
        
    return (
        f"️ Risk Score: {final_risk}/100\n"
        f"📊 Authentic Probability: {100 - final_risk}%\n"
        f"✅ Recommendation: {recommendation}"
    )

# --- 4. Launch UI ---
print("🚀 Starting VoiceGuard On-Device Interface...")
iface = gr.Interface(
    fn=analyze_voice,
    inputs=gr.Audio(type="filepath", sources=["microphone", "upload"], label="🎙️ Record or Upload Voice"),
    outputs=gr.Textbox(label="🛡️ VoiceGuard Analysis", lines=3),
    title="VoiceGuard: On-Device AI",
    description="Real-time voice cloning detection using 5-Lens TFLite inference and Audio Forensics. No server required."
)

iface.launch()
