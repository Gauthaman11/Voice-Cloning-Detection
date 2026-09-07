"""
models/inference/inference.py
Level 5: Real-Time Inference Engine
Loads the quantized TFLite model and normalizer to predict voice authenticity.
"""

import os
import numpy as np
import librosa
import pickle
import tensorflow as tf

class VoiceAuthenticityInference:
    def __init__(self, model_path, normalizer_path):
        print(f"📂 Loading TFLite model from {model_path}...")
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"📂 Loading feature normalizer from {normalizer_path}...")
        with open(normalizer_path, 'rb') as f:
            self.normalizer = pickle.load(f)
            
        self.sr = 16000
        self.chunk_samples = int(self.sr * 2.0) # 2 seconds as per Voice Guard Flow
        print("✅ Inference engine initialized successfully!")

    def extract_and_normalize_features(self, audio):
        """Extract 366D features and normalize them (mirroring Level 2)"""
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=128, n_fft=2048, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sr)
        
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)[0]
        
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        
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
        
        return (features - self.normalizer['mean']) / (self.normalizer['std'] + 1e-6)

    def predict(self, audio):
        """Run inference on a 2-second audio chunk"""
        # Ensure audio is exactly 2 seconds (32000 samples)
        if len(audio) < self.chunk_samples:
            audio = np.pad(audio, (0, self.chunk_samples - len(audio)), 'constant')
        else:
            audio = audio[:self.chunk_samples]
            
        features = self.extract_and_normalize_features(audio).astype(np.float32)
        input_data = np.expand_dims(features, axis=0)
        
        # Handle INT8 quantization if the model requires it
        input_details = self.input_details[0]
        if input_details['dtype'] == np.int8:
            scale, zero_point = input_details['quantization']
            input_data = (input_data / scale + zero_point).astype(np.int8)
            
        self.interpreter.set_tensor(input_details['index'], input_data)
        self.interpreter.invoke()
        
        output_details = self.output_details[0]
        output_data = self.interpreter.get_tensor(output_details['index'])
        
        # Dequantize output if necessary
        if output_details['dtype'] == np.int8:
            scale, zero_point = output_details['quantization']
            probability_authentic = float((output_data[0][0] - zero_point) * scale)
        else:
            probability_authentic = float(output_data[0][0])
            
        # Clamp to 0-1
        probability_authentic = max(0.0, min(1.0, probability_authentic))
        
        # Calculate Risk Score (0-100)
        risk_score = int((1.0 - probability_authentic) * 100)
        
        # Determine recommendation based on Voice Guard Flow PDF
        if risk_score <= 40:
            recommendation = "allow"
            status = "🟢 Safe"
        elif risk_score <= 70:
            recommendation = "mfa"
            status = "🟡 Suspicious (Require 2FA)"
        else:
            recommendation = "block"
            status = "🔴 High Risk (Block)"
            
        return {
            "probability_authentic": round(probability_authentic, 4),
            "risk_score": risk_score,
            "recommendation": recommendation,
            "status": status
        }