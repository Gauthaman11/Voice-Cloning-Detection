"""
models/training/feature_extractor.py
Level 2: The 5 Lenses Feature Extraction Engine
Extracts a 352-dimensional acoustic feature vector from audio.
"""

import librosa
import numpy as np
import pickle
from pathlib import Path

class AudioFeatureExtractor:
    def __init__(self, sr=16000, n_mfcc=13):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = 2048
        self.hop_length = 512
        
        # Precompute normalization stats
        self.mean = None
        self.std = None

    def extract_features(self, audio):
        """
        Extract acoustic feature vector (The 5 Lenses)
        """
        # 🔴 Lens 1: Mel-Spectrogram (256D)
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_mels=128,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 🟠 Lens 2: MFCCs + Deltas (78D)
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # 🟡 Lens 3: Chroma Features (24D)
        chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sr)
        
        # 🟢 Lens 4: Spectral Centroid & Rolloff (4D)
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)[0]
        
        # 🔵 Lens 5: Zero-Crossing Rate & Onset Envelope (4D)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        
                # Aggregate temporal features (mean + std for each)
        features = np.concatenate([
            np.mean(mel_spec_db, axis=1), np.std(mel_spec_db, axis=1),      # 256D
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),                    # 26D
            np.mean(mfcc_delta, axis=1), np.std(mfcc_delta, axis=1),        # 26D
            np.mean(mfcc_delta2, axis=1), np.std(mfcc_delta2, axis=1),      # 26D
            np.mean(chroma, axis=1), np.std(chroma, axis=1),                # 24D
            np.array([np.mean(spec_centroid)]), np.array([np.std(spec_centroid)]), # 2D
            np.array([np.mean(spec_rolloff)]), np.array([np.std(spec_rolloff)]), # 2D
            np.array([np.mean(zcr)]), np.array([np.std(zcr)]),              # 2D
            np.array([np.mean(onset_env)]), np.array([np.std(onset_env)])   # 2D
        ])
        
        return features

    def fit_normalizer(self, audio_files):
        """Compute mean/std for normalization using real human voices"""
        print(f"📊 Fitting normalizer on {len(audio_files)} files...")
        all_features = []
        for file in audio_files:
            audio, _ = librosa.load(file, sr=self.sr)
            features = self.extract_features(audio)
            all_features.append(features)
            
        all_features = np.array(all_features)
        self.mean = np.mean(all_features, axis=0)
        self.std = np.std(all_features, axis=0)
        
        # Save for inference
        pickle.dump({
            'mean': self.mean,
            'std': self.std
        }, open('feature_normalizer.pkl', 'wb'))
        print("✅ Saved feature_normalizer.pkl")

    def normalize(self, features):
        """Normalize features using training statistics"""
        if self.mean is None:
            raise ValueError("Must call fit_normalizer() first")
        return (features - self.mean) / (self.std + 1e-6)


# ============================================================
# TEST SCRIPT
# ============================================================
if __name__ == "__main__":
    print("🚀 Testing Feature Extraction Engine...\n")
    extractor = AudioFeatureExtractor(sr=16000)
    
    authentic_files = list(Path('../../data/authentic/train').glob('*.wav'))
    valid_files = [f for f in authentic_files if 'bad_short' not in str(f)]
    
    extractor.fit_normalizer(valid_files)
    
    test_file = valid_files[0]
    print(f"\n Extracting features from: {test_file.name}")
    audio, _ = librosa.load(test_file, sr=16000)
    
    raw_features = extractor.extract_features(audio)
    print(f"   Raw feature shape: {raw_features.shape}")
    
    norm_features = extractor.normalize(raw_features)
    print(f"   Normalized feature shape: {norm_features.shape}")
    
    print("\n🎉 Feature Extractor is working perfectly!")