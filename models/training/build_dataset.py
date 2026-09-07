"""
models/training/build_dataset.py
Level 2: Build the full training dataset using the 5 Lenses.
"""

import numpy as np
import librosa
from pathlib import Path
from feature_extractor import AudioFeatureExtractor

def main():
    print("🚀 Building Full Training Dataset...\n")
    
    # 1. Initialize Extractor
    extractor = AudioFeatureExtractor(sr=16000)
    
    # 2. Gather file paths (Relative to models/training/ folder)
    authentic_files = list(Path('../../data/authentic/train').glob('*.wav'))
    synthetic_files = list(Path('../../data/synthetic/google_tts').glob('*.wav'))
    
    # Filter out the intentionally bad file we created in Level 1
    authentic_files = [f for f in authentic_files if 'bad_short' not in str(f)]
    
    print(f" Found {len(authentic_files)} authentic files.")
    print(f"📂 Found {len(synthetic_files)} synthetic files.")
    
    # 3. Fit Normalizer (using authentic files as the "human baseline")
    print("\n📏 Fitting normalizer on authentic voices...")
    extractor.fit_normalizer(authentic_files)
    
    # 4. Extract and Normalize Features
    X = [] # Features
    y = [] # Labels (1 = Authentic, 0 = Synthetic)
    
    print("\n🔬 Extracting 352D features through the 5 Lenses...")
    
    # Process Authentic Voices (Label = 1)
    for file in authentic_files:
        audio, _ = librosa.load(file, sr=16000)
        feats = extractor.extract_features(audio)
        feats_norm = extractor.normalize(feats)
        X.append(feats_norm)
        y.append(1)
        
    # Process Synthetic Voices (Label = 0)
    for file in synthetic_files:
        audio, _ = librosa.load(file, sr=16000)
        feats = extractor.extract_features(audio)
        feats_norm = extractor.normalize(feats)
        X.append(feats_norm)
        y.append(0)
        
    # 5. Convert to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n✅ Dataset built successfully!")
    print(f"   Feature shape: {X.shape}  <-- (Should be (10, 352))")
    print(f"   Label shape:   {y.shape}  <-- (Should be (10,))")
    print(f"   Class balance: {np.sum(y==1)} Authentic, {np.sum(y==0)} Synthetic")
    
    # 6. Save to disk for Level 3
    np.save('features_train.npy', X)
    np.save('labels_train.npy', y)
    print("\n💾 Saved features_train.npy and labels_train.npy")
    print("\n🎉 LEVEL 2 COMPLETE! Ready for Level 3 (Model Training).")

if __name__ == "__main__":
    main()