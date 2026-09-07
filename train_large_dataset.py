import os
import numpy as np
import librosa
import pickle
import tensorflow as tf
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm

print("🚀 Initializing LARGE Dataset Training Pipeline (2000+ files)...")

DATA_DIR = 'data/raw_dataset'

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000, duration=3.0)
        if len(audio) < 16000: return None
        if len(audio) < 48000:
            audio = np.pad(audio, (0, max(0, 48000 - len(audio))), 'constant')
        else:
            audio = audio[:48000]
            
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
        
        return np.concatenate([
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
    except Exception:
        return None

X, y = [], []

# 1. Check for CSV first (Kaggle often uses CSVs)
csv_files = glob.glob(os.path.join(DATA_DIR, '**/*.csv'), recursive=True)
if csv_files:
    print(f"📂 Found CSV: {csv_files[0]}")
    df = pd.read_csv(csv_files[0])
    print(f"Columns: {df.columns.tolist()}")
    # Try to guess columns
    file_col = next((c for c in df.columns if 'file' in c.lower() or 'path' in c.lower() or 'name' in c.lower()), df.columns[0])
    label_col = next((c for c in df.columns if 'label' in c.lower() or 'type' in c.lower() or 'class' in c.lower()), df.columns[1])
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        f_name = str(row[file_col])
        label = str(row[label_col]).lower()
        f_path = os.path.join(DATA_DIR, f_name)
        if os.path.exists(f_path):
            feat = extract_features(f_path)
            if feat is not None:
                X.append(feat)
                y.append(0 if 'real' in label or 'human' in label or 'bonafide' in label or label=='0' else 1)
else:
    # 2. Fallback: Scan folders
    print(" Scanning folders for audio files...")
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith(('.wav', '.mp3', '.m4a', '.flac')):
                path = os.path.join(root, f)
                feat = extract_features(path)
                if feat is not None:
                    X.append(feat)
                    # If folder path contains fake/synthetic, it's 1, else 0
                    if 'fake' in root.lower() or 'synthetic' in root.lower() or 'spoof' in root.lower():
                        y.append(1)
                    else:
                        y.append(0)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print(f"✅ Loaded {len(X)} samples! (Real: {y.tolist().count(0)}, Fake: {y.tolist().count(1)})")

if len(X) < 100:
    print("⚠️ Not enough data found. Check folder structure.")
else:
    mean, std = np.mean(X, axis=0), np.std(X, axis=0)
    X = (X - mean) / (std + 1e-6)

    with open('models/training/feature_normalizer.pkl', 'wb') as f:
        pickle.dump({'mean': mean, 'std': std}, f)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🏗️ Building Robust Model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(366,)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    print(" Training on LARGE dataset... (This will take 5-10 minutes)")
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    print("📦 Saving Float32 TFLite model...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open('models/quantization/voice_authenticity_quantized.tflite', 'wb') as f:
        f.write(tflite_model)

    print("🎉 LARGE DATASET MODEL TRAINED SUCCESSFULLY!")
