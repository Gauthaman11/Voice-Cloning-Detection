import os
import numpy as np
import librosa
import pickle
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

print("🚀 Initializing Kaggle Production Training Pipeline (Float32 Fix)...")

DATA_DIR = 'data/raw_dataset'
EXCEL_FILE = os.path.join(DATA_DIR, 'dataset_voice_changing_metainfo.xlsx')

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

print(f"📂 Reading Kaggle metadata from {EXCEL_FILE}...")
df = pd.read_excel(EXCEL_FILE)

X, y = [], []
print("🎧 Extracting 5-Lens features from audio files...")

for index, row in tqdm(df.iterrows(), total=len(df)):
    country = str(row['country'])
    gender = str(row['gender'])
    audio_group = str(row['audio_group'])
    audio_name = str(row['audio_name'])
    
    folder_path = os.path.join(DATA_DIR, country, gender, audio_group)
    file_path = None
    if os.path.exists(folder_path):
        for f in os.listdir(folder_path):
            if f.startswith(audio_name):
                file_path = os.path.join(folder_path, f)
                break
                
    if file_path is None: continue
        
    features = extract_features(file_path)
    if features is not None:
        X.append(features)
        if 'original' in audio_name.lower(): y.append(0)
        else: y.append(1)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

if len(X) < 50:
    print(f"⚠️ WARNING: Only loaded {len(X)} samples.")
else:
    print(f"✅ Successfully loaded {len(X)} samples!")
    mean, std = np.mean(X, axis=0), np.std(X, axis=0)
    X = (X - mean) / (std + 1e-6)

    with open('models/training/feature_normalizer.pkl', 'wb') as f:
        pickle.dump({'mean': mean, 'std': std}, f)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("️ Building Robust Dense Network (Float32)...")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(366,)),
        tf.keras.layers.Dropout(0.3), # Prevents memorization
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    print("🧠 Training model...")
    model.fit(X_train, y_train, epochs=15, batch_size=8, validation_data=(X_test, y_test), verbose=1)

    print(" Saving as Float32 TFLite (Prevents weight collapse)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # We removed INT8 quantization to keep the math precise for this small dataset
    tflite_model = converter.convert()

    with open('models/quantization/voice_authenticity_quantized.tflite', 'wb') as f:
        f.write(tflite_model)

    print(" FLOAT32 PRODUCTION MODEL TRAINED SUCCESSFULLY!")
