from datasets import load_dataset, Audio
import os
import soundfile as sf
import numpy as np
import librosa
import io

print(" Downloading Real vs Fake Voice Dataset (Bypassing torchcodec)...")

os.makedirs("data/raw_dataset/real", exist_ok=True)
os.makedirs("data/raw_dataset/fake", exist_ok=True)

# Load dataset but DISABLE automatic audio decoding to avoid the torchcodec error
print("📦 Fetching dataset metadata...")
dataset = load_dataset("garystafford/deepfake-audio-detection", split="train")
dataset = dataset.cast_column("audio", Audio(decode=False))

real_count = 0
fake_count = 0

print("💾 Decoding and saving audio files manually...")
for i, item in enumerate(dataset):
    audio_data = item['audio']
    label = item['label']
    
    try:
        # Manually decode the audio bytes using soundfile
        if 'bytes' in audio_data and audio_data['bytes'] is not None:
            audio_array, sr = sf.read(io.BytesIO(audio_data['bytes']))
        elif 'path' in audio_data and audio_data['path'] is not None:
            audio_array, sr = sf.read(audio_data['path'])
        else:
            continue

        # Resample to 16kHz for our model
        if sr != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            sr = 16000

        filename = f"audio_{i:04d}.wav"
        
        if label == 0:
            filepath = os.path.join("data/raw_dataset/real", filename)
            real_count += 1
        else:
            filepath = os.path.join("data/raw_dataset/fake", filename)
            fake_count += 1
            
        sf.write(filepath, audio_array, sr)
        
        if i % 200 == 0:
            print(f"  Progress: {i}/{len(dataset)} files saved...")
    except Exception as e:
        print(f"  Skipped file {i} due to error: {e}")

print(f"✅ Download Complete!")
print(f"   Real voices saved: {real_count}")
print(f"   Fake voices saved: {fake_count}")
