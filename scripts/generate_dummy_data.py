"""
scripts/generate_dummy_data.py
Generates fake WAV files to test the DataValidator pipeline.
"""
import numpy as np
import soundfile as sf
import os

def generate_audio():
    sr = 16000
    duration = 3.0  # 3 seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 1. Generate "Authentic" style audio (Noisy, complex harmonics)
    os.makedirs('data/authentic/train', exist_ok=True)
    for i in range(5):
        # Base frequency + harmonics + random noise (simulates human voice texture)
        freq = 150 + np.random.randint(0, 100)
        audio = 0.4 * np.sin(2 * np.pi * freq * t) + \
                0.2 * np.sin(2 * np.pi * freq * 2 * t) + \
                0.1 * np.random.randn(len(t)) 
        
        # Add amplitude modulation (simulates speech envelope)
        env = np.abs(np.sin(2 * np.pi * 3 * t)) 
        audio = audio * env
        
        sf.write(f'data/authentic/train/authentic_{i:03d}.wav', audio.astype(np.float32), sr)
    print("✅ Generated 5 authentic dummy files.")

    # 2. Generate "Synthetic" style audio (Pure, clean, too perfect)
    os.makedirs('data/synthetic/google_tts', exist_ok=True)
    for i in range(5):
        freq = 200 + np.random.randint(0, 50)
        # Very clean sine wave, no noise (simulates basic TTS)
        audio = 0.5 * np.sin(2 * np.pi * freq * t) 
        sf.write(f'data/synthetic/google_tts/synthetic_{i:03d}.wav', audio.astype(np.float32), sr)
    print("✅ Generated 5 synthetic dummy files.")

    # 3. Generate one BAD file to test the validator's error catching
    bad_audio = np.zeros(int(sr * 0.5)) # 0.5 seconds (Too short!)
    sf.write('data/authentic/train/bad_short_file.wav', bad_audio.astype(np.float32), sr)
    print("✅ Generated 1 intentionally bad file (too short).")

if __name__ == "__main__":
    generate_audio()