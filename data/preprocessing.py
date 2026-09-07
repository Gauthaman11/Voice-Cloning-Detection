"""
data/preprocessing.py
Level 1: Data Validation Pipeline
Checks audio files for integrity, correct sample rate, and valid duration.
"""

import librosa
import numpy as np
from pathlib import Path

class DataValidator:
    def __init__(self, sr=16000):
        self.sr = sr

    def validate_audio_file(self, file_path):
        """Check a single audio file for integrity."""
        try:
            # Load audio at target sample rate
            audio, sr = librosa.load(file_path, sr=self.sr, mono=True)
            
            # Quality checks
            duration = len(audio) / sr
            
            if duration < 1.0:
                return False, f"Too short ({duration:.2f}s < 1.0s)"
            if duration > 300.0:
                return False, f"Too long ({duration:.2f}s > 300s)"
            if np.isnan(audio).any() or np.isinf(audio).any():
                return False, "Contains NaN or Inf values"
            if np.max(np.abs(audio)) < 0.01:
                return False, "Audio is essentially silent"
                
            return True, f"Valid ({duration:.1f}s)"
            
        except Exception as e:
            return False, f"Load error: {str(e)}"

    def validate_dataset(self, dataset_dir):
        """Validate an entire directory of audio files."""
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            print(f"❌ Directory not found: {dataset_dir}")
            return None

        results = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': [],
            'total_hours': 0.0
        }

        # Find all WAV files recursively
        audio_files = list(dataset_path.rglob('*.wav'))
        results['total_files'] = len(audio_files)
        
        if results['total_files'] == 0:
            print(f"️ No .wav files found in {dataset_dir}")
            return results

        print(f"🔍 Validating {results['total_files']} files in {dataset_dir}...")

        for audio_file in audio_files:
            valid, msg = self.validate_audio_file(audio_file)
            if valid:
                results['valid_files'] += 1
                # Calculate hours
                audio, _ = librosa.load(audio_file, sr=self.sr)
                results['total_hours'] += len(audio) / self.sr / 3600
            else:
                results['invalid_files'].append({
                    'file': str(audio_file),
                    'error': msg
                })

        # Print Summary Report
        valid_pct = (results['valid_files'] / results['total_files']) * 100
        print("\n" + "="*40)
        print("📊 DATASET VALIDATION SUMMARY")
        print("="*40)
        print(f"Directory:    {dataset_dir}")
        print(f"Total files:  {results['total_files']}")
        print(f"Valid files:  {results['valid_files']} ({valid_pct:.1f}%)")
        print(f"Invalid files:{len(results['invalid_files'])}")
        print(f"Total hours:  {results['total_hours']:.2f}")
        print("="*40 + "\n")

        if results['invalid_files']:
            print("❌ Failed Files:")
            for item in results['invalid_files'][:5]: # Show first 5
                print(f"  - {item['file']}: {item['error']}")

        return results

# THIS BLOCK IS CRUCIAL - IT TELLS THE SCRIPT TO RUN
if __name__ == "__main__":
    validator = DataValidator(sr=16000)
    print("🚀 Starting Data Validation Pipeline...\n")
    validator.validate_dataset('./data/authentic/train')
    validator.validate_dataset('./data/synthetic/google_tts')