"""
api/app.py
Level 5: Flask API for Real-Time Voice Analysis
"""

import os
import sys
import base64
import io
import librosa
import soundfile as sf
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add models/inference to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models', 'inference'))
from inference import VoiceAuthenticityInference

app = Flask(__name__)
CORS(app)

# Paths relative to the project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'quantization', 'voice_authenticity_quantized.tflite')
NORMALIZER_PATH = os.path.join(ROOT_DIR, 'models', 'training', 'feature_normalizer.pkl')

print("🚀 Initializing VoiceGuard Inference Engine...")
engine = VoiceAuthenticityInference(MODEL_PATH, NORMALIZER_PATH)
print("✅ Engine ready to receive requests!\n")

@app.route('/api/v1/analyze', methods=['POST'])
def analyze_voice():
    try:
        data = request.get_json()
        if not data or 'audio_base64' not in data:
            return jsonify({"error": "Missing 'audio_base64' in request body"}), 400
            
        audio_bytes = base64.b64decode(data['audio_base64'])
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        
        # Resample to 16kHz if necessary
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
        result = engine.predict(audio)
        
        return jsonify({
            "success": True,
            "risk_score": result['risk_score'],
            "recommendation": result['recommendation'],
            "status": result['status'],
            "probability_authentic": result['probability_authentic']
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)