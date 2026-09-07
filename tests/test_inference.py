"""
tests/test_inference.py
Automated pytest for the VoiceGuard API
"""
import base64
import pytest
import sys
import os

# Add the project root directory to the Python path
# This allows the test to find the 'api' module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from api.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_analyze_authentic_audio(client):
    # 1. Load dummy authentic audio
    audio_path = os.path.join(project_root, 'data/authentic/train/authentic_000.wav')
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()
    
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # 2. Send POST request
    response = client.post('/api/v1/analyze', json={'audio_base64': audio_base64})
    
    # 3. Assert response is valid
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert 'risk_score' in data
    assert 'recommendation' in data
    assert data['recommendation'] in ['allow', 'mfa', 'block']

def test_analyze_missing_audio(client):
    # Test error handling when no audio is provided
    response = client.post('/api/v1/analyze', json={})
    assert response.status_code == 400
    assert 'error' in response.get_json()