import base64
import requests
import json

print("🚀 Testing VoiceGuard API...\n")

# 1. Read the dummy audio file
audio_file = 'data/authentic/train/authentic_000.wav'
with open(audio_file, 'rb') as f:
    audio_bytes = f.read()

# 2. Encode to base64
audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

# 3. Send a POST request to our local API
print(f"📤 Sending {audio_file} to API...")
response = requests.post(
    'http://localhost:5001/api/v1/analyze',  # Make sure it's 5001!
    json={'audio_base64': audio_base64}
)

# 4. Print the result
print("\n✅ API Response:")
print(json.dumps(response.json(), indent=2))