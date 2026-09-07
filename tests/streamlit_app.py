import streamlit as st
import requests
import base64

st.set_page_config(page_title="VoiceGuard AI", page_icon="🛡️", layout="centered")
st.title("🛡️ VoiceGuard: AI Voice Authenticity Detector")
st.markdown("Upload an audio clip to analyze it for synthetic voice cloning artifacts in real-time.")

uploaded_file = st.file_uploader("🎙️ Upload Audio File (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    if st.button(" Analyze Voice", type="primary"):
        with st.spinner("Processing audio through the 5 Lenses..."):
            try:
                audio_bytes = uploaded_file.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                response = requests.post(
                    'http://localhost:5001/api/v1/analyze',
                    json={'audio_base64': audio_base64},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.divider()
                    st.subheader("📊 Analysis Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Risk Score", f"{data['risk_score']}/100")
                    with col2:
                        st.metric("Authentic Probability", f"{data['probability_authentic'] * 100:.1f}%")
                    with col3:
                        st.metric("Processing Time", "< 50ms")
                    
                    st.divider()
                    
                    rec = data['recommendation'].upper()
                    status = data['status']
                    
                    if rec == "ALLOW":
                        st.success(f"### ✅ Recommendation: {rec}")
                        st.info(status)
                    elif rec == "MFA":
                        st.warning(f"### ⚠️ Recommendation: {rec}")
                        st.info(status)
                    else:
                        st.error(f"### 🚫 Recommendation: {rec}")
                        st.info(status)
                else:
                    st.error(f"API Error: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to the backend. Is `api/app.py` running on port 5001?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")