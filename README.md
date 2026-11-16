# 🎙️ VoiceGuard – AI Voice Cloning Detection

**An Edge AI-powered voice authenticity detection system that identifies AI-generated or cloned voices using audio signal processing, machine learning, and TensorFlow Lite.**

VoiceGuard analyzes audio recordings to distinguish between genuine human voices and AI-generated synthetic voices. It uses five audio feature-extraction techniques and a Dense Neural Network deployed through TensorFlow Lite for lightweight inference.

---

## 🚀 Project Overview

Voice cloning technology has made it increasingly difficult to distinguish genuine human voices from AI-generated speech.

VoiceGuard aims to address this challenge by analyzing multiple acoustic characteristics of an audio recording and predicting its authenticity.

The system processes audio at 16 kHz, extracts 366 statistical audio features, normalizes them, and passes them to a trained Dense Neural Network.

### Key Features

- AI-generated voice and voice cloning detection.
- Five-lens audio feature extraction.
- 366-dimensional feature representation.
- Dense Neural Network classification.
- TensorFlow Lite model deployment.
- Audio preprocessing using Librosa.
- Real-time interactive analysis through Gradio.
- Microphone recording and audio file upload support.

---

## 🏗️ System Architecture

VoiceGuard follows a three-stage architecture:

**1. Ingestion → 2. Analysis → 3. Decision**

### Overall Workflow

```text
       User Audio Input
       (Microphone / Upload)
               |
               v
       Audio Preprocessing
          (16 kHz)
               |
               v
       Padding / Truncation
        (3 Seconds)
               |
               v
      Five-Lens Feature Extraction
               |
               v
       Feature Normalization
               |
               v
       TensorFlow Lite Model
               |
               v
        Probability Output
               |
               v
       Audio Forensics Check
               |
               v
         Final Decision
               |
          -------------
          |           |
         ALLOW       BLOCK
```

---

## 🔬 Five-Lens Feature Extraction

VoiceGuard uses five complementary audio feature-extraction techniques to capture different characteristics of voice recordings.

| Feature Extraction | Description |
|---|---|
| Mel Spectrogram | Represents frequency intensity over time. |
| MFCC | Captures important spectral characteristics associated with vocal sound. |
| Chroma Features | Represents pitch-class and harmonic information. |
| Spectral Centroid | Measures the spectral brightness of audio. |
| Zero Crossing Rate (ZCR) | Measures how frequently the audio waveform crosses zero. |

Additional temporal information is extracted using MFCC Delta, MFCC Delta-2, spectral rolloff, and onset strength.

### Feature Engineering

The extracted features include:

- Mel Spectrogram: Mean and Standard Deviation.
- MFCC: Mean and Standard Deviation.
- MFCC Delta: Mean and Standard Deviation.
- MFCC Delta-2: Mean and Standard Deviation.
- Chroma: Mean and Standard Deviation.
- Spectral Centroid: Mean and Standard Deviation.
- Spectral Rolloff: Mean and Standard Deviation.
- Zero Crossing Rate: Mean and Standard Deviation.
- Onset Strength: Mean and Standard Deviation.

**Total: 366 statistical audio features.**

---

## 🧠 Machine Learning Model

VoiceGuard uses a **Dense Neural Network (DNN)** rather than a Convolutional Neural Network (CNN).

### Model Architecture

```text
Input Layer
    |
    v
366 Features
    |
    v
Dense Layer (64 Neurons)
    |
    v
Dense Layer (32 Neurons)
    |
    v
Output Layer (1 Neuron)
    |
    v
Authenticity Probability
```

### Prediction Interpretation

The model produces a probability between 0.0 and 1.0.

| Probability | Interpretation |
|---|---|
| 0.0 | AI-generated / cloned voice |
| 1.0 | Genuine human voice |

The output represents the model's predicted authenticity probability, not a guaranteed determination of whether a voice is genuine.

---

## ⚙️ Audio Processing Pipeline

### Step 1: Audio Ingestion

Users can record audio directly through their microphone or upload an existing audio file using the Gradio interface.

### Step 2: Audio Preprocessing

Librosa loads and converts the audio into a standardized format.

- Sampling Rate: 16,000 Hz
- Audio Duration: 3 Seconds
- Total Samples: 48,000
- Short recordings: Zero-padded.
- Longer recordings: Truncated.

### Step 3: Feature Extraction

The five-lens feature extraction pipeline generates a 366-dimensional feature vector.

### Step 4: Feature Normalization

Features are normalized using the saved training normalizer.

\[
X_{normalized} = \frac{X-\mu}{\sigma+\epsilon}
\]

Where:

- \(X\) = Extracted features.
- \(\mu\) = Training mean.
- \(\sigma\) = Training standard deviation.
- \(\epsilon\) = Small numerical stability value.

### Step 5: TensorFlow Lite Inference

The normalized feature vector is passed to the TensorFlow Lite interpreter.

The model generates an authenticity probability.

### Step 6: Audio Forensics

An additional Zero Crossing Rate variance check is performed.

### Step 7: Final Decision

The current implementation combines its ZCR variance threshold with a rule-based risk score.

**Important:** The current decision logic is a prototype heuristic. ZCR variance alone cannot reliably establish whether a recording is human or AI-generated. The model probability should also be incorporated into a validated decision strategy before production deployment.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NumPy | Numerical computations |
| Librosa | Audio preprocessing and feature extraction |
| TensorFlow | Machine learning framework |
| TensorFlow Lite | Lightweight model inference |
| Pickle | Loading the trained feature normalizer |
| Gradio | Interactive user interface |

---

## 📁 Project Structure

```text
Voice-Cloning-Detection/
│
├── models/
│   ├── quantization/
│   │   └── voice_authenticity_quantized.tflite
│   │
│   └── training/
│       └── feature_normalizer.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 💻 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Gauthaman11/Voice-Cloning-Detection.git
```

### 2. Navigate to the Project

```bash
cd Voice-Cloning-Detection
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python app.py
```

Open the Gradio URL displayed in your terminal to access VoiceGuard.

---

## 🎯 Application Output

VoiceGuard displays:

- Risk Score (0–100).
- Authenticity Probability.
- Final Recommendation.

Example output:

```text
Risk Score: 25/100

Authentic Probability: 75%

Recommendation: ALLOW
```

The displayed result is based on the current prototype's decision logic and should not be treated as a verified forensic conclusion.

---

## 🌟 Future Improvements

- Improve detection accuracy using diverse real and synthetic voice datasets.
- Replace the simple ZCR threshold with a validated multi-feature decision strategy.
- Integrate model probability into the final risk calculation.
- Evaluate performance using Precision, Recall, F1-score, and ROC-AUC.
- Add support for longer audio recordings.
- Develop a REST API for external application integration.
- Optimize inference latency for mobile and edge devices.
- Evaluate robustness against compression, background noise, and different recording devices.

---

## 👨‍💻 Author

**Gauthaman S**

B.Tech – Information Technology

GitHub: https://github.com/Gauthaman11

---

## 📜 License

This project is intended for educational and research purposes. Add an appropriate open-source license before distributing it publicly.
