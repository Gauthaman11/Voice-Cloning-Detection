"""
on_device_app.py

The True On-Device Experience.

Records from mic, extracts features, and runs TFLite inference locally.
"""

import gradio as gr
import numpy as np
import librosa
import pickle
import tensorflow as tf
import os


# ============================================================
# 1. Load the Quantized Model & Normalizer
# ============================================================

MODEL_PATH = "models/quantization/voice_authenticity_quantized.tflite"
NORMALIZER_PATH = "models/training/feature_normalizer.pkl"

print("🚀 Loading On-Device AI Engine...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

if not os.path.exists(NORMALIZER_PATH):
    raise FileNotFoundError(f"❌ Normalizer not found: {NORMALIZER_PATH}")


# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

input_shape = input_details[0]["shape"]
input_dtype = input_details[0]["dtype"]
input_quantization = input_details[0]["quantization"]

output_dtype = output_details[0]["dtype"]
output_quantization = output_details[0]["quantization"]


# Load normalizer
with open(NORMALIZER_PATH, "rb") as f:
    normalizer = pickle.load(f)


# ============================================================
# 2. Validate Model & Normalizer
# ============================================================

EXPECTED_FEATURES = int(input_shape[-1])

print("\n========== MODEL INFORMATION ==========")
print("Input shape       :", input_shape)
print("Input dtype       :", input_dtype)
print("Input quantization:", input_quantization)
print("Output dtype      :", output_dtype)
print("Output quantization:", output_quantization)
print("Expected features :", EXPECTED_FEATURES)

print("\n========== NORMALIZER INFORMATION ==========")
print("Normalizer type   :", type(normalizer))

if isinstance(normalizer, dict):

    print("Normalizer keys   :", list(normalizer.keys()))

    if "mean" not in normalizer or "std" not in normalizer:
        raise ValueError(
            "❌ Normalizer dictionary must contain 'mean' and 'std'. "
            f"Found keys: {list(normalizer.keys())}"
        )

    NORMALIZER_MEAN = np.asarray(normalizer["mean"], dtype=np.float32)
    NORMALIZER_STD = np.asarray(normalizer["std"], dtype=np.float32)

else:
    raise ValueError(
        "❌ feature_normalizer.pkl is not a dictionary containing "
        "'mean' and 'std'."
    )


if len(NORMALIZER_MEAN) != EXPECTED_FEATURES:
    raise ValueError(
        f"\n❌ Normalizer feature mismatch!\n"
        f"Model expects : {EXPECTED_FEATURES}\n"
        f"Normalizer has: {len(NORMALIZER_MEAN)}"
    )


if len(NORMALIZER_STD) != EXPECTED_FEATURES:
    raise ValueError(
        f"\n❌ Normalizer STD mismatch!\n"
        f"Model expects : {EXPECTED_FEATURES}\n"
        f"STD has       : {len(NORMALIZER_STD)}"
    )


print("Normalizer features:", len(NORMALIZER_MEAN))
print("✅ Engine ready!")


# ============================================================
# 3. Feature Extraction
# ============================================================

def extract_features(audio, sr=16000):

    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    audio = np.asarray(audio, dtype=np.float32)

    # Remove NaN / infinity
    audio = np.nan_to_num(audio)

    # --------------------------------------------------------
    # 1. MEL SPECTROGRAM
    # --------------------------------------------------------

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )

    mel_spec_db = librosa.power_to_db(
        mel_spec,
        ref=np.max
    )

    # --------------------------------------------------------
    # 2. MFCC + DELTA + DELTA2
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    mfcc_delta = librosa.feature.delta(mfcc)

    mfcc_delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    # --------------------------------------------------------
    # 3. CHROMA
    # --------------------------------------------------------

    chroma = librosa.feature.chroma_cqt(
        y=audio,
        sr=sr
    )

    # --------------------------------------------------------
    # 4. SPECTRAL FEATURES
    # --------------------------------------------------------

    spec_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )[0]

    spec_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr
    )[0]

    # --------------------------------------------------------
    # 5. ZCR + ONSET
    # --------------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(
        audio
    )[0]

    onset_env = librosa.onset.onset_strength(
        y=audio,
        sr=sr
    )

    # --------------------------------------------------------
    # Combine all features
    #
    # MEL:
    # 128 mean + 128 std = 256
    #
    # MFCC:
    # 13 x 6 = 78
    #
    # Chroma:
    # 12 x 2 = 24
    #
    # Spectral:
    # 4
    #
    # ZCR + onset:
    # 4
    #
    # TOTAL = 366
    # --------------------------------------------------------

    features = np.concatenate([

        np.mean(mel_spec_db, axis=1),
        np.std(mel_spec_db, axis=1),

        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),

        np.mean(mfcc_delta, axis=1),
        np.std(mfcc_delta, axis=1),

        np.mean(mfcc_delta2, axis=1),
        np.std(mfcc_delta2, axis=1),

        np.mean(chroma, axis=1),
        np.std(chroma, axis=1),

        np.array([
            np.mean(spec_centroid),
            np.std(spec_centroid)
        ]),

        np.array([
            np.mean(spec_rolloff),
            np.std(spec_rolloff)
        ]),

        np.array([
            np.mean(zcr),
            np.std(zcr)
        ]),

        np.array([
            np.mean(onset_env),
            np.std(onset_env)
        ])

    ]).astype(np.float32)

    # --------------------------------------------------------
    # Verify feature count
    # --------------------------------------------------------

    if len(features) != EXPECTED_FEATURES:
        raise ValueError(
            f"Feature mismatch!\n"
            f"Model expects : {EXPECTED_FEATURES}\n"
            f"Extractor made: {len(features)}"
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    features = (
        features - NORMALIZER_MEAN
    ) / (
        NORMALIZER_STD + 1e-6
    )

    return features.astype(np.float32)


# ============================================================
# 4. TFLite Inference
# ============================================================

def run_inference(features):

    # Shape:
    # (366,) -> (1, 366)

    input_data = np.expand_dims(
        features,
        axis=0
    ).astype(np.float32)

    # --------------------------------------------------------
    # Quantize input for INT8 / UINT8 model
    # --------------------------------------------------------

    if input_dtype in [np.int8, np.uint8]:

        scale, zero_point = input_quantization

        if scale == 0:
            raise ValueError(
                "❌ Model input quantization scale is zero."
            )

        input_data = np.round(
            input_data / scale + zero_point
        )

        if input_dtype == np.int8:

            input_data = np.clip(
                input_data,
                -128,
                127
            ).astype(np.int8)

        else:

            input_data = np.clip(
                input_data,
                0,
                255
            ).astype(np.uint8)

    else:

        input_data = input_data.astype(
            np.float32
        )

    # --------------------------------------------------------
    # Run TFLite model
    # --------------------------------------------------------

    interpreter.set_tensor(
        input_index,
        input_data
    )

    interpreter.invoke()

    output_data = interpreter.get_tensor(
        output_index
    )

    # --------------------------------------------------------
    # Dequantize output
    # --------------------------------------------------------

    if output_dtype in [np.int8, np.uint8]:

        scale, zero_point = output_quantization

        if scale == 0:
            raise ValueError(
                "❌ Model output quantization scale is zero."
            )

        output_data = (
            output_data.astype(np.float32)
            - zero_point
        ) * scale

    else:

        output_data = output_data.astype(
            np.float32
        )

    # --------------------------------------------------------
    # Get probability
    # --------------------------------------------------------

    prob_authentic = float(
        np.asarray(output_data).flatten()[0]
    )

    prob_authentic = float(
        np.clip(
            prob_authentic,
            0.0,
            1.0
        )
    )

    return prob_authentic


# ============================================================
# 5. Analyze Voice
# ============================================================

def analyze_voice(audio_file_path):

    if audio_file_path is None:
        return "⚠️ Please record or upload audio first."

    try:

        # ----------------------------------------------------
        # Load audio
        # ----------------------------------------------------

        audio, sr = librosa.load(
            audio_file_path,
            sr=16000,
            mono=True
        )

        # ----------------------------------------------------
        # Exactly 2 seconds = 32000 samples
        # ----------------------------------------------------

        WINDOW_SIZE = 32000

        if len(audio) < WINDOW_SIZE:

            audio = np.pad(
                audio,
                (0, WINDOW_SIZE - len(audio)),
                mode="constant"
            )

        else:

            audio = audio[:WINDOW_SIZE]

        # ----------------------------------------------------
        # Extract features
        # ----------------------------------------------------

        features = extract_features(
            audio,
            sr=16000
        )

        # ----------------------------------------------------
        # TFLite inference
        # ----------------------------------------------------

        prob_authentic = run_inference(
            features
        )

        # ----------------------------------------------------
        # Risk score
        # ----------------------------------------------------

        risk_score = int(
            round(
                (1.0 - prob_authentic) * 100
            )
        )

        risk_score = max(
            0,
            min(
                100,
                risk_score
            )
        )

        # ----------------------------------------------------
        # Decision
        # ----------------------------------------------------

        if risk_score <= 40:

            rec = "ALLOW 🟢"
            status = "🟢 HUMAN / AUTHENTIC VOICE"

        elif risk_score <= 70:

            rec = "MFA 🟡"
            status = "🟡 SUSPICIOUS VOICE"

        else:

            rec = "BLOCK 🔴"
            status = "🔴 POSSIBLE AI / CLONED VOICE"

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        return (
            f"{status}\n\n"
            f"Risk Score: {risk_score}/100\n"
            f"Authentic Probability: "
            f"{prob_authentic * 100:.1f}%\n"
            f"Recommendation: {rec}"
        )

    except Exception as e:

        return (
            f"❌ Analysis failed:\n\n"
            f"{type(e).__name__}: {str(e)}"
        )


# ============================================================
# 6. Build Gradio Interface
# ============================================================

print("🌐 Starting Local App Interface...")

iface = gr.Interface(

    fn=analyze_voice,

    inputs=gr.Audio(
        type="filepath",
        sources=[
            "microphone",
            "upload"
        ],
        label="🎙️ Record or Upload Voice"
    ),

    outputs=gr.Textbox(
        label="🛡️ VoiceGuard Analysis",
        lines=5
    ),

    title="VoiceGuard: On-Device AI",

    description=(
        "🎙️ Record or upload voice audio. "
        "VoiceGuard extracts 366 audio features and "
        "runs the quantized TFLite model locally."
    )

)


# ============================================================
# 7. Launch
# ============================================================

iface.launch()