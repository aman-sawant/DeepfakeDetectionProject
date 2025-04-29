import os
import cv2
import numpy as np
import librosa
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model  # type: ignore
import joblib
from tensorflow.keras.utils import get_custom_objects # type: ignore
from tensorflow.keras.activations import swish # type: ignore
from tensorflow.keras.layers import Dropout # type: ignore
from tensorflow.keras.utils import get_custom_objects # type: ignore

class FixedDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)


get_custom_objects().update({
    'FixedDropout': FixedDropout,
    'swish': swish
})


get_custom_objects().update({'swish': swish})


app = Flask(__name__)

# Load models and scaler
voice_model = load_model("deepfake_voice_detection.h5")
image_model = load_model("deepfake_detector_3.h5")
scaler = joblib.load("scaler.save")

# Feature extraction for audio
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs = np.mean(mfcc, axis=1)
    return np.hstack([chroma_stft, rms, spec_cent, spec_bw, rolloff, zcr, mfccs])

@app.route('/predict-audio', methods=['POST'])
def predict_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    temp_path = 'temp.wav'
    file.save(temp_path)

    try:
        features = extract_features(temp_path)
        if features.shape[0] != 26:
            raise ValueError("Invalid feature shape")

        features_scaled = scaler.transform([features])
        features_reshaped = features_scaled.reshape(1, 26, 1)
        prediction = voice_model.predict(features_reshaped)[0][0]

        label = 'REAL' if prediction >= 0.5 else 'FAKE'
        confidence = float(prediction) if label == 'REAL' else 1 - float(prediction)

        return jsonify({
            'prediction': label,
            'confidence': f"{confidence:.2%}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/predict-image', methods=['POST'])
def predict_image_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    temp_path = 'temp_img.jpg'
    file.save(temp_path)

    try:
        img = cv2.imread(temp_path)
        img = cv2.resize(img, (224, 224)) / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = image_model.predict(img)[0][0]

        label = "Fake" if prediction > 0.5 else "Real"
        confidence = float(prediction) if label == "Fake" else 1 - float(prediction)

        return jsonify({
            'prediction': label,
            'confidence': f"{confidence:.2%}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
