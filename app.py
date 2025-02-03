from flask import Flask, request, jsonify
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load pre-trained models and scaler
cnn_model = load_model('cnn_model.keras')
svm_model = joblib.load('svm_model.pkl')
rf_model = joblib.load('rf_model.pkl')
meta_learner = joblib.load('meta_learner.pkl')
scaler = joblib.load('scaler.pkl')

def extract_features_from_audio(y, sr, fixed_length=40):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=fixed_length)
    mfccs = np.mean(mfccs.T, axis=0)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma = np.mean(chroma.T, axis=0)

    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel = np.mean(mel.T, axis=0)

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast = np.mean(contrast.T, axis=0)

    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    tonnetz = np.mean(tonnetz.T, axis=0)

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr = np.mean(zcr.T, axis=0)

    rmse = librosa.feature.rms(y=y)
    rmse = np.mean(rmse.T, axis=0)

    def pad_or_trim(feature, length):
        if len(feature) < length:
            return np.pad(feature, (0, length - len(feature)), mode='constant')
        else:
            return feature[:length]

    mfccs = pad_or_trim(mfccs, fixed_length)
    chroma = pad_or_trim(chroma, fixed_length)
    mel = pad_or_trim(mel, fixed_length)
    contrast = pad_or_trim(contrast, fixed_length)
    tonnetz = pad_or_trim(tonnetz, fixed_length)
    zcr = pad_or_trim(zcr, fixed_length)
    rmse = pad_or_trim(rmse, fixed_length)

    return np.concatenate([mfccs, chroma, mel, contrast, tonnetz, zcr, rmse])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    y, sr = librosa.load(file, sr=None)
    features = extract_features_from_audio(y, sr, fixed_length=40)
    features = scaler.transform([features])
    features_cnn = features[..., np.newaxis, np.newaxis]

    svm_pred_proba = svm_model.predict_proba(features)[0][1]
    rf_pred_proba = rf_model.predict_proba(features)[0][1]
    cnn_pred_proba = cnn_model.predict(features_cnn)[0][1]

    stacked_features = np.array([svm_pred_proba, rf_pred_proba, cnn_pred_proba]).reshape(1, -1)
    final_pred_proba = meta_learner.predict_proba(stacked_features)
    final_prediction = meta_learner.predict(stacked_features)

    label_map = {0: "Human", 1: "AI"}
    predicted_label = label_map[final_prediction[0]]
    ai_probability = final_pred_proba[0][1] * 100
    human_probability = final_pred_proba[0][0] * 100

    return jsonify({
        'prediction': predicted_label,
        'ai_probability': f"{ai_probability:.2f}%",
        'human_probability': f"{human_probability:.2f}%"
    })

if __name__ == '__main__':
    app.run(debug=True)
