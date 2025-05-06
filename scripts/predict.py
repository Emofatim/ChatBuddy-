import torch
import torch.nn as nn
import numpy as np
import joblib
import librosa
import os

# Load saved scaler
scaler = joblib.load("scaler.pkl")

# Define the same model class used during training
class EmotionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(40, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 7)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)

# Load model
model = EmotionClassifier()
model.load_state_dict(torch.load("emotion_model.pth", map_location=torch.device('cpu')))
model.eval()

# Emotion labels (adjust based on your dataset labels)
emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Function to extract features
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

# Main prediction function
def predict_emotion(audio_file):
    # Save uploaded audio temporarily
    file_path = "temp.wav"
    audio_file.save(file_path)

    try:
        features = extract_features(file_path)
        features = scaler.transform([features])
        features_tensor = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(features_tensor)
            predicted = torch.argmax(outputs, dim=1).item()
            emotion = emotion_labels[predicted]
            return emotion
    except Exception as e:
        print(f"Prediction failed: {e}")
        return "error"
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
