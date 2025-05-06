import os
import librosa
import numpy as np

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        return mfcc_scaled
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_dataset(data_dir):
    X, y = [], []
    emotions_map = {
        "angry": 0,
        "disgust": 1,
        "fear": 2,
        "happy": 3,
        "neutral": 4,
        "sad": 5,
        "surprise": 6
    }

    for dataset in os.listdir(data_dir):
        dataset_path = os.path.join(data_dir, dataset)
        if os.path.isdir(dataset_path):
            for root, _, files in os.walk(dataset_path):
                for file in files:
                    if file.endswith(".wav"):
                        file_path = os.path.join(root, file)

                        # Guess emotion label based on filename
                        file_lower = file.lower()
                        label = None
                        for emotion, idx in emotions_map.items():
                            if emotion in file_lower:
                                label = idx
                                break

                        if label is not None:
                            features = extract_features(file_path)
                            if features is not None:
                                X.append(features)
                                y.append(label)

    return np.array(X), np.array(y)
