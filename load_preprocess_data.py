import os
import librosa
import numpy as np

emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_emotion(filename):
    return emotion_map[filename.split("-")[2]]

def load_data(data_path="audio_speech_actors_01-24"):
    X, y = [], []
    for actor in os.listdir(data_path):
        actor_path = os.path.join(data_path, actor)
        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                file_path = os.path.join(actor_path, file)
                emotion = extract_emotion(file)
                y_val = list(emotion_map.keys())[list(emotion_map.values()).index(emotion)]

                audio, sr = librosa.load(file_path, sr=22050)
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                mfcc = mfcc.T

                if mfcc.shape[0] < 200:
                    mfcc = np.pad(mfcc, ((0, 200 - mfcc.shape[0]), (0, 0)))
                else:
                    mfcc = mfcc[:200, :]

                X.append(mfcc)
                y.append(int(y_val) - 1)

    X = np.array(X)
    y = np.array(y)

    # Save to disk
    np.save("X.npy", X)
    np.save("y.npy", y)

    return X, y

# Run this file directly to trigger loading and saving
if __name__ == "__main__":
    X, y = load_data("data/audio_speech_actors_01-24")
    print(f"Saved: X shape = {X.shape}, y shape = {y.shape}")
