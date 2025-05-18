import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import os

# Emotion labels (based on RAVDESS dataset)
EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Define your model
class CNNLSTM(nn.Module):
    def __init__(self, num_classes):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=40, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTM(num_classes=8)
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.eval()

# Preprocess uploaded audio
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < 200:
        pad_width = 200 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :200]
    return torch.tensor(mfcc.T, dtype=torch.float32).unsqueeze(0)

# Streamlit UI
st.title("🎙️ Speech Emotion Recognition")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    # Save uploaded file
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    # ▶️ Play the audio
    st.audio("temp.wav", format='audio/wav')

    # Extract features and predict
    features = extract_features("temp.wav")
    with torch.no_grad():
        prediction = model(features.to(device))
        predicted_class = torch.argmax(prediction, dim=1).item()
        st.success(f"Predicted Emotion: **{EMOTIONS[predicted_class]}**")
