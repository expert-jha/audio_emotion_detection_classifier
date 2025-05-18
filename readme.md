

## 📄 Description

**Speech Emotion Recognition (SER)** involves analyzing voice recordings to determine the speaker's emotional state. This project:
- Extracts **MFCC features** from audio
- Uses a **CNN + LSTM** based model for classification
- Supports real-time predictions via a **Streamlit web app**

---

## 📁 Dataset

We use the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

📥 **Download link:**  
[Kaggle RAVDESS Audio Speech Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

After downloading:
- Extract to a folder like: `audio_speech_actors_01-24/`
- Inside, you will find 24 folders, one for each actor (Actor_01 to Actor_24), each containing multiple `.wav` files.

---

## 🎯 Emotion Class Labels

Each audio file name follows this format:  
`03-01-01-01-01-01-01.wav`

| Segment | Description |
|---------|-------------|
| 03      | Modality (Speech) |
| 01      | Vocal Channel (Speech) |
| 01      | Emotion |
| 01      | Emotional Intensity |
| 01      | Statement |
| 01      | Repetition |
| 01      | Actor ID |

We use the **third digit** (Emotion) as our class label.

### 🎭 Emotion Mapping

| Code | Emotion       |
|------|---------------|
| 01   | Neutral       |
| 02   | Calm          |
| 03   | Happy         |
| 04   | Sad           |
| 05   | Angry         |
| 06   | Fearful       |
| 07   | Disgust       |
| 08   | Surprised     |

---

## 🛠️ Tech Stack & Libraries

- Python
- PyTorch
- Librosa (for audio processing)
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn (for EDA/visualization)
- Streamlit (for deployment)

---



--  Notes
The model was trained from scratch (no pre-trained weights).

MFCCs were used for feature extraction.

Audio resized and normalized uniformly.

CNN extracts spatial patterns from MFCCs, while LSTM captures temporal dynamics.




Author :

Govind Jha
expert.govindjha@gmail.com