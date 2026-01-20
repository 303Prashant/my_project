import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa as lb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ========== Model Definition ==========
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()

        # Convolutional Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2)
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3)
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.4)
        )

        # Compute flattened size dynamically
        self.flattened_size = self._get_flattened_size()

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.output = nn.Linear(128, 1)

    def _get_flattened_size(self):
        x = torch.zeros(1, 1, 64, 65)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x)
        return torch.sigmoid(x)
    
# ========== Helper Functions ==========

def create_mel_spectrogram(audio_data, sr=22050, n_mels=64):
    mel = lb.feature.melspectrogram(y=audio_data, sr=sr, n_mels=n_mels)
    mel_db = lb.power_to_db(mel, ref=np.max)
    return mel_db

def preprocess_audio(uploaded_file, sr=22050, duration=1.5, n_mels=64):
    y, _ = lb.load(uploaded_file, sr=sr)
    sample_length = int(sr * duration)

    if len(y) < sample_length:
        y = np.pad(y, (0, sample_length - len(y)), 'constant')
    else:
        y = y[:sample_length]

    mel = lb.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = lb.power_to_db(mel, ref=np.max)
    mel_db = np.abs(mel_db) / 80.0  # Normalize same as training

    mel_db = mel_db[:, :65]  # Ensure time axis = 65

    mel_tensor = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 64, 65]
    return mel_tensor.float(), y

def plot_waveform(y):
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(y, color='gray')
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

def plot_melspectrogram(mel):
    fig, ax = plt.subplots(figsize=(8, 4))
    img = lb.display.specshow(mel, sr=22050, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title("Mel Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    st.pyplot(fig)


# ========== Load Model ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EnhancedCNN().to(device)
model_path = "synthia/model/deep_audio_model.pth"
model.load_state_dict(torch.load(model_path))
model.eval()

# ========== Streamlit App ==========
st.title("🎙️ Audio Deepfake Detection")
st.markdown("Upload an audio file (`.wav` or `.mp3`) to detect if it's **Real** or **Fake (Deepfake)**.")

uploaded_file = st.file_uploader("🎵 Upload your audio file", type=["wav", "mp3"])

threshold = st.slider("🎚️ Prediction Threshold", min_value=0.0, max_value=1.0, value=0.67, step=0.01)

if uploaded_file is not None:
    with st.spinner("Processing audio..."):
        mel_tensor, raw_audio = preprocess_audio(uploaded_file)
        mel_tensor = mel_tensor.to(device)

        with torch.no_grad():
            output = model(mel_tensor).squeeze()
            prob = torch.sigmoid(output).item()
            prediction = int(prob > threshold)

    # Show prediction
    st.markdown(f"### 🔍 Confidence: `{prob:.2f}`")
    if prediction == 0:
        st.success("✅ Prediction: **Real Audio**")
    else:
        st.error("❌ Prediction: **Fake Audio (Deepfake)**")

    # Waveform + Mel Spectrogram
    st.subheader("🎧 Audio Waveform")
    st.audio(uploaded_file)
    plot_waveform(raw_audio)

    st.subheader("📊 Mel Spectrogram")
    mel_spec = create_mel_spectrogram(raw_audio)
    plot_melspectrogram(mel_spec)

    # Show static model metrics
    st.markdown("---")
    st.markdown(" Model Evaluation Metrics")
    st.write("Test Accuracy: 92.08%")
    st.write("F1 Score: 0.9212")
    st.write("Loss: 0.1666")
