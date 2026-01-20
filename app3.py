import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'synthia')))

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
from torchvision import transforms
from pathlib import Path

# Synthia utils
from synthia.utils.video_utils import extract_frames
from synthia.utils.image_utils import preprocess_image
from synthia.utils.detection import predict_frame
from synthia.utils.model_loader import load_deepfake_model
from synthia.utils.report import generate_report, generate_audio_report
from synthia.utils.audio_utils import save_audio_visualizations
from synthia.model import EnhancedCNN


# Audio utils
import librosa as lb
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn

# ==== Audio Utils ====
def preprocess_audio(uploaded_file, sr=22050, duration=1.5, n_mels=64):
    y, _ = lb.load(uploaded_file, sr=sr)
    sample_length = int(sr * duration)
    y = np.pad(y, (0, max(0, sample_length - len(y))), 'constant')[:sample_length]
    mel = lb.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = lb.power_to_db(mel, ref=np.max) / 80.0
    mel_db = mel_db[:, :65]
    mel_tensor = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0)
    return mel_tensor.float(), y

def plot_waveform(y):
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(y, color='gray')
    ax.set_title("Audio Waveform")
    st.pyplot(fig)

def plot_melspectrogram(y, sr=22050):
    mel = lb.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = lb.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(8, 4))
    img = lb.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title("Mel Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    st.pyplot(fig)

# ==== Setup ====
st.set_page_config(page_title="Synthia", layout="wide")
st.title("Synthia: Unified Deepfake Detection Platform")

# Setup history in session state
if "history" not in st.session_state:
    st.session_state.history = []

tabs = st.tabs(["🖼️ Image Detection", "🎥 Video Detection", "🔊 Audio Detection"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# ==== Tab 1: Image Detection ====
with tabs[0]:
    st.sidebar.subheader("Upload Image")
    uploaded_image = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        model = load_deepfake_model(device)
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        img_tensor = transform(image).unsqueeze(0).to(device)
        label = predict_frame(model, img_tensor)
        st.markdown(f"   Prediction: `{label}`")
        
        # Save to history
        st.session_state.history.append({
            "type": "Image",
            "prediction": label
        })

# ==== Tab 2: Video Detection ====
with tabs[1]:
    st.sidebar.subheader("Upload Video")
    uploaded_video = st.sidebar.file_uploader("Choose a video", type=["mp4", "mov", "avi"])
    FRAME_SKIP = st.sidebar.slider("Frame Sampling Interval", 1, 30, 10)
    if uploaded_video:
        model = load_deepfake_model(device)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_video.read())
            video_path = temp_file.name

        st.video(uploaded_video)

        frame_count, real_count, fake_count = 0, 0, 0
        suspicious_frames = []

        with st.spinner("Analyzing video frames..."):
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % FRAME_SKIP == 0:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    img_tensor = transform(img).unsqueeze(0).to(device)
                    label = predict_frame(model, img_tensor)
                    if label == "Real":
                        real_count += 1
                    else:
                        fake_count += 1
                        suspicious_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_count += 1
            cap.release()
            os.remove(video_path)

        st.metric("Real Frames", real_count)
        st.metric("Fake Frames", fake_count)
        prediction = 1 if fake_count > real_count else 0
        confidence = (fake_count / (real_count + fake_count)) * 100 if (real_count + fake_count) > 0 else 0
        report_path = generate_report(prediction, confidence, suspicious_frames[:3])
        st.markdown(f"   Final verdict: `{ 'Fake' if prediction == 1 else 'Real' }`")
        with open(report_path, "rb") as f:
            st.download_button("Download PDF Report", data=f.read(), file_name="synthia_video_report.pdf")
        # Save to history
        st.session_state.history.append({
            "type": "Video",
            "prediction": "Fake" if prediction == 1 else "Real",
            "confidence": f"{confidence:.2f}%"
        })

# ==== Tab 3: Audio Detection ====
with tabs[2]:
    st.sidebar.subheader("Upload Audio")
    uploaded_audio = st.sidebar.file_uploader("Choose a .wav or .mp3 file", type=["wav", "mp3"])
    threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.67, 0.01)

    if uploaded_audio:
        model = EnhancedCNN().to(device)
        audio_model_path = Path(__file__).resolve().parent / "synthia/model/deep_audio_model.pth"
        model.load_state_dict(torch.load(audio_model_path, map_location=device))
        model.eval()

        try:
            mel_tensor, raw_audio = preprocess_audio(uploaded_audio)
            mel_tensor = mel_tensor.to(device)
        except Exception as e:
            st.error(f"Failed to process audio: {e}")
            st.stop()

        with torch.no_grad():
            output = model(mel_tensor).squeeze()
            prob = torch.sigmoid(output).item()
            confidence = prob * 100
            prediction = int(prob > threshold)

        st.markdown(f" Confidence Score: `{prob:.2f}`")
        if prediction == 0:
            st.success(" Prediction: Real Audio")
        else:
            st.error(" Prediction: Fake Audio")

        st.audio(uploaded_audio)
        plot_waveform(raw_audio)
        plot_melspectrogram(raw_audio)

        waveform_img_path, mel_img_path = save_audio_visualizations(uploaded_audio)
        # PDF report for audio (reuse your existing function)
        report_path = generate_audio_report(prediction, confidence, [waveform_img_path, mel_img_path])

        with open(report_path, "rb") as f:
            st.download_button("Download PDF Report", f, file_name="audio_report.pdf", mime="application/pdf")

        st.subheader("Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", "92.08%")
        col2.metric("F1 Score", "0.9212")
        col3.metric("Loss", "0.1666")

        st.markdown("   Model Comparison (Simulated)")
        models = ["SimpleNN", "EnhancedCNN", "LSTM"]
        scores = [88.2, 92.1, 90.0]
        fig, ax = plt.subplots()
        sns.barplot(x=models, y=scores, palette="viridis", ax=ax)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Model Comparison")
        st.pyplot(fig)

        # Save to history
        st.session_state.history.append({
            "type": "Audio",
            "prediction": "Fake" if prediction == 1 else "Real",
            "confidence": f"{prob:.2f}"
        })

# ==== History Display ====
if st.session_state.history:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Recent Predictions")
    for i, record in enumerate(reversed(st.session_state.history[-5:])):
        if record["type"] == "Image":
            st.sidebar.write(f"🖼️ Image: {record['prediction']}")
        else:
            st.sidebar.write(f"{record['type']} - {record.get('prediction', '')} ({record.get('confidence', '')})")
