import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import torch
import cv2
import numpy as np
from torchvision import transforms, models
from PIL import Image
import tempfile
import os
from utils.video_utils import extract_frames 
from utils.image_utils import preprocess_image 
# from utils.detection import predict_frame 
from utils.detection import detect_image_file, predict_frame 
from utils.model_loader import load_deepfake_model
from utils.report import generate_report

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_deepfake_model(device)

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Streamlit UI
st.title("🧠 Deepfake Detection")
st.write("Upload an image or video to check for real vs fake content.")
option = st.radio("Choose input type:", ("Image", "Video"))

# ---------- IMAGE ----------
if option == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_tensor = transform(image).unsqueeze(0).to(device)

        label = predict_frame(model, img_tensor)
        st.markdown(f"### Prediction: `{label}`")

# ---------- VIDEO ----------
elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

    FRAME_SKIP = 10

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_video.read())
            video_path = temp_file.name

        st.video(uploaded_video)

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        real_count = 0
        fake_count = 0
        suspicious_frames = []  # ⬅️ Step 1: Collect fake frames

        st.write("Processing video frames...")

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
                    suspicious_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # ⬅️ Step 2: Add fake frame

            frame_count += 1

        cap.release()
        os.remove(video_path)

        st.write(f"🟢 Real frames: {real_count}")
        st.write(f"🔴 Fake frames: {fake_count}")

        prediction = 1 if fake_count > real_count else 0
        confidence = (fake_count / (real_count + fake_count)) * 100 if (real_count + fake_count) > 0 else 0
        top_frames = suspicious_frames[:3]  # ⬅️ Step 3: Pick top N suspicious frames

        report_path = generate_report(prediction, confidence, top_frames)
        st.markdown(f"### Final verdict: `{ 'Fake' if prediction == 1 else 'Real' }`")
        st.download_button("📄 Download Report", data=open(report_path, "rb").read(), file_name="synthia_report.pdf")
