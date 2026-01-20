import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import base64
from io import BytesIO
import pandas as pd

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'synthia')))

import torch
import cv2
import numpy as np
import tempfile
from torchvision import transforms
from pathlib import Path
import datetime

# Synthia utils
from synthia.utils.video_utils import extract_frames #future use
from synthia.utils.image_utils import preprocess_image
from synthia.utils.detection import predict_frame
from synthia.utils.model_loader import load_deepfake_model
from synthia.utils.report import generate_report, generate_audio_report,image_report
from synthia.utils.audio_utils import save_audio_visualizations
from synthia.model import EnhancedCNN
from synthia.utils.image2 import load_model, preprocess_image, predict

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


# Setup history in session state
if "history" not in st.session_state:
    st.session_state.history = []
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])



# Logo and fonts
logo_url = "C://Users/prash/Desktop/pp/synthia/tamplates/383276570968956934.png"  

# Custom CSS for font, navbar hover effect, and footer
def local_css():
    st.markdown("""
    <style>
    # /* Custom font for SYNTHIA - Use a techy/deepfake style font, e.g. Orbitron or similar */
    # @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap');

    
    /* Sidebar */
    .sidebar .sidebar-content {
        padding: 15px;
        background: #f0f2f6;
        border-radius: 8px;
        margin-top: 10px;
    }
    /* Footer styling */
    footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #222222;
        color: white;
        text-align: center;
        padding: 10px 0;
        font-family: 'Orbitron', sans-serif;
        font-size: 14px;
        opacity: 0.9;
        z-index: 9999;
    }
    footer a {
        color: #f72585;
        text-decoration: none;
        font-weight: bold;
    }
    footer a:hover {
        text-decoration: underline;
    }
    
    .navbar img.logo {
            height: 40px;
            margin-right: 15px;
            position: fixed;
        }
        .navbar h1 {
            font-size: 24px;
            margin: 0;
            
        }
    <style>
    .navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-image: url('{logo_url}');
        background-size: cover;
        background-position: center;
        padding: 10px 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
       }
    .logo {
        height: 40px;
    }
    .navbar h1 {
        color: white;
        font-size: 1.5rem;
        margin: 0;
     }
    </style>
    <div class="navbar">
    </div>
    
    """, unsafe_allow_html=True)

def footer():
    st.markdown("""
    <style>
    .footer {
        margin-top: 80px;
        padding: 20px 40px;
        background-color: dark-grey;
        color: white;
        font-family: 'Orbitron', sans-serif;
        font-size: 16px;
        line-height: 1.5;
        border-top-left-radius: 12px;
        border-top-right-radius: 12px;
    }
    .footer .container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        max-width: 1200px;
        margin: auto;
        gap: 80px;
    }
    .footer .left, .footer .center, .footer .right {
        flex: 1;
        min-width: 220px;
    }
    .footer .left h4, .footer .center h4, .footer .right h4 {
        margin-bottom: 10px;
        font-weight: 700;
        letter-spacing: 1.2px;
        font-size: 18px;
    }
    .footer a {
        color: #fce38a;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    .footer a:hover {
        color: #f72585;
        text-decoration: underline;
    }
    .footer .social-icons span {
        margin-right: 15px;
        font-size: 20px;
        cursor: pointer;
        user-select: none;
        transition: color 0.3s ease;
    }
    .footer .social-icons span:hover {
        color: #f72585;
    }
    </style>

    <div class="footer">
      <div class="container">
        <div class="left">
          <h4>Contact Us</h4>
          <p>Email: <a href="mailto:contact@synthia.ai">contact@synthia.ai</a></p>
          <p>Phone: +91 98765 43210</p>
          <p>Address: 123 AI Street, Tech City, India</p>
        </div>
        <div class="right">
          <h4>Follow Us</h4>
          <div class="social-icons">
            <span title="Twitter">🐦</span>
            <span title="LinkedIn">🔗</span>
            <span title="GitHub">🐙</span>
            <span title="Facebook">📘</span>
          </div>
          <p style="margin-top: 15px;">© 2025 SYNTHIA. All rights reserved.</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def home_page():
    st.title("Welcome to SYNTHIA - Deepfake Detection Platform")
    
    st.markdown("""
        <style>
        .hover-block {
            transition: transform 0.3s ease;
        }
        .hover-block:hover h1, 
        .hover-block:hover h2, 
        .hover-block:hover h3, 
        .hover-block:hover p, 
        .hover-block:hover li {
            transform: scale(1.1);
            transition: transform 0.3s ease;
        }
        </style>
    """, unsafe_allow_html=True)
      
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="hover-block">', unsafe_allow_html=True)
        st.header("What is Deepfake?")
        st.write("""
        Deepfake technology uses AI to create hyper-realistic fake videos or audios.  
        Detecting these manipulated media is critical for information authenticity.  
        SYNTHIA helps identify deepfakes through advanced image, video, and audio analysis.
        """)
    with col2:
        st.image("synthia/tamplates/video2.gif", width=350) 
    
    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        st.video("https://www.youtube.com/watch?v=BuufkPTFt0E")  
    with col4:
        st.markdown('<div class="hover-block">', unsafe_allow_html=True)
        st.header("How SYNTHIA Works")
        st.write("""
        1. Extract frames and audio features from media files.  
        2. Use trained models to classify real vs fake.  
        3. Provide detailed reports with confidence scores.  
        4. Continuously improve detection with latest AI advances.
        """)
    st.markdown("---")
    
    col5, col6 = st.columns(2)
    with col6:      
        st.image("synthia/tamplates/video1.gif",width=350) 
    with col5:
        st.markdown('<div class="hover-block">', unsafe_allow_html=True)
        st.header("Key Features")
        st.write("""
        - Real-time video and audio analysis
        - Multi-modal deepfake detection (image, audio, video)
        - Easy report download (PDF, JSON)
        - Accurate detection scores and explanations         
                 
       """)
    st.markdown("---")           
      
def feature_page(selected_tab):
        # Initialize session variables
    if "history" not in st.session_state:
        st.session_state.history = []
    if "reports" not in st.session_state:
        st.session_state.reports = []
        
    st.title(f"{selected_tab} - Deepfake Detection")
    if selected_tab == "image":
        st.header("Deepfake Image Detection")
        uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_files and st.button("Predict"):
            images = []
            img_names = []
            model_name = st.selectbox("Select Model", ["ResNet50", "MobileNetV2"])
            model = load_model(model_name)

            for file in uploaded_files:
                image = Image.open(file).convert('RGB')
                st.image(image, caption=f"Uploaded: {file.name}", use_container_width=True)
                img_tensor = preprocess_image(image)
                images.append(img_tensor)
                img_names.append(file.name)
    
            predictions = predict(model, images)

            # Display results
            st.write(" Results")
            results = []
            probs=[]
            for name, (label, prob) in zip(img_names, predictions):
                if({label}=="REAL"):
                    st.write(f"{label}   (Probability: {prob:.2f})")
                    st.success(f"Prediction is {label}image")
                else:
                    st.write(f"{label}   (Probability: {prob:.2f})")
                    st.success(f"Prediction is {label}image")    
                results.append({
                    "Image Name": name,
                    "Prediction": label,
                    "Probability": f"{prob:.2f}"
                })
                probs.append(prob)

                # Save to history
                st.session_state.history.append({
                    "type": "Image",
                    "file_name": name,
                    "prediction": label,
                    "probability": f"{prob:.2f}"
                })
        
            os.makedirs("reports", exist_ok=True)
            pdf_filename = image_report(results, img_names, probs, output_dir="reports")
            # st.success(f"PDF report saved: {pdf_filename}")
            with open(pdf_filename, "rb") as f:
                st.download_button("Download PDF Report", data=f, file_name=pdf_filename, mime="application/pdf")
            
    elif selected_tab == "video":
        uploaded_video = st.file_uploader("Choose a video", type=["mp4", "mov", "avi"])
        FRAME_SKIP = st.sidebar.slider("Frame Sampling Interval", 1, 30, 10)
    
        if uploaded_video:
            st.video(uploaded_video)
            
            if st.button("Predict"):
                try:
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
                    
                    color = "#FF0000" if prediction == 1 else "#4CAF50"
                    st.markdown(
                    f"<p style='font-size:30px; font-weight:bold; color: {color};'>Final Verdict: {'Fake' if prediction == 1 else 'Real'}</p>",
                     unsafe_allow_html=True
                     )

                    report_path = generate_report(prediction, confidence, suspicious_frames[:3])
                    report_filename = "synthia_video_report.pdf"

                    if report_filename not in st.session_state.reports:
                        st.session_state.reports.append(report_filename)

                    with open(report_path, "rb") as f:
                        st.download_button("Download PDF Report", data=f.read(), file_name=report_filename)

                    # Save to history
                    st.session_state.history.append({
                    "type": "Video",
                    "prediction": "Fake" if prediction == 1 else "Real",
                    "confidence": f"{confidence:.2f}%"
                })

                except Exception as e:
                    st.error(f"Error processing video: {e}")
                
                     
            
    elif selected_tab == "audio":
        uploaded_audio = st.file_uploader("Choose a .wav or .mp3 file", type=["wav", "mp3"])
        # threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.67, 0.01)

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
                prediction = int(prob > 0.67) # 0.67 = threshold

            st.markdown(f" Confidence Score: `{prob:.2f}`")
            if prediction == 0:
                st.success(" Prediction: Real Audio")
            else:
                st.error(" Prediction: Fake Audio")

            st.audio(uploaded_audio)
            plot_waveform(raw_audio)
            plot_melspectrogram(raw_audio)

            waveform_img_path, mel_img_path = save_audio_visualizations(uploaded_audio)
            
            
            report_path = generate_audio_report(prediction, confidence, [waveform_img_path, mel_img_path])
            report_filename = "audio_report.pdf"
            if report_filename not in st.session_state.reports:
               st.session_state.reports.append(report_filename)              
            with open(report_path, "rb") as f:
                st.download_button("Download PDF Report", f, file_name=report_filename, mime="application/pdf")

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
            
    elif selected_tab == "lipsync":
        st.subheader("Voice to Lip Sync Checker")
        st.info("This feature is coming soon. Stay tuned!")
        st.write("This feature is coming soon. Stay tuned!")


def about_page():
    # 💡 Global styling with CSS
    st.markdown("""
    <style>
    /* Standard font for the about page */
    .about-content {
        font-family: Arial, sans-serif;
        font-size: 15px;
        color: #ddd;
        line-height: 1.6;
    }
    .contributor-img {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        object-fit: cover;
        display: block;
    }
    </style>
    """, unsafe_allow_html=True)

    # 👇 Wrapping everything in a div for consistent font styling
    st.markdown('<div class="about-content">', unsafe_allow_html=True)

    st.title("About SYNTHIA")
    st.write("""
    **SYNTHIA** is an advanced deepfake detection platform built to combat misinformation and enhance digital trust.

    This app leverages state-of-the-art AI models to detect deepfakes in images, videos, and audio. It empowers journalists, researchers, and content creators to verify the authenticity of digital media with ease and confidence.
    
    ---
    """)

    st.header("✨ Features at a Glance")
    st.markdown("""
    -   Multi-modal Detection:  Analyze images, videos, and audio recordings for deepfake traces.
    -   Confidence Scores:  Get detailed predictions with confidence levels.
    -   Visual Analysis:  Explore extracted frames and confusion matrices.
    -   Downloadable Reports:  Generate comprehensive PDF reports.
    -   User-friendly Interface:  Designed for accessibility and ease of use.
    """)

    st.header("📚 Our Models & Datasets")
    st.markdown("""
    SYNTHIA utilizes:
    - **ResNet-50** for image and video analysis.
    - **EnhancedCNN** for audio deepfake detection.
    - Trained on datasets like **Celeb-DF**, **DeepFake Detection Challenge**, and **ASVspoof**.

    While powerful, SYNTHIA is not a replacement for professional forensic analysis.
    """)

    st.header("🚀 Meet the Contributors")

    contributors = [
        {"name": "Aman Tiwari", "role": "Lead Developer", "email": "aman@gmail.com", "image": "synthia/tamplates/aman.jpg"},
        {"name": "Ashish Gupta", "role": "Researcher", "email": "guptaashish@gmail.com", "image": "synthia/tamplates/ashish.jpg"},
        {"name": "Prashant Mishra", "role": "ML developer", "email": "pm303oracle@gmail.com", "image": "synthia/tamplates/babua.jpg"},
        {"name": "Salman Ahmad", "role": "UI/UX Designer", "email": "salman@gmail.com", "image": "synthia/tamplates/salman.jpg"},
        {"name": "Harsh Mangalam Pandey", "role": "Researcher", "email": "harsh@gmail.com", "image": "synthia/tamplates/Screenshot 2025-05-26 101744.png"},
    ]

    cols = st.columns(len(contributors))

    for i, c in enumerate(contributors):
        with cols[i]:
            try:
                image = Image.open(c["image"])
                # st.image(image, width=120, caption=f"{c['name']}", use_container_width=False, output_format="PNG")
                # Apply round shape with CSS
                st.markdown(
                    f"<img src='data:image/png;base64,{image_to_base64(image)}' width='120' class='contributor-img'/>",
                    unsafe_allow_html=True
                )
            except:
                st.warning(f"Could not load image for {c['name']}")
            st.markdown(f"<b>{c['name']}</b>", unsafe_allow_html=True)    
            st.markdown(f"{c['role']}")
            st.markdown(f"<small>{c['email']}</small>", unsafe_allow_html=True)

    st.markdown("---")

    st.header("💡 Future Plans")
    st.markdown("""
    - Adding real-time detection capabilities.
    - Expanding dataset support.
    - Enhanced model explainability and visualization tools.
    - Multi-language audio detection support.
    """)

    st.header("📬 Feedback & Contact")
    st.write("""
    We’re always looking to improve!  
    Reach out to us at any of the above emails with feedback, questions, or collaboration opportunities.

    Thank you for using SYNTHIA!
    """)

    st.markdown("</div>", unsafe_allow_html=True)  # Close the about-content div


# Helper function to convert image to base64 for inline CSS display

def image_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str


def contact_page():
    st.title("📬 Contact Us")
    st.write("""
    We're excited to hear from you!  
    Please fill out the form below and we'll respond as soon as possible.
    """)

    # Create a form container
    with st.form("contact_form"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("👤 Your Name", placeholder=" Enter Your Name")
        with col2:
            email = st.text_input("📧 Your Email", placeholder=" Enter email")

        message = st.text_area("💬 Your Message", placeholder="Type your message here...", height=150)

        submitted = st.form_submit_button(" Send Message")

        if submitted:
            if name.strip() and email.strip() and message.strip():
                st.success(" Thank you for reaching out to SYNTHIA!  We will get back to you soon. ✨")
                # Add your message-saving/sending logic here
            else:
                st.error(" Please fill in all the fields before submitting.")

    # Add a section for alternative contact
    st.markdown("---")
    st.write("""
    Prefer email?  Reach out directly at [contact@synthia-ai.com](mailto:contact@synthia-ai.com).
    
    Follow us on social media:  
    [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://linkedin.com)  
    [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com)  
    [![Twitter](https://img.shields.io/badge/-Twitter-1DA1F2?logo=twitter&logoColor=white)](https://twitter.com)
    """)

    st.markdown("---")
    
def sidebar_content():
    
        # Initialize session states if not exist
    if "history" not in st.session_state:
        st.session_state.history = []
    if "reports" not in st.session_state:
        st.session_state.reports = []
        
    st.sidebar.title("Recent Detections")

    if "history" in st.session_state and len(st.session_state.history) > 0:
        for item in reversed(st.session_state.history):  # Show most recent first
            if item["type"] == "Image":
                st.sidebar.markdown(
                    f"- 🖼️ **Image** - Prediction: **{item['prediction']}**"
                )
            elif item["type"] == "Video":
                confidence = item.get("confidence", "N/A")
                st.sidebar.markdown(
                    f"- 🎥 **Video** - Prediction: **{item['prediction']}**  \n"
                    f"  Confidence: `{confidence}`"
                )
            elif item["type"] == "Audio":
                confidence = item.get("confidence", "N/A")
                st.sidebar.markdown(
                    f"- 🔊 **Audio** - Prediction: **{item['prediction']}**  \n"
                    f"  Confidence: `{confidence}`"
                )
    else:
        st.sidebar.write("No recent detections yet.")

    st.sidebar.markdown("---")
    st.sidebar.title("Downloaded Reports")

    if "reports" in st.session_state and len(st.session_state.reports) > 0:
        for report in st.session_state.reports:
            st.sidebar.markdown(f"- 📄 {report}")
    else:
        st.sidebar.write("No reports downloaded yet.")


def main():
    st.set_page_config(page_title="SYNTHIA", layout="wide", page_icon="🕵️‍♂️",initial_sidebar_state="expanded")
    # navbar()
    local_css()
    
    # Sidebar main menu
    with st.sidebar:
        page = option_menu(
            menu_title=None,
            options=["Home", "About", "Features", "Contact"],
            icons=["house", "info-circle", "stars", "envelope"],
            # menu_icon="cast",
            default_index=0,
            orientation="vertical",
             styles={
                "container": {"padding": "5px", 
                "background-color": "dark grey",
                # "font-family": 'Orbitron', sans-serif";
                },
             },
        )
        st.sidebar.markdown("---")
        # Nested menu for Features only
        feature_tab = None
        if page == "Features":
            feature_tab = option_menu(
                menu_title="Features",
                options=["Image", "Video", "Audio", "Lip Sync (Coming Soon)"],
                icons=["image", "camera-video", "mic", "play-circle"],
                default_index=0,
                orientation="vertical",

            )
            st.sidebar.markdown("---")
    sidebar_content()        

    # Main content
    if page == "Home":
        home_page()
    elif page == "About":
        about_page()
    elif page == "Features":
        feature_page(feature_tab.split(" ")[0].lower())
    elif page == "Contact":
        contact_page()

    footer()
if __name__ == "__main__":
    main()



