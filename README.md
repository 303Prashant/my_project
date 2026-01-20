# SYNTHIA – Deepfake Detection Platform

SYNTHIA is a comprehensive Deepfake Detection tool designed to detect manipulated media including **images**, **videos**, and **audio**. It uses deep learning techniques and computer vision/audio processing to identify synthetic content.

---

## 🚀 Features

- 🔍 **Image Deepfake Detection** using ResNet-based classifiers
- 🎥 **Video Deepfake Detection** via frame-level classification
- 🎧 **Audio Deepfake Detection** using MFCCs and CNN
- 📊 Integrated Streamlit Web App Interface
- 📁 PDF Report Generation (with prediction, model confidence, and validation plots)
- 🧪 Modular codebase for easy experimentation

---

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python
- **Deep Learning**: PyTorch, ResNet18, Enhanced CNN
- **Audio Processing**: Librosa, NumPy
- **Computer Vision**: OpenCV, PIL

---

## 🧬 Model Details
- **Image/Video**: ResNet18 trained on FaceForensics++ and custom dataset
- **Audio**: CNN trained on DeepVoice and WaveFake datasets with MFCC features

---

## 📦 Folder Structure
utils/ # All utility scripts (audio, video, image, report)
  - model/ # Trained models (.pth files) [ignored in repo]
  - tamplates/ # Sample media and GIFs for demo [ignored in repo]
  - deep.py # Main detection logic  Streamlit UI
---
## 🧪 How to Run
### 1. Clone the repository
```bash
git clone https://github.com/303Prashant/my_project.git
cd synthia
pip install -r requirements.txt
streamlit run myapp.py
```
## 📸 App Interface
![Screenshot 2025-06-08 191120](https://github.com/user-attachments/assets/ac067406-8300-484c-b805-dd691ea8c201)

## 🧠 Future Work
Integrate real-time webcam detection
Extend audio model with speaker verification
Dockerize the app for easy deployment
---
## 📫 Contact :
- Prashant Mishra
- 📧 pm303oracle@gmail.com

