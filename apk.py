import streamlit as st
import pandas as pd
import datetime
from PIL import Image
from synthia.utils.image2 import load_model, preprocess_image, predict

st.title("🔥 Deepfake Image Detector")

model_name = st.selectbox("Select Model", ["ResNet50", "MobileNetV2"])

# Load model
model = load_model(model_name)

uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files and st.button("Predict"):
    images = []
    img_names = []
    for file in uploaded_files:
        image = Image.open(file).convert('RGB')
        img_tensor = preprocess_image(image)
        images.append(img_tensor)
        img_names.append(file.name)
    
    predictions = predict(model, images)
    
    # Display results
    st.write("## Results")
    results = []
    for name, (label, prob) in zip(img_names, predictions):
        st.write(f"**{name}** → **{label}** (Probability: {prob:.2f})")
        results.append({
            "Image Name": name,
            "Prediction": label,
            "Probability": prob
        })
    
    # Save logs
    df_results = pd.DataFrame(results)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"prediction_logs_{timestamp}.csv"
    df_results.to_csv(log_filename, index=False)
    
    st.success(f"Logs saved: {log_filename}")
    st.dataframe(df_results)
