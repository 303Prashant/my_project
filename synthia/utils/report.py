import os
import tempfile
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import datetime

import matplotlib.pyplot as plt
from fpdf import FPDF
from io import BytesIO


def generate_report(prediction, confidence, top_frames):
   
    result = "FAKE" if prediction == 1 else "REAL"
    report_path = os.path.join(tempfile.gettempdir(), "synthia_report.pdf")
    c = canvas.Canvas(report_path, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "Synthia Deepfake Detection Report")

    # Prediction details
    c.setFont("Helvetica", 14)
    c.drawString(50, height - 100, f"Prediction: {result}")
    c.drawString(50, height - 130, f"Confidence: {confidence:.2f}%")

    # Frames section
    y = height - 180
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Suspicious Frames:")
    y -= 20

    for idx, frame in enumerate(top_frames):
        temp_img_path = os.path.join(tempfile.gettempdir(), f"frame_{idx}.jpg")
        # Convert RGB to BGR for OpenCV saving
        cv2.imwrite(temp_img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        img = ImageReader(temp_img_path)
        c.drawImage(img, 50, y - 150, width=200, height=150)

        y -= 170
        if y < 100:
            c.showPage()
            y = height - 100

        os.remove(temp_img_path)

    c.save()
    return report_path




def generate_audio_report(prediction, confidence, image_paths):
    
    result = "FAKE" if prediction == 1 else "REAL"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(tempfile.gettempdir(), f"synthia_audio_report_{timestamp}.pdf")
    c = canvas.Canvas(report_path, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "Synthia Audio Deepfake Detection Report")

    # Prediction details
    c.setFont("Helvetica", 14)
    c.drawString(50, height - 100, f"Prediction: {result}")
    c.drawString(50, height - 130, f"Confidence: {confidence:.2f}%")

    # Images section (waveform and mel spectrogram)
    y = height - 180
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Audio Visualizations:")
    y -= 20

    max_width = 450
    max_height = 180

    for img_path in image_paths:
        if os.path.exists(img_path):
            img = ImageReader(img_path)
            # Preserve aspect ratio
            from PIL import Image
            with Image.open(img_path) as pil_img:
                iw, ih = pil_img.size
                aspect = iw / ih
                if iw > max_width:
                    iw = max_width
                    ih = iw / aspect
                if ih > max_height:
                    ih = max_height
                    iw = ih * aspect
            c.drawImage(img, 50, y - ih, width=iw, height=ih)
            y -= ih + 30
            if y < 100:
                c.showPage()
                y = height - 100

    c.save()
    return report_path

def image_report(results, img_names, probs, output_dir="reports"):
    # Save bar graph to image buffer
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(img_names, probs, color='skyblue')
    ax.set_xlabel('Probability')
    ax.set_title('Deepfake Detection Probabilities')
    ax.set_xlim(0, 1)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    buf = os.path.join(output_dir, "graph.png")
    plt.savefig(buf, format='png')
    plt.close(fig) 

    # Generate PDF
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"{output_dir}/deepfake_detection_report_{timestamp}.pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Deepfake Detection Report", ln=True, align="C")
    pdf.ln(10)

    for r in results:
        pdf.cell(0, 10, txt=f"{r['Image Name']}: {r['Prediction']} (Prob: {r['Probability']})", ln=True)

    # Insert graph
    pdf.ln(10)
    pdf.image(buf, x=10, w=pdf.w - 20)

    # Save PDF
    pdf.output(pdf_filename)
    
    os.remove(buf)
    return pdf_filename
