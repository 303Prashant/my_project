import torch
# from PIL import Image
# import cv2
# import librosa
# import numpy as np
# from torchvision import transforms
# import torchaudio

# # -----------------------------
# # Common image transform
# # -----------------------------
# image_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# # -----------------------------
# # Predict single image frame
# # -----------------------------
# def predict_frame(model, img_tensor):
#     with torch.no_grad():
#         output = model(img_tensor)
#         _, pred = torch.max(output, 1)
#         confidence = torch.nn.functional.softmax(output, dim=1)[0][pred.item()].item()
#     return "Real" if pred.item() == 0 else "Fake", confidence

# # -----------------------------
# # Detect from image file
# # -----------------------------
# def detect_image_file(image_path, model):
#     image = Image.open(image_path).convert("RGB")
#     img_tensor = image_transform(image).unsqueeze(0)
#     label, confidence = predict_frame(model, img_tensor)
#     return label, confidence


def predict_frame(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
    return "Real" if pred.item() == 0 else "Fake"
