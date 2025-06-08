import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image

# 🟢 Model Loader
def load_model(model_name):
    if model_name == "ResNet50":
        model = models.resnet50(weights='IMAGENET1K_V1')
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 1),
            nn.Sigmoid()
        )
        weight_path = "synthia/model/ResNet50_deepfake_detector.pth"  # your saved model
    elif model_name == "MobileNetV2":
        model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, 1),
            nn.Sigmoid()
        )
        weight_path = "synthia/model/mobilenetv2_deepfake_detector (1).pth"  # your saved model
    else:
        raise ValueError("Invalid model name!")

    # Load weights
    state_dict = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

# 🟢 Image Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0)  # batch dimension
    return img_tensor

# 🟢 Prediction
def predict(model, images):
    predictions = []
    with torch.no_grad():
        for img in images:
            output = model(img).item()
            label = "FAKE" if output > 0.5 else "REAL"
            predictions.append((label, output))
    return predictions
