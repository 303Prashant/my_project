import torch
from torchvision import models
from synthia.model import EnhancedCNN

def load_deepfake_model(device):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("synthia/model/deepfake_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from synthia.model import SimpleNN  # Ensure the class is correctly imported

# def load_audio_model(device, input_dim, num_labels):
#     model = SimpleNN(input_dim=input_dim, num_labels=num_labels)  # Initialize the model with input_dim and num_labels
#     model.load_state_dict(torch.load("model/audio_model.pth", map_location=device))  # Load weights
#     model.to(device)  # Move the model to the specified device (CPU or GPU)
#     model.eval()  # Set the model to evaluation mode
#     return model

# def load_audio_model(device):
#     model = EnhancedCNN()
#     model.load_state_dict(torch.load("model/deep_audio_model.pth", map_location="cpu"))
#     model.to(device)
#     model.eval()
#     return model