import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Load pretrained VGG16 model
vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
vgg.classifier = torch.nn.Identity()  # remove classifier
vgg.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = vgg(image)

    return features.numpy()

# Simple caption generator (rule-based)
def generate_caption(features):
    strength = np.mean(features)

    if strength > 0.5:
        return "A clear image with an object in focus."
    elif strength > 0.2:
        return "An image showing something interesting."
    else:
        return "A simple image with basic details."

# Main
img = "sample.jpg"   # your image path
features = extract_features(img)
caption = generate_caption(features)

print("Generated Caption:")
print(caption)