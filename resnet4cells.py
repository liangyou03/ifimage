
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

resnet18 = models.resnet18(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(resnet18.children())[:-1])
feature_extractor.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    """
    Extract features from an image using ResNet18
    
    Args:
        image_path (str): Path to input image file
        
    Returns:
        torch.Tensor: Extracted features (512-dimensional vector)
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    
    # Extract features
    with torch.no_grad():
        features = feature_extractor(input_batch)
    
    # Flatten features to 1D vector
    features = torch.flatten(features, 1)
    
    return features

# features = extract_features('example.jpg')
# print(f"Extracted features shape: {features.shape}")
# print(f"Sample features: {features[0, :10]}")