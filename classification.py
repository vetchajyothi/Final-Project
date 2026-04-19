import torch
import torch.nn as nn
from torchvision import models

class StrokeClassifier(nn.Module):
    """
    CNN for classifying Normal vs Stroke.
    Uses pre-trained ResNet-50 as the backbone for feature extraction.
    """
    def __init__(self, num_classes=2):
        super(StrokeClassifier, self).__init__()
        # Load pre-trained ResNet50
        self.model = models.resnet50(pretrained=True)
        
        # Replace the final fully connected layer to output num_classes (2)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

class StrokeTypeClassifier(nn.Module):
    """
    CNN for classifying Ischemic vs Hemorrhagic stroke.
    Also uses ResNet-50, but trained on a different objective.
    """
    def __init__(self, num_classes=2):
        super(StrokeTypeClassifier, self).__init__()
        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

def predict_class(model, image_tensor, class_names):
    """
    Helper function to run inference on a PyTorch model and return the predicted class name.
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        # outputs shape: [1, num_classes]
        _, predicted = torch.max(outputs, 1)
        # return the string label
        return class_names[predicted.item()]
