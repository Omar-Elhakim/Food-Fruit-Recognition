
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
model_path = "Models/Binary_Food_Fruit_Classification_model.pth" 
test_image = "Project Data/Fruit/Validation/Avocado/Images/66.jpg"          


norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = models.efficientnet_b0(weights=None) 
    num_features = model.classifier[1].in_features
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, 2)
    )
    
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

def predict_image(path):
    model=load_model(model_path)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    
    img = Image.open(path).convert("RGB")
    img_tensor = test_transforms(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, pred = torch.max(probabilities, 1)
        
    label = "Food" if pred.item() == 0 else "Fruit"
    return f"Prediction: {label}"

if __name__ == "__main__":
    print(predict_image(test_image))