from torchvision import models,transforms,datasets
import torch
import torch.nn as nn
from PIL import Image

#to get the classes' names
fruit_multiclass_train = datasets.ImageFolder(
    root="Project Data/Fruit/Train",
)
fruit_names = fruit_multiclass_train.classes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])


def test_image(img_path, model_path="Models/best_fruit_model.pth", class_names=fruit_names):
    
    # Load image
    img = Image.open(img_path).convert("RGB")
    img_tensor = train_transforms(img).unsqueeze(0).to(device)
    
    # Load model
    model_test = models.resnet18(weights=None)
    num_ftrs = model_test.fc.in_features
    model_test.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 30)
    )
    
    model_test.load_state_dict(torch.load(model_path, map_location=device))
    model_test.to(device)
    model_test.eval()
    
    # Predict
    with torch.no_grad():
        outputs = model_test(img_tensor)
        z, predict = torch.max(outputs, 1)

    predicted_class = class_names[predict.item()]
    print(f"Predicted Class: {predicted_class}")
    return predicted_class
    
if __name__ == "__main__":
    test_image('Project Data/Fruit/Validation/Apple_Gala/Images/59.jpg')
