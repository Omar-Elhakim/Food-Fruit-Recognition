import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
MODEL_PATH = 'Models/Binary_Segmentation_model.pth'
IMAGE_PATH = 'Project Data/Fruit/Validation/Carambola/Images/96.jpg' # Replace with actual path
ENCODER = 'resnet34'  # Must match training!
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model():
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,   # We load our own weights, so no need to download ImageNet
        in_channels=3,
        classes=1
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    # 3. Move to GPU/CPU and set to Evaluation Mode
    model.to(DEVICE)
    model.eval()
    
    return model

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img, (256, 256)) 
    
    img_norm = img.astype('float32') / 255.0
    
    img_transposed = np.moveaxis(img_norm, -1, 0)
    
    img_tensor = torch.from_numpy(img_transposed).unsqueeze(0).float()
    
    return img, img_tensor.to(DEVICE)

def predict(img_path):
    _,img_tensor=preprocess_image(img_path)
    model = load_model()

    with torch.no_grad():
        logits = model(img_tensor)
        
        pr_masks = logits.sigmoid()
        pred_mask = (pr_masks > 0.5).float()

        result = pred_mask.squeeze().cpu().numpy()

    return result

if __name__ == "__main__":
    original_img, _ = preprocess_image(IMAGE_PATH)

    mask = predict(IMAGE_PATH)
    
    # 4. Visualize
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(original_img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    plt.show()
    print("Inference Complete.")