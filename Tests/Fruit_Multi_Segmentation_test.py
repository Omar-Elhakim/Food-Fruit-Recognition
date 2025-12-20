import cv2
import numpy as np
import colorsys
import os
from tensorflow.keras.models import load_model  
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)
num_classes=31

def create_distinct_color_map(num_classes):
    """
    Create highly distinct colors for each class
    Background = black
    Fruits = visually different colors
    """
    colors = np.zeros((num_classes, 3), dtype=np.uint8)

    # Background
    colors[0] = [0, 0, 0]

    # Strong, visually distinct base palette
    base_palette = [
        (230, 25, 75),   # Red
        (60, 180, 75),   # Green
        (0, 130, 200),   # Blue
        (245, 130, 48),  # Orange
        (145, 30, 180),  # Purple
        (70, 240, 240),  # Cyan
        (240, 50, 230),  # Magenta
        (210, 245, 60),  # Lime
        (250, 190, 190), # Pink
        (0, 128, 128),   # Teal
        (230, 190, 255), # Lavender
        (170, 110, 40),  # Brown
        (255, 250, 200), # Beige
        (128, 0, 0),     # Maroon
        (170, 255, 195), # Mint
        (128, 128, 0),   # Olive
        (255, 215, 180), # Apricot
        (0, 0, 128),     # Navy
        (128, 128, 128), # Gray
        (255, 255, 255), # White
        (255, 0, 0),     # Strong Red
        (0, 255, 0),     # Strong Green
        (0, 0, 255),     # Strong Blue
        (255, 255, 0),   # Yellow
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Cyan
        (255, 165, 0),   # Orange Strong
        (138, 43, 226),  # Blue Violet
        (34, 139, 34),   # Forest Green
        (255, 20, 147),  # Deep Pink
    ]

    for i in range(1, num_classes):
        if i-1 < len(base_palette):
            colors[i] = base_palette[i-1]
        else:
            # fallback (Golden Ratio) لو زاد العدد
            hue = (i * 0.618033988749895) % 1
            r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
            colors[i] = [int(r*255), int(g*255), int(b*255)]

    return colors

COLOR_MAP = create_distinct_color_map(num_classes)

def mask_to_color(mask):
    h,w = mask.shape
    out = np.zeros((h,w,3), dtype=np.uint8)
    for c in range(num_classes):
        out[mask==c] = COLOR_MAP[c]
    return out

TRAIN_DIR = "Project Data/Fruit/Train"
fruits = sorted(os.listdir(TRAIN_DIR))

class_map = {fruit: i+1 for i, fruit in enumerate(fruits)}
reverse_class_map = {v: k for k, v in class_map.items()}

def extract_detected_fruits(mask, min_area_ratio=0.01):
    """
    mask: predicted class mask (H,W)
    min_area_ratio: minimum % of image area to count class
    """
    h, w = mask.shape
    total_pixels = h * w

    detected = []

    for c in np.unique(mask):
        if c == 0:
            continue

        area = np.sum(mask == c)
        ratio = area / total_pixels

        if ratio >= min_area_ratio:
            detected.append(reverse_class_map[c])

    return detected

def test_multi_fruit_image(image_path):
    """
    Test semantic segmentation on image containing multiple fruits
    """

    model = load_model("Models/Multi-Segmentation-model.h5")
    # -------- Read image --------
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError("Image not found!")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # -------- Preprocess --------
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    inp = img_resized.astype(np.float32) / 255.0
    inp = np.expand_dims(inp, axis=0)

    # -------- Predict --------
    pred = model.predict(inp, verbose=0)[0]
    pred_mask = np.argmax(pred, axis=-1)

    # -------- Resize mask back --------
    pred_mask = cv2.resize(
        pred_mask.astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    # -------- Color mask --------
    colored_mask = mask_to_color(pred_mask)

    # -------- Overlay --------
    overlay = cv2.addWeighted(
        img_rgb.astype(np.uint8), 0.6,
        colored_mask.astype(np.uint8), 0.4, 0
    )

    # -------- Detected classes --------
    detected_fruits = extract_detected_fruits(
    pred_mask,
    min_area_ratio=0.01   # 1% من الصورة
)

    return img_rgb, colored_mask, overlay, detected_fruits

if __name__ == "__main__":
    img, seg, overlay, fruits_found = test_multi_fruit_image('Test Cases Structure/Integerated Test/img1_100g.jpg')

    print("Detected fruits:", fruits_found)

    plt.figure(figsize=(18,5))
    plt.subplot(1,3,1); plt.imshow(img); plt.title("Original"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(seg); plt.title("Multi-Class Segmentation"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(overlay); plt.title("Overlay"); plt.axis("off")
    plt.show()
