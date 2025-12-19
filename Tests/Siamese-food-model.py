import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import time
import sys


ROOT = "Test Cases Structure/Siamese Case II Test/"
anchor = ROOT + "Anchor.JPG"
CONV_NET = sys.argv[1]
MODEL_WEIGHTS = sys.argv[2]
if not os.path.exists(MODEL_WEIGHTS):
    print(f"Error Model Weights: {MODEL_WEIGHTS} Doesn't Exist")
print(f"CONV_NET: {CONV_NET}")


testList = os.listdir(ROOT)

SIZE = (244, 244)


def read_and_process_image(path, size=SIZE):
    if not os.path.exists(path):
        print("Error : path " + str(path) + " Doesn't exist")
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    image = transforms.ToTensor()(image)
    # normalize the image
    # image = image / 255.0
    return image


class EmbeddingNet(nn.Module):
    """
    The base network for the Siamese architecture.
    """

    def __init__(self, embedding_dimension=128, conv_net="resnet50"):
        super(EmbeddingNet, self).__init__()

        mdls = {
            "vgg16": {
                "model": models.vgg16,
                "params": {
                    "weights": "IMAGENET1K_V1",
                },
            },
            "resnet50": {
                "model": models.resnet50,
                "params": {
                    "weights": "IMAGENET1K_V2",
                },
            },
            "mobilenetv3": {
                "model": models.mobilenet_v3_large,
                "params": {
                    "weights": "IMAGENET1K_V2",
                },
            },
            "googlenet": {
                "model": models.googlenet,
                "params": {
                    "aux_logits": False,
                },
            },
        }

        m = mdls[conv_net]
        self.model = m["model"](m["params"])

        # freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # different model have different names for lastest layers
        if "resnet" in conv_net or "googlenet" in conv_net:
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, embedding_dimension)
        elif "mobilenet" in conv_net:
            num_ftrs = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(num_ftrs, embedding_dimension)
        elif "vgg" in conv_net:
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, embedding_dimension)

    def forward(self, x):
        # Pass input through the modified Model
        x = self.model(x)

        # Normalize the embeddings (optional but often beneficial)
        x = F.normalize(x, p=2, dim=1)
        return x


def classify(query_img, support_embeddings, support_labels):
    model.eval()
    with torch.no_grad():
        query_emb = model(query_img)

    distances = torch.cdist(query_emb, support_embeddings)
    idx = distances.argmin()
    return support_labels[idx]


# --- 3. Instantiate and Use the Model ---

# Define the embedding dimension (e.g., 128)
EMBEDDING_DIM = 128

# Define a device
print("Cuda Available ? ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the Embedding Network
model = EmbeddingNet(embedding_dimension=EMBEDDING_DIM, conv_net=CONV_NET)
model.load_state_dict(torch.load(MODEL_WEIGHTS, weights_only=True,map_location=device))
model.to(device)

print(f"Model instantiated on: {device}")
print(f"Output embedding size: {EMBEDDING_DIM}")


try:
    testList.remove("Anchor.JPG")
    print("list: ", testList)
except Exception as e:
    print("Error: ", e)

query_img = read_and_process_image(ROOT + "Anchor.JPG").unsqueeze(0).to(device)
print(f"Image shape: {query_img.shape}")

ref_embeddings = []
for img_path in testList:
    img = read_and_process_image(ROOT + img_path).unsqueeze(0).to(device)
    emb = model(img)
    ref_embeddings.append(emb)

ref_embeddings = torch.stack(ref_embeddings)

label = classify(query_img, ref_embeddings, testList)
print("Label: ", label)
