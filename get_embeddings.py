# %%
import os
import sys
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import time

# %%
# ROOT="/kaggle/input/food-dataset/"
ROOT = "Project Data/"
# CONV_NET = "googlenet"
CONV_NET = sys.argv[1]
MODEL_WEIGHTS = sys.argv[2]
EMBEDDING_DIM = 64
SIZE = (244, 244)


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
                    # "weights": "IMAGENET1K_V2",
                },
            },
            "googlenet": {
                "model": models.googlenet,
                "params": {
                    # "weights": "IMAGENET1K_V1",
                    "init_weights": False,
                    "aux_logits": False,
                },
            },
        }

        m = mdls[conv_net]
        self.model = m["model"](**m["params"])

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


def get_embeddings(path, model):
    model.eval()
    with torch.no_grad():
        emb = model(read_and_process_image(path).unsqueeze(0).to(device))
        return emb


if not os.path.exists(MODEL_WEIGHTS):
    print(f"Error Model Weights: {MODEL_WEIGHTS} Doesn't Exist")
print(f"CONV_NET: {CONV_NET}")

print("Cuda Available ? ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


classes_folders = sorted(os.listdir(ROOT + "Food/Train"))
classes_list = []
ref_embeddings = []
model = EmbeddingNet(embedding_dimension=EMBEDDING_DIM, conv_net=CONV_NET)
model.load_state_dict(torch.load(MODEL_WEIGHTS, weights_only=True, map_location=device))
model.to(device)

# train_list = os.listdir(ROOT + "Food/Train")
for cls in classes_folders:
    images = os.listdir(ROOT + "Food/Train/" + cls)
    for img in images:
        emb = get_embeddings(ROOT + "Food/Train/" + cls + "/" + img, model)
        classes_list.append(cls)
        ref_embeddings.append(emb)
        break  # if you want one ref_embedding per class

ref_embeddings = torch.stack(ref_embeddings)
print("Ref Embedding Shape: ", ref_embeddings.shape)

torch.save(
    {
        "classes": classes_list,
        "ref_embeddings": ref_embeddings,
    },
    "./ref_embeddings.pt",
)

# ref_embd = torch.load("./ref_embeddings.pt")
