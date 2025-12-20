import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import sys
import warnings
import pickle


def read_and_process_image(path, size=(244, 244)):
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
            "mobilenetv3": {
                "model": models.mobilenet_v3_large,
                "params": {},
            },
            "googlenet": {
                "model": models.googlenet,
                "params": {
                    "init_weights": False,
                    "aux_logits": False,
                },
            },
        }

        m = mdls[conv_net]
        self.model = m["model"](**m["params"])

        # different model have different names for lastest layers
        if "googlenet" in conv_net:
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, embedding_dimension)
        elif "mobilenet" in conv_net:
            num_ftrs = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(num_ftrs, embedding_dimension)

    def forward(self, x):
        # Pass input through the modified Model
        x = self.model(x)

        # Normalize the embeddings (optional but often beneficial)
        x = F.normalize(x, p=2, dim=1)
        return x


def classify(
    query_img, model=None, support_embeddings=None, support_labels=None, loss_thr=1.8
):
    prinT = 0
    if not model:
        return

    model.eval()
    with torch.no_grad():
        query_emb = model(query_img)

    dists = torch.cdist(query_emb, support_embeddings).squeeze(0)
    idx = dists.argmin()
    dm = dists[idx]

    # idx = distances.argmin()
    if prinT:
        print(f"labels: {support_labels}")
        print(f"dist: {dists}")

    if dm >= loss_thr:
        return "No Match"
    else:
        return support_labels[idx]

    # return support_labels[idx]


def load_model(CONV_NET, MODEL_WEIGHTS, device):
    model = EmbeddingNet(embedding_dimension=64, conv_net=CONV_NET)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, weights_only=True))
    model.to(device)
    return model


def predict(img_path, device, model=None, ref_embeddings=None) -> str:
    warnings.filterwarnings("ignore")
    query_img = read_and_process_image(img_path).unsqueeze(0).to(device)
    if not model:
        model = load_model("googlenet", "Models/best_googlenet_2248.pth", device)
    if not ref_embeddings:
        path = "Models/ref_embeddings.pth"
        if not os.path.exists(path):
            print("Error: " + str(path) + " Doesn't Exist")
            return "No match"
        with open(path, "rb") as file:
            ref_embeddings = pickle.load(file)

    type = classify(
        query_img, model, ref_embeddings["ref_embeddings"], ref_embeddings["classes"]
    )
    return type


if __name__ == "__main__":
    ROOT = "Test Cases Structure/Siamese Case II Test/"
    anchor = ROOT + "Anchor.JPG"
    CONV_NET = sys.argv[1]
    MODEL_WEIGHTS = sys.argv[2]
    EMBEDDING_DIM = 64
    SIZE = (244, 244)

    if not os.path.exists(MODEL_WEIGHTS):
        print(f"Error Model Weights: {MODEL_WEIGHTS} Doesn't Exist")
    print(f"CONV_NET: {CONV_NET}")

    torch.manual_seed(40)
    torch.cuda.manual_seed_all(40)
    print("Cuda Available ? ", torch.cuda.is_available())

    testList = os.listdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(CONV_NET, MODEL_WEIGHTS, device)

    print(f"Model instantiated on: {device}")

    try:
        testList.remove("Anchor.JPG")
    except Exception as e:
        print("Error: ", e)

    ref_embeddings = []
    for img_path in testList:
        img = read_and_process_image(ROOT + img_path).unsqueeze(0).to(device)
        emb = model(img)
        ref_embeddings.append(emb)

    ref_embeddings = torch.cat(ref_embeddings, dim=0)

    label = predict(
        ROOT + "Anchor.JPG",
        device,
        model,
        ref_embeddings={"ref_embeddings": ref_embeddings, "classes": testList},
    )
    print("Label: ", label)
