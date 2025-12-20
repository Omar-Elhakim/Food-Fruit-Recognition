# %%
import os
import sys
import cv2
import torch
from torchvision import transforms
import pickle
from Siamese_food_model_test import EmbeddingNet


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


def get_embeddings(path, model, device):
    model.eval()
    with torch.no_grad():
        emb = model(read_and_process_image(path).unsqueeze(0).to(device))
        return emb


def main():
    # ROOT="/kaggle/input/food-dataset/"
    ROOT = "Project Data/"
    # CONV_NET = "googlenet"
    CONV_NET = sys.argv[1]
    MODEL_WEIGHTS = sys.argv[2]
    EMBEDDING_DIM = 64
    SIZE = (244, 244)

    if not os.path.exists(MODEL_WEIGHTS):
        print(f"Error Model Weights: {MODEL_WEIGHTS} Doesn't Exist")
    print(f"CONV_NET: {CONV_NET}")

    print("Cuda Available ? ", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes_folders = sorted(os.listdir(ROOT + "Food/Train"))
    classes_list = []
    ref_embeddings = []
    model = EmbeddingNet(embedding_dimension=EMBEDDING_DIM, conv_net=CONV_NET)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, weights_only=True))
    model.to(device)

    # train_list = os.listdir(ROOT + "Food/Train")
    for cls in classes_folders:
        images = os.listdir(ROOT + "Food/Train/" + cls)
        for img in images:
            emb = get_embeddings(ROOT + "Food/Train/" + cls + "/" + img, model, device)
            classes_list.append(cls.replace("_", " "))
            ref_embeddings.append(emb)
            break  # if you want one ref_embedding per class

    ref_embeddings = torch.cat(ref_embeddings, dim=0)
    print("Ref Embedding Shape: ", ref_embeddings.shape)

    with open("Models/ref_embeddings.pth", "wb") as file:
        pickle.dump(
            {
                "ref_embeddings": ref_embeddings,
                "classes": classes_list,
            },
            file,
        )


if __name__ == "__main__":
    main()
