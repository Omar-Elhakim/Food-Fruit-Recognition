# Food/Fruit Recognition and Calorie Estimation

A multi-model computer vision pipeline that decides whether a meal photo shows
**food** or **fruit**, recognizes the specific type, segments fruit from the
background, and estimates calories from the weight encoded in the image name.

## Overview

This is a Computer Vision course project (2026). The goal is to automate dietary
tracking: given a photograph of a meal, the system identifies what is in it and
how many calories it contains.

Recognition happens in two stages. The first stage classifies an image as food
or fruit. The second stage depends on that result: **fruit** images go to a
fixed multi-class classifier, while **food** images go to a one-shot Siamese
model — food categories constantly introduce novel dishes, so the model must
recognize unseen classes by comparison rather than a fixed label set. For fruit
images the pipeline also produces a binary mask (fruit vs. background) and a
multi-class mask (30 fruit types + background).

## Features

- **Stage 1 — Food vs. Fruit classification** (EfficientNet-B0).
- **Stage 2 (Fruit) — Fruit type classification** (ResNet18).
- **Stage 2 (Food) — One-shot food recognition** via a Siamese embedding network
  (GoogLeNet backbone) that matches a query image to reference embeddings by
  distance, returning `"No Match"` when nothing is close enough.
- **Binary fruit segmentation** (U-Net, `segmentation_models_pytorch`).
- **Multi-class fruit segmentation** into 31 classes / 30 fruits + background
  (Keras/TensorFlow model).
- **Integrated pipeline** that runs the right models per image, saves the
  segmentation masks, parses grams from the filename (e.g. `img1_180g.jpg`), and
  writes a per-image text report with type and total calories.

## Tech stack

- Python
- PyTorch + torchvision (classification, Siamese, binary segmentation)
- `segmentation_models_pytorch` (U-Net)
- TensorFlow / Keras (multi-class segmentation)
- OpenCV, NumPy, Pillow, Matplotlib

## Repository structure

```
src/         Inference scripts per part + the integrated pipeline
notebooks/   Jupyter notebooks used to train each model (with logs)
samples/     Sample images for the Siamese and integrated runs
docs/        Original project brief (ProjectDescription.pdf)
```

## Getting started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Required data and weights (not included)

Trained weights and the dataset are **not** committed to this repository
(`Models/` and `Project Data/` are git-ignored). To run the tests you need to
place:

- Trained model weights under `Models/` (e.g.
  `Models/Binary_Food_Fruit_Classification_model.pth`,
  `Models/best_googlenet_2248.pth`, `Models/ref_embeddings.pth`, …).
- The dataset under `Project Data/` following the `Food/` and `Fruit/`
  `Train`/`Validation` layout described in `docs/ProjectDescription.pdf`.

The models can be (re)trained from the notebooks in `notebooks/`.

## Usage

Run the full integrated pipeline from the repository root:

```bash
python src/integrated_pipeline.py
```

For each image in `samples/integrated/`, it classifies food vs. fruit,
recognizes the type, computes calories, and (for fruit) saves binary and
multi-class segmentation masks alongside a `.txt` result file.

The Siamese one-shot test can be run standalone with a backbone name and weights
path:

```bash
python src/food_siamese.py googlenet Models/best_googlenet_2248.pth
```

`src/get_embeddings.py` regenerates the Siamese reference embeddings from the
training set:

```bash
python src/get_embeddings.py googlenet Models/best_googlenet_2248.pth
```

## License

[MIT](LICENSE)
