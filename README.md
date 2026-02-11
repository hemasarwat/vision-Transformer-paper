# Vision Transformer for CIFAR-10 (Frozen Backbone)

This project is a practical, beginner-friendly implementation of Vision Transformers (ViT) on CIFAR-10.
The idea is simple: use a strong pre-trained ViT backbone, keep it frozen, and train only a small classifier head on top.
This setup is called **linear probing** and is a good way to adapt a large model with lower training cost.

## In Simple Words

- You give the model CIFAR-10 images.
- Images are resized to `224 x 224` to match ViT input requirements.
- The frozen ViT backbone extracts useful visual features.
- A lightweight final layer learns CIFAR-10 classes from those features.
- Training reports loss/accuracy and saves model weights for reuse.

## Why This Repository

- Keep the code easy to read and modify.
- Provide a clean baseline for ViT transfer learning on small datasets.
- Offer reusable utilities for data loading, training loops, and checkpoint saving.

## Project Scope

- Dataset: CIFAR-10
- Backbone: `torchvision.models.vit_b_16` with ImageNet pre-trained weights
- Training strategy: frozen backbone + trainable classifier head
- Input resolution: `224 x 224`
- Patch size: `16 x 16`

## Repository Structure

- `scripts/train.py`: training entrypoint for CIFAR-10 linear probing
- `src/data/dataset.py`: CIFAR-10 transforms and dataloaders
- `helping_functions/engine.py`: train/test loops and metric tracking
- `src/models/`: ViT building blocks and custom ViT implementation
- `src/utils.py`: reproducibility, device selection, and model saving helpers
- `notebooks/`: experiments and exploratory work

## Requirements

### System

- Python `>=3.12, <3.15`
- `pip`
- Optional: CUDA-capable GPU for faster training

### Python Packages

The project dependencies are listed in `requirements.txt`:

- `torch`
- `torchvision`
- `torchinfo`
- `torchmetrics`
- `numpy`
- `matplotlib`
- `requests`
- `tqdm`
- `pillow`

## Installation

From the `vit-implementation` directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Run CIFAR-10 training with a frozen ViT backbone:

```bash
python scripts/train.py \
  --epochs 5 \
  --batch-size 32 \
  --lr 1e-4 \
  --image-size 224 \
  --save-path checkpoints/vit_cifar10_frozen.pth
```

### Main Training Arguments

- `--epochs`: number of training epochs
- `--batch-size`: batch size for train and test dataloaders
- `--lr`: learning rate for the classifier head optimizer
- `--image-size`: input image size (default `224`)
- `--save-path`: output path for the model weights (`.pth` or `.pt`)

## Outputs

- Prints epoch-level train/test loss and accuracy
- Saves model weights to the path provided by `--save-path`

## Notes

- CIFAR-10 is automatically downloaded to `./data` if missing.
- Accuracy is computed at the epoch level using `torchmetrics`.
- `scripts/inference.py` is currently a placeholder and not implemented yet.
