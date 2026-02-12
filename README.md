# Vision Transformer for CIFAR-10 (Frozen Backbone)

This project is a practical, beginner-friendly implementation of Vision Transformers (ViT) on CIFAR-10.
It uses a pre-trained ViT backbone and trains only the final classifier head (linear probing), which keeps training simpler and cheaper.

## In Simple Words

- The model receives CIFAR-10 images.
- Images are resized to `224 x 224` to match ViT input size.
- The frozen backbone extracts visual features.
- A small trainable head maps those features to 10 CIFAR-10 classes.
- Training prints loss/accuracy and saves checkpoints.

## Project Scope

- Dataset: CIFAR-10
- Backbone: `torchvision.models.vit_b_16` with ImageNet pre-trained weights
- Training strategy: frozen backbone + trainable classifier head
- Input resolution: `224 x 224`
- Patch size: `16 x 16`

## Repository Structure

- `scripts/train.py`: training entrypoint
- `src/data/dataset.py`: CIFAR-10 transforms and dataloaders
- `helping_functions/engine.py`: train/test loops and metric tracking
- `src/models/`: ViT building blocks and custom ViT implementation
- `src/utils.py`: reproducibility, device selection, and checkpoint saving helpers
- `notebooks/`: exploratory experiments

## Requirements

### System

- Python `>=3.12, <3.15`
- `pip`
- Optional: CUDA-capable GPU

### Python Packages

Dependencies are defined in `requirements.txt`:

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

```bash
python scripts/train.py \
  --epochs 5 \
  --batch-size 32 \
  --lr 1e-4 \
  --image-size 224 \
  --save-path checkpoints/vit_cifar10_frozen.pth
```

## Outputs

- Prints epoch-level train/test loss and accuracy
- Saves model weights to `--save-path`

## Notes

- CIFAR-10 is downloaded to `./data` if missing.
- Accuracy is computed at epoch level with `torchmetrics`.
- `scripts/inference.py` is a placeholder.
