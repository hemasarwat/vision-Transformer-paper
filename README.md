# Vision Transformer (ViT) Implementation
*Treating images as sequences of patches*

A PyTorch implementation of the Vision Transformer paper: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2021).


## The Core Idea:

Transformers took over NLP a few years ago. Everyone thought they were just for text. Then this paper came along and said: "What if we treat images As a patch of sentences?"

The idea is simple: chop an image into fixed-size patches (like cutting a photo into a grid), flatten each patch into a vector, and feed them to a standard Transformer. No convolutions needed. The model learns to pay attention to different parts of the image, just like it would with words in a sentence.

Turns out, when you train it on massive datasets (we're talking hundreds of millions of images), it matches or beats CNNs. so where is the catch? It needs way more data than traditional computer vision models because it doesn't have built-in assumptions about images (like "nearby pixels are related"). It learns everything from scratch.


## Architecture Overview

**ViT-Base/16 specs:**
- Image size: `224×224`
- Patch size: `16×16` → 196 patches per image
- Embedding dimension: 768
- Transformer layers: 12
- Attention heads: 12 per layer
- Total parameters: ~86M

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

### Python Packages

The project dependencies are listed in `requirements.txt`

## Installation

From the `vit-implementation` directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Run CIFAR-10 training with ViT Model:

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
- Saves model weights to the path provided by `--save-path`

## Notes

- CIFAR-10 is automatically downloaded to `./data` if missing.

## References
```bibtex
@article{dosovitskiy2021image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and others},
  journal={ICLR},
  year={2021}
}
```
