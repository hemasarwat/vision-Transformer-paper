# Vision Transformer (ViT) Implementation

A PyTorch implementation of the Vision Transformer paper: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2021).

## The core idea:

Transformers took over NLP a few years ago. Everyone thought they were just for text. Then this paper came along and said: "What if we treat images the same way?"

The idea is surprisingly simple: chop an image into fixed-size patches (like cutting a photo into a grid), flatten each patch into a vector, and feed them to a standard Transformer. No convolutions needed. The model learns to pay attention to different parts of the image, just like it would with words in a sentence.

Turns out, when you train it on massive datasets (we're talking hundreds of millions of images), it matches or make CNNs. The catch? It needs way more data than traditional computer vision models because it doesn't have assumptions about images (like "nearby pixels are related"). It learns everything from scratch.

## What I Built

This repo contains:
- **Clean implementation** of ViT architecture (patch embedding, transformer blocks, classification head)
- **Fine-tuned model** on CIFAR10 using pre-trained ViT weights
- **Modular code structure** - each component in its own file
- **Training scripts** - ready to run

## Architecture Overview

**ViT-Base/16 specs:**
- Image size: `224×224`
- Patch size: `16×16` → 196 patches per image
- Embedding dimension: 768
- Transformer layers: 12
- Attention heads: 12 per layer
- Total parameters: ~86M

**How it works:**
1. Split image into 16×16 patches
2. Flatten and embed each patch
3. Add a learnable "class token" at the start
4. Add position embeddings (so the model knows where patches came from)
5. Feed through 12 transformer layers
6. Use the class token's final representation for classification

## Project Structure
```
vit-implementation/
├── src/
│   ├── models/
│   │   ├── vit.py              # Main ViT model
│   │   ├── patch_embedding.py  # Image → patches
│   │   └── transformer_block.py # Attention + MLP
│   ├── data/
│   │   └── dataset.py          # CIFAR10 dataloaders
│   └── utils/
│       ├── engine.py           # Training loop
│       └── helpers.py          # Utilities
├── scripts/
│   ├── train.py                # Training script
├── notebooks/
│   └── exploration.ipynb       # ViT walkthrough
└── checkpoints/                # Saved models
```

## Quick Start

### Installation
```bash
git clone https://github.com/yourusername/vit-implementation.git
cd vit-implementation
pip install -r requirements.txt
```

### Training

Fine-tune pre-trained ViT on CIFAR10:
```bash
python scripts/train.py --epochs 5 --batch-size 32
```

**Note:** This uses transfer learning - we freeze the pre-trained backbone and only train the classification head. Training from scratch would require days on CPU and millions of images.

### Inference

Test on your own image:
```bash
python scripts/inference.py checkpoints/vit_finetuned.pth your_image.jpg
```

## Results

| Approach | Dataset | Accuracy | Training Time |
|----------|---------|----------|---------------|
| Fine-tuned (head only) | CIFAR10 | ~87% | 30-60 min (CPU) |

**Why transfer learning?**
The paper trained on JFT-300M (300 million images) for pre-training. That's not feasible on a laptop. Fine-tuning a pre-trained model is the practical approach for limited resources.

## What I Learned

**Technical insights:**
- ViT needs massive data because it has no inductive bias (unlike CNNs which assume spatial locality)
- Patch embedding via Conv2d is cleaner than manual slicing
- Position embeddings can be 1D and still work (paper shows this)
- Layer normalization goes before attention/MLP blocks (pre-norm is more stable)

**Implementation lessons:**
- Building from paper teaches you way more than following tutorials
- Structuring code properly makes debugging 10x easier
- Transfer learning isn't optional - it's essential for transformers on small datasets
- Getting 10% accuracy on 225 images validated the paper's findings about data requirements

## Limitations & Future Work

**Current limitations:**
- Only tested on CIFAR10 (32×32 images upscaled to 224×224)
- Uses PyTorch's `nn.MultiheadAttention` instead of custom implementation
- No data augmentation or advanced training techniques

## References
```bibtex
@article{dosovitskiy2021image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and others},
  journal={ICLR},
  year={2021}
}
```

**Paper:** [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)