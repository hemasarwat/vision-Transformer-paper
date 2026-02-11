"""Fine-tune pre-trained ViT on CIFAR10"""
import argparse
from pathlib import Path
import torchvision
from torch import nn, optim
import timm
import torchmetrics
from src.utils import save_model, set_Seed, get_device
from src.data.dataset import get_CIFAR10_dataloaders
from helping_functions.engine import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ViT on CIFAR10")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--save-path", type=str, default="checkpoints/vit_finetuned.pth")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # ----- Reproducibility ----- #
    set_Seed(1)
    # ----- Set Device ----- #
    device = get_device()
    print(f"Using device: {device}\n")

    # Data
    print("Loading CIFAR10...")
    train_loader, test_loader = get_CIFAR10_dataloaders(
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}\n")

    # Load pre-trained ViT
    print("Loading pre-trained ViT-Base-16...")
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=True,
        num_classes= 10  # CIFAR10 10 classes
    ).to(device)

    # Get a pre-trained model
    pretrained_ViT_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    pretrained_ViT_model = torchvision.models.vit_b_16(weights=pretrained_ViT_weights)

    # Freeze the base parameters
    for param in pretrained_ViT_model.parameters():
        param.requires_grad = False

    # Update the classifier
    pretrained_ViT_model.heads.head = nn.Linear(in_features=768, out_features=10)
    model = pretrained_ViT_model.to(device)
    # Training setup
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    accuracy_fn = torchmetrics.Accuracy(
        task="multiclass",
        num_classes=10
    ).to(device)

    # Train
    print("Starting training...\n")
    results = train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=args.epochs,
        accuracy_fn=accuracy_fn,
        device=device
    )

    # Save
    save_path = save_model(
        model=model,
        target_dir=str(Path(args.save_path).parent),
        model_name=Path(args.save_path).name
    )
    # Print final results
    print(f"\nFinal Results:")
    print(f"Train Acc: {results['train_acc'][-1]:.2%}")
    print(f"Test Acc: {results['test_acc'][-1]:.2%}")


if __name__ == "__main__":
    main()