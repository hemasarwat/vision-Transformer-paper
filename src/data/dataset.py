"""Dataset and dataloader utilities for ViT experiments."""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_vit_transform(image_size: int = 224) -> transforms.Compose:
    """ViT-Base standard transforms (Resize + Normalize)."""
    return transforms.Compose(
        [transforms.Resize(image_size),
         transforms.ToTensor(),
         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                              std=(0.229, 0.224, 0.225))]
    )


def get_CIFAR10_dataloaders(
    batch_size: int = 32,
    image_size: int = 224,
    data_root: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """Return CIFAR10 train/test dataloaders using ViT transforms."""
    vit_transform = get_vit_transform(image_size=image_size)

    train_data = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=vit_transform,
    )
    test_data = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=vit_transform,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, test_loader
