"""
Contains functions to create PyTorch DataLoaders for image classification.
"""

import os
from typing import List, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_WORKERS = os.cpu_count() 


def get_vit_transform(image_size: int = 224) -> transforms.Compose:
    """Return standard ViT transform for CIFAR-10 images."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ]
    )


def create_cifar10_dataloaders(
    batch_size: int,
    image_size: int = 224,
    data_root: str = "data",
    transform: transforms.Compose | None = None,
    num_workers: int = NUM_WORKERS,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Create CIFAR-10 training and testing DataLoaders.

    Args:
        batch_size: Number of samples per batch.
        image_size: Image size that is ViT Paper.
        data_root: Root directory for CIFAR-10.
        transform:  defaults to ViT transform.
        num_workers: Number of subprocesses for data loading.
        download: Whether to download CIFAR-10 if missing.

    Returns:
        (train_dataloader, test_dataloader, class_names)
    """
    if transform is None:
        transform = get_vit_transform(image_size=image_size)

    train_data = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=download,
        transform=transform,
    )
    test_data = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=download,
        transform=transform,
    )

    class_names = train_data.classes

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_dataloader, test_dataloader, class_names


def create_dataloader(
    train_dir: str | None = None,
    test_dir: str | None = None,
    transform: transforms.Compose | None = None,
    batch_size: int = 32,
    num_workers: int = NUM_WORKERS,
    image_size: int = 224,
    data_root: str = "data",
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Backward-compatible wrapper that returns CIFAR-10 dataloaders.
    """
    _ = (train_dir, test_dir)
    return create_cifar10_dataloaders(
        batch_size=batch_size,
        image_size=image_size,
        data_root=data_root,
        transform=transform,
        num_workers=num_workers,
        download=download,
    )
