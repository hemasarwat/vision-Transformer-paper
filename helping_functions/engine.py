"""
Contains functions for training and testing PyTorch models.
"""

from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from tqdm.auto import tqdm


def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    accuracy_fn: MulticlassAccuracy,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Trains a PyTorch model for a single epoch.

    Returns:
        (train_loss, train_accuracy)
    """
    model.train()
    accuracy_fn.reset()

    train_loss = 0.0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        accuracy_fn.update(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_acc = accuracy_fn.compute().item()

    return train_loss, train_acc


def test_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    accuracy_fn: MulticlassAccuracy,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluates a PyTorch model for a single epoch.

    Returns:
        (test_loss, test_accuracy)
    """
    model.eval()
    accuracy_fn.reset()

    test_loss = 0.0

    with torch.inference_mode():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            y_pred = model(x)

            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            accuracy_fn.update(y_pred, y)

    test_loss /= len(dataloader)
    test_acc = accuracy_fn.compute().item()

    return test_loss, test_acc


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    accuracy_fn: MulticlassAccuracy,
    device: torch.device,
) -> Dict[str, List[float]]:
    """
    Trains and evaluates a model for multiple epochs.

    Returns:
        Dictionary with lists of train/test loss & accuracy.
    """
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device,
        )

        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device,
        )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
