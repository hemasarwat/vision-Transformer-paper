"""
Utility functions for training and saving PyTorch models.
"""
from pathlib import Path
import numpy as np
import torch

# ----- Reproducibility ----- #
def set_Seed(seed: int = 1) -> None: # Set random seeds for reproducibility.
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.manual_seed(seed)

# ----- Device helper ----- #
def get_device():
    """Returns the available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----- Model saving ----- #
"""
Contains utility functions for saving PyTorch models.
"""

from pathlib import Path

import torch
from torch import nn


def save_model(model: nn.Module, target_dir: str, model_name: str) -> None:
    """
    Saves a PyTorch model into a target directory.

    Args:
        model: PyTorch model to save.
        target_dir: Directory to store the model.
        model_name: File name for the model (must end with .pth or .pt).

    example:
        save_model(model= model_0,
                   target_dir= "models/",
                   model_name= "ViT_Model.pth")
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith((".pth", ".pt")), "model_name must end with .pth or .pt"

    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(model.state_dict(), model_save_path)
