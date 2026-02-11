
import torch
from torch import nn

from .transformer_block import TransformerEncoder
from .patch_embedding import  ViTInputLayer
class ViT(nn.Module):
    """Vision Transformer implementation"""

    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        embedding_dim: int = 768,
        patch_size: int = 16,
        num_heads: int = 12,
        mlp_size: int = 3072,
        num_transformer_layers: int = 12,
        dropout: float = 0.1,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        # ViT input layer: patching + CLS token + positional embeddings

        self.input_layer = ViTInputLayer(
            img_size=img_size,
            in_channels=in_channels,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
        )
        # Transformer encoder blocks (MLP + multi-head self attention)
        self.transformer_encoder = nn.Sequential(
            *[
                TransformerEncoder(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_size=mlp_size,
                    dropout=dropout
                )
                for layer in range(num_transformer_layers)
            ]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f"Expected (B,C,H,W), got shape {x.shape}"

        x = self.input_layer(x)
        x = self.transformer_encoder(x)
        x = x[:, 0]
        return self.classifier(x) # return (B, num_classes) 
