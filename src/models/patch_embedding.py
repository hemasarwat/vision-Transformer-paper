import torch
from torch import nn

class PatchEmbedding(nn.Module):
    """Turns a 2D image into a 1D sequence of patch embeddings."""

    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 768,
        patch_size: int = 16,
    ) -> None:
        """
        Args:
            in_channels: Number of input channels. default is 3 for RGB images.
            embedding_dim: Dimension of the output embeddings. default is 768.
            patch_size: Size of each patch. default is 16.
        """
        super().__init__()
        self.patch_size = patch_size
        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f"Expected (B,C,H,W), got shape {x.shape}"
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Image size {image_resolution} must be divisible by patch size {self.patch_size}."

        x = self.patcher(x)
        x = self.flatten(x)
        return x.permute(0, 2, 1)


class ViTInputLayer(nn.Module):
    """Patching + CLS token + positional embeddings."""

    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        patch_size: int = 16,
        embedding_dim: int = 768,
    ) -> None:
        """
        Args:
            img_size: Size of the input image. default is 224.
            in_channels: Number of input channels. default is 3 for RGB images.
            patch_size: Size of each patch. default is 16.
            embedding_dim: Dimension of the output embeddings. default is 768.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.flatten = nn.Flatten(2, 3)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, "Input should be a batch of images with shape (B, C, H, W)"
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Image size {image_resolution} must be divisible by patch size {self.patch_size}."
        
        x = self.patcher(x)
        x = self.flatten(x)
        x = x.permute(0, 2, 1)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x + self.pos_embedding
