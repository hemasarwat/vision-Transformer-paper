from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention block."""

    def __init__(
        self,
        embedding_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
            embedding_dim: Dimension of input and output embeddings. default is 768.
            num_heads: Number of attention heads. default is 12.
            dropout: Dropout probability applied inside attention. default is 0.0.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ln = nn.LayerNorm(normalized_shape=embedding_dim)
        self.msa = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x):
        x_norm = self.ln(x)
        attn_output, _ = self.msa(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            need_weights=False,
        )
        return attn_output + x


class MLPBlock(nn.Module):
    """MLP block from the ViT paper (Linear -> GELU -> Dropout -> Linear)."""

    def __init__(
        self,
        embedding_dim: int = 768,
        expanded_features: int = 3072,
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
            embedding_dim: Dimension of input and output embeddings. default is 768.
            expanded_features: Number of hidden features in the MLP. default is 3072.
            dropout: Dropout probability between MLP layers. default is 0.0.
        """
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=expanded_features),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=expanded_features, out_features=embedding_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return x + self.mlp(self.ln(x))


class TransformerEncoder(nn.Module):
    """Transformer encoder block used in ViT."""

    def __init__(
        self,
        embedding_dim: int = 768,
        num_heads: int = 12,
        mlp_size: int = 3072,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            embedding_dim: Dimension of input and output embeddings. default is 768.
            num_heads: Number of attention heads. default is 12.
            mlp_size: Hidden size of the feed-forward MLP (expansion size). default is 3072.
            dropout: Dropout probability applied in attention and MLP. default is 0.1.
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.ln2 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.msa = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        x = x + self.msa(
            query=self.ln1(x),
            key=self.ln1(x),
            value=self.ln1(x),
            need_weights=False,
        )[0]
        x = x + self.mlp(self.ln2(x))
        return x
