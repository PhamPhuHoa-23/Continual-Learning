"""
Embedding Wrapper/Transformer

Provides wrappers to transform embeddings before clustering.
Useful for future extensions like submanifold projection, dimensionality reduction, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BaseEmbeddingWrapper(nn.Module):
    """
    Base class for embedding transformations.

    Wrapper around embeddings that can apply transformations
    before clustering. By default, returns embeddings as-is.
    """

    def __init__(self, input_dim: int, output_dim: Optional[int] = None):
        """
        Args:
            input_dim: Input embedding dimension
            output_dim: Output dimension (None = same as input)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Transform embeddings.

        Args:
            embeddings: Input embeddings (N, D_in)

        Returns:
            transformed: Transformed embeddings (N, D_out)
        """
        raise NotImplementedError

    def inverse(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Inverse transformation (if applicable).

        Args:
            embeddings: Transformed embeddings (N, D_out)

        Returns:
            original: Original space embeddings (N, D_in)
        """
        raise NotImplementedError


class IdentityWrapper(BaseEmbeddingWrapper):
    """
    Identity wrapper - returns embeddings unchanged.

    This is the default wrapper when no transformation is needed.
    """

    def __init__(self, input_dim: int, output_dim: Optional[int] = None):
        super().__init__(input_dim, output_dim or input_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return embeddings as-is."""
        return embeddings

    def inverse(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return embeddings as-is."""
        return embeddings


class LinearProjectionWrapper(BaseEmbeddingWrapper):
    """
    Linear projection wrapper.

    Projects embeddings to a different dimension using a linear layer.
    Useful for dimensionality reduction before clustering.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_bias: bool = True,
        normalize: bool = False
    ):
        """
        Args:
            input_dim: Input embedding dimension
            output_dim: Output dimension
            use_bias: Whether to use bias in projection
            normalize: Whether to L2-normalize output
        """
        super().__init__(input_dim, output_dim)
        self.normalize = normalize

        self.projection = nn.Linear(input_dim, output_dim, bias=use_bias)

        # Initialize with Xavier
        nn.init.xavier_uniform_(self.projection.weight)
        if use_bias:
            nn.init.zeros_(self.projection.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project embeddings."""
        out = self.projection(embeddings)
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out

    def inverse(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Pseudo-inverse using least squares."""
        # x = (A^T A)^{-1} A^T y
        weight = self.projection.weight  # (output_dim, input_dim)
        return torch.matmul(embeddings, weight)


class NonLinearWrapper(BaseEmbeddingWrapper):
    """
    Non-linear projection with MLP.

    Projects embeddings using a multi-layer perceptron.
    More expressive than linear projection.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        activation: str = 'relu',
        normalize: bool = False
    ):
        """
        Args:
            input_dim: Input embedding dimension
            output_dim: Output dimension
            hidden_dim: Hidden dimension (default: (input_dim + output_dim) // 2)
            num_layers: Number of layers
            activation: Activation function ('relu', 'gelu', 'tanh')
            normalize: Whether to L2-normalize output
        """
        super().__init__(input_dim, output_dim)
        self.normalize = normalize

        hidden_dim = hidden_dim or (input_dim + output_dim) // 2

        # Build MLP
        layers = []
        in_dim = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function."""
        if name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        elif name == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {name}")

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Transform embeddings."""
        out = self.mlp(embeddings)
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out

    def inverse(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Not implemented for non-linear."""
        raise NotImplementedError(
            "Inverse not available for non-linear wrapper")


class SubmanifoldWrapper(BaseEmbeddingWrapper):
    """
    Submanifold projection wrapper.

    Projects embeddings onto a learned submanifold using an encoder-decoder.
    Useful for denoising and finding low-dimensional structure.

    This is a placeholder for your future extension ideas.
    """

    def __init__(
        self,
        input_dim: int,
        manifold_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2
    ):
        """
        Args:
            input_dim: Input embedding dimension
            manifold_dim: Dimension of the submanifold
            hidden_dim: Hidden dimension for encoder/decoder
            num_layers: Number of layers in encoder/decoder
        """
        super().__init__(input_dim, manifold_dim)

        hidden_dim = hidden_dim or max(input_dim, manifold_dim) * 2

        # Encoder: input -> manifold
        encoder_layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            in_dim = hidden_dim
        encoder_layers.append(nn.Linear(in_dim, manifold_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: manifold -> input
        decoder_layers = []
        in_dim = manifold_dim
        for _ in range(num_layers - 1):
            decoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            in_dim = hidden_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project to submanifold."""
        return self.encoder(embeddings)

    def inverse(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Reconstruct from submanifold."""
        return self.decoder(embeddings)

    def reconstruct(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full reconstruction: encode then decode.

        Returns:
            manifold: Embeddings on manifold (N, manifold_dim)
            reconstructed: Reconstructed embeddings (N, input_dim)
        """
        manifold = self.forward(embeddings)
        reconstructed = self.inverse(manifold)
        return manifold, reconstructed

    def reconstruction_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss for training the wrapper."""
        _, reconstructed = self.reconstruct(embeddings)
        return F.mse_loss(reconstructed, embeddings)
