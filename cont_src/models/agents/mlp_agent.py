"""Agent implementations."""

import torch
import torch.nn as nn
from typing import Dict, Any

from cont_src.core.base_module import BaseAgent
from cont_src.core.registry import AGENT_REGISTRY


@AGENT_REGISTRY.register("mlp")
class MLPAgent(BaseAgent):
    """
    Multi-Layer Perceptron agent.

    Maps slot representations to hidden representations: s_k -> h_k
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 3,
        activation: str = "relu",
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        **kwargs
    ):
        """
        Initialize MLP agent.

        Args:
            input_dim: Input dimension (slot_dim)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (d_h)
            num_layers: Number of MLP layers
            activation: Activation function (relu, gelu, leaky_relu)
            dropout: Dropout probability
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__(config={
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "num_layers": num_layers,
            "activation": activation,
            "dropout": dropout,
            "use_layer_norm": use_layer_norm,
        })

        # Activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "gelu":
            act_fn = nn.GELU
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build MLP
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act_fn())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            slots: Slot representations
                - Shape (B, K, D_slot) for batch of slots
                - Shape (B, D_slot) for single slot per image
                - Shape (D_slot,) for single slot

        Returns:
            Hidden representations with same batch structure
        """
        return self.network(slots)


@AGENT_REGISTRY.register("mlp_with_decoder")
class MLPAgentWithDecoder(MLPAgent):
    """
    MLP Agent with decoder for reconstruction loss (L_agent).

    Prevents agent collapse by requiring ability to reconstruct slots.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add decoder: h_k -> s_k
        input_dim = self.config["input_dim"]
        output_dim = self.config["output_dim"]
        hidden_dim = self.config["hidden_dim"]

        decoder_layers = [
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        ]

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional reconstruction.

        Args:
            slots: Slot representations

        Returns:
            Dict with:
                - hidden: Hidden representations h_k
                - reconstructed: Reconstructed slots (if training)
        """
        hidden = self.network(slots)

        output = {"hidden": hidden}

        if self.training:
            reconstructed = self.decoder(hidden)
            output["reconstructed"] = reconstructed

        return output


@AGENT_REGISTRY.register("identity")
class IdentityAgent(BaseAgent):
    """
    Identity agent for debugging.

    Simply passes through slots without transformation.
    """

    def __init__(self, input_dim: int = 64, **kwargs):
        super().__init__(config={"input_dim": input_dim})
        self.projection = nn.Identity()

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """Pass through slots unchanged."""
        return self.projection(slots)
