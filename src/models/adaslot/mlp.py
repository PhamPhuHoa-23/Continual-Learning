"""
MLP module used in AdaSlot (ff_mlp for slot attention).

Mirrors: ocl.conditioning.MLP
Checkpoint key prefix: models.perceptual_grouping.slot_attention.ff_mlp.module.*
"""

import torch
import torch.nn as nn
from typing import Optional


class MLP(nn.Module):
    """
    Multi-layer perceptron with optional LayerNorm and residual connection.

    The internal structure uses nn.ModuleList with named modules:
        dense_mlp_0, dense_mlp_0_act, ..., dense_mlp_N

    This matches the original ocl.conditioning.MLP exactly to ensure
    checkpoint key compatibility.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_hidden_layers: int = 1,
        activation_fn: type = nn.ReLU,
        layernorm: Optional[str] = None,
        activate_output: bool = False,
        residual: bool = False,
        weight_init=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.activation_fn = activation_fn
        self.layernorm = layernorm
        self.activate_output = activate_output
        self.residual = residual
        self.weight_init = weight_init

        if self.layernorm == "pre":
            self.layernorm_module = nn.LayerNorm(input_size, eps=1e-6)
        elif self.layernorm == "post":
            self.layernorm_module = nn.LayerNorm(output_size, eps=1e-6)

        # Build MLP with named modules (matching original key structure)
        self.model = nn.ModuleList()
        self.model.add_module("dense_mlp_0", nn.Linear(self.input_size, self.hidden_size))
        self.model.add_module("dense_mlp_0_act", self.activation_fn())
        for i in range(1, self.num_hidden_layers):
            self.model.add_module(f"den_mlp_{i}", nn.Linear(self.hidden_size, self.hidden_size))
            self.model.add_module(f"dense_mlp_{i}_act", self.activation_fn())
        self.model.add_module(
            f"dense_mlp_{self.num_hidden_layers}",
            nn.Linear(self.hidden_size, self.output_size),
        )
        if self.activate_output:
            self.model.add_module(
                f"dense_mlp_{self.num_hidden_layers}_act", self.activation_fn()
            )

        # Xavier initialization for linear layers
        for name, module in self.model.named_children():
            if "act" not in name:
                nn.init.xavier_uniform_(module.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        if self.layernorm == "pre":
            x = self.layernorm_module(x)
        for layer in self.model:
            x = layer(x)
        if self.residual:
            x = x + inputs
        if self.layernorm == "post":
            x = self.layernorm_module(x)
        return x
