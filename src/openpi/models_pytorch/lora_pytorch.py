"""Minimal PyTorch LoRA helper layers (attention/FFN agnostic).

These modules mirror the Flax helpers in `openpi.models.lora` but are kept light
and framework-specific for PyTorch. They are designed to wrap linear-like weight
matrices and add rank-decomposed updates.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


class LoRAConfig:
    """Configuration for LoRA adapters."""

    def __init__(
        self,
        rank: int,
        alpha: float = 1.0,
        rslora: bool = False,
        init_std: float = 0.01,
    ) -> None:
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.rslora = bool(rslora)
        self.init_std = float(init_std)

    @property
    def scaling_value(self) -> float:
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank


class LoRALinear(nn.Module):
    """Linear layer with a LoRA branch.

    The base weight is expected to be provided by the caller (e.g., an existing
    Linear module). LoRA parameters are created here and added as a low-rank
    update during the forward pass.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        lora_config: LoRAConfig,
        *,
        fan_in_fan_out: bool = False,
    ) -> None:
        super().__init__()
        self.fan_in_fan_out = fan_in_fan_out
        self.lora_config = lora_config

        # Keep base weights as parameters but freeze them (so checkpoint loading works with the same keys).
        self.weight = nn.Parameter(weight.clone().detach(), requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias.clone().detach(), requires_grad=False)
        else:
            self.bias = None

        # torch.nn.Linear stores weight as (out_features, in_features)
        in_features, out_features = (weight.shape[1], weight.shape[0]) if not fan_in_fan_out else (weight.shape[0], weight.shape[1])
        rank = lora_config.rank

        # LoRA A and B projections (down/up)
        self.lora_A = nn.Parameter(torch.zeros((rank, in_features), device=weight.device, dtype=weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank), device=weight.device, dtype=weight.dtype))

        # Init
        nn.init.normal_(self.lora_A, std=lora_config.init_std)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        b = self.bias
        x = x.to(w.dtype)
        if self.fan_in_fan_out:
            w = w.t()
        base = torch.matmul(x, w.t()) if not self.fan_in_fan_out else torch.matmul(x, w)

        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t()
        out = base + lora_out * self.lora_config.scaling_value
        if b is not None:
            out = out + b
        return out


def apply_lora_to_linear(module: nn.Linear, lora_config: LoRAConfig) -> LoRALinear:
    """Wrap an existing nn.Linear with a LoRA branch, freezing the base weight.

    Returns a LoRALinear module that holds the original weight/bias as buffers and
    trains only the LoRA parameters.
    """
    # Clone weights to detach from future optimizer steps on the original module.
    weight = module.weight.detach().clone()
    bias = module.bias.detach().clone() if module.bias is not None else None
    return LoRALinear(weight, bias, lora_config, fan_in_fan_out=False)
