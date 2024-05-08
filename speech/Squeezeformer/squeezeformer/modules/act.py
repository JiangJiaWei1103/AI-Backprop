"""
Activation functions.
Author: JiaWei Jiang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Swish(nn.Module):
    """Swish activation function."""

    def __init__(self, in_place: bool = False) -> None:
        super().__init__()

        self.in_place = in_place

    def forward(self, x: Tensor) -> Tensor:
        if self.in_place:
            x.mul_(F.sigmoid(x))
        else:
            x = x * F.sigmoid(x)

        return x


class GLU(nn.Module):
    """Gated liner unit.

    Args:
        dim: The dimension to split the tensor.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()

        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        x_lin, x_gate = torch.chunk(x, chunks=2, dim=self.dim)
        x = x_lin * F.sigmoid(x_gate)

        return x
