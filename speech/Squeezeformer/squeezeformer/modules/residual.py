"""
Residual Block.
Author: JiaWei Jiang
"""
import torch.nn as nn
from torch import Tensor


class ResidualBlock(nn.Module):
    """Residual Block.

    Args:
        module: Non-linear module.
        module_factor: Factor used to adjust the weight of the output
            of non-linear module.
    """

    def __init__(
        self,
        module: nn.Module,
        module_factor: float = 1.0,
    ) -> None:
        super().__init__()

        self.module = module
        self.module_factor = module_factor

    def forward(self, x: Tensor) -> Tensor:
        x_resid = x
        x = self.module(x)
        x = x * self.module_factor + x_resid

        return x
