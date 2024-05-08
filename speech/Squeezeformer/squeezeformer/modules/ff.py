"""
Feed Forward Module.
Author: JiaWei Jiang
"""
import torch.nn as nn
from torch import Tensor

from .act import Swish
from .layers import AdaptiveScalingLayer


class FeedForwardModule(nn.Module):
    """Feed Forward Module.

    Args:
        in_features: Number of input features.
        expansion_factor: Expand factor of hidden dimension.
        dropout: Dropout rate.
        adaptive_scaling: If True, a learnable scaling layer is applid
            to the input activations.

    Shape:
        Input: (*, C), where C is the number of input features.
        Output: (*, C), same shape as the input.
    """

    def __init__(
        self,
        in_features: int,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        adaptive_scaling: bool = True,
    ) -> None:
        super().__init__()

        self.adaptive_scaling = adaptive_scaling

        if adaptive_scaling:
            self.scaling_layer = AdaptiveScalingLayer(in_features)
        self.ff = nn.Sequential(
            nn.Linear(in_features, in_features * expansion_factor),
            Swish(),  # nn.SiLU()
            nn.Dropout(dropout),
            nn.Linear(in_features * expansion_factor, in_features),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.adaptive_scaling:
            x = self.scaling_layer(x)
        x = self.ff(x)

        return x
