"""
Multi-Head Attention Module.
Author: JiaWei Jiang
"""
import torch.nn as nn
from torch import Tensor

from .layers import AdaptiveScalingLayer, MultiheadAttention


class MHAModule(nn.Module):
    """Multi-Head Attention Module.

    Args:
        mha_type: Type of positional encoding and self-attention layer.
        d_model: Total dimension of the model.
        num_heads: Number of parallel attention heads.
        dropout: Dropout rate.
        adaptive_scaling: If True, a learnable scaling layer is applid
            to the input activations.

    Shape:
        Input: (B, L, E), where B is the batch size, L is the sequence
            length, and E denotes the encoder dimension.
        Output: Same shape as the input.
    """

    def __init__(
        self,
        mha_type: str,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        adaptive_scaling: bool = True,
    ) -> None:
        super().__init__()
        assert mha_type == "abs", "Currently support canonical MHA with absolute PE only."

        self.adaptive_scaling = adaptive_scaling

        if adaptive_scaling:
            self.scaling_layer = AdaptiveScalingLayer(d_model)
        # self.pe = PositionalEncoding(d_model)
        self.mha = MultiheadAttention(d_model, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        if self.adaptive_scaling:
            x = self.scaling_layer(x)

        # pos_enc = self.pe(x)
        x = self.mha(query=x, key=x, value=x, mask=None)
        x = self.dropout(x)

        return x
