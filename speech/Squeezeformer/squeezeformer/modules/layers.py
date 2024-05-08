"""
Common layers.
Author: JiaWei Jiang
"""
import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class AdaptiveScalingLayer(nn.Module):
    """Adaptive scaling layer.

    Replace preLN with a learnable scaling layer to control the weight
    of activations within a residual block.

    Args:
        in_features: Number of input features.
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(in_features))
        self.beta = nn.Parameter(torch.zeros(in_features))

    def forward(self, x: Tensor) -> Tensor:
        x = x * self.gamma + self.beta

        return x


class PointWiseConv1d(nn.Module):
    """Pointwise 1D convolution layer.

    Apply 1x1 pointwise convolution to map cross-channel correlation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride length.
        padding: Padding added to both sides of the input.
        bias: If True, add bias to the output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.pw_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.pw_conv(x)

        return x


class PointWiseConv2d(nn.Module):
    """Pointwise 2D convolution layer.

    Apply 1x1 pointwise convolution to map cross-channel correlation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride length.
        padding: Padding added to all four sides of the input.
        bias: If True, add bias to the output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.pw_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.pw_conv(x)

        return x


class DepthWiseConv1d(nn.Module):
    """Depthwise 1D convolution layer.

    The depth multiplier is equal to (out_channels / in_channels).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of convolving kernel.
        stride: Stride length.
        padding: Padding added to both sides of the input.
        bias: If True, add bias to the output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert out_channels % in_channels == 0, "out_channels/in_channles must be a constant integer multiplier."

        self.dw_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.dw_conv(x)

        return x


class DepthWiseConv2d(nn.Module):
    """Depthwise 2D convolution layer.

    The depth multiplier is equal to (out_channels / in_channels).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of convolving kernel.
        stride: Stride length.
        padding: Padding added to all four sides of the input.
        bias: If True, add bias to the output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert out_channels % in_channels == 0, "out_channels/in_channles must be a constant integer multiplier."

        self.dw_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.dw_conv(x)

        return x


class MultiheadAttention(nn.Module):
    """Vanilla multi-head attention layer.

    Args:
        d_model: Total dimension of the model.
        num_heads: Number of parallel attention heads.
        dropout: Dropout rate.

    Shape:
        Input: (B, L, E), where B is the batch size, L is the sequence
            length, and E denotes the embedding dimension.
        Output: Same shape as the input.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), "d_model must be divisible by num_heads, because d_model will be splitted across num_heads."

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scaling_factor = self.d_head**0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            mask: Indicate which elements in key should be ignored.
                Note that mask = key_padding_mask + attn_mask
        """
        batch_size, seq_len, emb_dim = query.shape

        # Input projection
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        # Split attention heads
        query = query.reshape(batch_size, seq_len, self.num_heads, self.d_head)
        key = key.reshape(batch_size, seq_len, self.num_heads, self.d_head)
        value = value.reshape(batch_size, seq_len, self.num_heads, self.d_head)

        # Scaled Dot-Product Attention
        # Note that torch.einsum can be replaced by torch.matmul
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        attn_scores = torch.einsum("bnld,bnsd->bnls", query, key) / self.scaling_factor
        if mask is not None:
            # e^(-inf) = 0 in softmax
            attn_scores = attn_scores.mask_fill(mask == 1, float("-inf"))
        attn_scores = torch.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)
        query = torch.einsum("bnls,bnsd->bnld", attn_scores, value)

        # Concatenate all heads
        query = query.transpose(1, 2).reshape(batch_size, seq_len, -1)

        return query


class PositionalEncoding(nn.Module):
    """Positional Encoding.

    Args:
        d_model: Embedding dimension.
        max_len: The maximum sequence length that the model might ever
            be used with.

    Shape:
        Input: (B, L, E), where B is the batch size, L is the sequence
            length, and E denotes the embedding dimension.
        Output: Same shape as the input.
    """

    BASE: int = 10000

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()

        self.register_buffer("pe_table", self._get_pe_table(d_model, max_len))

    def _get_pe_table(self, d_model: int, max_len: int) -> Tensor:
        # def _get_pos_angle_vec(pos: int) -> List[float]:
        #     return [pos / self.BASE**(2 * (j // 2) / d_model) for j in range(d_model)]

        # pe_table = torch.tensor(
        #     [_get_pos_angle_vec(pos) for pos in range(max_len)],
        #     dtype=torch.float32,
        # )
        # pe_table[:, 0::2] = torch.sin(pe_table[:, 0::2])  # Dimension 2i
        # pe_table[:, 1::2] = torch.cos(pe_table[:, 1::2])  # Dimension 2i+1

        pos_vec = torch.arange(max_len).unsqueeze(dim=1)
        omega_vec = torch.exp(torch.arange(0, d_model, 2) * (-math.log(self.BASE) / d_model))  # Angular frequency
        pe_table = torch.zeros(max_len, d_model)
        pe_table[:, 0::2] = torch.sin(pos_vec * omega_vec)
        pe_table[:, 1::2] = torch.cos(pos_vec * omega_vec)
        pe_table = pe_table.unsqueeze(dim=0)  # Add batch dim

        return pe_table

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe_table[:, : x.size(1), :]

        return x


# class RelativeMultiHeadAttention(nn.Module):
#     """Relative multi-head attention layer.

#     Assume _qkv_same_emb_dim is True.

#     * [ ] Init in in_proj Linears is diff from torch MHA layer


#     Args:
#         d_model: Total dimension of the model.
#         num_heads: Number of parallel attention heads.

#         dropout: Dropout rate.

#     Shape:
#         Input: (*, C), where C is the number of input features.
#         Output: (*, C), same shape as the input.
#     """

#     def __init__(
#         self,
#         d_model: int,
#         num_heads: int,
#         dropout: float = 0.1,
#     ) -> None:
#         super().__init__()
#         assert (
#             d_model % num_heads == 0
#         ), "d_model must be divisible by num_heads, because d_model will be splitted across num_heads."
#         self.d_head = d_model // num_heads

#         # Input projection layers with bias=True
#         self.q_proj = nn.Linear(d_model, d_model)
#         self.k_proj = nn.Linear(d_model, d_model)
#         self.v_proj = nn.Linear(d_model, d_model)

#         self.pos_proj = nn.Linear(d_model, d_model)

#         self.dropout = nn.Dropout(dropout)

#     def forward(
#         self,
#         query: Tensor,
#         key: Tensor,
#         value: Tensor,
#         pos: Tensor,
#     ) -> Tensor:
#         # Input projection
#         query = self.q_proj(query)
#         key = self.k_proj(key)
#         value = self.v_proj(value)

#         pos = self.pos_proj(pos)

#         return x
