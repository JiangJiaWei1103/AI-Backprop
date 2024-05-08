"""
Convolution Module.
Author: JiaWei Jiang
"""
import torch.nn as nn
from torch import Tensor

from .act import GLU, Swish
from .layers import AdaptiveScalingLayer, DepthWiseConv1d, DepthWiseConv2d, PointWiseConv1d, PointWiseConv2d


class DWSepConv2dSubsampling(nn.Module):
    """Depthwise Separable Subsampling.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.

    Shape:
        Input: (B, L, D), where B is the batch size, L is the sequence
            length, and D is the number of input features (e.g., freq
            bands).
        Output: (B, L', D'), where L' is the output sequence length,
            and D' is the number of output features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=3 // 2)
        self.act = nn.ReLU()
        self.conv2 = nn.Sequential(
            DepthWiseConv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=3 // 2),
            PointWiseConv2d(out_channels, out_channels, stride=1),
        )

        # Output projection
        # If output projection is performed here, number of input feats
        # should be passed in
        # out_feats = int((in_feats // 2) // 2)
        # self.out_proj = nn.Linear(out_channels * out_feats, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        # Add channel dimension
        x = x.unsqueeze(dim=1)
        # Conv1
        x = self.conv1(x)
        x = self.act(x)
        # Conv2
        x = self.conv2(x)
        x = self.act(x)

        # Flatten channel and freq dims
        b, c, t, f = x.size()
        x = x.transpose(1, 2).reshape(b, t, -1)
        # Output projection
        # x = self.out_proj(x)

        return x


class ConvolutionModule(nn.Module):
    """Convolution Module.

    Args:
        in_channels: Number of input channels.
        expansion_factor: Expand factor of hidden dimension.
        kernel_size: Size of convolving kernel in depthwise conv.
        dropout: Dropout rate.
        use_glu: If True, the first activation function is replaced by
            gated linear unit.
        adaptive_scaling: If True, a learnable scaling layer is applid
            to the input activations.

    Shape:
        Input: (B, L, C), where L is the sequence length, and C is the
            number of input channels.
        Output: (B, L', C), where L' is the output sequence length.
    """

    def __init__(
        self,
        in_channels: int,
        expansion_factor: int = 2,
        kernel_size: int = 31,
        dropout: float = 0.1,
        use_glu: bool = False,
        adaptive_scaling: bool = True,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, 'kernel_size should be a odd number for "same" padding.'
        assert expansion_factor == 2, "Only support expansion_factor of 2."

        self.adaptive_scaling = adaptive_scaling

        if adaptive_scaling:
            self.scaling_layer = AdaptiveScalingLayer(in_channels)
        self.pw_conv1 = PointWiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0)
        if use_glu:
            # Split the tensor into two along the channel dim in GLU
            self.act1 = GLU()
            h_dim = in_channels
        else:
            self.act1 = Swish()
            h_dim = in_channels * expansion_factor
        self.dw_conv = DepthWiseConv1d(
            h_dim, h_dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm1d(h_dim)
        self.act2 = Swish()
        self.pw_conv2 = PointWiseConv1d(h_dim, in_channels, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        if self.adaptive_scaling:
            x = self.scaling_layer(x)
        x = x.transpose(1, 2)
        x = self.pw_conv1(x)
        x = self.act1(x)
        x = self.dw_conv(x)
        x = self.bn(x)
        x = self.act2(x)
        x = self.pw_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)

        return x


class TimeReductionModule(nn.Module):
    """Time Reudction Module.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.

    Shape:
        Input: (B, L, C), where L is the sequence length, and C is the
            number of input channels.
        Output: (B, L', C'), where L' is the output sequence length,
            and C' is the number of output channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.dw_conv = DepthWiseConv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=3 // 2)
        self.pw_conv = PointWiseConv1d(out_channels, out_channels, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        x = x.transpose(1, 2)

        return x
