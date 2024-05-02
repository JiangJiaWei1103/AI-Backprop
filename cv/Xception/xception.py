"""
Xception (torch implementation).
Author: JiaWei Jiang
"""
import torch.nn as nn
from torch import Tensor


class Xception(nn.Module):
    """Xception model architecture."""

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 10,
    ) -> None:
        """Initialize Xception.

        Args:
            in_channels: Number of input channels.
            n_classes: Number of output classes.
        """
        super().__init__()

        self.relu = nn.ReLU()

        # Entry flow
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=3 // 2, bias=False),
            nn.BatchNorm2d(32),
            self.relu,
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=3 // 2, bias=False),
            nn.BatchNorm2d(64),
            self.relu,
        )
        self.conv_block1 = _ConvBlock(64, 128, stride=2, n_layers=2, relu_first=False)
        self.conv_block2 = _ConvBlock(128, 256, stride=2, n_layers=2)
        self.conv_block3 = _ConvBlock(256, 728, stride=2, n_layers=2)

        # Middle flow (conv_block4 ~ 11)
        self.middle_flow = nn.Sequential()
        for _ in range(8):
            self.middle_flow.append(_ConvBlock(728, 728, stride=1, n_layers=3))

        # Exit flow
        self.conv_block12 = _ConvBlock(728, 1024, stride=2, n_layers=2, expand_ch_first=False)
        self.out_conv = nn.Sequential(
            _SeparableConv(1024, 1536, 3, stride=1, padding=3 // 2, bias=False),
            nn.BatchNorm2d(1536),
            self.relu,
            _SeparableConv(1536, 2048, 3, stride=1, padding=3 // 2, bias=False),
            nn.BatchNorm2d(2048),
            self.relu,
        )
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(2048, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Entry flow
        x = self.in_conv(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # Middle flow (conv_block4 ~ 11)
        x = self.middle_flow(x)

        # Exit flow
        x = self.conv_block12(x)
        x = self.out_conv(x)
        output = self.output(x)

        return output


class _ConvBlock(nn.Module):
    """Main convolution block.

    A main convolution block is defined as a convolution module having
    linear residual connection around it,

     |------------------------- conv_1x1|bn -----------------------|
     |                                                             |
    -|- (relu)|sep_conv|bn - relu|sep_conv|bn - . - (maxpooling) - + ->

    , where a conv submodule can be defined as (relu)|sep_conv|bn.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        n_layers: int,
        relu_first: bool = True,
        expand_ch_first: bool = True,
    ) -> None:
        """Initialize a main convolution block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride length of max pooling, determining the
                sub-sampling ratio.
            n_layers: Number of conv submodules.
            relu_first: If True, the conv module starts with relu.
            expand_ch_first: If True, the channel dimension is expanded
                in the first conv submodule.
        """
        super().__init__()

        self.conv_block = []
        self.relu = nn.ReLU()

        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.skip_bn = nn.BatchNorm2d(out_channels)
        else:
            self.skip_conv = None
            self.skip_bn = None

        if expand_ch_first:
            self.conv_block.append(self.relu)
            self.conv_block.append(
                _SeparableConv(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=3 // 2,
                    bias=False,
                )
            )
            self.conv_block.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels

        for _ in range(n_layers - 1):
            self.conv_block.append(self.relu)
            self.conv_block.append(
                _SeparableConv(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=3 // 2,
                    bias=False,
                )
            )
            self.conv_block.append(nn.BatchNorm2d(in_channels))

        if not expand_ch_first:
            self.conv_block.append(self.relu)
            self.conv_block.append(
                _SeparableConv(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=3 // 2,
                    bias=False,
                )
            )
            self.conv_block.append(nn.BatchNorm2d(out_channels))

        if not relu_first:
            self.conv_block = self.conv_block[1:]

        if stride != 1:
            self.conv_block.append(nn.MaxPool2d(kernel_size=3, stride=stride, padding=3 // 2))

        self.conv_block = nn.Sequential(*self.conv_block)

    def forward(self, x: Tensor) -> Tensor:
        x_resid = x
        if self.skip_conv is not None:
            x_resid = self.skip_conv(x_resid)
            x_resid = self.skip_bn(x_resid)
        x = self.conv_block(x)  # type: ignore

        # Add residual component
        x = x + x_resid

        return x


class _SeparableConv(nn.Module):
    """Depthwise separable convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias: bool,
    ) -> None:
        """Initialize a depthwise separable convolution layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels in pointwise conv.
            kernel_size: Size of convolving kernel in depthwise conv.
            stride: Stride length in depthwise conv.
            padding: Padding added to all four sides of the input in
                depthwise conv.
            bias: If True, add bias to the output.
        """
        super().__init__()

        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # Depth multiplier of 1
            bias=False,
        )
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        return x
