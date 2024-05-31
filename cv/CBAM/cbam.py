"""
CBAM (torch implementation).
Author: JiaWei Jiang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Args:
        in_channels: Number of input channels.
        reduction_ratio: Reduction ratio of the hidden size in the
            channel attention module.
        bn: If True, batch normalization is applied to the attn map in
            the spatial attentio module.

    Shape:
        Input: (B, C, H, W), where B is the batch size, C is the number
            of input channels, H and W are the height and width of the
            input feature map.
        Output: Same shape as the input.
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        bn: bool = True,
    ) -> None:
        super().__init__()

        self.channel_attn = _ChannelAttnModule(in_channels, reduction_ratio)
        self.spatial_attn = _SpatialAttnModule(bn)

    def forward(self, x: Tensor) -> Tensor:
        x = self.channel_attn(x)
        output = self.spatial_attn(x)

        return output


class _ChannelAttnModule(nn.Module):
    """Channel Attention Module.

    Args:
        in_channels: Number of input channels.
        reduction_ratio: Reduction ratio of the hidden size.

    Shape:
        Input: (B, C, H, W)
        Output: Same shape as the input.
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
    ) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        *_, h, w = x.shape

        x_avg = F.avg_pool2d(x, kernel_size=(h, w))
        x_max = F.max_pool2d(x, kernel_size=(h, w))

        # Shared MLP
        x_avg = self.mlp(x_avg.squeeze((2, 3)))
        x_max = self.mlp(x_max.squeeze((2, 3)))

        attn = self.act(x_avg + x_max)[:, :, None, None]
        x = x * attn

        return x


class _SpatialAttnModule(nn.Module):
    """Spatial Attention Module.

    Args:
        bn: If True, batch normalization is applied to the attn map.

    Shape:
        Input: (B, C, H, W)
        Output: Same shape as the input.
    """

    def __init__(
        self,
        bn: bool = True,
    ) -> None:
        super().__init__()

        self.avg_pool = _ChannelPool2d("avg")
        self.max_pool = _ChannelPool2d("max")
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=7,
            stride=1,
            padding=7 // 2,
            bias=False if bn else True,
        )
        if bn:
            self.bn = nn.BatchNorm2d(1)
        else:
            self.bn = None
        self.act = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)

        x_cat = torch.cat([x_avg, x_max], dim=1)
        x_cat = self.conv(x_cat)
        if self.bn is not None:
            x_cat = self.bn(x_cat)
        attn = self.act(x_cat)
        x = x * attn

        return x


class _ChannelPool2d(nn.Module):
    """Pooling operation along channel dimension.

    Args:
        pool_type: Pooling type.

    Shape:
        Input: (B, C, *)
        Output: (B, 1, *)
    """

    def __init__(self, pool_type: str = "avg") -> None:
        super().__init__()

        if pool_type == "avg":
            self.pool_fn = torch.mean
        elif pool_type == "max":
            self.pool_fn = torch.amax

    def forward(self, x: Tensor) -> None:
        x = self.pool_fn(x, dim=1, keepdim=True)

        return x


if __name__ == "__main__":
    x = torch.rand((2, 3, 224, 224))

    model = CBAM(in_channels=3)
    assert model(x).shape == (2, 3, 224, 224)
