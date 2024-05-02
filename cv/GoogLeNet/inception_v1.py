"""
GoogLeNet - Inception V1 (torch implementation).
Author: JiaWei Jiang
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class GoogLeNet(nn.Module):
    """GoogLeNet model architecture."""

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 10,
        use_aux_clf: bool = True,
    ) -> None:
        """Initialize GoogLeNet.

        Args:
            in_channels: Number of input channels.
            n_classes: Number of output classes.
            use_aux_clf: If True, auxiliary classifiers are used in
                training phase.
        """
        super().__init__()

        self.use_aux_clf = use_aux_clf

        # Input convolution module
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=7 // 2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=3 // 2),
            nn.LocalResponseNorm(size=64),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=3 // 2),
            nn.LocalResponseNorm(size=192),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=3 // 2),
        )

        # Core Inception architecture
        self.inception_3a = _InceptionModule(
            in_channels=192,
            out_channels1=64,
            out_channels2_red=96,
            out_channels2=128,
            out_channels3_red=16,
            out_channels3=32,
            out_channels4=32,
        )
        self.inception_3b = _InceptionModule(
            in_channels=256,
            out_channels1=128,
            out_channels2_red=128,
            out_channels2=192,
            out_channels3_red=32,
            out_channels3=96,
            out_channels4=64,
        )

        self.inception_4a = _InceptionModule(
            in_channels=480,
            out_channels1=192,
            out_channels2_red=96,
            out_channels2=208,
            out_channels3_red=16,
            out_channels3=48,
            out_channels4=64,
        )
        self.inception_4b = _InceptionModule(
            in_channels=512,
            out_channels1=160,
            out_channels2_red=112,
            out_channels2=224,
            out_channels3_red=24,
            out_channels3=64,
            out_channels4=64,
        )
        self.inception_4c = _InceptionModule(
            in_channels=512,
            out_channels1=128,
            out_channels2_red=128,
            out_channels2=256,
            out_channels3_red=24,
            out_channels3=64,
            out_channels4=64,
        )
        self.inception_4d = _InceptionModule(
            in_channels=512,
            out_channels1=112,
            out_channels2_red=144,
            out_channels2=288,
            out_channels3_red=32,
            out_channels3=64,
            out_channels4=64,
        )
        self.inception_4e = _InceptionModule(
            in_channels=528,
            out_channels1=256,
            out_channels2_red=160,
            out_channels2=320,
            out_channels3_red=32,
            out_channels3=128,
            out_channels4=128,
        )

        self.inception_5a = _InceptionModule(
            in_channels=832,
            out_channels1=256,
            out_channels2_red=160,
            out_channels2=320,
            out_channels3_red=32,
            out_channels3=128,
            out_channels4=128,
        )
        self.inception_5b = _InceptionModule(
            in_channels=832,
            out_channels1=384,
            out_channels2_red=192,
            out_channels2=384,
            out_channels3_red=48,
            out_channels3=128,
            out_channels4=128,
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=3 // 2)

        # Auxiliary classfiers
        if use_aux_clf:
            self.aux_clf_4a = _AuxClassifier(512, n_classes)
            self.aux_clf_4d = _AuxClassifier(528, n_classes)

        # Output module
        self.output = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1), nn.Flatten(start_dim=1), nn.Dropout(0.4), nn.Linear(1024, n_classes)
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        # Input convolution module
        x = self.in_conv(x)

        # Core Inception architecture
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.maxpool(x)

        x = self.inception_4a(x)
        if self.training and self.use_aux_clf:
            aux_output1 = self.aux_clf_4a(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        if self.training and self.use_aux_clf:
            aux_output2 = self.aux_clf_4d(x)
        x = self.inception_4e(x)
        x = self.maxpool(x)

        x = self.inception_5a(x)
        x = self.inception_5b(x)

        # Output module
        output = self.output(x)

        if self.training and self.use_aux_clf:
            return output, (aux_output1, aux_output2)
        else:
            return output, None


class _InceptionModule(nn.Module):
    """Inception module."""

    def __init__(
        self,
        in_channels: int,
        out_channels1: int,
        out_channels2_red: int,
        out_channels2: int,
        out_channels3_red: int,
        out_channels3: int,
        out_channels4: int,
    ) -> None:
        """Initialize an Inception module.

        Args:
            in_channels: Number of input channels.
            out_channels1: Number of output channels in tower 1.
            out_channels2_red: Number of output channels of reduction
                layer in tower 2.
            out_channels2: Number of output channels in tower 2.
            out_channels3_red: Number of output channels of reduction
                layer in tower 3.
            out_channels3: Number of output channels in tower 3.
            out_channels4: Number of output channels in tower 4.
        """
        super().__init__()

        self.tower1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels1, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.tower2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels2_red, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(out_channels2_red, out_channels2, kernel_size=3, stride=1, padding=3 // 2),
            nn.ReLU(),
        )
        self.tower3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels3_red, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(out_channels3_red, out_channels3, kernel_size=5, stride=1, padding=5 // 2),
            nn.ReLU(),
        )
        self.tower4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=3 // 2),
            nn.Conv2d(in_channels, out_channels4, kernel_size=1, stride=1),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.tower1(x)
        x2 = self.tower2(x)
        x3 = self.tower3(x)
        x4 = self.tower4(x)

        # DepthConcat - Concatenate along channel dimension
        x = torch.cat([x1, x2, x3, x4], dim=1)

        return x


class _AuxClassifier(nn.Module):
    """Auxiliary classifier."""

    def __init__(self, in_channels: int, n_classes: int = 10) -> None:
        super().__init__()

        self.clf = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(in_channels, 128, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(1024, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        output = self.clf(x)

        return output
