"""
Squeezeformer (torch implementation).
Author: JiaWei Jiang
"""
import torch
import torch.nn as nn
from modules.conv import ConvolutionModule, DWSepConv2dSubsampling, TimeReductionModule
from modules.ff import FeedForwardModule
from modules.layers import PositionalEncoding
from modules.mha import MHAModule
from modules.residual import ResidualBlock
from torch import Tensor


class Squeezeformer(nn.Module):
    """Squeezeformer model architecture.

    Args:
        n_classes: Number of output classes.
        in_feats: Number of input features.
        enc_dim: Dimension of Squeezeformer encoder.
        n_layers: Number of Squeezeformer blocks.
        time_reduce_idx: Index of layer to perform subsampling along
            time axis.
        time_recover_idx: Index of layer to perform upsampling along
            time axis, recovering the resolution.
        mha_type: Type of positional encoding and self-attention layer.
        mha_num_heads: Number of parallel attention heads in MHAModule.
        mha_dropout: Dropout rate of MHAModule.
        ff_expansion_factor: Expand factor of hidden dimension in
            FeedForwardModule.
        ff_dropout: Dropout rate of FeedForwardModule.
        ff_resid_factor: Factor used to adjust the weight of the output
            of FeedForwardModule.
        conv_expansion_factor: Expand factor of hidden dimension in
            ConvolutionModule.
        conv_dropout: Dropout rate of ConvolutionModule.
        conv_use_glu: If True, the first activation function is replaced
            by gated linear unit in ConvolutionModule.

    Shape:
        Input: (B, L, D), where B is the batch size, L is the sequence
            length, and D denotes the number of input features.
        Output: (B, O), where O denotes the number of output classes.
    """

    def __init__(
        self,
        n_classes: int,
        in_feats: int = 80,
        enc_dim: int = 144,
        n_layers: int = 16,
        time_reduce_idx: int = 7,
        time_recover_idx: int = 15,
        mha_type: str = "abs",
        mha_num_heads: int = 8,
        mha_dropout: float = 0.1,
        ff_expansion_factor: int = 4,
        ff_dropout: float = 0.1,
        ff_resid_factor: float = 1.0,
        conv_expansion_factor: int = 2,
        conv_kernel_size: int = 31,
        conv_dropout: float = 0.1,
        conv_use_glu: bool = False,
        pe_max_len: int = 5000,
    ) -> None:
        super().__init__()

        # Encoder
        self.encoder = SqueezeformerEncoder(
            in_feats=in_feats,
            enc_dim=enc_dim,
            n_layers=n_layers,
            time_reduce_idx=time_reduce_idx,
            time_recover_idx=time_recover_idx,
            mha_type=mha_type,
            mha_num_heads=mha_num_heads,
            mha_dropout=mha_dropout,
            ff_expansion_factor=ff_expansion_factor,
            ff_dropout=ff_dropout,
            ff_resid_factor=ff_resid_factor,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            conv_dropout=conv_dropout,
            conv_use_glu=conv_use_glu,
            pe_max_len=pe_max_len,
        )
        # Decoder
        self.output = nn.Linear(enc_dim, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        output = self.output(x)

        return output


class SqueezeformerEncoder(nn.Module):
    """Squeezeformer encoder.

    Args:
        in_feats: Number of input features.
        enc_dim: Dimension of Squeezeformer encoder.
        n_layers: Number of Squeezeformer blocks.
        time_reduce_idx: Index of layer to perform subsampling along
            time axis.
        time_recover_idx: Index of layer to perform upsampling along
            time axis, recovering the resolution.
        mha_type: Type of positional encoding and self-attention layer.
        mha_num_heads: Number of parallel attention heads in MHAModule.
        mha_dropout: Dropout rate of MHAModule.
        ff_expansion_factor: Expand factor of hidden dimension in
            FeedForwardModule.
        ff_dropout: Dropout rate of FeedForwardModule.
        ff_resid_factor: Factor used to adjust the weight of the output
            of FeedForwardModule.
        conv_expansion_factor: Expand factor of hidden dimension in
            ConvolutionModule.
        conv_dropout: Dropout rate of ConvolutionModule.
        conv_use_glu: If True, the first activation function is replaced
            by gated linear unit in ConvolutionModule.
        pe_max_len: The maximum sequence length that the model might ever
            be used with.

    Shape:
        Input: (B, L, D), where B is the batch size, L is the sequence
            length, and D denotes the number of input features.
        Output: (B, L', E), where L' is the output sequence length, and
            E denotes the encoder dimension.
    """

    def __init__(
        self,
        in_feats: int = 80,
        enc_dim: int = 144,
        n_layers: int = 16,
        time_reduce_idx: int = 7,
        time_recover_idx: int = 15,
        mha_type: str = "abs",
        mha_num_heads: int = 8,
        mha_dropout: float = 0.1,
        ff_expansion_factor: int = 4,
        ff_dropout: float = 0.1,
        ff_resid_factor: float = 1.0,
        conv_expansion_factor: int = 2,
        conv_kernel_size: int = 31,
        conv_dropout: float = 0.1,
        conv_use_glu: bool = False,
        pe_max_len: int = 5000,
    ) -> None:
        super().__init__()

        self.time_reduce_idx = time_reduce_idx
        self.time_recover_idx = time_recover_idx

        # Subsampling
        # Use two Conv2D with stride=2
        self.conv_subsampling = DWSepConv2dSubsampling(in_channels=1, out_channels=enc_dim)
        self.in_proj = nn.Linear(enc_dim * int((in_feats // 2) // 2), enc_dim)

        self.pe = PositionalEncoding(d_model=enc_dim, max_len=pe_max_len)
        self.pre_ln = nn.LayerNorm(enc_dim)
        self.time_reduction = TimeReductionModule(enc_dim, enc_dim)
        self.time_recovery = nn.Linear(enc_dim, enc_dim)

        # Squeezeformer blocks
        self.layers = nn.ModuleList()
        for lth in range(n_layers):
            self.layers.append(
                SqueezeformerBlock(
                    enc_dim=enc_dim,
                    mha_type=mha_type,
                    mha_num_heads=mha_num_heads,
                    mha_dropout=mha_dropout,
                    ff_expansion_factor=ff_expansion_factor,
                    ff_dropout=ff_dropout,
                    ff_resid_factor=ff_resid_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    conv_kernel_size=conv_kernel_size,
                    conv_dropout=conv_dropout,
                    conv_use_glu=conv_use_glu,
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        # Depthwise Separable Subsampling
        x = self.conv_subsampling(x)
        # Project to encoder dimension
        x = self.in_proj(x)

        # Positional Encoding
        x = self.pe(x)

        # Squeezeformer Blocks
        x = self.pre_ln(x)
        for lth, layer in enumerate(self.layers):
            if lth == self.time_reduce_idx:
                # Downsampling
                x_skip = x
                x = self.time_reduction(x)

            if lth == self.time_recover_idx:
                # Upsampling
                # Upsample along time axis (might be longer than x_skip)
                x = torch.repeat_interleave(x, repeats=2, dim=1)
                x = x[:, : x_skip.size(1), :]
                x = self.time_recovery(x)
                x = x + x_skip

            x = layer(x)

        return x


class SqueezeformerBlock(nn.Module):
    """Squeezeformer block.

    The Squeezeformer block follows an MF/CF structure.

    Args:
        enc_dim: Dimension of Squeezeformer encoder.
        mha_type: Type of positional encoding and self-attention layer.
        mha_num_heads: Number of parallel attention heads in MHAModule.
        mha_dropout: Dropout rate of MHAModule.
        ff_expansion_factor: Expand factor of hidden dimension in
            FeedForwardModule.
        ff_dropout: Dropout rate of FeedForwardModule.
        ff_resid_factor: Factor used to adjust the weight of the output
            of FeedForwardModule.
        conv_expansion_factor: Expand factor of hidden dimension in
            ConvolutionModule.
        conv_dropout: Dropout rate of ConvolutionModule.
        conv_use_glu: If True, the first activation function is replaced
            by gated linear unit in ConvolutionModule.

    Shape:
        Input: (B, L, E), where B is the batch size, L is the sequence
            length, and E denotes the encoder dimension.
        Output: Same shape as the input.
    """

    def __init__(
        self,
        enc_dim: int,
        mha_type: str = "abs",
        mha_num_heads: int = 8,
        mha_dropout: float = 0.1,
        ff_expansion_factor: int = 4,
        ff_dropout: float = 0.1,
        ff_resid_factor: float = 1.0,
        conv_expansion_factor: int = 2,
        conv_kernel_size: int = 31,
        conv_dropout: float = 0.1,
        conv_use_glu: bool = False,
    ) -> None:
        super().__init__()

        self.squeezeformer_block = nn.Sequential(
            # MF
            ResidualBlock(
                module=MHAModule(
                    mha_type=mha_type,
                    d_model=enc_dim,
                    num_heads=mha_num_heads,
                    dropout=mha_dropout,
                    adaptive_scaling=True,
                ),
            ),
            nn.LayerNorm(enc_dim),
            ResidualBlock(
                module=FeedForwardModule(
                    in_features=enc_dim,
                    expansion_factor=ff_expansion_factor,
                    dropout=ff_dropout,
                    adaptive_scaling=True,
                ),
                module_factor=ff_resid_factor,
            ),
            nn.LayerNorm(enc_dim),
            # CF
            ResidualBlock(
                module=ConvolutionModule(
                    in_channels=enc_dim,
                    expansion_factor=conv_expansion_factor,
                    kernel_size=conv_kernel_size,
                    dropout=conv_dropout,
                    use_glu=conv_use_glu,
                    adaptive_scaling=True,
                ),
            ),
            nn.LayerNorm(enc_dim),
            ResidualBlock(
                module=FeedForwardModule(
                    in_features=enc_dim,
                    expansion_factor=ff_expansion_factor,
                    dropout=ff_dropout,
                    adaptive_scaling=True,
                ),
                module_factor=ff_resid_factor,
            ),
            nn.LayerNorm(enc_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.squeezeformer_block(x)

        return x


if __name__ == "__main__":
    model = SqueezeformerEncoder()
    x = torch.rand(2, 100, 80)
    x = model(x)
    assert x.shape == (2, 25, 144)
