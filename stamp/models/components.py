import torch
from torch import nn


class Pad(nn.Module):
    """Pad a tensor with zeros according to causal or non-causal 1D padding scheme.

    Args:
        kernel_size (int): Size of the convolution kernel.
        dilation (int): Dilation factor.
        causal (bool, optional): Whether to use causal padding. Defaults to True.
    """

    def __init__(self, kernel_size: int, dilation: int, causal: bool = True):
        super().__init__()
        pad = dilation * (kernel_size - 1)
        if not causal:
            pad //= 2
            self.padding = (pad, pad)
        else:
            self.padding = (pad, 0)

    def forward(self, x):
        return nn.functional.pad(x, self.padding)


class FiLM(nn.Module):
    """Apply clip-level affine conditioning across the time axis."""

    def __init__(self, film_embedding_size: int, input_channels: int):
        super().__init__()
        self.net = nn.Linear(film_embedding_size, 2 * input_channels)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor):
        gamma, beta = self.net(embedding).chunk(2, dim=-1)
        return gamma[..., None] * x + beta[..., None]


class TemporalBlock(nn.Module):
    """Local temporal processing without normalization across modes or time."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            5,
            padding=2 * dilation,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv1d(channels * 2, channels, 1),
        )

    def forward(self, x, mask):
        b, k, t, c = x.shape
        y = self.norm(x).masked_fill(~mask[..., None], 0)
        y = y.reshape(b * k, t, c).transpose(1, 2)
        y = self.pointwise(self.depthwise(y))
        y = y.transpose(1, 2).reshape(b, k, t, c)
        return (x + y).masked_fill(~mask[..., None], 0)
