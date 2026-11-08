import torch
from torch import nn

from stamp.models.components import FiLM, Pad


class _DilatedResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        causal: bool,
        embedding_size: int | None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            Pad(kernel_size, dilation, causal=causal),
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation),
        )
        self.film = (
            FiLM(embedding_size, channels) if embedding_size is not None else None
        )
        self.activation = nn.GELU()
        self.residual = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor | None):
        affine = None
        if self.film is not None:
            if embedding is None:
                raise ValueError("A clip embedding is required for FiLM.")
            affine = self.film.net(embedding)
        return self.apply_affine(x, affine)

    def apply_affine(self, x: torch.Tensor, affine: torch.Tensor | None):
        y = self.net(x)
        if affine is not None:
            gamma, beta = affine.unsqueeze(-1).chunk(2, dim=1)
            y = gamma * y + beta
        return self.activation(y) + self.residual(x)


class TCN(nn.Module):
    """Dilated residual waveform network with optional clip-level FiLM."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        num_layers: int,
        kernel_size: int,
        dilation_base: int = 2,
        causal: bool = True,
        film_embedding_size: int | None = None,
    ):
        super().__init__()
        self.in_projection = nn.Conv1d(in_channels, hidden_channels, 1)
        self.net = nn.ModuleList(
            _DilatedResidualBlock(
                hidden_channels,
                kernel_size,
                dilation_base**i,
                causal,
                film_embedding_size,
            )
            for i in range(num_layers)
        )
        self.out_projection = nn.Conv1d(hidden_channels, out_channels, 1)
        self.left_context = (
            (kernel_size - 1) * sum(dilation_base**i for i in range(num_layers))
            if causal
            else None
        )

    def forward(self, x: torch.Tensor, embedding: torch.Tensor | None = None):
        x = self.in_projection(x)
        for layer in self.net:
            x = layer(x, embedding)
        return self.out_projection(x)

    def film_coefficients(self, embedding: torch.Tensor):
        return tuple(layer.film.net(embedding) for layer in self.net)

    def forward_with_film(
        self, x: torch.Tensor, coefficients: tuple[torch.Tensor, ...]
    ):
        x = self.in_projection(x)
        for layer, affine in zip(self.net, coefficients):
            x = layer.apply_affine(x, affine)
        return self.out_projection(x)
