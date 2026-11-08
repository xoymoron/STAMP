import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from stamp.models.components import Pad


class _SoundStreamResidualUnit(nn.Module):
    def __init__(
        self,
        width: int,
        dilation: int,
        kernel_size: int = 7,
        causal: bool = False,
    ):
        super().__init__()

        self.net = nn.Sequential(
            Pad(kernel_size, dilation, causal=causal),
            nn.Conv1d(width, width, kernel_size, dilation=dilation, padding=0),
            nn.ELU(),
            nn.Conv1d(width, width, 1),
        )

        self.final_activation = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return x + self.final_activation(y)


class _SoundStreamEncoderBlock(nn.Module):
    def __init__(
        self,
        width: int,
        stride: int,
        kernel_size: int = 7,
        causal: bool = False,
        aligned: bool = False,
    ):
        super().__init__()

        self.net = nn.ModuleList(
            [
                _SoundStreamResidualUnit(
                    width // 2,
                    1,
                    kernel_size,
                    causal=causal,
                ),
                _SoundStreamResidualUnit(
                    width // 2,
                    3,
                    kernel_size,
                    causal=causal,
                ),
                _SoundStreamResidualUnit(
                    width // 2,
                    9,
                    kernel_size,
                    causal=causal,
                ),
            ]
        )
        self.output = nn.Sequential(
            nn.Identity() if aligned else Pad(2 * stride, 1, causal=causal),
            nn.Conv1d(
                width // 2,
                width,
                2 * stride,
                stride=stride,
                padding=stride // 2 if aligned else 0,
            ),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.net:
            x = layer(x)
        return self.output(x)


class SoundStreamEncoder(nn.Module):
    """Convolutional waveform encoder from SoundStream model, without vector
    quantization.

    Args:
        input_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        output_channels (int): Number of output channels.
        kernel_size (int, optional): Kernel size. Defaults to 7.
        strides (tuple[int, ...], optional): Strides. Defaults to (2, 2, 4, 4).
        causal (bool, optional): Whether to use causal padding. Defaults to False.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        kernel_size: int = 7,
        strides: tuple[int, ...] = (2, 2, 4, 4),
        causal: bool = False,
        transpose_output: bool = False,
        aligned: bool = False,
        checkpoint_length: int | None = None,
    ):
        super().__init__()
        if aligned and (causal or any(s < 2 or s % 2 for s in strides)):
            raise ValueError(
                "Aligned downsampling requires even strides and symmetric padding."
            )
        self.checkpoint_length = checkpoint_length

        self.input = nn.Sequential(
            Pad(kernel_size, 1, causal=causal),
            nn.Conv1d(input_channels, hidden_channels, kernel_size, padding=0),
        )

        encoder_blocks = []
        for stride in strides:
            hidden_channels *= 2
            encoder_blocks.append(
                _SoundStreamEncoderBlock(
                    hidden_channels,
                    stride,
                    kernel_size=kernel_size,
                    causal=causal,
                    aligned=aligned,
                )
            )
        self.encoder_blocks = nn.ModuleList(encoder_blocks)

        self.output = nn.Sequential(
            Pad(3, 1, causal=causal),
            nn.Conv1d(hidden_channels, output_channels, 3, padding=0),
        )
        self.transpose_output = transpose_output
        self.minimum_input_length = 1
        if not causal and not aligned:
            # Noncausal even-kernel strided blocks need at least two inputs.
            for stride in reversed(strides):
                self.minimum_input_length = (self.minimum_input_length - 1) * stride + 2

    def forward(self, x: torch.Tensor, *, checkpoint_activations=True) -> torch.Tensor:
        if x.shape[-1] < 1:
            raise ValueError("SoundStream requires nonempty audio")
        if x.shape[-1] < self.minimum_input_length:
            x = nn.functional.pad(x, (0, self.minimum_input_length - x.shape[-1]))
        recompute = (
            checkpoint_activations
            and torch.is_grad_enabled()
            and self.checkpoint_length is not None
            and x.shape[-1] > self.checkpoint_length
        )
        x = self.input(x)

        for encoder_block in self.encoder_blocks:
            if recompute:
                x = checkpoint(
                    encoder_block, x, use_reentrant=False, preserve_rng_state=False
                )
            else:
                x = encoder_block(x)

        if self.transpose_output:
            x = self.output(x)
            x = x.transpose(1, 2)
            return x

        return self.output(x)
