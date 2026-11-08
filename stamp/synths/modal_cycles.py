"""FP32 modal rendering for the cycles/sample joint-model contract."""

import math
from functools import lru_cache
from typing import Optional

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from stamp.utils.modal_tracks import _fill_inactive_frequencies


@lru_cache
def _cuda_carry():
    try:
        from stamp.synths._cycle_cuda import wrapped_block_starts
    except ImportError:
        return None
    return wrapped_block_starts


def _block_starts(totals):
    carry = _cuda_carry() if totals.is_cuda else None
    if carry is not None:
        return carry(totals)
    starts = torch.empty_like(totals)
    state = torch.zeros_like(totals[..., 0])
    compensation = torch.zeros_like(state)
    for index in range(totals.shape[-1]):
        starts[..., index] = state
        increment = totals[..., index].remainder(1.0) - compensation
        updated = state + increment
        compensation = (updated - state) - increment
        state = updated.remainder(1.0)
    return starts


class _WrappedCycleSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frequency, block_size):
        length = frequency.shape[-1]
        padded = F.pad(frequency, (0, (-length) % block_size))
        blocks = padded.reshape(*frequency.shape[:-1], -1, block_size)
        local = blocks.cumsum(-1)
        tree_width = 1 << (block_size - 1).bit_length()
        tree = F.pad(blocks, (0, tree_width - block_size))
        while tree.shape[-1] > 1:
            tree = tree[..., ::2] + tree[..., 1::2]
        starts = _block_starts(tree.squeeze(-1))
        wrapped = (local + starts.unsqueeze(-1)).remainder(1.0)
        return wrapped.flatten(-2)[..., :length]

    @staticmethod
    def backward(ctx, gradient):
        # Modulo has unit derivative; reverse cumsum is the prefix-sum adjoint.
        # The enclosing sine makes integer-turn boundaries periodic.
        return gradient.flip(-1).cumsum(-1).flip(-1), None


def _wrapped_cycle_cumsum(frequency: torch.Tensor, block_size: int = 256):
    """Accumulate FP32 cycles with pairwise block totals and wrapped Kahan carry.

    Like DDSP angular_cumsum, this bounds phase before evaluating sine:
    https://github.com/magenta/ddsp/blob/main/ddsp/core.py#L724
    Unlike that radians-based implementation, this consumes cycles/sample and
    uses compensated block carries. The output is for periodic functions only,
    not a discontinuous phase target or neural feature. No tensors are saved
    for backward: the analytical prefix-sum adjoint avoids retaining the Kahan
    recurrence graph. Integer-turn boundaries use its periodic continuation.
    """
    if frequency.dtype != torch.float32 or frequency.shape[-1] < 1:
        raise ValueError("Expected nonempty FP32 cycles/sample frequency.")
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size < 1
    ):
        raise ValueError("block_size must be a positive integer.")
    return _WrappedCycleSum.apply(frequency, block_size)


def modal_synth_cycles(
    freqs: torch.Tensor,
    amps: torch.Tensor,
    num_samples: int,
    phase: Optional[torch.Tensor] = None,
    mode_chunk_size: int = 32,
    phase_block_size: int = 256,
    analysis_num_samples: Optional[int] = None,
    recompute: bool = False,
) -> torch.Tensor:
    """Render cycles/sample frequencies with FP32-only phase accumulation.

    Inputs have shape [batch, modes, frames]. Initial sine phase remains in
    radians, read from frame zero; sample zero includes the first frequency
    increment. analysis_num_samples preserves the original align_corners=False
    clock when the physical audio is shorter than the analysis extent. Cache
    radians must be converted by the data adapter, never by this graph.
    """
    if freqs.ndim != 3 or freqs.shape != amps.shape or min(freqs.shape) < 1:
        raise ValueError("Expected matching nonempty [batch,modes,frames] tensors.")
    if freqs.dtype != torch.float32 or amps.dtype != torch.float32:
        raise ValueError("Cycles modal synthesis requires FP32 inputs.")
    if phase is not None and (
        phase.shape != freqs.shape or phase.dtype != torch.float32
    ):
        raise ValueError("Phase must match frequencies and use FP32 radians.")
    if num_samples < 1 or mode_chunk_size < 1:
        raise ValueError("num_samples and mode_chunk_size must be positive.")
    extent = num_samples if analysis_num_samples is None else analysis_num_samples
    if extent < num_samples:
        raise ValueError("Analysis extent must cover the requested audio.")
    freqs = _fill_inactive_frequencies(freqs, amps)
    position = (
        (torch.arange(num_samples, device=freqs.device, dtype=torch.float32) + 0.5)
        * (freqs.shape[-1] / extent)
        - 0.5
    ).clamp(0, freqs.shape[-1] - 1)
    left = position.floor().long()
    right = (left + 1).clamp(max=freqs.shape[-1] - 1)
    fraction = position - left

    def render_chunk(frequency, amplitude, initial_phase):
        f_left, f_right = frequency[..., left], frequency[..., right]
        a_left = amplitude[..., left]
        a_right = amplitude[..., right]
        left_active = a_left != 0
        right_active = a_right != 0
        w = torch.lerp(f_left, f_right, fraction)
        w = torch.where(left_active & ~right_active, f_left, w)
        w = torch.where(~left_active & right_active, f_right, w)
        a = torch.lerp(a_left, a_right, fraction)
        angle = _wrapped_cycle_cumsum(w, phase_block_size) * (2 * math.pi)
        if initial_phase is not None:
            angle = angle + initial_phase
        return (a * angle.sin()).sum(1)

    output = freqs.new_zeros((freqs.shape[0], num_samples))
    for start in range(0, freqs.shape[1], mode_chunk_size):
        modes = slice(start, start + mode_chunk_size)
        inputs = (
            freqs[:, modes],
            amps[:, modes],
            None if phase is None else phase[:, modes, :1],
        )
        if recompute and torch.is_grad_enabled():
            chunk = checkpoint(
                render_chunk, *inputs, use_reentrant=False, preserve_rng_state=False
            )
        else:
            chunk = render_chunk(*inputs)
        output = output + chunk
    return output


class CyclesModalSynth(torch.nn.Module):
    """Render FP32 cycles/sample tracks with initial phase in radians."""

    def __init__(self, phase_block_size: int = 256, mode_chunk_size: int = 32):
        super().__init__()
        self.phase_block_size = phase_block_size
        self.mode_chunk_size = mode_chunk_size

    def forward(
        self,
        params: torch.Tensor,
        num_samples: int,
        analysis_num_samples=None,
        *,
        recompute: bool = False,
    ):
        if params.ndim != 4 or params.shape[1] not in (2, 3):
            raise ValueError("Expected [batch,2 or 3,modes,frames] parameters.")
        phase = params[:, 2] if params.shape[1] == 3 else None
        return modal_synth_cycles(
            params[:, 0],
            params[:, 1],
            num_samples,
            phase,
            self.mode_chunk_size,
            self.phase_block_size,
            analysis_num_samples,
            recompute,
        ).unsqueeze(1)
