"""Validate sample/frame counts without silently changing their time axis."""

import torch


def validate_lengths(lengths, *, batch_size: int, max_length: int, device):
    """Return integer counts [B], or None when the entire stored axis is valid."""
    if lengths is None:
        return None
    if isinstance(lengths, (list, tuple)) and any(isinstance(x, bool) for x in lengths):
        raise ValueError("lengths must contain integer counts, not booleans")
    lengths = torch.as_tensor(lengths, device=device)
    if lengths.ndim != 1 or lengths.numel() != batch_size:
        raise ValueError(
            f"Expected lengths with shape [{batch_size}], got {tuple(lengths.shape)}"
        )
    if lengths.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("lengths must contain integer counts")
    if max_length < 1 or torch.any(lengths < 1) or torch.any(lengths > max_length):
        raise ValueError(f"lengths must be in [1, {max_length}]")
    return lengths.long()
