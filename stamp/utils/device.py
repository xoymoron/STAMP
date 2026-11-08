"""Shared CPU/CUDA selection for offline audio processing."""

from typing import Union

import torch


def resolve_device(device: Union[str, torch.device] = "cpu") -> torch.device:
    """Resolve an explicit device or choose an available GPU for ``auto``.

    Explicit CUDA requests never silently fall back to CPU. CUDA indices refer
    to visible devices, so CUDA_VISIBLE_DEVICES is respected.
    """
    if str(device) == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        selected = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("Use cpu, cuda, cuda:N, or auto for device.") from exc
    if selected.type not in ("cpu", "cuda"):
        raise ValueError("Use cpu, cuda, cuda:N, or auto for device.")
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable. Check the "
                "PyTorch CUDA build and NVIDIA driver, or use --device cpu."
            )
        index = selected.index
        if index is None:
            index = torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is not available.")
        selected = torch.device("cuda", index)
    return selected
