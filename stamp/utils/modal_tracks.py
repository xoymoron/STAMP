import torch


def _fill_inactive_frequencies(freqs: torch.Tensor, amps: torch.Tensor) -> torch.Tensor:
    """Hold the last observed frequency in gaps and fill leading frames.

    Activity is independent of frequency values, so this also repairs cached
    fmin padding. Fully silent modes use zero frequency. Gather/interpolation
    keep gradients connected to the observed frequency values. This stored
    placeholder does not define interpolation through missing observations;
    the cycles renderer applies the active-endpoint rule at sample resolution.
    """
    frames = freqs.shape[-1]
    if frames < 1:
        raise ValueError("Modal parameters must contain at least one frame.")
    active = amps != 0
    index = torch.arange(frames, device=freqs.device).expand_as(freqs)
    previous = torch.where(active, index, -1).cummax(dim=-1).values
    following = (
        torch.where(active, index, frames).flip(-1).cummin(dim=-1).values.flip(-1)
    )
    source = torch.where(previous >= 0, previous, following).clamp(0, frames - 1)
    filled = freqs.gather(-1, source)
    return torch.where(
        active.any(dim=-1, keepdim=True), filled, torch.zeros_like(filled)
    )
