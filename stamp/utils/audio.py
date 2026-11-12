"""
Audio utility functions
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

from stamp.utils.device import resolve_device


def read_float_wav(path: Path) -> tuple[torch.Tensor, int]:
    """Decode WAV samples as float32 in channels-first order."""
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), sample_rate


def write_float_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Write channels-first samples as IEEE float32 WAV at their current level."""
    sf.write(
        path,
        waveform.detach().cpu().float().T.numpy(),
        sample_rate,
        format="WAV",
        subtype="FLOAT",
    )


def preprocess_audio_file(
    input_file: Path,
    output_file: Path,
    sample_rate: int,
    num_samples: Optional[int] = None,
    mono: bool = True,
    silence_threshold_db: float = -65.0,
    remove_start_silence: bool = True,
    remove_end_silence: bool = False,
    tail_silence_threshold_db: float = -70.0,
    tail_peak_ratio: float = 0.001,
    min_tail_silence_ms: float = 50.0,
    frame_size: int = 256,
    hop_size: int = 256,
    max_duration_sec: Optional[float] = 20.0,
    device: Union[str, torch.device] = "cpu",
):
    """Convert one WAV file into the canonical training-audio representation.

    The variable-length pipeline is: load, select one channel when requested,
    resample, find the first frame above ``silence_threshold_db``, trim silence,
    reject samples longer than the configured limit, and save. The onset scan
    is also the all-silent rejection check, so the waveform is scanned once.
    No loudness or peak normalization is applied. Signal operations use
    ``device`` (cpu, cuda, cuda:N, or auto); decoding and saving use CPU tensors.
    """
    compute_device = resolve_device(device)
    waveform, orig_freq = read_float_wav(input_file)
    assert waveform.ndim == 2, "Expecting a 2D tensor, channels x samples"
    waveform = waveform.to(compute_device)

    if mono:
        waveform = select_highest_rms_channel(waveform)
        assert waveform.shape[0] == 1, "Expecting a mono signal"

    if orig_freq != sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_freq, new_freq=sample_rate
        ).to(device=compute_device, dtype=waveform.dtype)
        waveform = resampler(waveform)

    onset_sample = first_non_silent_sample_multichannel(
        waveform,
        frame_size=frame_size,
        hop_size=hop_size,
        threshold_db=silence_threshold_db,
    )
    if onset_sample is None:
        raise ValueError(f"silent_all: below {silence_threshold_db}dB")
    if remove_start_silence:
        waveform = waveform[:, onset_sample:]

    if remove_end_silence:
        min_tail_silence_samples = int((min_tail_silence_ms / 1000.0) * sample_rate)

        waveform = cut_end_silence(
            waveform,
            frame_size=frame_size,
            hop_size=hop_size,
            threshold_db=tail_silence_threshold_db,
            min_silence_samples=min_tail_silence_samples,
            peak_ratio=tail_peak_ratio,
        )

    if max_duration_sec is not None and (num_samples is None):
        max_len = int(max_duration_sec * sample_rate)
        if waveform.shape[1] > max_len:
            raise ValueError(
                f"too_long: {waveform.shape[1]} samples > {max_len} samples"
            )

    if num_samples is not None and waveform.shape[1] != num_samples:
        if waveform.shape[1] > num_samples:
            waveform = waveform[:, :num_samples]
        else:
            num_pad = num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, num_pad))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_float_wav(output_file, waveform, sample_rate)


def select_highest_rms_channel(waveform: torch.Tensor) -> torch.Tensor:
    """Return the physical input channel with the largest RMS amplitude.

    ``torch.argmax`` deterministically selects the first channel when RMS
    values tie. A one-channel waveform is returned unchanged.
    """
    assert waveform.ndim == 2, "Expecting a 2D tensor, channels x samples"
    if waveform.shape[0] == 1:
        return waveform

    channel_rms = torch.sqrt(torch.mean(waveform.square(), dim=1))
    channel_index = int(torch.argmax(channel_rms).item())
    return waveform[channel_index : channel_index + 1, :]


def first_non_silent_sample(
    x: torch.Tensor,
    frame_size: int = 256,
    hop_size: int = 256,
    threshold_db: float = -65.0,
) -> Union[int, None]:
    """
    Returns the index of the first non-silent sample in a waveform.
    Implementation based on Essentia StartStopCut.
    """
    assert x.ndim == 1, "Expecting a 1D tensor"
    threshold_power = float(np.power(10.0, threshold_db / 10.0))

    for start in range(0, x.shape[-1], hop_size):
        frame = x[start : start + frame_size]
        power = torch.inner(frame, frame) / frame.shape[-1]
        if power > threshold_power:
            return start
    return None


def first_non_silent_sample_multichannel(
    x: torch.Tensor,
    frame_size: int = 256,
    hop_size: int = 256,
    threshold_db: float = -65.0,
) -> Optional[int]:
    """Return the earliest frame above the threshold in any channel."""
    assert x.ndim == 2, "Expecting (channels, num_samples)"
    start_samples = [
        first_non_silent_sample(
            channel,
            frame_size=frame_size,
            hop_size=hop_size,
            threshold_db=threshold_db,
        )
        for channel in x
    ]
    valid_starts = [start for start in start_samples if start is not None]
    return min(valid_starts) if valid_starts else None


def cut_end_silence(
    x: torch.Tensor,
    frame_size: int = 256,
    hop_size: int = 256,
    threshold_db: float = -60.0,
    min_silence_samples: int = 0,
    peak_ratio: float = 0.02,
) -> torch.Tensor:
    """
    Removes silent samples from the end of a waveform.

    Cut condition:
      - trailing region length >= min_silence_samples
      - tail peak <= peak_ratio * global_peak
      - tail frames (mean power) are below threshold_db (power dB)
    """
    assert x.ndim == 2, "Expecting (channels, num_samples)"
    _, T = x.shape
    if T == 0:
        return x

    global_peak = float(x.abs().max().item())
    if global_peak <= 1e-12:
        raise ValueError("Entire wavfile is (near) zero")

    thr_power = float(np.power(10.0, threshold_db / 10.0))

    num_frames = int(math.ceil(T / hop_size))
    frame_starts = [i * hop_size for i in range(num_frames)]

    silent_flags = []
    for start in frame_starts:
        end = min(start + frame_size, T)
        frame = x[:, start:end]
        if frame.numel() == 0:
            silent = True
        else:
            power = (frame * frame).mean(dim=1)  # [C]
            silent = bool(torch.all(power <= thr_power).item())
        silent_flags.append(silent)

    # Find the last non-silent frame.
    last_non_silent_idx = None
    for i in range(num_frames - 1, -1, -1):
        if not silent_flags[i]:
            last_non_silent_idx = i
            break

    if last_non_silent_idx is None:
        raise ValueError(f"Entire wavfile below threshold level {threshold_db}dB")

    candidate_cut = min(frame_starts[last_non_silent_idx] + frame_size, T)
    trailing_len = T - candidate_cut
    if trailing_len < min_silence_samples:
        return x

    tail_peak = (
        float(x[:, candidate_cut:].abs().max().item()) if candidate_cut < T else 0.0
    )
    if tail_peak <= peak_ratio * global_peak:
        return x[:, :candidate_cut]

    return x
