"""CUDA FFT front end for the existing CPU modal peak tracker.

Only spectral transforms run on CUDA. Each bounded block returns to NumPy so
the peak fitting, association, and exported features keep their CPU semantics.
"""

import numpy as np
import torch
import torch.nn.functional as functional
from scipy import signal


def cqt_coefficients(
    reduced,
    sample_rate,
    centers,
    frequencies,
    lengths,
    width,
    transform_size,
    decimation,
    device,
    batch_size,
):
    """Evaluate the same centered CQT kernels as the SciPy path in CUDA batches."""
    from stamp.utils.modal_analysis import _window_gain

    spectrum = torch.fft.fft(torch.as_tensor(reduced, device=device), n=transform_size)
    positions = centers / decimation
    left = np.floor(positions).astype(np.int64)
    fraction = positions - left
    first_index = torch.as_tensor(
        width // 2 + np.clip(left, 0, len(reduced) - 1), device=device
    )
    second_index = torch.as_tensor(
        width // 2 + np.clip(left + 1, 0, len(reduced) - 1), device=device
    )
    fraction = torch.as_tensor(fraction, dtype=torch.float32, device=device)
    output = np.empty((len(centers), len(frequencies)), np.complex64)

    for start in range(0, len(frequencies), batch_size):
        stop = min(start + batch_size, len(frequencies))
        kernels = np.zeros((stop - start, width), np.complex64)
        gains_first = np.empty((stop - start, len(centers)), np.float32)
        gains_second = np.empty_like(gains_first)
        for row, index in enumerate(range(start, stop)):
            frequency = frequencies[index]
            length = lengths[index]
            offsets = np.arange(length) - length // 2
            window = signal.windows.hann(length, sym=True)
            positions_to_normalize = np.r_[
                np.clip(left, 0, len(reduced) - 1),
                np.clip(left + 1, 0, len(reduced) - 1),
            ]
            gains = _window_gain(window, positions_to_normalize, len(reduced))
            gains_first[row], gains_second[row] = np.split(gains, 2)
            offset = (width - length) // 2
            kernels[row, offset : offset + length] = (
                2
                * window
                / window.sum()
                * np.exp(2j * np.pi * frequency * offsets / sample_rate)
            )

        kernels = torch.as_tensor(kernels, device=device)
        convolved = torch.fft.ifft(
            torch.fft.fft(kernels, n=transform_size, dim=-1) * spectrum,
            dim=-1,
        )
        first = convolved.index_select(-1, first_index)
        second = convolved.index_select(-1, second_index)
        first *= torch.as_tensor(gains_first, dtype=first.real.dtype, device=device)
        second *= torch.as_tensor(gains_second, dtype=second.real.dtype, device=device)
        frequency = torch.as_tensor(
            frequencies[start:stop], dtype=torch.float32, device=device
        )
        rotation = torch.exp(2j * torch.pi * frequency / sample_rate)[:, None]
        phase = torch.exp(
            2j * torch.pi * frequency[:, None] * fraction[None] / sample_rate
        )
        coefficients = (
            (1 - fraction)[None] * first + fraction[None] * second / rotation
        ) * phase
        output[:, start:stop] = coefficients.transpose(0, 1).cpu().numpy()
    return output


def stft_blocks(audio, centers, size, frame_block_size, device):
    """Yield centered, normalized STFT blocks on CPU after CUDA transforms."""
    from stamp.utils.modal_analysis import _window_gain

    sample_count = len(audio)
    window_numpy = signal.windows.hann(size, sym=False).astype(np.float32)
    audio = torch.as_tensor(audio, device=device)
    padded = functional.pad(audio, (size // 2, size // 2))
    windows = padded.unfold(0, size, 1)
    window = torch.as_tensor(window_numpy, device=device)
    rotation = torch.where(
        torch.arange(size // 2 + 1, device=device) % 2 == 0, 1.0, -1.0
    )
    for start in range(0, len(centers), frame_block_size):
        stop = min(start + frame_block_size, len(centers))
        locations = np.rint(centers[start:stop]).astype(np.int64)
        indices = torch.as_tensor(locations, device=device)
        spectra = torch.fft.rfft(windows.index_select(0, indices) * window, dim=-1)
        spectra *= (2 / window.sum()) * rotation
        gains = _window_gain(window_numpy, locations, sample_count)
        spectra *= torch.as_tensor(gains, dtype=spectra.real.dtype, device=device)[
            :, None
        ]
        yield start, locations, spectra.cpu().numpy()
