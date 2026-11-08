"""Fixed noise carriers with sample-aligned, learned band envelopes."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from scipy import fft, signal
from torch import nn
from torch.nn import functional as F

SAMPLE_RATE = 48_000
HOP = 128
BANDS = 256
BANK_LENGTH = 65_536
TIME_CHUNK = 65_536


def build_noise_bank(seed: int = 42):
    """Build the offline asset using the NoiseBandNet filter layout."""
    rng = np.random.default_rng(seed)
    edges = np.r_[
        0,
        np.linspace(20, 6000, 128)[:-1],
        np.geomspace(6000, SAMPLE_RATE / 2, 128, endpoint=False),
        SAMPLE_RATE / 2,
    ]
    bank = np.empty((BANDS, BANK_LENGTH), dtype=np.float32)
    metadata = []
    bin_hz = SAMPLE_RATE / BANK_LENGTH
    for band, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        taps, beta = signal.kaiserord(50, 0.2 * (high - low) / (SAMPLE_RATE / 2))
        taps = 2 * (taps // 2) + 1
        if taps > BANK_LENGTH:
            raise ValueError("The noise period is shorter than the FIR response.")
        cutoff = [high] if band == 0 else [low] if band == BANDS - 1 else [low, high]
        impulse = signal.firwin(
            taps, cutoff, window=("kaiser", beta), fs=SAMPLE_RATE, pass_zero=band == 0
        ).astype(np.float32)
        response = np.abs(fft.rfft(impulse, n=BANK_LENGTH))
        phase = rng.uniform(-np.pi, np.pi, response.size).astype(np.float32)
        spectrum = response * np.exp(np.complex64(1j) * phase)
        spectrum[[0, -1]] = 0
        carrier = fft.irfft(spectrum, n=BANK_LENGTH)
        rms = np.sqrt(np.mean(carrier**2))
        bank[band] = carrier / rms
        passband = np.flatnonzero(response >= response.max() / np.sqrt(2))
        metadata.append(
            {
                "edges_hz": [float(low), float(high)],
                "fir_taps": taps,
                "response_peak_hz": float(response.argmax() * bin_hz),
                "response_3db_hz": [
                    float(passband[0] * bin_hz),
                    float(passband[-1] * bin_hz),
                ],
                "rms_before_normalization": float(rms),
                "rms": float(np.sqrt(np.mean(bank[band] ** 2))),
            }
        )
    return {
        "bank": torch.from_numpy(bank),
        "metadata": {
            "version": 1,
            "sample_rate": SAMPLE_RATE,
            "seed": seed,
            "attenuation_db": 50,
            "transition_fraction": 0.2,
            "bands": metadata,
        },
    }


def load_noise_bank(path: str | Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata, bank = payload["metadata"], payload["bank"]
    if (
        metadata["version"] != 1
        or metadata["sample_rate"] != SAMPLE_RATE
        or bank.shape != (BANDS, BANK_LENGTH)
        or bank.dtype != torch.float32
        or not torch.isfinite(bank).all()
    ):
        raise ValueError("Invalid FP32 48 kHz noise bank asset.")
    return bank, hashlib.sha256(bank.contiguous().numpy()).hexdigest()


class _NoiseBandRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx, amplitudes, bank, offsets, length):
        amps = amplitudes.squeeze(0).transpose(0, 1)
        ctx.save_for_backward(bank, offsets)
        ctx.frames = amps.shape[1]
        ctx.chunk_size = TIME_CHUNK
        output = []
        for start in range(0, length, ctx.chunk_size):
            stop = min(start + ctx.chunk_size, length)
            samples = torch.arange(start, stop, device=amps.device)
            first = max(0, (start - HOP // 2) // HOP)
            last = min(amps.size(1), (stop - 1 - HOP // 2) // HOP + 2)
            sample_start = start - first * HOP
            chunk = 0
            for band in range(0, BANDS, 64):
                bands = slice(band, band + 64)
                envelope = F.interpolate(
                    amps[None, bands, first:last],
                    scale_factor=HOP,
                    mode="linear",
                    align_corners=False,
                )[0, :, sample_start : sample_start + samples.numel()]
                carrier = bank[bands].gather(
                    1, (samples + offsets[bands, None]) % BANK_LENGTH
                )
                chunk = chunk + (envelope * carrier).sum(0)
            output.append(chunk)
        return torch.cat(output)[None, None, :]

    @staticmethod
    def backward(ctx, gradient):
        bank, offsets = ctx.saved_tensors
        gradient = gradient.flatten()
        relative = (torch.arange(HOP, device=gradient.device) + 0.5) / HOP - 0.5
        own = 1 - relative.abs()
        prev = -relative[: HOP // 2]
        following = relative[HOP // 2 :]
        # Halo bins collect the contributions held at the two endpoints.
        result = gradient.new_zeros((BANDS, ctx.frames + 2))
        for start in range(0, gradient.numel(), ctx.chunk_size):
            stop = min(start + ctx.chunk_size, gradient.numel())
            upstream = gradient[start:stop]
            pad = (-upstream.numel()) % HOP
            samples = torch.arange(start, stop + pad, device=gradient.device)
            upstream = F.pad(upstream, (0, pad))
            first = start // HOP
            last = first + upstream.numel() // HOP
            for band in range(0, BANDS, 64):
                bands = slice(band, band + 64)
                carrier = bank[bands].gather(
                    1, (samples + offsets[bands, None]) % BANK_LENGTH
                )
                product = (carrier * upstream).reshape(64, -1, HOP)
                result[bands, first:last] += (product[..., : HOP // 2] * prev).sum(-1)
                result[bands, first + 1 : last + 1] += (product * own).sum(-1)
                result[bands, first + 2 : last + 2] += (
                    product[..., HOP // 2 :] * following
                ).sum(-1)
        result[:, 1] += result[:, 0]
        result[:, -2] += result[:, -1]
        return result[:, 1:-1].transpose(0, 1).unsqueeze(0), None, None, None


class NoiseBandSynth(nn.Module):
    def __init__(self, bank: torch.Tensor):
        super().__init__()
        if bank.shape != (BANDS, BANK_LENGTH) or bank.dtype != torch.float32:
            raise ValueError("Expected a [256,65536] FP32 carrier bank.")
        self.register_buffer("bank", bank.contiguous(), persistent=False)

    def forward(self, amplitudes: torch.Tensor, offsets: torch.Tensor, length: int):
        return _NoiseBandRender.apply(amplitudes, self.bank, offsets, length)
