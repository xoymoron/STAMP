"""Inference-only measurements for frozen latent controls."""

import hashlib
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

SCOPES = ("modal", "noise", "transient", "mix")
EPS = 1e-8


def runtime_sources(entrypoint):
    files = {"entrypoint": Path(entrypoint)}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("stamp.") and path and Path(path).suffix == ".py":
            files[name] = Path(path)
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in sorted(files.items())
    }


def redundancy_valid(neutral_rms, edited_rms, effect_norm):
    return (neutral_rms >= 1e-5) & (edited_rms >= 1e-5).all(-1) & (effect_norm > 1e-6)


def relative_norms(value, reference):
    valid = reference.square().mean(-1).sqrt() > EPS
    ratio = value.norm(dim=-1) / reference.norm(dim=-1).clamp_min(1e-12)
    return [float(x) if keep else None for x, keep in zip(ratio, valid)]


def branches(result):
    return torch.stack(
        [
            result.modal_waveform.flatten(),
            result.noise_waveform.flatten(),
            result.transient_waveform.flatten(),
            result.waveform.flatten(),
        ]
    )


class SpectralReference:
    """GPU equivalent of the audition's RMS-matched spectral-change metric."""

    def __init__(self, wave):
        self.length = wave.shape[-1]
        self.windows = [
            torch.hann_window(min(n, self.length), device=wave.device)
            for n in (512, 2048, 4096)
        ]
        self.valid = wave.square().mean(-1).sqrt() > EPS
        crest = wave.abs().amax(-1, keepdim=True) / wave.square().mean(
            -1, keepdim=True
        ).sqrt().clamp_min(1e-12)
        self.quiet_target = (0.95 / crest.clamp_min(1e-12)).clamp_max(0.1)
        self.spectra = self.spectra_for(wave)

    def spectra_for(self, wave):
        rms = wave.square().mean(-1, keepdim=True).sqrt()
        # matched_waves leaves sub-floor edits unscaled; cancel only the reference target.
        wave = torch.where(
            rms > EPS, wave / rms.clamp_min(1e-12), wave / self.quiet_target
        )
        spectra = []
        for window in self.windows:
            n = len(window)
            hop = n - n * 3 // 4
            padding = (-(wave.shape[-1] - n % 2)) % hop
            value = torch.stft(
                F.pad(wave, (0, padding)),
                n,
                hop_length=hop,
                window=window,
                center=True,
                pad_mode="constant",
                return_complex=True,
            ).abs()
            spectra.append(value)
        return spectra

    def change(self, wave):
        values = (
            torch.stack(
                [
                    (a - b).flatten(1).norm(dim=1)
                    / b.flatten(1).norm(dim=1).clamp_min(1e-12)
                    for a, b in zip(self.spectra_for(wave), self.spectra)
                ]
            )
            .mean(0)
            .tolist()
        )
        return [v if valid else None for v, valid in zip(values, self.valid.tolist())]


def compare_branches(result, neutral, spectral=None):
    current, origin = branches(result), branches(neutral)
    if not torch.isfinite(current).all():
        raise FloatingPointError("Nonfinite synthesis output")
    metric = spectral if spectral is not None else SpectralReference(origin)
    changes = metric.change(current)
    rms = current.square().mean(-1).sqrt()
    original_rms = origin.square().mean(-1).sqrt()
    delta = (current - origin).square().mean(-1).sqrt()
    peak = current.abs().amax(-1)
    raw = torch.stack((rms, original_rms, delta, peak)).cpu().tolist()
    rows = {}
    for i, scope in enumerate(SCOPES):
        r, r0, d, p = (a[i] for a in raw)
        rows.update(
            {
                scope + "/spectral_change": changes[i],
                scope + "/delta_over_mix": d / raw[1][-1] if raw[1][-1] > EPS else None,
                scope + "/delta_over_branch": d / r0 if r0 > EPS else None,
                scope + "/rms_ratio": r / r0 if r0 > EPS else None,
                scope + "/rms": r,
                scope + "/peak": p,
                scope + "/neutral_silent": r0 <= EPS,
            }
        )
    return rows
