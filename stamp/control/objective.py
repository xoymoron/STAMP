"""Frozen joint synthesis and branchwise post-hoc intervention observations."""

from math import expm1, log, sqrt

import torch
from torch import nn
from torch.nn import functional as F

from stamp.control.editing import DIMENSION, AcousticView, BoundedReader
from stamp.models.joint_resynthesis import MODE_CHUNK

SCOPES = ("modal", "noise", "mix")


def gain_from_knob(position):
    return (1 if position >= 0 else -1) * expm1(abs(position) * log(65))


def joint_prefix(model, ref, h, end):
    """Exact S/N prefix; preserve all frame clocks and preceding phase gradients."""
    if not 0 < end <= ref.length:
        raise ValueError("Prefix endpoint must be inside the valid clip.")
    params, _ = model._refine(ref, h, False)
    modal = model.modal_synth(params, end, ref.analysis_num_samples).flatten()[None]
    count = ref.observed.shape[-1]
    updated = torch.cat(
        [
            model._modal_tokens(p, mask, False)
            for p, mask in zip(
                params[..., :count].split(MODE_CHUNK, 2),
                ref.observed.split(MODE_CHUNK, 1),
            )
        ],
        1,
    )
    cm = model._context(
        updated,
        ref.observed,
        ref.audio_times.clamp_max(ref.length - 1),
        ref.analysis_num_samples,
        params.shape[-1],
        False,
    )
    envelope = model.noise_head(torch.cat((h, cm), -1))
    noise = model.noise_synth(envelope, ref.noise_offsets, end).flatten()[None]
    return {"modal": modal, "noise": noise, "mix": modal + noise}


class JointObjective(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.observers = nn.ModuleDict({k: AcousticView() for k in SCOPES})
        self.readers = nn.ModuleDict({k: BoundedReader(features) for k in SCOPES})

    @torch.no_grad()
    def views(self, neutral, rng):
        views = {}
        for scope, wave in neutral.items():
            if float(wave.square().mean().sqrt()) < 1e-5:
                continue
            observer = self.observers[scope]
            rows = []
            for _ in range(2):
                jitter = int(rng.integers(128))
                scale = 10 ** (-float(rng.uniform(30, 40)) / 20)
                first, second = (
                    torch.randn_like(wave) * scale,
                    torch.randn_like(wave) * scale,
                )
                origin = observer(wave, jitter, first)
                zero = (observer(wave, jitter, second) - origin) / sqrt(DIMENSION)
                rows.append((jitter, second, origin, zero))
            views[scope] = rows
        return views

    def effects(self, waves, views):
        return {
            scope: torch.cat(
                [
                    (self.observers[scope](waves[scope], jitter, noise) - origin)
                    / sqrt(DIMENSION)
                    for jitter, noise, origin, _ in rows
                ]
            )
            for scope, rows in views.items()
        }

    def loss(self, effects, group, position, knob, *, frozen=False):
        if not effects:
            raise ValueError("No audible neutral scope for the intervention objective.")
        losses, metrics = [], {}
        for scope, effect in effects.items():
            loss, measured = self.readers[scope].loss(
                effect, group, position, knob, frozen=frozen
            )
            losses.append(loss)
            metrics.update(
                {f"{scope}/{key}": float(value) for key, value in measured.items()}
            )
            metrics[f"{scope}/loss"] = float(loss.detach())
        return torch.stack(losses).mean(), metrics

    def zero_loss(self, views, group):
        losses = []
        for scope, rows in views.items():
            zero = torch.cat([row[3] for row in rows])
            _, amount = self.readers[scope](zero)
            losses.append(
                F.smooth_l1_loss(amount[:, group], torch.zeros_like(amount[:, group]))
            )
        return 0.25 * torch.stack(losses).mean()
