"""FP32 temporal SAE and boundary-safe batches for frozen joint latents."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from stamp.models.joint_resynthesis import HOP

CHANNELS = 256
FEATURES = 512
CORE = 256
HALO = 8
BATCH = 16
DEAD_WINDOW = 1_000_000
CORE_SLICE = slice(HALO, HALO + CORE)


class FrameCache:
    """A packed read-only memmap with crops that never cross clip boundaries."""

    def __init__(self, directory, split, rows):
        self.rows = rows
        self.samples = np.array([row["samples"] for row in rows], dtype=np.int64)
        self.frames = (self.samples + HOP - 1) // HOP
        self.offsets = np.r_[0, self.frames.cumsum()]
        counts = (self.frames + CORE - 1) // CORE
        self.clip_indices = np.repeat(np.arange(len(rows)), counts)
        self.starts = CORE * (
            np.arange(counts.sum()) - np.repeat(np.r_[0, counts.cumsum()[:-1]], counts)
        )
        self.values = np.load(directory / f"{split}.npy", mmap_mode="r")
        if (
            self.values.shape != (self.offsets[-1], CHANNELS)
            or self.values.dtype != np.float32
        ):
            raise ValueError(f"Invalid {split} frame cache.")

    def batch(self, indices):
        clips = self.clip_indices[indices]
        frames = self.starts[indices, None] + np.arange(-HALO, CORE + HALO)
        weights = np.clip(self.samples[clips, None] - frames * HOP, 0, HOP)
        weights[frames < 0] = 0
        positions = np.clip(frames, 0, self.frames[clips, None] - 1)
        hidden = self.values[self.offsets[clips, None] + positions]
        return torch.from_numpy(hidden), torch.from_numpy(
            weights.astype(np.float32) / HOP
        )


class ControlSAE(nn.Module):
    def __init__(self, features=FEATURES):
        super().__init__()
        if features < 8:
            raise ValueError("Require at least eight SAE features.")
        self.features, self.top_k = features, 8
        self.encoder = nn.Conv1d(CHANNELS, features, 9, padding=4)
        self.dictionary = nn.Parameter(torch.randn(features, CHANNELS, 9) * 0.02)
        self.bias = nn.Parameter(torch.zeros(CHANNELS))

    def preactivations(self, hidden, mask):
        return self.encoder(
            hidden.masked_fill(~mask[..., None], 0).transpose(1, 2)
        ).relu()

    def atoms(self):
        norm = self.dictionary.flatten(1).norm(dim=1).clamp_min(1e-6)
        return self.dictionary / norm[:, None, None]

    def render_codes(self, codes, *, atoms=None, auxiliary=False):
        return F.conv_transpose1d(
            codes.transpose(1, 2),
            self.atoms() if atoms is None else atoms,
            bias=None if auxiliary else self.bias,
            padding=4,
        ).transpose(1, 2)

    @staticmethod
    def select(pre, mask, k):
        values, indices = pre.topk(k, dim=1)
        values = values.clamp_min(0) * mask[:, None]
        codes = torch.zeros_like(pre).scatter(1, indices, values)
        return codes.transpose(1, 2), indices, values

    def reconstruct(self, hidden, mask):
        codes, _, _ = self.select(self.preactivations(hidden, mask), mask, self.top_k)
        return self.render_codes(codes), codes


def weighted_error(error, weights):
    return (error.square() * weights[..., None]).sum()


def training_loss(
    sae, hidden, weights, last_seen, frame_count, frame_total, *, return_core=False
):
    mask = weights > 0
    pre = sae.preactivations(hidden, mask)
    codes, indices, values = sae.select(pre, mask, sae.top_k)
    atoms = sae.atoms()
    core_weights = weights[:, CORE_SLICE]
    core = sae.render_codes(codes, atoms=atoms)[:, CORE_SLICE]
    residual = hidden[:, CORE_SLICE] - core
    error = weighted_error(residual, core_weights)

    # Track the exact last activation on the valid core, never on repeated halo.
    clock = (core_weights > 0).flatten().cumsum(0).view_as(core_weights) + frame_total
    seen = torch.where(values[:, :, CORE_SLICE] > 0, clock[:, None], -1)
    last_seen.scatter_reduce_(
        0, indices[:, :, CORE_SLICE].reshape(-1), seen.reshape(-1), reduce="amax"
    )
    end = frame_total + frame_count
    if end >= DEAD_WINDOW:
        dead = last_seen <= end - DEAD_WINDOW
        if dead.any():
            # Masked slots can fill TopK when fewer than 128 features are dead.
            aux, _, _ = sae.select(
                pre.masked_fill(~dead[None, :, None], -torch.inf),
                mask,
                min(128, sae.features),
            )
            estimate = sae.render_codes(aux, atoms=atoms, auxiliary=True)[:, CORE_SLICE]
            error = (
                error + weighted_error(residual.detach() - estimate, core_weights) / 32
            )
    loss = error / (core_weights.sum() * CHANNELS)
    return (loss, core) if return_core else loss
