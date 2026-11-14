"""An indexed, memory-mapped cache of variable-length reconstruction latents."""

import tempfile
from pathlib import Path

import numpy as np
import torch

from stamp.synths.noise_band import HOP
from stamp.utils.io import read_json, sha256, write_json

CHANNELS = 256
SPLITS = ("train", "val", "test")
SEED = 20261106


class Corpus:
    def __init__(self, root, split):
        self.split = split
        path = Path(root) / split
        index_path = path / "index.json"
        index = read_json(index_path)
        if index["version"] != 1 or index["split"] != split:
            raise ValueError(f"Invalid SAE cache index: {index_path}")
        self.identity = sha256(index_path)
        self.source = index["source"]
        self.rows = index["clips"]
        self.ids = [row["id"] for row in self.rows]
        self.by_id = {row["id"]: row for row in self.rows}
        if not self.rows or len(self.by_id) != len(self.rows):
            raise ValueError(f"Empty cache or duplicate clip IDs: {index_path}")
        self.lengths = torch.tensor([row["samples"] for row in self.rows])
        self.starts = torch.tensor([row["start"] for row in self.rows])
        frames = (self.lengths + HOP - 1) // HOP
        if (self.lengths <= 0).any() or not self.starts.equal(
            frames.cumsum(0) - frames
        ):
            raise ValueError(f"Invalid cached frame extents: {index_path}")
        self.frames = int(frames.sum())
        shape = (self.frames, CHANNELS)
        data_path = path / "latents.f32"
        if data_path.stat().st_size != self.frames * CHANNELS * 4:
            raise ValueError(f"Incomplete latent cache: {data_path}")
        self.values = np.memmap(data_path, dtype="<f4", mode="r", shape=shape)

    def __len__(self):
        return len(self.rows)

    def get(self, sample_id):
        row = self.by_id[sample_id]
        first = row["start"]
        count = (row["samples"] + HOP - 1) // HOP
        return torch.from_numpy(self.values[first : first + count].copy())

    def sample(self, count, generator):
        clips = torch.randint(len(self), (count,), generator=generator)
        # A uniform sample index gives each frame its valid-sample weight.
        positions = (
            torch.rand(count, dtype=torch.float64, generator=generator)
            * self.lengths[clips]
        ).long()
        indices = self.starts[clips] + positions // HOP
        return torch.from_numpy(self.values[indices.numpy()])

    def weighted_clips(self):
        for row in self.rows:
            x = self.get(row["id"])
            valid = torch.full((len(x),), HOP, dtype=torch.float64)
            valid[-1] = row["samples"] - (len(x) - 1) * HOP
            yield x, valid / row["samples"]

    def normalization(self):
        mean = torch.zeros(CHANNELS, dtype=torch.float64)
        second = torch.zeros((), dtype=torch.float64)
        for x, weights in self.weighted_clips():
            x = x.double()
            mean += weights @ x
            second += weights @ x.square().sum(dim=1)
        mean /= len(self)
        scale = (second / len(self) - mean.square().sum()).clamp_min(0).sqrt()
        if not torch.isfinite(scale) or scale <= 0:
            raise ValueError("Training cache has no finite latent variance.")
        return mean.float(), scale.float()

    def pca(self, mean, scale, device="cpu"):
        covariance = torch.zeros(CHANNELS, CHANNELS, dtype=torch.float64, device=device)
        mean = mean.to(device, torch.float64)
        scale = scale.to(device, torch.float64)
        for h, weights in self.weighted_clips():
            x = (h.to(device, torch.float64) - mean) / scale
            covariance.addmm_(x.T, x * weights.to(device)[:, None])
        _, directions = torch.linalg.eigh(covariance)
        return directions.flip((1,)).float()


def write_split(root, split, records, source=None):
    """Publish a split only after every clip and its index have been written."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / split
    if destination.exists():
        raise FileExistsError(f"Cached split already exists: {destination}")
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        staging = Path(tmp) / split
        staging.mkdir()
        clips = []
        ids = set()
        start = 0
        with (staging / "latents.f32").open("wb") as stream:
            for sample_id, h, samples, onset in records:
                count = (samples + HOP - 1) // HOP
                if (
                    samples <= 0
                    or h.shape != (count, CHANNELS)
                    or not 0 <= onset < count
                    or sample_id in ids
                    or not torch.isfinite(h).all()
                ):
                    raise ValueError(f"Invalid latent cache row for {sample_id!r}.")
                ids.add(sample_id)
                h.detach().cpu().numpy().astype("<f4", copy=False).tofile(stream)
                clips.append(
                    {
                        "id": sample_id,
                        "start": start,
                        "samples": samples,
                        "onset": onset,
                    }
                )
                start += count
        if not clips:
            raise ValueError(f"No clips in {split} split.")
        write_json(
            staging / "index.json",
            {"version": 1, "split": split, "clips": clips, "source": source or {}},
        )
        staging.rename(destination)
