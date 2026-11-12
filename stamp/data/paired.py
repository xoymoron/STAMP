"""Shared data order for STAMP and the variable-length DrumBlender baseline."""

from pathlib import PurePosixPath, PureWindowsPath

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, DistributedSampler, Sampler, Subset


def _index_paths(metadata):
    result = {}
    for key, row in metadata.items():
        path = PurePosixPath(row["orig_relpath"].replace("\\", "/"))
        if (
            path == PurePosixPath(".")
            or path.is_absolute()
            or PureWindowsPath(path).drive
            or ".." in path.parts
        ):
            raise ValueError(
                f"Expected a processed-root-relative waveform path: {path}"
            )
        name = path.as_posix()
        if name in result:
            raise ValueError(f"Duplicate waveform path: {name}")
        result[name] = key
    return result


def paired_rows(stamp_metadata, og_metadata, audio_dir):
    stamp_paths = _index_paths(stamp_metadata)
    og_paths = _index_paths(og_metadata)
    if stamp_paths.keys() != og_paths.keys():
        missing_in_stamp = sorted(og_paths.keys() - stamp_paths.keys())
        missing_in_og = sorted(stamp_paths.keys() - og_paths.keys())
        raise ValueError(
            "Caches disagree on waveform paths. "
            f"Missing in STAMP: {missing_in_stamp[:3]}. Missing in OG: {missing_in_og[:3]}."
        )
    splits = {split: [] for split in ("train", "val", "test")}
    for path in sorted(stamp_paths):
        stamp_key, og_key = stamp_paths[path], og_paths[path]
        stamp_row, og_row = stamp_metadata[stamp_key], og_metadata[og_key]
        if stamp_row["num_samples"] != og_row["num_samples"]:
            raise ValueError(f"Caches disagree on num_samples: {path}")
        if og_row.get("split") not in splits:
            raise ValueError(f"Missing or invalid frozen OG split: {path}")
        if not (audio_dir / path).is_file():
            raise FileNotFoundError(audio_dir / path)
        splits[og_row["split"]].append((path, stamp_key, og_key))
    return splits


class EpochSampler(Sampler):
    """Shuffle globally, keep complete optimizer batches, then shard by rank."""

    def __init__(self, size, *, batch_size, accumulate, seed, replicas, rank):
        self.size = size
        self.seed = seed
        self.rank = rank
        self.replicas = replicas
        self.epoch = 0
        group = batch_size * accumulate * replicas
        self.count = (size // group) * group
        if not self.count:
            raise ValueError(
                f"Need at least {group} training clips for one optimizer step."
            )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator)
        return iter(indices[self.rank : self.count : self.replicas].tolist())

    def __len__(self):
        return self.count // self.replicas


class PairedDataModule(pl.LightningDataModule):
    def __init__(self, dataset, keys, rows, key_column, collate_fn, data, settings):
        super().__init__()
        positions = {key: i for i, key in enumerate(keys)}
        self.datasets = {}
        for split in ("train", "val"):
            indices = []
            for row in rows[split]:
                key = row[key_column]
                dataset.metadata[key]["orig_relpath"] = row[0]
                indices.append(positions[key])
            if not indices:
                raise ValueError(f"Empty {split} split.")
            self.datasets[split] = Subset(dataset, indices)
        self.collate_fn = collate_fn
        self.batch_size = data["batch_size"]
        self.num_workers = data["num_workers"]
        self.seed = settings["seed"]
        self.accumulate = settings["accumulate_grad_batches"]
        self.pin_memory = settings["accelerator"] != "cpu" and torch.cuda.is_available()

    def _loader(self, split):
        dataset = self.datasets[split]
        replicas, rank = self.trainer.world_size, self.trainer.global_rank
        if split == "train":
            sampler = EpochSampler(
                len(dataset),
                batch_size=self.batch_size,
                accumulate=self.accumulate,
                seed=self.seed,
                replicas=replicas,
                rank=rank,
            )
        else:
            sampler = DistributedSampler(
                dataset, num_replicas=replicas, rank=rank, shuffle=False
            )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self):
        return self._loader("train")

    def val_dataloader(self):
        return self._loader("val")
