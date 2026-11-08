"""Frozen-model data access and sampling for the final joint SAE protocol."""

from collections import OrderedDict

import torch

from stamp.control import FrameCache
from stamp.control.cache import dataset_for, normalization, open_cache, open_joint
from stamp.control.editing import EXCERPT, cached_reference


class RunData:
    def __init__(self, args):
        self.args = args
        self.manifest = open_cache(args.cache)
        self.model, self.seed, self.identity = open_joint(
            args, self.manifest["identity"]
        )
        self.mean, self.std = normalization(self.manifest, args.device)
        self.datasets, self.caches = {}, {}
        for split in ("train", "val"):
            dataset = dataset_for(args, self.seed, split)
            rows = self.manifest["splits"][split]
            if dataset.file_list != [r["id"] for r in rows] or dataset.lengths != [
                r["samples"] for r in rows
            ]:
                raise ValueError(f"Cache membership mismatch: {split}")
            self.datasets[split] = dataset
            self.caches[split] = FrameCache(args.cache, split, rows)
        self.references = OrderedDict()

    def hidden(self, split, index):
        cache = self.caches[split]
        values = cache.values[cache.offsets[index] : cache.offsets[index + 1]].copy()
        return torch.from_numpy(values)[None].to(self.args.device)

    def reference(self, split, index):
        key = (split, int(index))
        if key not in self.references:
            item = self.datasets[split][int(index)]
            self.references[key] = cached_reference(
                self.model, item, self.hidden(split, index), self.seed
            )
            if len(self.references) > 4:
                self.references.popitem(last=False)
        self.references.move_to_end(key)
        return self.references[key]

    def codes(self, sae, ref):
        q = (ref.h_ref - self.mean) / self.std
        return sae.select(sae.preactivations(q, ref.rho > 0), ref.rho > 0, sae.top_k)[0]


def crop_bounds(ref, frame):
    center = frame * 128 + 64
    left = max(0, min(center - EXCERPT // 2, ref.length - EXCERPT))
    return left, min(ref.length, left + EXCERPT)


def choose_group(codes, cohort, counts, rng):
    for _ in range(16):
        frame = int(rng.integers(codes.shape[1]))
        eligible = (
            torch.nonzero(codes[0, frame, cohort].detach() > 0).flatten().cpu().numpy()
        )
        if len(eligible) >= 4:
            weights = 1 / (1 + counts[eligible])
            picked = rng.choice(eligible, 4, replace=False, p=weights / weights.sum())
            return frame, torch.tensor(picked, device=codes.device)
    return None
