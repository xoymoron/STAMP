"""Cache frozen joint latents and train the temporal SAE warm-up."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from stamp.control import (
    BATCH,
    CHANNELS,
    CORE_SLICE,
    FEATURES,
    ControlSAE,
    FrameCache,
    training_loss,
    weighted_error,
)
from stamp.data import JointResynthesisDataset
from stamp.models.joint_resynthesis import HOP
from stamp.synths.noise_band import SAMPLE_RATE, load_noise_bank
from stamp.tasks.joint_resynthesis import JointResynthesisTask, clip_offsets
from stamp.utils.device import resolve_device
from stamp.utils.io import read_json, sha256, write_json


def open_joint(args, expected=None):
    bank, checksum = load_noise_bank(args.noise_bank)
    identity = {
        "checkpoint": sha256(args.checkpoint),
        "bank": checksum,
        "manifest": sha256(args.data_dir / args.meta_file),
        "modal_config": sha256(args.data_dir / "modal_config.json"),
    }
    if expected is not None and identity != expected:
        raise ValueError("The checkpoint, bank, or dataset differs from the h cache.")
    task = JointResynthesisTask.load_from_checkpoint(
        str(args.checkpoint), map_location="cpu", bank=bank, bank_sha256=checksum
    )
    return (
        task.model.eval().requires_grad_(False).to(args.device),
        task.hparams.seed,
        identity,
    )


def dataset_for(args, seed, split):
    dataset = JointResynthesisDataset(
        args.data_dir,
        args.meta_file,
        SAMPLE_RATE,
        None,
        split=split,
        audio_dir=args.audio_dir,
        seed=seed,
    )
    if not len(dataset):
        raise ValueError(f"Empty {split} split.")
    return dataset


def reference(model, item, seed, device):
    return model.encode_reference(
        item["audio"][None].to(device, non_blocking=True),
        item["params"][None].to(device, non_blocking=True),
        item["analysis_num_samples"],
        clip_offsets(item["sample_id"], seed).to(device),
        checkpoint_activations=False,
    )


@torch.inference_mode()
def cache_latents(args):
    model, seed, identity = open_joint(args)
    datasets = {s: dataset_for(args, seed, s) for s in ("train", "val")}
    args.cache.mkdir(parents=True, exist_ok=False)
    manifest = {"version": 1, "identity": identity, "seed": seed, "splits": {}}
    count = 0.0
    total = np.zeros(CHANNELS, dtype=np.float64)
    squares = total.copy()
    for split, data in datasets.items():
        rows = [
            {
                "id": key,
                "samples": size,
                "pack": data.sample_pack_key(data.metadata[key]),
            }
            for key, size in zip(data.file_list, data.lengths)
        ]
        nframes = sum((n + HOP - 1) // HOP for n in data.lengths)
        cache = np.lib.format.open_memmap(
            args.cache / f"{split}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(nframes, CHANNELS),
        )
        loader = DataLoader(
            data,
            batch_size=None,
            num_workers=args.workers,
            pin_memory=args.device.type == "cuda",
        )
        cursor = 0
        for row, item in tqdm(
            zip(rows, loader), total=len(rows), desc=f"Cache {split}"
        ):
            if item["sample_id"] != row["id"]:
                raise ValueError("DataLoader order differs from the cache manifest.")
            h = reference(model, item, seed, args.device).h_ref[0].cpu().numpy()
            frames = (row["samples"] + HOP - 1) // HOP
            if h.shape != (frames, CHANNELS) or not np.isfinite(h).all():
                raise ValueError(f"Invalid latent for {row['id']}.")
            cache[cursor : cursor + frames] = h
            cursor += frames
            if split == "train":
                rho = np.ones(frames, dtype=np.float64)
                rho[-1] = (row["samples"] - (frames - 1) * HOP) / HOP
                count += rho.sum()
                total += np.einsum("tc,t->c", h, rho, dtype=np.float64)
                squares += np.einsum("tc,tc,t->c", h, h, rho, dtype=np.float64)
        cache.flush()
        del cache
        manifest["splits"][split] = rows
        print(f"{split}: {nframes:,} frames, {nframes * CHANNELS * 4 / 2**30:.2f} GiB")
    mean = total / count
    manifest["mean"] = mean.tolist()
    manifest["std"] = np.sqrt(np.maximum(squares / count - mean**2, 1e-12)).tolist()
    write_json(args.cache / "cache.json", manifest)


def open_cache(path):
    manifest = read_json(path / "cache.json")
    if manifest.get("version") != 1:
        raise ValueError("Unsupported h cache format.")
    return manifest


def batches(cache, order, device):
    groups = (order[start : start + BATCH] for start in range(0, len(order), BATCH))
    if device.type != "cuda":
        for ids in groups:
            yield ids, cache.batch(ids)
        return

    def pinned(ids):
        return tuple(tensor.pin_memory() for tensor in cache.batch(ids))

    with ThreadPoolExecutor(max_workers=1) as pool:
        ids = next(groups, None)
        if ids is None:
            return
        pending = pool.submit(pinned, ids)
        for following in groups:
            batch = pending.result()
            pending = pool.submit(pinned, following)
            yield ids, batch
            ids = following
        yield ids, pending.result()


def normalization(manifest, device):
    return tuple(
        torch.tensor(manifest[k], dtype=torch.float32, device=device)
        for k in ("mean", "std")
    )


def prepare_batch(batch, mean, std):
    hidden, weights = batch
    hidden = hidden.to(mean.device, non_blocking=True)
    weights = weights.to(mean.device, non_blocking=True)
    return (hidden - mean) / std, weights


@torch.inference_mode()
def validate(sae, cache, mean, std, deadline):
    summary = torch.zeros(2, device=mean.device, dtype=torch.float64)
    live = torch.zeros(sae.features, device=mean.device, dtype=torch.bool)
    for _, batch in batches(cache, np.arange(len(cache.starts)), mean.device):
        if time.monotonic() >= deadline:
            return None
        hidden, weights = prepare_batch(batch, mean, std)
        reconstruction, codes = sae.reconstruct(hidden, weights > 0)
        core = weights[:, CORE_SLICE]
        error = reconstruction[:, CORE_SLICE] - hidden[:, CORE_SLICE]
        summary += torch.stack((weighted_error(error, core), core.sum() * CHANNELS))
        live |= (codes[:, CORE_SLICE] > 0).any(dim=(0, 1))
    return float(summary[0] / summary[1]), int((~live).sum())


def save_best(path, state):
    temporary = path.with_suffix(".partial")
    torch.save(state, temporary)
    temporary.replace(path)


def train_sae(args):
    from stamp.control.protocol import WARMUP_KIND

    manifest = open_cache(args.cache)
    train, val = (
        FrameCache(args.cache, s, manifest["splits"][s]) for s in ("train", "val")
    )
    mean, std = normalization(manifest, args.device)
    torch.manual_seed(manifest["seed"])
    rng = np.random.default_rng(manifest["seed"])
    sae = ControlSAE().to(args.device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=3e-4, foreach=True)
    last_seen = torch.full((FEATURES,), -1, dtype=torch.int64, device=args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    ckpt = args.output / "best.pt"
    identity = sha256(args.cache / "cache.json")
    frame_total = step = 0
    best = float("inf")
    start = time.monotonic()
    target = start + args.minutes * 60
    deadline = start + args.max_minutes * 60
    epoch = 0
    history = []
    progress = tqdm(
        total=args.minutes * 60,
        desc="SAE warm-up",
        unit="s",
        dynamic_ncols=True,
        mininterval=1,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}{postfix}]",
    )
    while time.monotonic() < target or step == 0:
        epoch += 1
        for _, batch in batches(train, rng.permutation(len(train.starts)), args.device):
            if step and time.monotonic() >= target:
                break
            nframes = int(batch[1][:, CORE_SLICE].count_nonzero())
            hidden, weights = prepare_batch(batch, mean, std)
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(sae, hidden, weights, last_seen, nframes, frame_total)
            loss.backward()
            optimizer.step()
            step += 1
            frame_total += nframes
            progress.update(min(time.monotonic() - start, progress.total) - progress.n)
            if step == 20 or step % 100 == 0:
                current_loss = float(loss.detach())
                elapsed = time.monotonic() - start
                progress.set_postfix(
                    step=step, loss=f"{current_loss:.5f}", refresh=False
                )
                tqdm.write(
                    f"{step:,} steps | {frame_total / elapsed:,.0f} valid frames/s (run average) | MSE+AuxK {current_loss:.5f}",
                )

        progress.set_description("Warm-up validation")
        measured = validate(sae, val, mean, std, deadline)
        progress.update(min(time.monotonic() - start, progress.total) - progress.n)
        progress.set_description("SAE warm-up")
        if measured is None:
            break
        score, dead = measured
        entry = {
            "epoch": epoch,
            "step": step,
            "frames": frame_total,
            "elapsed_s": time.monotonic() - start,
            "val_mse": score,
            "val_dead": dead,
        }
        history.append(entry)
        write_json(args.output / "training.json", history)
        tqdm.write(json.dumps(entry))
        if score < best:
            best = score
            save_best(
                ckpt,
                {
                    "state_dict": sae.state_dict(),
                    "cache_sha256": identity,
                    "identity": manifest["identity"],
                    "seed": manifest["seed"],
                    "kind": WARMUP_KIND,
                    **entry,
                },
            )
    progress.close()
    if not ckpt.is_file():
        raise RuntimeError("The time cap expired before a complete validation pass.")
    # Budget the entire invocation, not the time of the best validation epoch.
    artifact = torch.load(ckpt, map_location="cpu", weights_only=True)
    artifact.update(
        warmup_seconds=time.monotonic() - start,
        warmup_total_steps=step,
        loss="L_h + L_AuxK / 32",
        initialization="random",
        optimizer="Adam",
        learning_rate=3e-4,
    )
    save_best(ckpt, artifact)
    write_json(
        args.output / "timing.json",
        {
            "warmup_seconds": artifact["warmup_seconds"],
            "target_minutes": args.minutes,
            "max_minutes": args.max_minutes,
            "selected_step": artifact["step"],
            "total_steps": step,
        },
    )
    print(f"Best checkpoint: {ckpt}")


@torch.inference_mode()
def clip_activity(sae, cache, mean, std):
    activity = np.zeros((len(cache.rows), FEATURES), dtype=np.float64)
    for ids, batch in batches(cache, np.arange(len(cache.starts)), mean.device):
        h, w = prepare_batch(batch, mean, std)
        mask = w > 0
        codes, _, _ = sae.select(sae.preactivations(h, mask), mask, sae.top_k)
        sums = (codes[:, CORE_SLICE].square() * w[:, CORE_SLICE, None]).sum(dim=1)
        np.add.at(activity, cache.clip_indices[ids], sums.cpu().numpy())
    return np.sqrt(activity / (cache.samples[:, None] / HOP))


def shortlist(activity, rows, limit):
    packs = np.asarray([row["pack"] for row in rows])
    pack_total = len(set(packs))
    ranked = []
    for feature, strength in enumerate(activity.T):
        active = np.flatnonzero(strength > 0)
        n_packs = len(set(packs[active]))
        if len(active) < 3 or n_packs < min(2, pack_total):
            continue
        order = active[np.argsort(-strength[active], kind="stable")]
        distinct, seen = [], set()
        for i in order:
            if packs[i] not in seen:
                distinct.append(int(i))
                seen.add(packs[i])
        examples = (distinct + [int(i) for i in order if int(i) not in distinct])[:3]
        ranked.append(
            (
                len(active),
                n_packs,
                float(np.median(strength[active])),
                feature,
                examples,
            )
        )
    ranked.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3]))
    return ranked[:limit]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    cache = stages.add_parser("cache", help="Encode the train and validation splits.")
    train = stages.add_parser(
        "train", help="Train one SAE and keep the best validation checkpoint."
    )
    for sub in (cache, train):
        sub.add_argument("--cache", type=Path, required=True)
        sub.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    for sub in (cache,):
        sub.add_argument("--checkpoint", type=Path, required=True)
        sub.add_argument("--noise-bank", type=Path, required=True)
        sub.add_argument("--data-dir", type=Path, required=True)
        sub.add_argument("--audio-dir", type=Path)
        sub.add_argument("--meta-file", default="metadata.json")
    cache.add_argument("--workers", type=int, default=2)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--minutes", type=float, default=25)
    train.add_argument("--max-minutes", type=float, default=30)
    args = parser.parse_args(argv)
    try:
        args.device = resolve_device(args.device)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    if args.stage == "train" and not 0 < args.minutes < args.max_minutes <= 180:
        parser.error("Require 0 < --minutes < --max-minutes <= 180.")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    {"cache": cache_latents, "train": train_sae}[args.stage](args)


if __name__ == "__main__":
    main()
