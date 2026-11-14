"""Train and audition a JumpReLU SAE on a frozen STAMP checkpoint."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm import tqdm

from stamp.sae.corpus import SEED, SPLITS, Corpus, write_split
from stamp.sae.experiment import (
    FRACTIONS,
    SWEEP_GAINS,
    ListeningBundle,
    candidates,
    paired_edits,
    sample_protocol,
    sweep,
)
from stamp.sae.model import fit, measure, read_sae, save_sae
from stamp.sae.runtime import Runtime
from stamp.synths.noise_band import HOP
from stamp.utils.device import resolve_device
from stamp.utils.io import sha256, write_json

PENALTIES = (0.001, 0.003, 0.01)


def _add_assets(parser):
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--noise-bank", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--meta-file", default="metadata.json")


def _add_selection(parser):
    parser.add_argument("--features", type=int, nargs="+", required=True)
    parser.add_argument(
        "--pca-axes",
        type=int,
        nargs="+",
        help="Zero-based PCA axes selected on validation. Defaults to leading axes.",
    )
    parser.add_argument("--sweep-gains", type=float, nargs="+", default=SWEEP_GAINS)
    parser.add_argument("--triples", type=int, default=12)
    parser.add_argument("--sweep-clips", type=int, default=24)


def _parser():
    parser = argparse.ArgumentParser(prog="stamp sae", description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    cache = sub.add_parser(
        "cache", help="Encode the existing data splits with frozen recon."
    )
    _add_assets(cache)
    cache.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS)
    train = sub.add_parser(
        "train", help="Fit one JumpReLU SAE per sparsity coefficient."
    )
    train.add_argument("--penalties", type=float, nargs="+", default=PENALTIES)
    train.add_argument("--width", type=int, default=512)
    train.add_argument("--steps", type=int, default=20000)
    train.add_argument("--batch-size", type=int, default=4096)
    train.add_argument("--lr", type=float, default=7e-5)
    train.add_argument("--lr-warmup", type=int, default=1000)
    train.add_argument("--penalty-warmup", type=int, default=2000)
    sparsity = sub.add_parser("sparsity", help="Measure L0, FVU and feature coverage.")
    sparsity.add_argument("--sae", type=Path, nargs="+", required=True)
    audition = sub.add_parser(
        "audition", help="Browse validation features and hear their sweeps."
    )
    _add_assets(audition)
    audition.add_argument("--sae", type=Path, required=True)
    audition.add_argument("--candidates", type=int, default=10)
    compare = sub.add_parser(
        "compare", help="Compare transfer and morph settings on validation."
    )
    evaluate = sub.add_parser(
        "evaluate",
        help="Render held-out transfers, morphs and sweeps with fixed choices.",
    )
    for command, split in ((compare, "val"), (evaluate, "test")):
        _add_assets(command)
        command.add_argument("--sae", type=Path, required=True)
        _add_selection(command)
        command.set_defaults(split=split)
    for command in (cache, train, sparsity, audition, compare, evaluate):
        command.add_argument("--cache", type=Path, required=True)
        command.add_argument(
            "--device", default="auto", help="auto, cpu, cuda, or cuda:N"
        )
    for command in (train, sparsity, audition, compare, evaluate):
        command.add_argument("--output", type=Path, required=True)
    for command in (train, audition, compare, evaluate):
        command.add_argument("--seed", type=int, default=SEED)
    sparsity.add_argument("--split", choices=("val", "test"), default="val")
    return parser


def _runtime(args, device):
    return Runtime(
        args.checkpoint,
        args.noise_bank,
        args.data_dir,
        args.audio_dir,
        args.meta_file,
        device,
    )


def _cache(args, device):
    runtime = _runtime(args, device)
    for split in dict.fromkeys(args.splits):
        dataset = runtime.dataset(split)
        records = tqdm(runtime.cache_records(dataset), total=len(dataset), desc=split)
        write_split(args.cache, split, records, runtime.source)


def _train(args, device):
    train, val = Corpus(args.cache, "train"), Corpus(args.cache, "val")
    if (
        any(not math.isfinite(p) or p < 0 for p in args.penalties)
        or len(set(args.penalties)) != len(args.penalties)
        or not math.isfinite(args.lr)
        or min(args.lr_warmup, args.penalty_warmup) < 0
    ):
        raise ValueError(
            "Use distinct finite nonnegative penalties and valid optimizer settings."
        )
    args.output.mkdir(parents=True, exist_ok=False)
    paths = []
    for penalty in args.penalties:
        print(f"Training JumpReLU SAE, lambda={penalty:g}", flush=True)
        model, log = fit(
            train,
            val,
            width=args.width,
            steps=args.steps,
            batch_size=args.batch_size,
            seed=args.seed,
            penalty=penalty,
            device=device,
            lr=args.lr,
            lr_warmup=args.lr_warmup,
            penalty_warmup=args.penalty_warmup,
        )
        folder = args.output / f"lambda-{penalty}"
        folder.mkdir()
        meta = {
            "source": train.source,
            "train_cache": train.identity,
            "val_cache": val.identity,
            "seed": args.seed,
            "penalty": penalty,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lr_warmup": args.lr_warmup,
            "penalty_warmup": args.penalty_warmup,
        }
        save_sae(folder / "sae.pt", model, meta)
        write_json(folder / "training.json", log)
        paths.append(str(folder / "sae.pt"))
        stats = log[-1]["validation"]
        print(
            f"  validation mean L0={stats['mean_l0']:.2f}, FVU={stats['fvu']}",
            flush=True,
        )
    write_json(args.output / "checkpoints.json", paths)


def _read_model(args, path, split, device):
    model, metadata = read_sae(path, device)
    corpus = Corpus(args.cache, split)
    if metadata["source"] != corpus.source:
        raise ValueError(f"SAE and {split} cache use different reconstruction assets.")
    if split in ("train", "val") and metadata[f"{split}_cache"] != corpus.identity:
        raise ValueError(f"SAE was trained with a different {split} cache.")
    return model, metadata, corpus


def _sparsity(args, device):
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    reports = []
    for path in args.sae:
        model, meta, corpus = _read_model(args, path, args.split, device)
        reports.append(
            {
                "sae": str(path),
                "sae_sha256": sha256(path),
                "penalty": meta["penalty"],
                **measure(model, corpus),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(
            {"split": args.split, "cache": corpus.identity, "runs": reports},
            stream,
            indent=2,
        )
        stream.write("\n")


def _listening_inputs(args, device, split):
    model, meta, corpus = _read_model(args, args.sae, split, device)
    runtime = _runtime(args, device)
    if runtime.source != corpus.source:
        raise ValueError(
            "Recon checkpoint, noise bank, or dataset differs from the SAE cache."
        )
    dataset = runtime.dataset(split)
    if dataset.file_list != corpus.ids:
        raise ValueError(f"Dataset clip order or split has changed: {split}")
    index = {sample_id: i for i, sample_id in enumerate(dataset.file_list)}

    def shot(sample_id):
        return runtime.shot(dataset[index[sample_id]], model, corpus)

    protocol = {
        "stage": args.action,
        "split": split,
        "seed": args.seed,
        "source": runtime.source,
        "cache": corpus.identity,
        "sae_sha256": sha256(args.sae),
        "sae_training": meta,
    }
    return model, corpus, runtime, shot, protocol


def _audition(args, device):
    if args.candidates < 1:
        raise ValueError("Use a positive candidate count.")
    model, corpus, runtime, shot, protocol = _listening_inputs(args, device, "val")
    choices = candidates(model, corpus, args.seed, args.candidates)
    protocol["sweep_gains"] = SWEEP_GAINS
    protocol["candidates"] = []
    with ListeningBundle(args.output) as bundle:
        for feature, count, sample_ids, active in choices:
            protocol["candidates"].append(
                {
                    "id": feature,
                    "active_clips": count,
                    "examples": sample_ids,
                    "curated": active,
                }
            )
            for position, sample_id in enumerate(sample_ids):
                role = "high activation" if position < active else "random"
                example = shot(sample_id)
                clips = sweep(runtime, model, example, feature, SWEEP_GAINS)
                if position == 0:
                    projected = model.approximate(example.h)
                    clips.append(
                        ("SAE reconstruction", runtime.render(example, projected))
                    )
                bundle.add(
                    f"Feature {feature} · {role} · {sample_id}",
                    clips,
                    {"active": bool(example.code[:, feature].any())},
                    onset=example.onset,
                )
        bundle.finish(protocol)


def _compare(args, device):
    model, corpus, runtime, shot, protocol = _listening_inputs(args, device, args.split)
    features = args.features
    pca_ids = args.pca_axes if args.pca_axes is not None else list(range(len(features)))
    if (
        len(features) not in (1, 2)
        or len(set(features)) != len(features)
        or min(features) < 0
        or max(features) >= model.width
    ):
        raise ValueError("Provide one or two distinct valid zero-based feature IDs.")
    if (
        len(set(pca_ids)) != len(features)
        or len(pca_ids) != len(features)
        or min(pca_ids) < 0
        or max(pca_ids) >= model.mean.numel()
    ):
        raise ValueError(
            "Provide the same number of distinct valid zero-based PCA axes."
        )
    if (
        len(set(args.sweep_gains)) != len(args.sweep_gains)
        or 1 not in args.sweep_gains
        or any(not math.isfinite(v) or v < 0 for v in args.sweep_gains)
    ):
        raise ValueError(
            "Sweep gains must be distinct, finite, nonnegative and include 1."
        )
    if not 1 <= args.triples <= len(corpus) // 3 or not 1 <= args.sweep_clips <= len(
        corpus
    ):
        raise ValueError(
            "Requested examples exceed the split or contain an empty draw."
        )
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    train = Corpus(args.cache, "train")
    if train.identity != protocol["sae_training"]["train_cache"]:
        raise ValueError("PCA requires the training split used for this SAE.")
    axes = train.pca(model.mean, model.scale, device)[:, pca_ids]
    triples, sweep_ids = sample_protocol(
        corpus, args.seed, args.triples, args.sweep_clips
    )
    protocol.update(
        features=features,
        pca_axes=pca_ids,
        fractions=FRACTIONS,
        sweep_gains=args.sweep_gains,
        triples=triples,
        sweeps=sweep_ids,
        feature_coverage=measure(model, corpus),
    )
    with ListeningBundle(args.output) as bundle:
        for source_id, b_id, c_id in tqdm(
            triples, desc=f"{args.split} transfer and morph"
        ):
            source = shot(source_id)
            group = []
            donors = []
            details = {}
            for slot, donor_id in (("B", b_id), ("C", c_id)):
                donor = shot(donor_id)
                donors.append((slot, donor))
                transfers, morphs, diagnostics = paired_edits(
                    runtime, model, source, donor, features, axes
                )
                for stats, sample in (("source", source), ("donor", donor)):
                    diagnostics[f"{stats}_active"] = (
                        sample.code[:, features].any(dim=0).tolist()
                    )
                details[slot] = {"donor": donor_id, **diagnostics}
                group.extend((f"{slot} · {name}", audio) for name, audio in transfers)
                bundle.add(
                    f"Morph · {source_id} ↔ {donor_id}",
                    morphs,
                    diagnostics,
                    onset=max(source.onset, donor.onset),
                )
            common_onset = max(source.onset, *(item.onset for _, item in donors))
            aligned_donors = [
                (
                    f"{slot} · Donor neutral",
                    F.pad(item.neutral, ((common_onset - item.onset) * HOP, 0)),
                )
                for slot, item in donors
            ]
            left = (common_onset - source.onset) * HOP
            bundle.add(
                f"Transfer · {source_id} ← B: {b_id} / C: {c_id}",
                aligned_donors
                + [(name, F.pad(audio, (left, 0))) for name, audio in group],
                details,
                onset=common_onset,
            )
        for sample_id in tqdm(sweep_ids, desc=f"{args.split} feature sweeps"):
            example = shot(sample_id)
            for feature in features:
                bundle.add(
                    f"Sweep · feature {feature} · {sample_id}",
                    sweep(runtime, model, example, feature, args.sweep_gains),
                    {"active": bool(example.code[:, feature].any())},
                    onset=example.onset,
                )
        bundle.finish(protocol)


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    handlers = {
        "cache": _cache,
        "train": _train,
        "sparsity": _sparsity,
        "audition": _audition,
        "compare": _compare,
        "evaluate": _compare,
    }
    try:
        device = resolve_device(args.device)
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        handlers[args.action](args, device)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0
