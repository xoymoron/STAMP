from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import monotonic, perf_counter
from typing import Dict, List, Tuple

import torch
import torchaudio
from tqdm import tqdm

from stamp.data.modal_stats import (
    result_details,
    summarize_metadata,
    write_json_atomic,
)
from stamp.utils.modal_analysis import ANALYZER_VERSION, ModalAnalyzer


def stable_id(rel_path: str) -> str:
    # deterministic numeric-ish id from path
    h = hashlib.md5(rel_path.replace("\\", "/").encode("utf-8")).hexdigest()
    return str(int(h[:12], 16))


def list_wavs(root: Path) -> List[Path]:
    wavs = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"]
    wavs.sort()
    return wavs


def make_pack_key(
    processed_root: Path,
    wav_path: Path,
    pack_depth: int = 1,
) -> Tuple[str, str, str]:
    """
    Build a pack key directly from the top-level folder(s) under processed_root.

    Example:
      processed/pack_1/a.wav            -> pack=pack_1
      processed/pack_1/sub/x.wav        -> pack=pack_1
      processed/pack_2/sub/deep/y.wav   -> pack=pack_2

    If pack_depth > 1, the first N directory levels are joined.
    This still ignores deeper subdirectory structure for split grouping.
    """
    rel = wav_path.relative_to(processed_root)
    parts = rel.parts

    # Use the top-level folder under processed_root as the pack id.
    # Subdirectories under the pack are ignored for pack grouping.
    type_name = "custom"
    inst_name = "unlabeled"

    inner_dirs = list(parts[:-1])  # all directories before filename
    depth = max(1, int(pack_depth))
    if len(inner_dirs) == 0:
        pack_name = "__root__"
    else:
        pack_name = "/".join(inner_dirs[:depth])

    pack = pack_name
    return type_name, inst_name, pack


def make_splits_within_pack(
    pack_keys: List[str],
    seed: int,
    train: float = 0.8,
    val: float = 0.1,
) -> List[str]:
    """
    Split files within each pack.
    """
    if train < 0.0 or val < 0.0 or (train + val) > 1.0:
        raise ValueError(
            "Invalid split ratios: require train >= 0, val >= 0, train+val <= 1"
        )

    by_pack: Dict[str, List[int]] = {}
    for idx, pack in enumerate(pack_keys):
        by_pack.setdefault(pack, []).append(idx)

    g = torch.Generator().manual_seed(seed)
    out = ["train"] * len(pack_keys)

    for pack in sorted(by_pack.keys()):
        idxs = by_pack[pack]
        perm = torch.randperm(len(idxs), generator=g).tolist()
        shuffled = [idxs[i] for i in perm]

        n = len(shuffled)
        n_train = int(n * train)
        n_val = int(n * val)

        # Ensure each pack contributes at least one training sample where possible.
        if n > 0 and n_train == 0:
            n_train = 1
        if n_train + n_val > n:
            n_val = max(0, n - n_train)

        for rank, original_idx in enumerate(shuffled):
            if rank < n_train:
                out[original_idx] = "train"
            elif rank < (n_train + n_val):
                out[original_idx] = "val"
            else:
                out[original_idx] = "test"

    return out


def parse_args(argv=None):
    """Resolve backend-specific defaults before touching output directories."""
    ap = argparse.ArgumentParser()

    # inputs
    ap.add_argument("--processed_root", type=str, required=True)

    # outputs
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--meta_name", type=str, default="metadata.json")

    # modal params
    ap.add_argument("--sample_rate", type=int, default=48000)
    ap.add_argument(
        "--modal_backend",
        choices=("hybrid", "stft", "cqt"),
        default="hybrid",
    )
    ap.add_argument(
        "--num_modes",
        type=int,
        default=64,
    )

    ap.add_argument("--hop_length", type=int, default=256)
    ap.add_argument("--fmin", type=int, default=20)
    ap.add_argument("--n_bins", type=int, default=240)
    ap.add_argument("--bins_per_octave", type=int, default=24)
    ap.add_argument(
        "--min_length",
        type=int,
        default=4,
    )
    ap.add_argument(
        "--threshold_db",
        type=float,
        default=-90.0,
    )
    ap.add_argument("--diff_threshold", type=float, default=5.0)
    ap.add_argument("--cqt_max_frequency", type=float, default=700.0)
    ap.add_argument("--short_window", type=int, default=2048)
    ap.add_argument("--long_window", type=int, default=8192)
    ap.add_argument("--max_gap", type=int, default=2)
    ap.add_argument("--max_deviation_hz", type=float, default=40.0)
    ap.add_argument("--relative_threshold_db", type=float, default=-80.0)
    ap.add_argument("--min_relative_score_db", type=float, default=-40.0)
    ap.add_argument("--min_prominence_db", type=float, default=2.0)
    ap.add_argument(
        "--compute_device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Spectral transforms run here; peak tracking stays on CPU.",
    )
    ap.add_argument(
        "--gpu_cqt_batch_size",
        type=int,
        default=16,
        help="CUDA CQT filters per FFT batch; lower to reduce VRAM.",
    )
    ap.add_argument(
        "--no_refine",
        action="store_true",
        help="Disable complex-lobe fitting and phase frequency refinement.",
    )

    # behavior
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_files", type=int, default=0, help="0 = all files")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument(
        "--no_tqdm",
        action="store_true",
        help="Hide the worker bar when a multi-GPU parent displays shard bars.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed files with matching analysis settings.",
    )
    ap.add_argument(
        "--checkpoint_every",
        type=int,
        default=100,
        help="Save metadata after this many new files.",
    )
    ap.add_argument(
        "--pack_depth",
        type=int,
        default=1,
        help=(
            "Number of top-level path segments under processed_root used as pack id. "
            "1 means top-level folder only (e.g., processed/pack_name/...)."
        ),
    )
    ap.add_argument(
        "--write_split",
        action="store_true",
        help=(
            "Write split labels into metadata during preprocessing. "
            "Default is OFF so split is done at training time in the dataset."
        ),
    )

    ap.add_argument(
        "--pad_to",
        type=int,
        default=0,
        help="If >0, right-pad audio shorter than this to this length (samples).",
    )
    ap.add_argument(
        "--min_duration_ms",
        type=float,
        default=0.0,
        help="If >0, skip files shorter than this (ms). Set 0 to not skip.",
    )
    args = ap.parse_args(argv)
    # Preserve the metadata key used by existing hybrid extraction resumes.
    args.short_cqt_boundary_policy = "not_applicable"
    args.pad_short = True
    if args.num_modes < 1:
        ap.error("--num_modes must be positive")
    if args.checkpoint_every < 1:
        ap.error("--checkpoint_every must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        ap.error("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    if args.gpu_cqt_batch_size < 1:
        ap.error("--gpu_cqt_batch_size must be positive")
    if args.write_split and args.num_shards > 1:
        ap.error("--write_split is not supported with sharded extraction")
    if args.num_shards > 1:
        args.out_dir = str(Path(args.out_dir) / f"shard_{args.shard_index}")
    return args


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv)

    processed_root = Path(args.processed_root)
    if not processed_root.exists():
        raise FileNotFoundError(processed_root)

    out_dir = Path(args.out_dir)
    args.analyzer_version = ANALYZER_VERSION
    if not args.resume and (
        (out_dir / args.meta_name).exists()
        or (out_dir / "modal_config.json").exists()
        or any((out_dir / "features").glob("*.pt"))
        or any((out_dir / "audio").glob("*.wav"))
    ):
        raise FileExistsError(
            "Output is not empty; use --resume or a new output directory."
        )
    config_path = out_dir / "modal_config.json"
    meta_path = out_dir / args.meta_name
    if args.resume and config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        for name, value in vars(args).items():
            previous_value = previous.get(name)
            if (
                name not in {"resume", "checkpoint_every", "max_files", "no_tqdm"}
                and previous_value != value
            ):
                raise ValueError(
                    f"Cannot resume with changed {name}: {previous_value!r} != {value!r}"
                )
    elif args.resume and meta_path.exists():
        raise ValueError("Cannot resume metadata without modal_config.json.")
    audio_dir = out_dir / "audio"
    feat_dir = out_dir / "features"
    audio_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    wavs = list_wavs(processed_root)
    total_wavs = len(wavs)
    wavs = wavs[args.shard_index :: args.num_shards]
    if args.max_files and args.max_files > 0:
        wavs = wavs[: args.max_files]

    print(
        f"[scan] {processed_root} -> {total_wavs} wavs; "
        f"shard {args.shard_index}/{args.num_shards} -> {len(wavs)} wavs"
    )
    if len(wavs) == 0:
        raise RuntimeError("No wavs found.")

    typed: List[Tuple[str, str, str]] = []
    packs: List[str] = []
    for p in wavs:
        type_name, inst_name, pack = make_pack_key(
            processed_root,
            p,
            pack_depth=args.pack_depth,
        )
        typed.append((type_name, inst_name, pack))
        packs.append(pack)

    split_by_index = None
    if args.write_split:
        # Keep this optional path disabled by default so the dataset class
        # computes split dynamically from sample_pack_key and seed.
        split_by_index = make_splits_within_pack(packs, seed=args.seed)

    analyzer_options = dict(
        backend=args.modal_backend,
        cqt_max_frequency=args.cqt_max_frequency,
        short_window=args.short_window,
        long_window=args.long_window,
        max_gap=args.max_gap,
        max_deviation_hz=args.max_deviation_hz,
        relative_threshold_db=args.relative_threshold_db,
        min_relative_score_db=args.min_relative_score_db,
        min_prominence_db=args.min_prominence_db,
        refine=not args.no_refine,
        compute_device=args.compute_device,
        gpu_cqt_batch_size=args.gpu_cqt_batch_size,
    )
    modal = ModalAnalyzer(
        args.sample_rate,
        hop_length=args.hop_length,
        fmin=args.fmin,
        n_bins=args.n_bins,
        bins_per_octave=args.bins_per_octave,
        min_length=args.min_length,
        num_modes=args.num_modes,
        threshold=args.threshold_db,
        diff_threshold=args.diff_threshold,
        **analyzer_options,
    )
    config_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    meta: Dict[str, Dict] = (
        json.loads(meta_path.read_text(encoding="utf-8"))
        if args.resume and meta_path.exists()
        else {}
    )
    failed = 0
    kept = 0
    skipped = 0
    attempted = 0
    run_started = monotonic()
    last_progress_write = 0.0
    progress_path = out_dir / "progress.json"
    summary_path = out_dir / "run_stats.json"
    failures_path = out_dir / "failures.jsonl"

    def write_progress(last: dict | None = None, *, force: bool = False) -> None:
        nonlocal last_progress_write
        now = monotonic()
        if not force and now - last_progress_write < 0.5:
            return
        elapsed = max(now - run_started, 1e-6)
        write_json_atomic(
            progress_path,
            {
                "device": args.compute_device,
                "shard_index": args.shard_index,
                "total": len(wavs),
                "attempted": attempted,
                "kept": kept,
                "skipped": skipped,
                "failed": failed,
                "elapsed_seconds": round(elapsed, 3),
                "files_per_second": round(attempted / elapsed, 4),
                "estimated_remaining_seconds": (
                    round((len(wavs) - attempted) * elapsed / attempted, 1)
                    if attempted
                    else None
                ),
                "last": last,
            },
        )
        last_progress_write = now

    def save_metadata() -> None:
        # A replace keeps a usable checkpoint if the process stops mid-write.
        write_json_atomic(meta_path, meta)
        write_json_atomic(
            summary_path,
            {
                "run": {
                    "device": args.compute_device,
                    "shard_index": args.shard_index,
                    "total": len(wavs),
                    "attempted": attempted,
                    "new": kept,
                    "resumed": skipped,
                    "failed": failed,
                    "elapsed_seconds": round(monotonic() - run_started, 3),
                },
                "completed": summarize_metadata(meta, args.num_modes, args.sample_rate),
            },
            indent=2,
        )

    def right_pad_to(w: torch.Tensor, target: int) -> torch.Tensor:
        t = int(w.shape[-1])
        if t >= target:
            return w
        return torch.nn.functional.pad(w, (0, target - t))

    def extract_feat(w: torch.Tensor):
        """Return modal parameters, diagnostics, and analysis time."""
        result = modal.analyze(w[0].numpy())
        modal_freqs, modal_amps, modal_phases = (
            torch.from_numpy(values)[None] for values in result.parameters()
        )
        analysis_seconds = result.analysis_seconds
        if modal_freqs.numel() == 0:
            return w.new_zeros((3, 0, modal_freqs.shape[-1])), result, analysis_seconds

        modal_freqs = 2 * torch.pi * modal_freqs / args.sample_rate
        feat = torch.stack([modal_freqs, modal_amps, modal_phases])  # (3,1,M,F)
        feat = feat.squeeze(1)  # (3,M,F)
        return feat, result, analysis_seconds

    pbar = tqdm(
        zip(wavs, typed),
        total=len(wavs),
        desc=f"modal {args.compute_device} {args.shard_index + 1}/{args.num_shards}",
        unit="file",
        dynamic_ncols=True,
        disable=args.no_tqdm,
    )
    write_progress(force=True)

    for idx, (wav_path, (type_name, inst_name, pack)) in enumerate(pbar):
        rel = wav_path.relative_to(processed_root)
        key = stable_id(str(rel))
        file_started = perf_counter()
        if args.resume and key in meta:
            existing = meta[key]
            if (out_dir / existing["filename"]).is_file() and (
                out_dir / existing["feature_file"]
            ).is_file():
                skipped += 1
                attempted += 1
                write_progress({"source": str(rel), "status": "resumed"})
                pbar.set_postfix(kept=kept, skipped=skipped, failed=failed)
                continue

        last = {"source": str(rel), "status": "failed"}
        try:
            wav, sr = torchaudio.load(str(wav_path))  # [C,T]
            if wav.ndim != 2:
                raise ValueError(f"bad wav shape: {tuple(wav.shape)}")

            # no resample: enforce preprocessed sample rate
            if sr != args.sample_rate:
                raise ValueError(
                    f"sample_rate mismatch: {sr} != {args.sample_rate} for {rel}"
                )

            # mono-only policy: always channel 0
            if wav.shape[0] == 0:
                raise ValueError("empty channel dim")
            wav = wav[:1, :]

            # optional hard skip for too-short
            if args.min_duration_ms and args.min_duration_ms > 0:
                min_samples = int(args.sample_rate * (args.min_duration_ms / 1000.0))
                if wav.shape[-1] < min_samples:
                    raise ValueError(
                        f"too_short: {wav.shape[-1]} < {min_samples} samples"
                    )

            # optional fixed padding floor
            if args.pad_to and args.pad_to > 0:
                wav = right_pad_to(wav, args.pad_to)

            analysis_num_samples = int(wav.shape[-1])
            feat, result, analysis_seconds = extract_feat(wav)

            # force fixed num_modes
            p, m, f = feat.shape
            active_modes = int(torch.any(feat[1] != 0, dim=-1).sum())
            if m < args.num_modes:
                pad = feat.new_zeros((p, args.num_modes - m, f))
                feat = torch.cat([feat, pad], dim=1)
            elif m > args.num_modes:
                feat = feat[:, : args.num_modes, :]

            # save outputs under out_dir
            out_wav = audio_dir / f"{key}.wav"
            out_feat = feat_dir / f"{key}.pt"
            torchaudio.save(str(out_wav), wav, args.sample_rate)
            torch.save(feat, out_feat)

            meta_item = {
                "filename": str(out_wav.relative_to(out_dir)),
                "feature_file": str(out_feat.relative_to(out_dir)),
                "sample_pack_key": pack,
                "instrument": inst_name,
                "type": type_name,
                "num_samples": int(wav.shape[-1]),
                "analysis_num_samples": analysis_num_samples,
                "orig_relpath": str(rel),
                "modal_backend": args.modal_backend,
                "analyzer_version": args.analyzer_version,
                "modal_slots": args.num_modes,
                "active_modes": active_modes,
            }
            meta_item.update(result_details(result, args.cqt_max_frequency))
            meta_item["file_seconds"] = round(perf_counter() - file_started, 6)
            if split_by_index is not None:
                meta_item["split"] = split_by_index[idx]
            meta[key] = meta_item

            kept += 1
            last = {
                "source": str(rel),
                "status": "completed",
                "modes": active_modes,
                "lf_modes": meta_item.get("modal_lf_modes"),
                "hf_modes": meta_item.get("modal_hf_modes"),
                "candidates": meta_item.get("modal_candidates_pre_limit"),
                "seconds": meta_item["file_seconds"],
            }
        except Exception as e:
            failed += 1
            last["error"] = repr(e)
            with failures_path.open("a", encoding="utf-8") as failure_log:
                failure_log.write(json.dumps(last, ensure_ascii=False) + "\n")
            tqdm.write(f"fail: {rel} -> {e!r}")

        attempted += 1
        write_progress(last)
        if kept and kept % args.checkpoint_every == 0 and last["status"] == "completed":
            save_metadata()
        pbar.set_postfix(
            kept=kept,
            skipped=skipped,
            failed=failed,
            modes=last.get("modes", "-"),
            lf=last.get("lf_modes", "-"),
            hf=last.get("hf_modes", "-"),
            candidates=last.get("candidates", "-"),
            sec=last.get("seconds", "-"),
        )

    save_metadata()
    write_progress(force=True)

    print(
        "[done] total:", len(meta), "new:", kept, "skipped:", skipped, "failed:", failed
    )
    print("stats:", summary_path)
    print("out_dir:", out_dir)
    print("meta:", meta_path)
    if failed:
        raise RuntimeError(
            f"{failed} file(s) failed; inspect {failures_path} and rerun with --resume."
        )


if __name__ == "__main__":
    main()
