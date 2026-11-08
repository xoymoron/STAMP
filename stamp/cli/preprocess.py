"""Preprocess raw WAV sample packs while preserving their folder structure."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional


@dataclass
class Config:
    """Input/output locations and audio preprocessing policy."""

    raw_root: Path
    processed_root: Path
    rejected_root: Path
    logs_root: Path
    sample_rate: int = 48_000
    num_samples: Optional[int] = None
    max_duration_sec: Optional[float] = 20.0
    mono: bool = True
    silence_threshold_db: float = -65.0
    remove_start_silence: bool = True
    remove_end_silence: bool = False
    tail_silence_threshold_db: float = -70.0
    tail_peak_ratio: float = 0.001
    min_tail_silence_ms: float = 50.0
    frame_size: int = 256
    hop_size: int = 256
    copy_rejected: bool = True
    device: str = "cpu"


def parse_args(argv: Optional[list[str]] = None) -> Config:
    """Parse explicit input/output paths and optional signal-processing settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", "--raw_root", dest="raw_root", type=Path, required=True
    )
    parser.add_argument(
        "--output", "--processed_root", dest="processed_root", type=Path, required=True
    )
    parser.add_argument(
        "--rejected", "--rejected_root", dest="rejected_root", type=Path, required=True
    )
    parser.add_argument(
        "--logs", "--logs_root", dest="logs_root", type=Path, required=True
    )

    parser.add_argument("--sample-rate", type=int, default=Config.sample_rate)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=Config.num_samples,
        help="Fixed output length in samples. Default keeps variable-length audio.",
    )
    parser.add_argument(
        "--max-duration-sec",
        type=float,
        default=Config.max_duration_sec,
        help="Reject longer variable-length samples; use 0 to disable.",
    )
    parser.add_argument(
        "--mono",
        action=argparse.BooleanOptionalAction,
        default=Config.mono,
        help="Keep only the channel with the highest RMS (default: enabled).",
    )
    parser.add_argument(
        "--silence-threshold-db", type=float, default=Config.silence_threshold_db
    )
    parser.add_argument(
        "--remove-start-silence",
        action=argparse.BooleanOptionalAction,
        default=Config.remove_start_silence,
    )
    parser.add_argument(
        "--remove-end-silence",
        action=argparse.BooleanOptionalAction,
        default=Config.remove_end_silence,
    )
    parser.add_argument(
        "--tail-silence-threshold-db",
        type=float,
        default=Config.tail_silence_threshold_db,
    )
    parser.add_argument("--tail-peak-ratio", type=float, default=Config.tail_peak_ratio)
    parser.add_argument(
        "--min-tail-silence-ms", type=float, default=Config.min_tail_silence_ms
    )
    parser.add_argument("--frame-size", type=int, default=Config.frame_size)
    parser.add_argument("--hop-size", type=int, default=Config.hop_size)
    parser.add_argument(
        "--copy-rejected",
        action=argparse.BooleanOptionalAction,
        default=Config.copy_rejected,
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Processing device: cpu, cuda, cuda:N, or auto.",
    )

    values = vars(parser.parse_args(argv))
    if values["max_duration_sec"] == 0:
        values["max_duration_sec"] = None
    config = Config(**values)
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    """Reject invalid signal settings and unsafe/overlapping directory paths."""
    if config.sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if config.num_samples is not None and config.num_samples <= 0:
        raise ValueError("num_samples must be positive when specified")
    if config.max_duration_sec is not None and config.max_duration_sec <= 0:
        raise ValueError("max_duration_sec must be positive when specified")
    if config.frame_size <= 0 or config.hop_size <= 0:
        raise ValueError("frame_size and hop_size must be positive")
    if config.min_tail_silence_ms < 0:
        raise ValueError("min_tail_silence_ms cannot be negative")
    if config.tail_peak_ratio < 0:
        raise ValueError("tail_peak_ratio cannot be negative")
    numeric_settings = [
        config.silence_threshold_db,
        config.tail_silence_threshold_db,
        config.tail_peak_ratio,
        config.min_tail_silence_ms,
    ]
    if config.max_duration_sec is not None:
        numeric_settings.append(config.max_duration_sec)
    if not all(math.isfinite(value) for value in numeric_settings):
        raise ValueError("Audio thresholds and durations must be finite numbers")
    if config.device not in {"cpu", "cuda", "auto"} and not re.fullmatch(
        r"cuda:\d+", config.device
    ):
        raise ValueError("device must be cpu, cuda, cuda:N, or auto")

    path_fields = ("raw_root", "processed_root", "rejected_root", "logs_root")
    for field in path_fields:
        setattr(config, field, getattr(config, field).expanduser().resolve())
    if not config.raw_root.is_dir():
        raise NotADirectoryError(config.raw_root)

    roots = [getattr(config, field) for field in path_fields]
    for index, left in enumerate(roots):
        for right in roots[index + 1 :]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError(
                    f"Processing directories must not overlap: {left}, {right}"
                )


def classify_reason(error: Exception) -> str:
    """Map preprocessing errors to the established rejection categories."""
    message = str(error).lower()
    if "too_long" in message:
        return "too_long"
    if "silent_all" in message:
        return "silent_all"
    if "near zero" in message or "near) zero" in message:
        return "zero"
    if "sox" in message or "ffmpeg" in message:
        return "decode_error"
    return "error"


def append_manifest(path: Path, record: dict[str, object]) -> None:
    """Append one UTF-8 JSON record to a JSON Lines manifest."""
    with path.open("a", encoding="utf-8") as manifest:
        manifest.write(json.dumps(record, ensure_ascii=False) + "\n")


def timestamp_utc() -> str:
    """Return an ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def process_file(
    input_path: Path,
    config: Config,
    device: object,
    ok_manifest: Path,
    bad_manifest: Path,
    preprocess: Callable[..., None],
) -> bool:
    """Process one WAV, preserving its relative path in the output tree."""
    relative_path = input_path.relative_to(config.raw_root)
    output_path = config.processed_root / relative_path

    try:
        preprocess(
            input_file=input_path,
            output_file=output_path,
            sample_rate=config.sample_rate,
            num_samples=config.num_samples,
            mono=config.mono,
            silence_threshold_db=config.silence_threshold_db,
            remove_start_silence=config.remove_start_silence,
            remove_end_silence=config.remove_end_silence,
            tail_silence_threshold_db=config.tail_silence_threshold_db,
            tail_peak_ratio=config.tail_peak_ratio,
            min_tail_silence_ms=config.min_tail_silence_ms,
            frame_size=config.frame_size,
            hop_size=config.hop_size,
            max_duration_sec=config.max_duration_sec,
            device=device,
        )
    except Exception as error:
        if output_path.exists():
            output_path.unlink()

        reason = classify_reason(error)
        rejected_path = config.rejected_root / reason / relative_path
        copy_error = None
        if config.copy_rejected:
            try:
                rejected_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(input_path, rejected_path)
            except OSError as exc:
                copy_error = str(exc)

        record: dict[str, object] = {
            "ts": timestamp_utc(),
            "status": "bad",
            "reason": reason,
            "input": str(input_path),
            "error": str(error),
        }
        if copy_error is not None:
            record["copy_error"] = copy_error
        append_manifest(bad_manifest, record)
        return False

    append_manifest(
        ok_manifest,
        {
            "ts": timestamp_utc(),
            "status": "ok",
            "input": str(input_path),
            "output": str(output_path),
        },
    )
    return True


def main(argv: Optional[list[str]] = None) -> None:
    """Preprocess every WAV under the input root and print a summary."""
    config = parse_args(argv)

    from tqdm import tqdm

    from stamp.utils.audio import preprocess_audio_file
    from stamp.utils.device import resolve_device

    # Validate CUDA once before creating outputs or classifying any files.
    device = resolve_device(config.device)
    print(f"[device] signal processing: {device}; audio I/O: cpu")

    config.processed_root.mkdir(parents=True, exist_ok=True)
    config.rejected_root.mkdir(parents=True, exist_ok=True)
    config.logs_root.mkdir(parents=True, exist_ok=True)
    ok_manifest = config.logs_root / "manifest_ok.jsonl"
    bad_manifest = config.logs_root / "manifest_bad.jsonl"

    wav_paths = sorted(
        path
        for path in config.raw_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    )
    print(f"[scan] {config.raw_root} -> {len(wav_paths)} WAV files")

    succeeded = 0
    progress = tqdm(wav_paths, desc="preprocess", unit="file", dynamic_ncols=True)
    for input_path in progress:
        if process_file(
            input_path,
            config,
            device,
            ok_manifest,
            bad_manifest,
            preprocess_audio_file,
        ):
            succeeded += 1
        progress.set_postfix(processed=succeeded, rejected=len(wav_paths) - succeeded)

    rejected = len(wav_paths) - succeeded
    print(f"[done] processed={succeeded}, rejected={rejected}")
    print(f"  processed root: {config.processed_root}")
    print(f"  rejected root: {config.rejected_root}")
    print(f"  manifests: {ok_manifest}, {bad_manifest}")


if __name__ == "__main__":
    main()
