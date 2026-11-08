"""Extract modal features with independent CPU or GPU workers.

Each worker runs an independent shard in a child process. Once every shard
finishes, the metadata merger verifies complete coverage of the processed WAVs.
Run with ``stamp extract-parallel`` and explicit input/output paths.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from time import monotonic, sleep

from tqdm import tqdm

from stamp.cli.extract import list_wavs
from stamp.data.modal_stats import write_json_atomic


def _selected_devices(value):
    """Map logical GPU indices through an existing CUDA visibility mask."""
    try:
        indices = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise ValueError("--gpus must be comma-separated GPU indices") from error
    if (
        not indices
        or any(index < 0 for index in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("--gpus needs distinct, nonnegative GPU indices")

    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask is None:
        return [(index, str(index)) for index in indices]
    visible = [part.strip() for part in mask.split(",") if part.strip()]
    if any(index >= len(visible) for index in indices):
        raise ValueError(
            f"Requested GPUs {indices}, but CUDA_VISIBLE_DEVICES exposes only "
            f"{len(visible)} device(s): {mask!r}"
        )
    return [(index, visible[index]) for index in indices]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    hardware = parser.add_mutually_exclusive_group()
    hardware.add_argument("--gpus", help="Visible GPU indices, e.g. 0,1")
    hardware.add_argument(
        "--cpu_workers",
        type=int,
        default=None,
        help="CPU processes; defaults to one when --gpus is omitted.",
    )
    parser.add_argument("--processed_root", type=Path, required=True)
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--modal_backend",
        choices=("hybrid", "stft", "cqt"),
        default="hybrid",
        help="Analysis backend; hybrid combines CQT and STFT.",
    )
    parser.add_argument("--num_modes", type=int, default=64)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--gpu_cqt_batch_size", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry_run", action="store_true", help="Print worker commands without running"
    )
    args = parser.parse_args(argv)
    if args.num_modes < 1 or args.checkpoint_every < 1 or args.gpu_cqt_batch_size < 1:
        parser.error(
            "Mode count, checkpoint interval, and CQT batch size must be positive"
        )
    if args.gpus is None:
        workers = args.cpu_workers if args.cpu_workers is not None else 1
        if workers < 1:
            parser.error("--cpu_workers must be positive")
        devices = [(f"CPU {index}", None, "cpu") for index in range(workers)]
    else:
        try:
            devices = [
                (f"GPU {logical}", physical, "cuda")
                for logical, physical in _selected_devices(args.gpus)
            ]
        except ValueError as error:
            parser.error(str(error))

    working_directory = Path.cwd()
    processed_root = args.processed_root.resolve()
    out_dir = args.out_dir.resolve()
    if not args.dry_run and not processed_root.is_dir():
        parser.error(f"Processed WAV directory does not exist: {processed_root}")

    worker_commands = []
    for shard_index, (label, physical_gpu, compute_device) in enumerate(devices):
        command = [
            sys.executable,
            "-m",
            "stamp.cli.extract",
            "--modal_backend",
            args.modal_backend,
            "--processed_root",
            str(processed_root),
            "--out_dir",
            str(out_dir),
            "--num_modes",
            str(args.num_modes),
            "--compute_device",
            compute_device,
            "--num_shards",
            str(len(devices)),
            "--shard_index",
            str(shard_index),
            "--checkpoint_every",
            str(args.checkpoint_every),
            "--gpu_cqt_batch_size",
            str(args.gpu_cqt_batch_size),
            "--no_tqdm",
        ]
        if args.resume:
            command.append("--resume")
        worker_commands.append((label, physical_gpu, command))

    merge_command = [
        sys.executable,
        "-m",
        "stamp.cli.merge_shards",
        "--out_dir",
        str(out_dir),
        "--num_shards",
        str(len(devices)),
    ]
    if args.dry_run:
        for label, physical_gpu, command in worker_commands:
            mask = (
                f"CUDA_VISIBLE_DEVICES={physical_gpu} "
                if physical_gpu is not None
                else ""
            )
            print(f"{label}: {mask}{shlex.join(command)}")
        if len(devices) > 1:
            print(f"Merge: {shlex.join(merge_command)}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    wav_count = len(list_wavs(processed_root))
    if wav_count == 0:
        parser.error(f"No WAVs found under {processed_root}")
    run_started = monotonic()
    workers = []
    with ExitStack() as stack:
        bars = [
            stack.enter_context(
                tqdm(
                    total=len(range(index, wav_count, len(devices))),
                    desc=label,
                    unit="file",
                    position=index,
                    dynamic_ncols=True,
                )
            )
            for index, (label, _, _) in enumerate(devices)
        ]
        try:
            for shard_index, (label, physical_gpu, command) in enumerate(
                worker_commands
            ):
                log_path = out_dir / f"shard_{shard_index}.log"
                shard_dir = (
                    out_dir / f"shard_{shard_index}" if len(devices) > 1 else out_dir
                )
                shard_dir.mkdir(parents=True, exist_ok=True)
                # Reset a previous run's live snapshot before polling it.
                write_json_atomic(
                    shard_dir / "progress.json",
                    {"attempted": 0, "kept": 0, "skipped": 0, "failed": 0},
                )
                log = stack.enter_context(log_path.open("a", encoding="utf-8"))
                log.write(f"\nStarting {label}, shard {shard_index}\n")
                log.flush()
                environment = os.environ.copy()
                if physical_gpu is not None:
                    environment["CUDA_VISIBLE_DEVICES"] = physical_gpu
                environment.setdefault("OMP_NUM_THREADS", "1")
                environment.setdefault("MKL_NUM_THREADS", "1")
                process = subprocess.Popen(
                    command,
                    cwd=working_directory,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                workers.append((shard_index, process, log_path))
                tqdm.write(
                    f"Started {label} as shard {shard_index} "
                    f"(PID {process.pid}); log: {log_path}"
                )

            reported = set()
            while len(reported) < len(workers):
                for shard_index, process, log_path in workers:
                    progress_dir = (
                        out_dir / f"shard_{shard_index}"
                        if len(devices) > 1
                        else out_dir
                    )
                    progress_path = progress_dir / "progress.json"
                    try:
                        progress = json.loads(progress_path.read_text(encoding="utf-8"))
                    except (FileNotFoundError, json.JSONDecodeError):
                        progress = {}
                    attempted = min(
                        progress.get("attempted", 0), bars[shard_index].total
                    )
                    bars[shard_index].update(max(0, attempted - bars[shard_index].n))
                    last = progress.get("last") or {}
                    bars[shard_index].set_postfix(
                        kept=progress.get("kept", 0),
                        skipped=progress.get("skipped", 0),
                        failed=progress.get("failed", 0),
                        modes=last.get("modes", "-"),
                        lf=last.get("lf_modes", "-"),
                        hf=last.get("hf_modes", "-"),
                        candidates=last.get("candidates", "-"),
                    )
                    status = process.poll()
                    if status is None or shard_index in reported:
                        continue
                    if status != 0:
                        raise RuntimeError(
                            f"Shard {shard_index} failed with exit code {status}; "
                            f"see {log_path}. Rerun with --resume after fixing it."
                        )
                    reported.add(shard_index)
                    tqdm.write(f"Shard {shard_index} completed.")
                if len(reported) == len(workers):
                    break
                sleep(1)
        except BaseException:
            for _, process, _ in workers:
                if process.poll() is None:
                    process.terminate()
            for _, process, _ in workers:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise

    if len(devices) > 1:
        subprocess.run(merge_command, cwd=working_directory, check=True)
    root_stats_path = out_dir / "run_stats.json"
    if root_stats_path.is_file():
        stats = json.loads(root_stats_path.read_text(encoding="utf-8"))
        elapsed = monotonic() - run_started
        stats["wall_seconds"] = round(elapsed, 3)
        stats["files_per_second"] = round(wav_count / elapsed, 4)
        write_json_atomic(root_stats_path, stats, indent=2)
    print(f"Modal extraction complete: {out_dir}")


if __name__ == "__main__":
    main()
