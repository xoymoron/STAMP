"""Export paired WAVs and branch diagnostics from a joint model checkpoint."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm

from stamp.data import JointResynthesisDataset
from stamp.metrics import DEFAULT_METRICS_CONFIG
from stamp.synths.noise_band import SAMPLE_RATE, load_noise_bank
from stamp.tasks.joint_resynthesis import (
    DIAGNOSTICS,
    LOSS_NAMES,
    JointResynthesisTask,
    reconstruction_diagnostics,
)
from stamp.utils.device import resolve_device


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def render_sinusoidal(item, synth):
    """Render the analyzed amplitudes and phases on their original frame clock."""
    if item["params"].shape[1] == 0:
        return torch.zeros_like(item["audio"])
    return synth(item["params"][None], item["length"], item["analysis_num_samples"])[0]


@torch.inference_mode()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--noise-bank", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--meta-file", default="metadata.json")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--branches", action="store_true", help="Also write S, N, T and T-off."
    )
    args = parser.parse_args(argv)
    try:
        device = resolve_device(args.device)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    bank, checksum = load_noise_bank(args.noise_bank)
    task = (
        JointResynthesisTask.load_from_checkpoint(
            str(args.checkpoint),
            map_location="cpu",
            bank=bank,
            bank_sha256=checksum,
        )
        .eval()
        .to(device)
    )
    dataset = JointResynthesisDataset(
        args.data_dir,
        args.meta_file,
        SAMPLE_RATE,
        None,
        split=args.split,
        audio_dir=args.audio_dir,
        seed=task.hparams.seed,
    )
    if not len(dataset):
        parser.error(f"Empty {args.split} split.")

    args.output.mkdir(parents=True, exist_ok=False)
    folders = ["target", "recon", "sinusoidal", "evaluation", "configs"]
    if args.branches:
        folders.extend(("modal", "noise", "transient", "t_off"))
    for folder in folders:
        (args.output / folder).mkdir()
    shutil.copyfile(
        DEFAULT_METRICS_CONFIG, args.output / "configs" / "evaluation_metrics.yaml"
    )

    losses, diagnostics = [], []
    for item in tqdm(dataset, total=len(dataset), unit="clip"):
        sample_id = item["sample_id"]
        source = dataset.metadata[sample_id]
        name = hashlib.blake2b(sample_id.encode(), digest_size=12).hexdigest() + ".wav"
        item["audio"] = item["audio"].to(device)
        item["params"] = item["params"].to(device)
        result, target, terms = task.evaluate_item(item)
        values = reconstruction_diagnostics(
            result, target, item["analysis_num_samples"]
        )
        row = {
            "sample_id": sample_id,
            "source_filename": name,
            "orig_relpath": source.get("orig_relpath", ""),
            "sample_pack_key": source.get("sample_pack_key", "__root__"),
            "length": item["length"],
        }
        losses.append({**row, **dict(zip(LOSS_NAMES, terms.cpu().tolist()))})
        diagnostics.append({**row, **dict(zip(DIAGNOSTICS, values.cpu().tolist()))})
        signals = {
            "target": item["audio"],
            "recon": result.waveform * result.gain,
            "sinusoidal": render_sinusoidal(item, task.model.modal_synth),
        }
        if args.branches:
            signals.update(
                modal=result.modal_waveform * result.gain,
                noise=result.noise_waveform * result.gain,
                transient=result.transient_waveform * result.gain,
                t_off=(result.modal_waveform + result.noise_waveform) * result.gain,
            )
        for folder, wave in signals.items():
            sf.write(
                args.output / folder / name,
                wave.flatten().cpu().numpy(),
                SAMPLE_RATE,
                subtype="FLOAT",
            )
    write_rows(args.output / "evaluation" / "per_file_loss.csv", losses)
    write_rows(args.output / "evaluation" / "per_file_diagnostics.csv", diagnostics)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "items": len(dataset),
        "split": args.split,
        "seed": task.hparams.seed,
        "sinusoidal": "Direct cached modal synthesis before learned corrections",
    }
    (args.output / "evaluation" / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
