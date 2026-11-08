"""Combine completed modal extraction shards into one dataset metadata file.

The shard directories remain in place. Metadata paths are prefixed with their
shard names, so no WAV or feature tensor needs to be copied or renamed.
"""

import argparse
import json
from pathlib import Path

from stamp.cli.extract import list_wavs, stable_id
from stamp.data.modal_stats import summarize_metadata, write_json_atomic


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    args = parser.parse_args(argv)
    if args.num_shards < 2:
        parser.error("--num_shards must be at least 2")

    merged = {}
    baseline = None
    ignored_settings = {
        "out_dir",
        "shard_index",
        "resume",
        "checkpoint_every",
        "max_files",
        "no_tqdm",
    }
    for index in range(args.num_shards):
        shard_name = f"shard_{index}"
        shard_dir = args.out_dir / shard_name
        config = json.loads(
            (shard_dir / "modal_config.json").read_text(encoding="utf-8")
        )
        if config["num_shards"] != args.num_shards or config["shard_index"] != index:
            raise ValueError(f"Incorrect shard settings in {shard_dir}")
        settings = {
            key: value for key, value in config.items() if key not in ignored_settings
        }
        if baseline is None:
            baseline = settings
            first_config = config
        elif settings != baseline:
            raise ValueError(f"Analysis settings differ in {shard_dir}")

        metadata = json.loads(
            (shard_dir / config["meta_name"]).read_text(encoding="utf-8")
        )
        for key, item in metadata.items():
            if key in merged:
                raise ValueError(f"Duplicate sample ID: {key}")
            item = item.copy()
            for field in ("filename", "feature_file"):
                relative = Path(item[field])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe {field} in {shard_dir}: {relative}")
                if not (shard_dir / relative).is_file():
                    raise FileNotFoundError(shard_dir / relative)
                item[field] = str(Path(shard_name) / relative)
            merged[key] = item

    processed_root = Path(first_config["processed_root"])
    if not processed_root.is_dir():
        raise FileNotFoundError(processed_root)
    waveforms = list_wavs(processed_root)
    if not waveforms:
        raise RuntimeError(f"No WAVs found under {processed_root}")
    expected = {stable_id(str(path.relative_to(processed_root))) for path in waveforms}
    actual = set(merged)
    if actual != expected:
        raise ValueError(
            f"Extraction incomplete: {len(expected - actual)} missing, "
            f"{len(actual - expected)} unexpected IDs. Resume both shards first."
        )

    output_metadata = args.out_dir / first_config["meta_name"]
    write_json_atomic(output_metadata, merged)
    combined_config = first_config.copy()
    combined_config["out_dir"] = str(args.out_dir)
    combined_config["shard_index"] = None
    combined_config["merged_shards"] = args.num_shards
    (args.out_dir / "modal_config.json").write_text(
        json.dumps(combined_config, indent=2), encoding="utf-8"
    )
    write_json_atomic(
        args.out_dir / "run_stats.json",
        {
            "shards": args.num_shards,
            "completed": summarize_metadata(
                merged, first_config["num_modes"], first_config["sample_rate"]
            ),
        },
        indent=2,
    )
    print(f"Merged {len(merged)} samples into {output_metadata}")


if __name__ == "__main__":
    main()
