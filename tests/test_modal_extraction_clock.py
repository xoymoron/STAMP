"""Exercise short CQT extraction using real audio and feature caches."""

import json
import sys

import numpy as np
import soundfile as sf

from stamp.cli import extract as build_modal_features


def extract(tmp_path, monkeypatch, backend, length, *options):
    source = tmp_path / "source"
    output = tmp_path / "cache"
    source.mkdir(exist_ok=True)
    waveform = 0.5 * np.sin(np.arange(length) * 2 * np.pi * 1500 / 8000)
    sf.write(source / "hit.wav", waveform, 8000, subtype="FLOAT")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_modal_features",
            "--processed_root",
            str(source),
            "--out_dir",
            str(output),
            "--modal_backend",
            backend,
            "--sample_rate",
            "8000",
            "--hop_length",
            "32",
            "--fmin",
            "1000",
            "--n_bins",
            "12",
            "--bins_per_octave",
            "12",
            "--num_modes",
            "1",
            "--min_length",
            "1",
            "--no_tqdm",
            *options,
        ],
    )
    build_modal_features.main()
    metadata = json.loads((output / "metadata.json").read_text())
    return output, metadata


def test_explicit_padding_is_saved_with_the_analysis_length(tmp_path, monkeypatch):
    output, metadata = extract(tmp_path, monkeypatch, "hybrid", 100, "--pad_to", "320")
    (item,) = metadata.values()
    info = sf.info(output / item["filename"])
    assert info.frames == 320 and info.subtype == "FLOAT"
    assert item["num_samples"] == item["analysis_num_samples"] == 320
