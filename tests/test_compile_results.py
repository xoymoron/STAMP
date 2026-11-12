"""Regression coverage for the default reconstruction tables and numeric reports."""

import csv
import json
import shutil

import numpy as np
import pytest
import soundfile as sf
import torch

from stamp.cli import evaluate as compile_results
from stamp.metrics import (
    DEFAULT_METRICS_CONFIG,
    audio_pair_fingerprint,
    evaluation_score_keys,
    load_evaluation_metrics,
    score_reconstruction,
)


def test_default_metrics_clip_at_epoch135_bounds():
    metrics, _ = load_evaluation_metrics()
    prediction = torch.full((1, 1, 4096), 1.5)
    target = torch.ones_like(prediction)
    scores = score_reconstruction(metrics, prediction, target, sample_rate=48000)
    assert scores["test/log_rms_envelope"] == pytest.approx(0.0)


@pytest.mark.parametrize("multiple_models", [False, True])
@pytest.mark.parametrize("sinusoidal", [False, True])
def test_default_table_includes_all_metrics_and_keeps_pack_statistics(
    tmp_path, multiple_models, sinusoidal
):
    bundle = tmp_path / "example_model"
    evaluation = bundle / "evaluation"
    evaluation.mkdir(parents=True)
    (bundle / "configs").mkdir()
    config = bundle / "configs" / "evaluation_metrics.yaml"
    shutil.copyfile(DEFAULT_METRICS_CONFIG, config)
    metrics, config_sha256 = load_evaluation_metrics(config)
    loss_rows, metric_rows, sinusoidal_rows = [], [], []
    for pack, score in (("PackA", 1.23456), ("PackB", 2.23456)):
        relative = f"{pack}/hit.wav"
        for kind in (
            ("target", "recon", "sinusoidal") if sinusoidal else ("target", "recon")
        ):
            path = bundle / kind / relative
            path.parent.mkdir(parents=True)
            sf.write(path, np.full(2048, 0.1, dtype=np.float32), 48000, subtype="FLOAT")
        source = {
            "source_filename": relative,
            "sample_pack_key": pack,
            "length": 2048,
        }
        loss_rows.append(source)
        metric_rows.append(
            {
                **source,
                **dict.fromkeys(evaluation_score_keys(metrics), score),
                "metric_config_sha256": config_sha256,
                "audio_sha256": audio_pair_fingerprint(bundle, relative),
            }
        )
        if sinusoidal:
            sinusoidal_rows.append(
                {
                    **source,
                    **dict.fromkeys(evaluation_score_keys(metrics), score + 2),
                    "metric_config_sha256": config_sha256,
                    "audio_sha256": audio_pair_fingerprint(
                        bundle, relative, "sinusoidal"
                    ),
                }
            )
    reports = [
        ("per_file_loss.csv", loss_rows),
        ("per_file_metrics.csv", metric_rows),
    ]
    if sinusoidal:
        reports.append(("per_file_sinusoidal_metrics.csv", sinusoidal_rows))
    for name, rows in reports:
        with (evaluation / name).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (evaluation / "summary.json").write_text("{}", encoding="utf-8")

    if multiple_models:
        shutil.copytree(bundle, tmp_path / "other_model")
    root = tmp_path if multiple_models else bundle
    report = root / "reports" if multiple_models else evaluation
    assert compile_results.main([str(root), "--device", "cpu"]) == 0

    tex = (report / "summary_table.tex").read_text(encoding="utf-8")
    expected_header = (
        r"Model & MSS $\downarrow$ & LSD $\downarrow$ & LSD low $\downarrow$ & "
        r"LSD mid $\downarrow$ & LSD high $\downarrow$ & SF $\downarrow$ & "
        r"MFCC $\downarrow$ \\"
    )
    assert expected_header in tex
    model_line = next(
        line for line in tex.splitlines() if line.startswith(r"example\_model &")
    )
    assert model_line.count(" & ") == 7
    assert model_line.count(r"$1.73 \pm 0.50$") == 7
    assert "PackA" not in tex and "PackB" not in tex
    text = (report / "summary_table.txt").read_text(encoding="utf-8")
    assert (
        "| Model | MSS ↓ | LSD ↓ | LSD low ↓ | LSD mid ↓ | LSD high ↓ | SF ↓ | MFCC ↓ |"
        in text
    )
    assert text.count("1.73 ± 0.50") == (14 if multiple_models else 7)
    if sinusoidal:
        assert text.count("3.73 ± 0.50") == (14 if multiple_models else 7)
        lines = [line for line in tex.splitlines() if line.endswith(r"\\")]
        model_index = lines.index(model_line)
        assert lines[model_index + 1].startswith("Sinusoidal &")
        assert lines[model_index + 1].count(r"$3.73 \pm 0.50$") == 7
        assert (evaluation / "statistics_sinusoidal.csv").is_file()

    with (report / "summary_table.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == (6 if multiple_models else 3) * (2 if sinusoidal else 1)
    assert {row["pack"] for row in rows} == {"all", "PackA", "PackB"}
    overall = next(row for row in rows if row["pack"] == "all")
    for _, prefix, _ in compile_results.TABLE_METRICS:
        assert float(overall[f"{prefix}_mean"]) == pytest.approx(1.73456)
        assert float(overall[f"{prefix}_std"]) == pytest.approx(0.5)
    summary = json.loads((evaluation / "summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["test/mfcc"] == pytest.approx(1.73456)
    if sinusoidal:
        assert summary["sinusoidal_metrics"]["test/mfcc"] == pytest.approx(3.73456)
        with (evaluation / "per_file_sinusoidal_metrics.csv").open(
            encoding="utf-8", newline=""
        ) as file:
            cached = list(csv.DictReader(file))
        with (evaluation / "per_file_loss.csv").open(
            encoding="utf-8", newline=""
        ) as file:
            sources = list(csv.DictReader(file))
        assert compile_results._cache_matches(
            bundle, cached, sources, metrics, config_sha256, "sinusoidal"
        )
        (bundle / "sinusoidal" / "PackA" / "hit.wav").unlink()
        assert not compile_results._cache_matches(
            bundle, cached, sources, metrics, config_sha256, "sinusoidal"
        )
        with pytest.raises(FileNotFoundError, match="sinusoidal"):
            compile_results.main([str(bundle), "--device", "cpu"])
