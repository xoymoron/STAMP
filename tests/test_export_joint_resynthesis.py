"""Direct modal exports preserve cache amplitude, phase and analysis timing."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import soundfile as sf
import torch

from stamp.cli import evaluate as compile_results
from stamp.cli import reconstruct as export_joint_resynthesis
from stamp.cli.reconstruct import render_sinusoidal
from stamp.loss import LengthAwareMRSTFTLoss
from stamp.synths.modal_cycles import CyclesModalSynth
from stamp.tasks import drumblender_og


@pytest.mark.parametrize("empty", [False, True])
def test_sinusoidal_export_uses_raw_tracks_and_the_full_analysis_clock(empty):
    params = torch.tensor([[[0.01, 0.02]], [[0.003, 0.009]], [[0.7, 0.7]]])
    if empty:
        params = params[:, :0]
    original = params.clone()
    item = {
        "audio": torch.full((1, 127), 0.2),
        "params": params,
        "length": 127,
        "analysis_num_samples": 256,
    }
    synth = CyclesModalSynth()
    actual = render_sinusoidal(item, synth)
    expected = (
        torch.zeros_like(item["audio"]) if empty else synth(params[None], 127, 256)[0]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(params, original, rtol=0, atol=0)
    assert actual.shape == item["audio"].shape
    assert not actual.requires_grad


def test_joint_export_and_evaluation_include_sinusoidal_by_default(
    tmp_path, monkeypatch
):
    params = torch.tensor([[[0.01, 0.012]], [[0.003, 0.006]], [[0.7, 0.7]]])
    item = {
        "sample_id": "example",
        "audio": torch.full((1, 4097), 0.02),
        "params": params,
        "length": 4097,
        "analysis_num_samples": 4200,
    }

    class Dataset:
        metadata = {
            "example": {"orig_relpath": "Pack/hit.wav", "sample_pack_key": "Pack"}
        }

        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 1

        def __iter__(self):
            yield item

    class Task:
        model = SimpleNamespace(modal_synth=CyclesModalSynth())
        hparams = SimpleNamespace(seed=42)

        @classmethod
        def load_from_checkpoint(cls, *args, **kwargs):
            return cls()

        def eval(self):
            return self

        def to(self, device):
            return self

        def evaluate_item(self, item):
            result = SimpleNamespace(waveform=torch.full((1, 1, 4097), 0.125), gain=0.2)
            return result, item["audio"][None], torch.zeros(3)

    monkeypatch.setattr(export_joint_resynthesis, "JointResynthesisDataset", Dataset)
    monkeypatch.setattr(export_joint_resynthesis, "JointResynthesisTask", Task)
    monkeypatch.setattr(
        export_joint_resynthesis, "load_noise_bank", lambda _: (None, "bank")
    )
    monkeypatch.setattr(
        export_joint_resynthesis,
        "reconstruction_diagnostics",
        lambda *args: torch.zeros(len(export_joint_resynthesis.DIAGNOSTICS)),
    )
    output = tmp_path / "joint"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_joint_resynthesis",
            "--checkpoint",
            "test.ckpt",
            "--noise-bank",
            "bank.pt",
            "--data-dir",
            "cache",
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
    )
    export_joint_resynthesis.main()
    saved = next((output / "sinusoidal").glob("*.wav"))
    wave, rate = sf.read(saved, dtype="float32")
    expected = render_sinusoidal(item, CyclesModalSynth()).flatten()
    torch.testing.assert_close(torch.from_numpy(wave), expected, rtol=0, atol=0)
    assert rate == 48000 and sf.info(saved).subtype == "FLOAT"
    assert compile_results.main([str(output), "--device", "cpu"]) == 0
    assert "Sinusoidal &" in (output / "evaluation/summary_table.tex").read_text()
    summary = json.loads((output / "evaluation/summary.json").read_text())
    assert "test/mfcc" in summary["metrics"]
    assert "test/mfcc" in summary["sinusoidal_metrics"]
    columns = (
        (output / "evaluation/per_file_sinusoidal_metrics.csv")
        .read_text()
        .splitlines()[0]
        .split(",")
    )
    assert not {"loss", "spec", "env", "test/loss"}.intersection(columns)


def test_og_checkpoint_exports_unclipped_float_and_uses_shared_evaluator(
    tmp_path, monkeypatch
):
    class Dataset:
        metadata = {"hit": {"orig_relpath": "Pack/hit.wav", "sample_pack_key": "Pack"}}

        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 1

        def __iter__(self):
            yield torch.ones(1, 4096), torch.zeros(3, 1, 16), "hit"

    baseline = ModuleType("drumblender.baseline")
    baseline.BaselineDataset = Dataset
    monkeypatch.setitem(sys.modules, "drumblender.baseline", baseline)

    class Model(torch.nn.Module):
        loss_fn = LengthAwareMRSTFTLoss()

        def forward(self, target, params):
            return 1.5 * target

        def modal_synth(self, params, length):
            return torch.zeros(1, 1, length)

    model = Model()
    monkeypatch.setattr(drumblender_og, "load_og_model", lambda _: model)
    monkeypatch.setattr(
        drumblender_og.DrumBlenderOGTask,
        "load_from_checkpoint",
        lambda *args, **kwargs: drumblender_og.DrumBlenderOGTask(
            kwargs["model"], seed=20261106, run_seed=20261106, lr=1e-4
        ),
    )
    output = tmp_path / "og"
    export_joint_resynthesis.main(
        [
            "--model",
            "drumblender_og",
            "--og-config",
            "og.yaml",
            "--checkpoint",
            "og.ckpt",
            "--data-dir",
            "cache",
            "--device",
            "cpu",
            "--output",
            str(output),
        ]
    )
    path = next((output / "recon").glob("*.wav"))
    samples, rate = sf.read(path, dtype="float32")
    assert sf.info(path).subtype == "FLOAT" and rate == 48000
    assert (samples == 1.5).all()
    assert compile_results.main([str(output), "--device", "cpu"]) == 0
    summary = json.loads((output / "evaluation/summary.json").read_text())
    assert summary["metrics"]["test/log_rms_envelope"] == pytest.approx(0.0)
