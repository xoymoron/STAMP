import sys
import types

import numpy as np
import torch

from stamp.control.audio import difference_score
from stamp.control.editing import cached_reference
from stamp.control.evaluation import SpectralReference, compare_branches
from stamp.models.joint_resynthesis import JointResynthesisModel


def test_spectral_metric_matches_existing_screening_and_handles_silence():
    torch.set_num_threads(2)
    rng = np.random.default_rng(4)
    x = rng.normal(size=5101).astype(np.float32)
    y = (0.3 * x + rng.normal(size=len(x)) * 0.2).astype(np.float32)
    metric = SpectralReference(torch.from_numpy(x)[None])
    assert (
        abs(metric.change(torch.from_numpy(y)[None])[0] - difference_score(y, x)) < 2e-5
    )
    assert metric.change(torch.from_numpy(x * 7)[None])[0] < 2e-6
    assert SpectralReference(torch.zeros(1, 5101)).change(torch.ones(1, 5101)) == [None]
    assert abs(metric.change(torch.zeros(1, 5101))[0] - 1) < 1e-6


def test_noise_only_freezes_modal_transient_and_isolates_context_path():
    torch.manual_seed(31)
    torch.set_num_threads(2)
    model = (
        JointResynthesisModel(torch.randn(256, 65536) * 0.1)
        .eval()
        .requires_grad_(False)
    )
    with torch.no_grad():
        model.modal_out.weight.normal_(std=0.04)
        params = torch.zeros(3, 2, 12)
        params[0] = 0.01
        params[1] = torch.rand(2, 12) * 0.1
        ref = cached_reference(
            model,
            dict(
                audio=torch.randn(1, 1201),
                params=params,
                length=1201,
                analysis_num_samples=2048,
                sample_id="selected-eval",
            ),
            torch.randn(1, 10, 256),
            9,
        )
        neutral = model.decode(ref, checkpoint_activations=False)
        h = ref.h_ref + torch.randn_like(ref.h_ref) * 2
        joint = model.decode(ref, h, checkpoint_activations=False)
        noise = model.decode(
            ref, h, control_scope="noise", checkpoint_activations=False
        )
        assert torch.equal(noise.modal_waveform, neutral.modal_waveform)
        assert torch.equal(noise.transient_waveform, neutral.transient_waveform)
        cm = ref.neutral[1]
        expected = model.noise_synth(
            model.noise_head(torch.cat((h, cm), -1)), ref.noise_offsets, ref.length
        )
        torch.testing.assert_close(noise.noise_waveform, expected, rtol=0, atol=0)
        assert not torch.equal(joint.noise_waveform, noise.noise_waveform)
        result = compare_branches(neutral, neutral)
        assert result["modal/spectral_change"] < 1e-7
        assert result["noise/delta_over_mix"] == 0


def test_tiny_nonzero_edit_keeps_original_audition_silence_rule():
    rng = np.random.default_rng(11)
    x = rng.normal(size=5101).astype(np.float32)
    y = x * 1e-10
    measured = SpectralReference(torch.from_numpy(x)[None]).change(
        torch.from_numpy(y)[None]
    )[0]
    assert abs(measured - difference_score(y, x)) < 2e-5


def test_source_lock_covers_changed_imported_dependency(tmp_path, monkeypatch):
    from stamp.control.evaluation import runtime_sources

    path = tmp_path / "dependency.py"
    path.write_text("v = 1\n")
    module = types.ModuleType("stamp._evaluation_probe")
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    before = runtime_sources(path)
    path.write_text("v = 2\n")
    after = runtime_sources(path)
    assert before[module.__name__] != after[module.__name__]
    assert "stamp.control.editing" in before


def test_redundancy_floor_and_grid_silence_have_explicit_validity():
    from stamp.control.evaluation import redundancy_valid, relative_norms

    assert redundancy_valid(1e-7, torch.ones(2, 2), torch.ones(2)).tolist() == [
        False,
        False,
    ]
    assert redundancy_valid(
        1.0, torch.tensor([[1.0, 1.0], [1e-8, 1.0]]), torch.ones(2)
    ).tolist() == [True, False]
    reference = torch.stack((torch.zeros(32), torch.ones(32)))
    assert relative_norms(torch.ones_like(reference), reference) == [None, 1.0]


def test_seeded_test_demo_does_not_claim_active_validation_clips(tmp_path):
    from stamp.cli.evaluate_control import page

    page(
        tmp_path,
        "index",
        [dict(atom=69, active=6, packs=2, clips=[])],
        "Demo",
        "Seeded test subset",
    )
    text = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "active validation clips" not in text
    assert "seed-selected test examples" in text
