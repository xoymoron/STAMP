import json
import math

import numpy as np
import torch

from stamp.cli.train_control import restore_trace, selection_rows
from stamp.control.editing import cached_reference
from stamp.control.objective import JointObjective, gain_from_knob, joint_prefix
from stamp.models.joint_resynthesis import JointResynthesisModel


def test_joint_prefix_matches_full_renderer_and_input_gradient():
    torch.set_num_threads(2)
    torch.manual_seed(21)
    model = JointResynthesisModel(torch.randn(256, 65536) * 0.1).eval()
    with torch.no_grad():
        model.modal_out.weight.normal_(std=0.03)
    model.requires_grad_(False)
    params = torch.zeros(3, 2, 21)
    params[0] = torch.linspace(0.002, 0.01, 21)
    params[1] = torch.rand(2, 21) * 0.1
    params[2] = 0.3
    item = dict(
        audio=torch.randn(1, 1499),
        params=params,
        length=1499,
        analysis_num_samples=3107,
        sample_id="joint-prefix",
    )
    ref = cached_reference(model, item, torch.randn(1, 12, 256), 9)
    h = ref.h_ref.clone().requires_grad_(True)
    result = model.decode(ref, h, checkpoint_activations=False)
    short = joint_prefix(model, ref, h, 997)
    expected = {
        "modal": result.modal_waveform.flatten()[None, :997],
        "noise": result.noise_waveform.flatten()[None, :997],
    }
    for scope in expected:
        torch.testing.assert_close(short[scope], expected[scope], atol=2e-6, rtol=1e-5)
        a = torch.autograd.grad(
            short[scope][:, 123:].square().sum(), h, retain_graph=True
        )[0]
        b = torch.autograd.grad(
            expected[scope][:, 123:].square().sum(), h, retain_graph=True
        )[0]
        torch.testing.assert_close(a, b, atol=3e-5, rtol=2e-4)
        assert a.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())


def test_joint_reader_gradient_split_and_silent_scope():
    objective = JointObjective(4)
    neutral = {k: torch.randn(1, 1600) for k in ("modal", "noise", "mix")}
    neutral["noise"].zero_()
    views = objective.views(neutral, np.random.default_rng(1))
    assert set(views) == {"modal", "mix"}
    waves = {k: v.clone().requires_grad_(True) for k, v in neutral.items()}
    effect = objective.effects(waves, views)
    group = torch.arange(4)
    loss, _ = objective.loss(effect, group, 1, 0.6, frozen=True)
    loss.backward()
    assert waves["modal"].grad.isfinite().all() and waves["mix"].grad.abs().sum() > 0
    assert all(p.grad is None for p in objective.readers.parameters())
    previous = waves["modal"].grad.clone()
    loss, _ = objective.loss({k: e.detach() for k, e in effect.items()}, group, 1, 0.6)
    (loss + objective.zero_loss(views, group)).backward()
    torch.testing.assert_close(waves["modal"].grad, previous)
    assert any(p.grad is not None for p in objective.readers.parameters())
    assert (
        objective.views(
            {k: torch.zeros_like(v) for k, v in neutral.items()},
            np.random.default_rng(1),
        )
        == {}
    )


def test_exponential_knob_targets_and_large_range():
    assert gain_from_knob(0) == 0
    assert math.isclose(gain_from_knob(1), 64)
    assert math.isclose(gain_from_knob(-1), -64)
    for alpha in (-64, -16, -4, -1, 1, 4, 16, 64):
        knob = np.sign(alpha) * np.log1p(abs(alpha)) / np.log(65)
        assert -1 <= knob <= 1
        assert math.isclose(gain_from_knob(knob), alpha)


def test_resume_preserves_discarded_provenance_without_duplicate_steps(tmp_path):
    rows = [{"step": 100}, {"step": 200}, {"step": 300}]
    (tmp_path / "training.json").write_text(json.dumps(rows))
    for name in ("sampling.jsonl", "validation.jsonl"):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    restore_trace(tmp_path, 200)
    assert json.loads((tmp_path / "training.json").read_text()) == rows[:2]
    assert len((tmp_path / "sampling.jsonl").read_text().splitlines()) == 2
    backup = list(tmp_path.glob("interrupted-*.json"))
    assert len(backup) == 1
    assert json.loads(backup[0].read_text())["training.json"] == rows[2:]
    restore_trace(tmp_path, 200)
    assert len(list(tmp_path.glob("interrupted-*.json"))) == 1


def test_all_cohort_features_get_fixed_examples_and_eligibility_reason():
    activity = np.zeros((6, 512))
    activity[:3, 2] = 1
    rows = [{"pack": str(i % 2)} for i in range(6)]
    result = selection_rows(activity, rows, [2, 5], 11)
    assert {r["atom"] for r in result} == {2, 5}
    assert result[0]["eligible"] and not result[1]["eligible"]
    assert result[1]["reason"] and len(set(result[1]["examples"])) == 3
