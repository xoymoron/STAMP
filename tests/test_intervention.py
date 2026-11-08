import torch

from stamp.control.editing import (
    AcousticView,
    BoundedReader,
    InterventionSAE,
    cached_reference,
    edit_direction,
)
from stamp.models.joint_resynthesis import JointResynthesisModel
from stamp.tasks.joint_resynthesis import clip_offsets


def test_edit_budget_zero_feature_and_gradient():
    torch.manual_seed(2)
    sae = InterventionSAE()
    q = torch.randn(1, 24, 256)
    _, codes = sae.reconstruct(q, torch.ones(1, 24, dtype=torch.bool))
    atom = int(codes.sum((0, 1)).argmax())
    delta, _ = edit_direction(sae, codes, atom, torch.tensor(1e-4))
    assert delta.square().mean(-1).sqrt().max() <= 1.00001
    delta.square().sum().backward()
    assert sae.dictionary.grad is not None
    silent_codes = torch.zeros_like(codes)
    zero, _ = edit_direction(sae, silent_codes, atom, torch.tensor(1e-4))
    assert torch.equal(zero, torch.zeros_like(zero))


def test_acoustic_view_is_gain_invariant_and_short_audio_is_valid():
    observer = AcousticView()
    x = torch.randn(1, 3300)
    torch.testing.assert_close(observer(x), observer(x * 3.7), atol=1e-6, rtol=1e-5)
    short = torch.randn(1, 87, requires_grad=True)
    result = observer(short)
    assert result.shape == (1, 6144) and result.isfinite().all()
    result.sum().backward()
    assert short.grad.isfinite().all()


def test_reader_bound_and_zero_input():
    reader = BoundedReader(4)
    x = torch.randn(3, 6144) * 0.01
    perturbation = torch.randn_like(x) * 0.001
    logits, _ = reader(x)
    changed, _ = reader(x + perturbation)
    bound = perturbation.norm(dim=-1) / reader.tau
    assert ((changed - logits).abs().max(-1).values <= bound + 1e-6).all()
    a, b = reader(torch.zeros_like(x))
    assert not a.any() and not b.any()


def test_cached_reference_and_prefix_preserve_phase_lattice_and_gradient():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = JointResynthesisModel(torch.zeros(256, 65536)).eval()
    with torch.no_grad():
        model.modal_out.weight.normal_(std=0.02)
    model.requires_grad_(False)
    params = torch.zeros(3, 3, 19)
    params[0] = torch.linspace(0.003, 0.017, 19)
    params[1] = torch.rand(3, 19) * 0.1
    params[2] = torch.rand(3, 19)
    item = dict(
        audio=torch.randn(1, 1333),
        params=params,
        length=1333,
        analysis_num_samples=4001,
        sample_id="reference-test",
    )
    with torch.no_grad():
        original = model.encode_reference(
            item["audio"][None],
            params[None],
            4001,
            clip_offsets(item["sample_id"], 8),
            checkpoint_activations=False,
        )
    rebuilt = cached_reference(model, item, original.h_ref, 8)
    for field in ("u", "p0", "h_ref", "rho", "modal_times", "s_ref", "noise_offsets"):
        torch.testing.assert_close(getattr(original, field), getattr(rebuilt, field))
    h = original.h_ref.detach().clone().requires_grad_(True)
    full = model.decode(
        original, h, checkpoint_activations=False
    ).modal_waveform.flatten()[None]
    prefix = model.decode(
        rebuilt, h, checkpoint_activations=False
    ).modal_waveform.flatten()[None, :997]
    torch.testing.assert_close(prefix, full[:, :997])
    g_full = torch.autograd.grad(full[:, 200:997].square().sum(), h)[0]
    g_prefix = torch.autograd.grad(prefix[:, 200:].square().sum(), h)[0]
    torch.testing.assert_close(g_full, g_prefix, atol=1e-5, rtol=1e-4)
    assert g_full.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
