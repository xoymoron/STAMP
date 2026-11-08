import pytest
import torch

from stamp.loss import JointReconstructionLoss
from stamp.models import joint_resynthesis
from stamp.models.joint_resynthesis import JointResynthesisModel
from stamp.synths import noise_band
from stamp.synths.modal_cycles import CyclesModalSynth, _block_starts, _cuda_carry
from stamp.synths.noise_band import NoiseBandSynth


def test_reference_identity_and_noise_only_edit():
    torch.manual_seed(7)
    model = JointResynthesisModel(torch.randn(256, 65536)).eval()
    audio = torch.randn(1, 1, 129) * 0.05
    params = torch.tensor([[[[0.01, 0.012]], [[0.02, 0.01]], [[0.4, 0.4]]]])
    with torch.no_grad():
        ref = model.encode_reference(audio, params, 256)
        neutral = model.decode(ref)
        edited = model.decode(ref, ref.h_ref + 0.5, control_scope="noise")
    torch.testing.assert_close(neutral.refined_params, ref.p0, rtol=0, atol=0)
    expected = CyclesModalSynth()(ref.p0, 129, 256)
    torch.testing.assert_close(neutral.modal_waveform, expected)
    assert ref.rho.tolist() == [[1, 1 / 128]]
    assert neutral.transient_waveform.count_nonzero() == 0
    assert edited.modal_waveform is neutral.modal_waveform
    assert edited.transient_waveform is neutral.transient_waveform
    assert not torch.equal(edited.noise_waveform, neutral.noise_waveform)
    assert sum(p.numel() for p in model.parameters()) == 3_066_484


@pytest.mark.parametrize("length", [31, 8201])
def test_noise_adjoint_matches_dense_interpolation(monkeypatch, length):
    monkeypatch.setattr(noise_band, "TIME_CHUNK", 4096)
    bank = torch.linspace(-1, 1, 65536).expand(256, -1)
    amplitudes = torch.randn(1, (length + 127) // 128, 256, requires_grad=True)
    offsets = torch.arange(256) * 263
    prediction = NoiseBandSynth(bank)(amplitudes, offsets, length)
    expected = torch.nn.functional.interpolate(
        amplitudes.transpose(1, 2),
        scale_factor=128,
        mode="linear",
        align_corners=False,
    )[..., :length]
    carriers = bank.gather(1, (torch.arange(length) + offsets[:, None]) % 65536)
    expected = (expected * carriers).sum(dim=1, keepdim=True)
    torch.testing.assert_close(prediction, expected, atol=2e-5, rtol=2e-5)
    upstream = torch.linspace(-0.7, 1.3, length)[None, None, :]
    (derivative,) = torch.autograd.grad(prediction, amplitudes, upstream)
    (expected_derivative,) = torch.autograd.grad(expected, amplitudes, upstream)
    torch.testing.assert_close(derivative, expected_derivative, atol=1e-4, rtol=1e-5)


def test_silent_loss_is_mean_square_with_finite_gradient():
    prediction = torch.full((1, 1, 19), 0.1, requires_grad=True)
    loss, spec, env = JointReconstructionLoss()(
        prediction, torch.zeros_like(prediction)
    )
    torch.testing.assert_close(loss, prediction.square().mean())
    torch.testing.assert_close(spec, loss)
    assert env == 0
    loss.backward()
    torch.testing.assert_close(
        prediction.grad, 2 * prediction.detach() / prediction.numel()
    )


def test_transient_retained_and_checkpointed_paths_match(monkeypatch):
    monkeypatch.setattr(joint_resynthesis, "TIME_CHUNK", 64)
    bank = torch.zeros(1, 65536).expand(256, -1)
    model = JointResynthesisModel(bank)
    # Open the zero-initialized paths to isolate the intended gradient routes.
    with torch.no_grad():
        model.transient.net[-1].film.net.weight.fill_(0.01)
        model.transient.out_projection.weight.fill_(0.01)
    modal = torch.randn(1, 1, 71, requires_grad=True)
    strike = torch.randn(1, 16, requires_grad=True)
    film = model.transient.film_coefficients(strike)
    output = model._transient(modal, film)
    retained = model._transient(modal, film, checkpoint_activations=False)
    torch.testing.assert_close(output, retained)
    modal_grad, strike_grad = torch.autograd.grad(
        output.sum(), (modal, strike), allow_unused=True, retain_graph=True
    )
    (retained_grad,) = torch.autograd.grad(retained.sum(), strike)
    torch.testing.assert_close(strike_grad, retained_grad)
    assert modal_grad is None
    assert strike_grad.count_nonzero() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_fused_cycle_carry_matches_cpu():
    carry = _cuda_carry()
    if carry is None:
        pytest.skip("Requires Triton")
    totals = torch.rand(2, 16, 3750, generator=torch.Generator().manual_seed(9)) * 128
    reference = _block_starts(totals)
    torch.testing.assert_close(carry(totals.cuda()).cpu(), reference, rtol=0, atol=0)
