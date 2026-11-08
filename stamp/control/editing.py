"""Post-hoc sparse interventions; the reconstruction model stays unchanged."""

from math import ceil, log, sqrt

import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.functional import melscale_fbanks

from stamp.control import ControlSAE
from stamp.models.joint_resynthesis import HOP, MODE_CHUNK, ReferenceState
from stamp.tasks.joint_resynthesis import clip_offsets

BETA = 0.25
EXCERPT = 32768
DIMENSION = 6144


class InterventionSAE(ControlSAE):
    """Same dictionary as the latent warm-up; every feature remains eligible."""


def atom_delta(sae, codes, feature):
    atom = sae.atoms()[feature : feature + 1]
    return F.conv_transpose1d(
        codes[..., feature].unsqueeze(1), atom, padding=4
    ).transpose(1, 2)


def edit_direction(sae, codes, feature, floor):
    delta = atom_delta(sae, codes, feature)
    radius = torch.linalg.vector_norm(delta, dim=-1).amax() / sqrt(256)
    denominator = torch.maximum(radius, torch.as_tensor(floor, device=delta.device))
    return delta / denominator.clamp_min(1e-6), radius


def strike_code(model, h, rho):
    features = model.strike_norm(h)
    weights = (model.strike_score(features).squeeze(-1) + rho.log()).softmax(-1)
    return model.strike_out((weights[..., None] * features).sum(1))


@torch.no_grad()
def cached_reference(model, item, h, seed):
    """Rebuild the exact reference from cached h without the waveform encoder."""
    device = h.device
    audio = item["audio"].to(device)
    params = item["params"][None].to(device)
    length, extent = item["length"], item["analysis_num_samples"]
    if params.shape[2] == 0:
        params = params.new_zeros(1, 3, 1, params.shape[-1])
    frames = params.shape[-1]
    count = max(1, min(frames, ceil((length - 0.5) * frames / extent - 0.5) + 1))
    halo = min(frames, count + 8)
    gain = audio.abs().amax().clamp_min(1e-4)
    freq, amp, phase = params.unbind(1)
    p0 = torch.stack((freq, amp / gain, phase[..., :1].expand_as(phase)), 1)
    mask_halo = p0[:, 1, :, :halo] > 0
    observed = mask_halo[..., :count]
    u = torch.cat(
        [
            model._modal_tokens(p, m, True)
            for p, m in zip(
                p0[..., :halo].split(MODE_CHUNK, 2), mask_halo.split(MODE_CHUNK, 1)
            )
        ],
        1,
    )[..., :count, :]
    starts = torch.arange(h.shape[1], device=device, dtype=torch.float32) * HOP
    rho = ((length - starts).clamp(0, HOP) / HOP)[None]
    times = (torch.arange(count, device=device) + 0.5) * (extent / frames) - 0.5
    strike = strike_code(model, h, rho)
    return ReferenceState(
        gain,
        length,
        extent,
        times,
        starts + 63.5,
        rho,
        p0,
        observed,
        observed.any(-1),
        u,
        h,
        strike,
        model.transient.film_coefficients(strike),
        clip_offsets(item["sample_id"], seed).to(device),
    )


class AcousticView(nn.Module):
    def __init__(self):
        super().__init__()
        self.sizes = (512, 2048, 4096)
        self.register_buffer("power_scale", torch.ones(3))
        self.register_buffer("feature_scale", torch.ones(3))
        for n in self.sizes:
            self.register_buffer(f"window_{n}", torch.hann_window(n))
            self.register_buffer(
                f"mel_{n}", melscale_fbanks(n // 2 + 1, 0, 24000, 64, 48000)
            )

    def powers(self, wave, jitter=0, noise=None):
        wave = wave.reshape(-1, wave.shape[-1])
        wave = wave / wave.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-5)
        if noise is not None:
            wave = wave + noise
        support = F.pad(torch.ones_like(wave), (jitter, 0))
        wave = F.pad(wave, (jitter, 0))
        result = []
        for n in self.sizes:
            window = getattr(self, f"window_{n}")
            spectrum = (
                torch.stft(
                    wave,
                    n,
                    hop_length=n // 4,
                    window=window,
                    center=True,
                    pad_mode="constant",
                    return_complex=True,
                )
                .abs()
                .square()
            )
            mel = torch.einsum("fm,bft->bmt", getattr(self, f"mel_{n}"), spectrum)
            weight = (
                F.conv1d(
                    support[:, None],
                    window.square()[None, None],
                    stride=n // 4,
                    padding=n // 2,
                )
                / window.square().sum()
            )
            result.append((mel, weight))
        return result

    def forward(self, wave, jitter=0, noise=None):
        features = []
        for i, (power, weight) in enumerate(self.powers(wave, jitter, noise)):
            compressed = torch.log1p(power / self.power_scale[i].clamp_min(1e-12))
            pooled = F.adaptive_avg_pool1d(compressed * weight, 32)
            pooled = pooled / F.adaptive_avg_pool1d(weight, 32).clamp_min(1e-6)
            features.append(pooled.flatten(1) / self.feature_scale[i].clamp_min(1e-6))
        return torch.cat(features, 1)

    @torch.no_grad()
    def calibrate(self, waves):
        energy = torch.zeros(3, device=self.power_scale.device)
        count = energy.clone()
        for wave in waves:
            for i, (power, weight) in enumerate(self.powers(wave)):
                energy[i] += (power.square() * weight).sum()
                count[i] += weight.sum() * 64
        self.power_scale.copy_((energy / count.clamp_min(1)).sqrt().clamp_min(1e-12))
        energy.zero_()
        count.zero_()
        for wave in waves:
            feature = self(wave).reshape(-1, 3, 2048)
            energy += feature.square().sum((0, 2))
            count += feature.shape[0] * 2048
        self.feature_scale.copy_((energy / count.clamp_min(1)).sqrt().clamp_min(1e-6))


class BoundedReader(nn.Module):
    def __init__(self, features, tau=0.1):
        super().__init__()
        self.tau = tau
        self.identity = nn.Parameter(torch.randn(features * 2, DIMENSION) * 0.01)
        self.amount = nn.Parameter(torch.randn(features, DIMENSION) * 0.01)

    def forward(self, effect, *, frozen=False):
        wi, wa = self.identity, self.amount
        if frozen:
            wi, wa = wi.detach(), wa.detach()
        logits = F.linear(effect, F.normalize(wi, dim=-1)) / self.tau
        amount = (F.linear(effect, F.normalize(wa, dim=-1)) / self.tau).tanh()
        return logits, amount

    def loss(self, effect, group, position, alpha, *, frozen=False):
        logits, amounts = self(effect, frozen=frozen)
        classes = torch.stack((2 * group, 2 * group + 1), -1).flatten()
        target = 2 * position + int(alpha > 0)
        selected = logits[:, classes]
        identity = F.cross_entropy(
            selected, torch.full((len(effect),), target, device=effect.device)
        ) / log(8)
        estimate = amounts[:, group[position]]
        regression = F.smooth_l1_loss(estimate, torch.full_like(estimate, alpha))
        others = selected.clone()
        others[:, target] = -torch.inf
        return identity + 0.25 * regression, {
            "accuracy": (selected.argmax(-1) == target).float().mean().detach(),
            "amount_mae": (estimate - alpha).abs().mean().detach(),
            "effect_norm": effect.norm(dim=-1).mean().detach(),
            "margin": (selected[:, target] - others.max(-1).values).mean().detach(),
        }
