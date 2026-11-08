"""Joint modal, band-noise, and transient reconstruction."""

from dataclasses import dataclass, field
from math import ceil, log

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from stamp.models.components import TemporalBlock
from stamp.models.soundstream import SoundStreamEncoder
from stamp.synths.modal_cycles import CyclesModalSynth
from stamp.synths.noise_band import (
    BANDS,
    HOP,
    SAMPLE_RATE,
    NoiseBandSynth,
)
from stamp.synths.transient import TransientTCN

MODE_CHUNK = 16
TIME_CHUNK = 65_536


def _sample_frames(values, position):
    position = position.clamp(0, values.shape[-2] - 1)
    left = position.long()
    right = (left + 1).clamp_max(values.shape[-2] - 1)
    return torch.lerp(
        values[..., left, :], values[..., right, :], (position - left)[:, None]
    )


def _projection(source, target):
    return nn.Sequential(nn.Linear(source, target), nn.LayerNorm(target), nn.SiLU())


def _residual_mlp(width):
    return nn.Sequential(
        nn.LayerNorm(width),
        nn.Linear(width, 2 * width),
        nn.SiLU(),
        nn.Linear(2 * width, width),
    )


def _zero_linear(layer):
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


def _run(module, *inputs, recompute):
    if recompute and torch.is_grad_enabled():
        return checkpoint(
            module, *inputs, use_reentrant=False, preserve_rng_state=False
        )
    return module(*inputs)


@dataclass
class ReferenceState:
    gain: torch.Tensor
    length: int
    analysis_num_samples: int
    modal_times: torch.Tensor
    audio_times: torch.Tensor
    rho: torch.Tensor
    p0: torch.Tensor
    observed: torch.Tensor
    mode_valid: torch.Tensor
    u: torch.Tensor
    h_ref: torch.Tensor
    s_ref: torch.Tensor
    film_ref: tuple[torch.Tensor, ...]
    noise_offsets: torch.Tensor
    neutral: tuple["SynthesisResult", torch.Tensor] | None = field(
        default=None, repr=False
    )


@dataclass
class SynthesisResult:
    waveform: torch.Tensor
    modal_waveform: torch.Tensor
    noise_waveform: torch.Tensor
    transient_waveform: torch.Tensor
    refined_params: torch.Tensor
    noise_amplitudes: torch.Tensor
    log_corrections: torch.Tensor
    strike_latent: torch.Tensor
    gain: torch.Tensor


class JointResynthesisModel(nn.Module):
    """Encode one valid clip and decode its cached temporal latent."""

    def __init__(self, bank: torch.Tensor):
        super().__init__()
        self.wave_encoder = SoundStreamEncoder(
            2,
            16,
            192,
            strides=(2, 2, 4, 8),
            aligned=True,
            transpose_output=True,
            checkpoint_length=TIME_CHUNK,
        )
        self.stem = nn.Sequential(
            nn.Linear(5, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU()
        )
        self.temporal = nn.ModuleList(TemporalBlock(64, dilation=d) for d in (1, 3))
        self.pool_norm = nn.LayerNorm(64)
        self.pool_queries = nn.Parameter(torch.randn(4, 64) * 0.02)
        self.pool_attention = nn.MultiheadAttention(64, 4, batch_first=True)
        self.pool_out = nn.Sequential(nn.Linear(256, 128), nn.LayerNorm(128))

        self.audio_projection = _projection(192, 256)
        self.modal_projection = _projection(128, 128)
        self.fusion = _projection(384, 256)
        self.gru = nn.GRU(256, 256, num_layers=2, batch_first=True)
        self.post_gru = nn.LayerNorm(256)
        self.ffns = nn.ModuleList(_residual_mlp(256) for _ in range(2))

        self.modal_u = nn.Linear(64, 64)
        self.modal_h = nn.Linear(256, 64)
        self.modal_norm = nn.LayerNorm(64)
        self.modal_ffn = _residual_mlp(64)
        self.modal_out = nn.Linear(64, 2)
        _zero_linear(self.modal_out)
        self.noise_head = nn.Sequential(
            nn.Linear(384, 256), nn.SiLU(), nn.Linear(256, BANDS), nn.Softplus()
        )
        nn.init.normal_(self.noise_head[2].weight, std=1e-3)
        nn.init.constant_(self.noise_head[2].bias, -7)

        self.strike_norm = nn.LayerNorm(256)
        self.strike_score = nn.Linear(256, 1)
        self.strike_out = nn.Linear(256, 16)
        _zero_linear(self.strike_score)
        self.transient = TransientTCN(
            1, 16, 1, num_layers=10, kernel_size=3, film_embedding_size=16
        )
        for layer in self.transient.net:
            affine = layer.film.net
            _zero_linear(affine)
            with torch.no_grad():
                affine.bias[:16].fill_(1)
        _zero_linear(self.transient.out_projection)
        self.modal_synth = CyclesModalSynth(mode_chunk_size=MODE_CHUNK)
        self.noise_synth = NoiseBandSynth(bank)

    def _modal_tokens(self, params, observed, temporal):
        freq, amp, phase = params.unbind(dim=1)
        initial = phase[..., :1]
        features = torch.stack(
            (
                (freq * SAMPLE_RATE).clamp_min(1).div(1000).log2().div(8),
                amp.clamp_min(1e-5).log().div(8),
                initial.sin().expand_as(freq),
                initial.cos().expand_as(freq),
                observed.to(freq.dtype),
            ),
            dim=-1,
        )
        u = self.stem(features.masked_fill(~observed[..., None], 0))
        u = u.masked_fill(~observed[..., None], 0)
        if temporal:
            for layer in self.temporal:
                u = layer(u, observed)
        return u

    def _pool(self, tokens, observed):
        # One dummy key keeps all-silent frames finite.
        tokens = self.pool_norm(tokens.squeeze(0).transpose(0, 1))
        mask = ~observed.squeeze(0).transpose(0, 1)
        empty = mask.all(dim=1)
        tokens = tokens.masked_fill(empty[:, None, None], 0)
        mask = torch.cat((mask[:, :1] & ~empty[:, None], mask[:, 1:]), dim=1)
        queries = self.pool_queries.expand(tokens.shape[0], -1, -1)
        value, _ = self.pool_attention(
            queries, tokens, tokens, key_padding_mask=mask, need_weights=False
        )
        return (
            self.pool_out(value.flatten(1)).masked_fill(empty[:, None], 0).unsqueeze(0)
        )

    def _context(self, tokens, observed, audio_times, extent, native_frames, recompute):
        pooled = _run(
            self._pool, tokens, observed, recompute=recompute and tokens.numel() > 2**20
        )
        position = (audio_times + 0.5) * (native_frames / extent) - 0.5
        return _sample_frames(pooled, position)

    def encode_reference(
        self,
        audio,
        modal_params,
        analysis_num_samples,
        noise_offsets=None,
        *,
        checkpoint_activations=True,
    ):
        """Build a reference from one unpadded [1,1,L] clip and its native cache."""
        if audio.ndim != 3 or audio.shape[:2] != (1, 1) or audio.shape[-1] == 0:
            raise ValueError("Expected one nonempty [1,1,L] waveform.")
        if (
            modal_params.ndim != 4
            or modal_params.shape[:2] != (1, 3)
            or not modal_params.shape[-1]
        ):
            raise ValueError("Expected modal parameters with shape [1,3,M,F].")
        if audio.dtype != torch.float32 or modal_params.dtype != torch.float32:
            raise ValueError("Joint resynthesis uses FP32 audio and modal parameters.")
        length = audio.shape[-1]
        extent = analysis_num_samples
        if extent < length:
            raise ValueError("Analysis extent must cover all valid audio samples.")
        # Keep a silent slot so every DDP rank uses all modal parameters.
        if modal_params.shape[2] == 0:
            modal_params = modal_params.new_zeros((1, 3, 1, modal_params.shape[-1]))
        frames = modal_params.shape[-1]
        # The right interpolation endpoint can be outside the physical clip.
        last_position = ((length - 0.5) * frames / extent) - 0.5
        contributing = max(1, min(frames, ceil(last_position) + 1))
        halo = min(frames, contributing + 8)
        with torch.no_grad():
            gain = audio.abs().amax().clamp_min(1e-4)
            freq, amplitude, phase = modal_params.unbind(1)
            p0 = torch.stack(
                (freq, amplitude / gain, phase[..., :1].expand_as(phase)), dim=1
            )
            observed_halo = p0[:, 1, :, :halo] > 0
            observed = observed_halo[..., :contributing]
            m0 = self.modal_synth(p0, length, extent)
            audio = audio / gain
            pair = F.pad(torch.cat((audio, audio - m0), dim=1), (0, (-length) % HOP))
            starts = (
                torch.arange(
                    pair.shape[-1] // HOP, device=audio.device, dtype=torch.float32
                )
                * HOP
            )
            rho = ((length - starts).clamp(0, HOP) / HOP).unsqueeze(0)
            audio_times = starts + 63.5
            modal_times = (
                torch.arange(contributing, device=audio.device, dtype=torch.float32)
                + 0.5
            ) * (extent / frames) - 0.5
            if noise_offsets is None:
                noise_offsets = torch.zeros(
                    BANDS, dtype=torch.long, device=audio.device
                )

        za = self.wave_encoder(pair, checkpoint_activations=checkpoint_activations)
        u = torch.cat(
            [
                _run(
                    self._modal_tokens,
                    params,
                    mask,
                    True,
                    recompute=checkpoint_activations and p0.shape[2] * halo > 4096,
                )
                for params, mask in zip(
                    p0[..., :halo].split(MODE_CHUNK, dim=2),
                    observed_halo.split(MODE_CHUNK, dim=1),
                )
            ],
            dim=1,
        )[..., :contributing, :]
        cm = self._context(
            u,
            observed,
            audio_times.clamp_max(length - 1),
            extent,
            frames,
            checkpoint_activations,
        )
        v = self.fusion(
            torch.cat((self.audio_projection(za), self.modal_projection(cm)), dim=-1)
        )
        h = self.post_gru(v + self.gru(v)[0])
        for ffn in self.ffns:
            h = h + ffn(h)
        strike_features = self.strike_norm(h)
        weights = (self.strike_score(strike_features).squeeze(-1) + rho.log()).softmax(
            -1
        )
        strike = self.strike_out((weights[..., None] * strike_features).sum(dim=1))
        return ReferenceState(
            gain,
            length,
            extent,
            modal_times,
            audio_times,
            rho,
            p0,
            observed,
            observed.any(dim=-1),
            u,
            h,
            strike,
            self.transient.film_coefficients(strike),
            noise_offsets,
        )

    def _correction(self, u, h, observed):
        e = self.modal_norm(self.modal_u(u) + h[:, None]).masked_fill(
            ~observed[..., None], 0
        )
        e = (e + self.modal_ffn(e)).masked_fill(~observed[..., None], 0)
        return self.modal_out(e).tanh().masked_fill(~observed[..., None], 0)

    def _refine(self, ref, h, recompute):
        hm = self.modal_h(_sample_frames(h, (ref.modal_times - 63.5) / HOP))
        delta = torch.cat(
            [
                _run(
                    self._correction,
                    u,
                    hm,
                    mask,
                    recompute=recompute and ref.observed.numel() > 4096,
                )
                for u, mask in zip(
                    ref.u.split(MODE_CHUNK, dim=1),
                    ref.observed.split(MODE_CHUNK, dim=1),
                )
            ],
            dim=1,
        )
        df, da = (delta[..., 0] * 0.03, delta[..., 1] * log(4))
        count = df.shape[-1]
        freq, amplitude, phase = ref.p0[..., :count].unbind(1)
        # These bounds are the neighboring representable FP32 values.
        corrected = (freq * df.exp()).clamp(2.0**-149, 0.5 - 2.0**-25)
        freq = torch.where(ref.observed, corrected, freq)
        active = torch.stack((freq, amplitude * da.exp(), phase), dim=1)
        params = torch.cat((active, ref.p0[..., count:]), dim=-1)
        return params, torch.stack((df, da), dim=-1)

    def _transient(self, modal, film, *, checkpoint_activations=True):
        signal = modal.detach()
        if not checkpoint_activations or signal.shape[-1] <= TIME_CHUNK:
            return self.transient.forward_with_film(signal, film)
        # The exact receptive-field halo preserves convolution across chunks.
        outputs = []
        for start in range(0, signal.shape[-1], TIME_CHUNK):
            left = max(0, start - self.transient.left_context)
            chunk = signal[..., left : start + TIME_CHUNK]
            value = _run(self.transient.forward_with_film, chunk, film, recompute=True)
            outputs.append(value[..., start - left :])
        return torch.cat(outputs, dim=-1)

    def decode(
        self,
        reference,
        h_edit=None,
        control_scope="joint",
        *,
        checkpoint_activations=True,
    ):
        """Decode normalized branches while keeping reference FiLM and offsets fixed."""
        ref = reference
        if control_scope not in ("joint", "noise"):
            raise ValueError("control_scope must be 'joint' or 'noise'.")
        cacheable = not self.training and not torch.is_grad_enabled()
        if h_edit is None and cacheable and ref.neutral is not None:
            return ref.neutral[0]
        h = ref.h_ref if h_edit is None else h_edit
        if h.shape != ref.h_ref.shape:
            raise ValueError("Edited latent must match the reference latent shape.")

        if control_scope == "noise" and h_edit is not None:
            if not cacheable:
                raise ValueError("Noise-only editing requires eval mode and no_grad.")
            if ref.neutral is None:
                self.decode(ref)
            neutral, cm = ref.neutral
            params, delta = neutral.refined_params, neutral.log_corrections
            modal, transient = neutral.modal_waveform, neutral.transient_waveform
        else:
            params, delta = self._refine(ref, h, checkpoint_activations)
            modal = self.modal_synth(
                params,
                ref.length,
                ref.analysis_num_samples,
                recompute=checkpoint_activations
                and ref.length * params.shape[2] > 2**20,
            )
            count = ref.observed.shape[-1]
            updated = torch.cat(
                [
                    _run(
                        self._modal_tokens,
                        p,
                        mask,
                        False,
                        recompute=checkpoint_activations
                        and ref.observed.numel() > 4096,
                    )
                    for p, mask in zip(
                        params[..., :count].split(MODE_CHUNK, dim=2),
                        ref.observed.split(MODE_CHUNK, dim=1),
                    )
                ],
                dim=1,
            )
            cm = self._context(
                updated,
                ref.observed,
                ref.audio_times.clamp_max(ref.length - 1),
                ref.analysis_num_samples,
                params.shape[-1],
                checkpoint_activations,
            )
            transient = self._transient(
                modal, ref.film_ref, checkpoint_activations=checkpoint_activations
            )
        envelope = self.noise_head(torch.cat((h, cm), dim=-1))
        noise = self.noise_synth(envelope, ref.noise_offsets, ref.length)
        result = SynthesisResult(
            modal + noise + transient,
            modal,
            noise,
            transient,
            params,
            envelope,
            delta,
            ref.s_ref,
            ref.gain,
        )
        if h_edit is None and cacheable:
            ref.neutral = result, cm
        return result

    def forward(
        self,
        audio,
        modal_params,
        analysis_num_samples,
        noise_offsets,
        *,
        checkpoint_activations=True,
    ):
        return self.decode(
            self.encode_reference(
                audio,
                modal_params,
                analysis_num_samples,
                noise_offsets,
                checkpoint_activations=checkpoint_activations,
            ),
            checkpoint_activations=checkpoint_activations,
        )
