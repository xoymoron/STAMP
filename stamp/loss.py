from typing import Optional

import torch
import torch.nn.functional as F

from stamp.utils.lengths import validate_lengths


def _build_default_mrstft_loss() -> torch.nn.Module:
    import auraloss

    return auraloss.freq.MultiResolutionSTFTLoss()


class LengthAwareMRSTFTLoss(torch.nn.Module):
    """
    Mean of MR-STFT losses evaluated independently on valid item waveforms.

    STFT settings are inherited from the supplied auraloss module. Even equal-
    length items are evaluated separately so spectral convergence cannot couple
    their energies. Clips too short for centered reflection padding are extended
    with zeros to the minimum supported length, independently of batch padding.
    """

    def __init__(self, mrstft: Optional[torch.nn.Module] = None):
        super().__init__()
        self.mrstft = mrstft if mrstft is not None else _build_default_mrstft_loss()

    @staticmethod
    def _normalize_lengths(
        lengths,
        *,
        batch_size: int,
        max_length: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        return validate_lengths(
            lengths, batch_size=batch_size, max_length=max_length, device=device
        )

    @staticmethod
    def _is_uniform_length(lengths: Optional[torch.Tensor]) -> bool:
        if lengths is None:
            return True
        return bool(torch.all(lengths == lengths[0]).item())

    @staticmethod
    def _iter_valid_pairs(
        pred: torch.Tensor,
        target: torch.Tensor,
        lengths: Optional[torch.Tensor],
    ):
        if lengths is None:
            for idx in range(pred.shape[0]):
                yield pred[idx : idx + 1], target[idx : idx + 1]
            return

        for idx in range(pred.shape[0]):
            length = int(lengths[idx].item())
            yield pred[idx : idx + 1, :, :length], target[idx : idx + 1, :, :length]

    def forward(self, pred: torch.Tensor, target: torch.Tensor, lengths=None):
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have identical shape, got {pred.shape} vs {target.shape}"
            )
        lengths = self._normalize_lengths(
            lengths,
            batch_size=pred.shape[0],
            max_length=pred.shape[-1],
            device=pred.device,
        )
        stft_losses = getattr(self.mrstft, "stft_losses", ())
        min_length = max((term.fft_size // 2 + 1 for term in stft_losses), default=1)
        losses = []
        for pred_i, target_i in self._iter_valid_pairs(pred, target, lengths):
            padding = max(0, min_length - pred_i.shape[-1])
            if padding:
                pred_i = F.pad(pred_i, (0, padding))
                target_i = F.pad(target_i, (0, padding))
            losses.append(self.mrstft(pred_i.contiguous(), target_i.contiguous()))
        return torch.stack(losses).mean()


class JointReconstructionLoss(torch.nn.Module):
    """Full-sum spectral convergence, log magnitude and log RMS losses."""

    fft_sizes = (256, 1024, 4096, 8192)

    def __init__(self, env_weight=0.1):
        super().__init__()
        self.env_weight = env_weight
        for size in self.fft_sizes:
            self.register_buffer(
                f"hann_{size}", torch.hann_window(size), persistent=False
            )

    @staticmethod
    def _log_rms(waveform):
        length = waveform.shape[-1]
        frames = max(1, (length - 128 + 63) // 64 + 1)
        end = (frames - 1) * 64 + 128
        power = F.pad(waveform.square(), (0, end - length))
        count = (length - torch.arange(frames, device=waveform.device) * 64).clamp_max(
            128
        )
        mean_power = power.unfold(-1, 128, 64).sum(-1) / count
        return ((mean_power + 1e-8).sqrt() + 1e-4).log()

    def forward(self, prediction, target):
        if prediction.shape != target.shape or prediction.shape[:2] != (1, 1):
            raise ValueError("Expected matching [1,1,L] valid waveforms.")
        y, x = prediction.flatten(), target.detach().flatten()
        spectral = []
        for size in self.fft_sizes:
            window = getattr(self, f"hann_{size}")
            kwargs = dict(
                n_fft=size,
                hop_length=size // 4,
                window=window,
                center=True,
                pad_mode="constant",
                return_complex=True,
            )
            target_mag = torch.stft(x, **kwargs).abs()
            pred_mag = torch.stft(y, **kwargs).abs()
            target_norm = torch.linalg.vector_norm(target_mag)
            eps = 1e-4 * (target_norm / target_mag.numel() ** 0.5).clamp_min(1e-3)
            spectral.append(
                torch.linalg.vector_norm(pred_mag - target_mag) / (target_norm + eps)
                + ((pred_mag + eps).log() - (target_mag + eps).log()).abs().mean()
            )
        silent = x.count_nonzero() == 0
        spec = torch.where(silent, y.square().mean(), torch.stack(spectral).mean())
        env = (self._log_rms(y) - self._log_rms(x)).abs().mean()
        env = torch.where(silent, 0.0, env)
        return spec + self.env_weight * env, spec, env
