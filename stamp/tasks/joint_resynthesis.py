"""Single-stage reconstruction task and distributed validation."""

import hashlib
from math import log

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F

from stamp.loss import JointReconstructionLoss, LengthAwareMRSTFTLoss
from stamp.models.joint_resynthesis import JointResynthesisModel
from stamp.synths.noise_band import BANDS, BANK_LENGTH, SAMPLE_RATE

LOSS_NAMES = ("loss", "spec", "env")
TAIL_DIAGNOSTICS = (
    "tail_modal_energy",
    "tail_noise_energy",
    "tail_transient_energy",
    "tail_target_energy",
    "tail_inner_ts",
    "tail_correlation",
    "tail_transient_ratio",
)
DIAGNOSTICS = (
    "modal_rms",
    "noise_rms",
    "transient_rms",
    "noise_amp_mean",
    "noise_amp_p95",
    "df_p50",
    "df_p95",
    "da_p50",
    "da_p95",
    "df_saturation",
    "da_saturation",
    *TAIL_DIAGNOSTICS,
)


def reconstruction_seed(sample_id, seed, *, epoch=None, rank=0):
    key = (
        f"{seed}:validation:{sample_id}"
        if epoch is None
        else f"{seed}:train:{epoch}:{rank}:{sample_id}"
    )
    return int.from_bytes(
        hashlib.blake2b(key.encode(), digest_size=8).digest(), "little"
    )


def clip_offsets(sample_id, seed, *, epoch=None, rank=0):
    """Offsets are reproducible and independent of activation recomputation."""
    number = reconstruction_seed(sample_id, seed, epoch=epoch, rank=rank)
    return torch.randint(
        BANK_LENGTH, (BANDS,), generator=torch.Generator().manual_seed(number)
    )


def _weighted_quantiles(value, weight, probs):
    value, order = value.sort()
    cumulative = weight[order].cumsum(0)
    indices = torch.searchsorted(cumulative, probs * cumulative[-1]).clamp_max(
        value.numel() - 1
    )
    return torch.where(cumulative[-1] > 0, value[indices], float("nan"))


@torch.no_grad()
def tail_diagnostics(signals):
    """Measure the modal, noise, transient and target channels after onset."""
    length = signals.shape[-1]
    device = signals.device
    samples = torch.arange(length, device=device)
    # Detect the target onset only for diagnostics.
    n_windows = max(1, (length - 128 + 31) // 32 + 1)
    end = (n_windows - 1) * 32 + 128
    counts = (length - torch.arange(n_windows, device=device) * 32).clamp_max(128)
    target_power = F.pad(signals[3].square(), (0, end - length))
    power = target_power.unfold(0, 128, 32).sum(-1) / counts
    peak = power.max()
    onset = (power > peak * 1e-4).int().argmax() * 32
    tail = (samples >= onset + 4800) & (peak > 0)
    energy = (signals.square() * tail).sum(-1)
    inner = (signals[0] * signals[2] * tail).sum()
    floor = tail.sum() * 1e-12
    present = tail.any()
    correlation = torch.where(
        present & (energy[0] > floor) & (energy[2] > floor),
        inner / ((energy[0] * energy[2]).sqrt() + floor),
        float("nan"),
    )
    ratio = torch.where(
        present & (energy[3] > floor), energy[2] / (energy[3] + floor), float("nan")
    )
    return torch.cat(
        (
            torch.where(present, energy, float("nan")),
            torch.stack(
                (torch.where(present, inner, float("nan")), correlation, ratio)
            ),
        )
    )


@torch.no_grad()
def reconstruction_diagnostics(result, target, analysis_extent):
    signals = torch.cat(
        (
            result.modal_waveform,
            result.noise_waveform,
            result.transient_waveform,
            target,
        ),
        dim=1,
    ).squeeze(0)
    length = signals.shape[-1]
    device = signals.device
    samples = torch.arange(length, device=device)
    frames = result.refined_params.shape[-1]
    pos = ((samples + 0.5) * (frames / analysis_extent) - 0.5).clamp(0, frames - 1)
    left = pos.long()
    right = (left + 1).clamp_max(frames - 1)
    frac = pos - left
    beta = (
        target.new_zeros(frames)
        .index_add_(0, left, 1 - frac)
        .index_add_(0, right, frac)
    )
    df, da = result.log_corrections.abs().unbind(-1)
    observed = result.refined_params[:, 1, :, : df.shape[-1]] > 0
    weight = (beta[: df.shape[-1]] * observed).flatten()
    probs = target.new_tensor([0.5, 0.95])
    df, da = df.flatten(), da.flatten()
    quantiles = torch.cat(
        (_weighted_quantiles(df, weight, probs), _weighted_quantiles(da, weight, probs))
    )
    total_weight = weight.sum()
    saturation = (
        torch.stack(
            (
                (weight * (df >= 0.95 * 0.03)).sum(),
                (weight * (da >= 0.95 * log(4))).sum(),
            )
        )
        / total_weight
    )

    envelope = result.noise_amplitudes.flatten()
    return torch.cat(
        (
            signals[:3].square().mean(-1).sqrt(),
            torch.stack((envelope.mean(), torch.quantile(envelope, 0.95))),
            quantiles,
            saturation,
            tail_diagnostics(signals),
        )
    )


def unique_validation_rows(shards):
    rows = {}
    for shard in shards:
        for sample_id, row in shard:
            rows.setdefault(sample_id, row)
    if not rows:
        raise ValueError("Validation split contains no samples.")
    return rows


def checkpoint_plan(batch, budget_mib):
    """Choose which clips to checkpoint using CPU-side shape estimates."""
    budget = budget_mib * 2**20
    costs = [
        # Waveform encoder/TCN, sample-rate modal, and frame-rate modal activations.
        8192 * item["length"]
        + 24 * item["length"] * max(1, item["mode_count"])
        + 8192 * item["frame_length"] * max(1, item["mode_count"])
        for item in batch
    ]
    checkpointed = [True] * len(batch)
    for index in sorted(range(len(batch)), key=costs.__getitem__):
        if costs[index] > budget:
            break
        checkpointed[index] = False
        budget -= costs[index]
    return checkpointed


class ReconstructionTask(pl.LightningModule):
    diagnostics = ()

    def __init__(self, *, log_mss):
        super().__init__()
        self.validation_mss = LengthAwareMRSTFTLoss() if log_mss else None
        self.validation_rows = []
        self.audition = {"train": {}, "val": {}}

    def _audition_logger(self):
        trainer = self._trainer
        if trainer is None or not trainer.is_global_zero or trainer.sanity_checking:
            return None
        return next(
            (logger for logger in trainer.loggers if isinstance(logger, WandbLogger)),
            None,
        )

    @torch.no_grad()
    def _capture_audition(self, split, target, reconstruction, branches):
        signals = {
            name: signal.detach().flatten().float().cpu()
            for name, signal in dict(
                target=target, reconstruction=reconstruction, **branches
            ).items()
        }
        # Use common headroom so playback keeps the level ratios within the clip.
        headroom = max(1.0, max(float(x.abs().max()) for x in signals.values()) / 0.99)
        clips = self.audition[split]
        clips.setdefault("audio", []).extend(
            (signals["target"] / headroom, signals["reconstruction"] / headroom)
        )
        for name in branches:
            clips.setdefault(f"audio_{name}", []).append(signals[name] / headroom)

    def _log_audition(self, split):
        clips = self.audition[split]
        if not clips:
            return
        logger = self._audition_logger()
        if logger is not None:
            from wandb import Audio

            logger.log_metrics(
                {
                    f"{split}/{name}": Audio(
                        torch.cat(waves).numpy(),
                        sample_rate=SAMPLE_RATE,
                        caption=f"{split}/{name}",
                    )
                    for name, waves in clips.items()
                },
                step=self.global_step,
            )
        clips.clear()

    def on_train_epoch_end(self):
        self._log_audition("train")

    def log_training_terms(self, terms, batch_size, batch_idx):
        detached = terms.detach()
        # Lightning emits step metrics after an optimizer update.
        batch_number = batch_idx + 1
        if (
            batch_number % self.trainer.accumulate_grad_batches == 0
            or batch_number == self.trainer.num_training_batches
        ):
            step_terms = self.trainer.strategy.reduce(
                detached.clone(), reduce_op="mean"
            )
            self.log_dict(
                dict(zip(self.training_step_metrics, step_terms)),
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
            )
        self.log_dict(
            dict(zip(self.training_epoch_metrics, detached)),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return terms[0]

    def on_validation_epoch_start(self):
        self.validation_rows.clear()

    def on_validation_epoch_end(self):
        local = []
        if self.validation_rows:
            ids, values = zip(*self.validation_rows)
            local = list(zip(ids, torch.stack(values).cpu().tolist()))
        self.validation_rows.clear()
        if dist.is_available() and dist.is_initialized():
            shards = [None] * dist.get_world_size()
            dist.all_gather_object(shards, local)
        else:
            shards = [local]
        rows = unique_validation_rows(shards)
        values = torch.tensor(list(rows.values()))
        loss_names = self.validation_metrics
        if self.validation_mss is not None:
            loss_names += ("validation/mss",)
        count = len(loss_names)
        mean = torch.cat(
            (
                values[:, :count].mean(dim=0),
                values[:, count:].nanmean(dim=0),
            )
        )
        # Keep the monitor key for existing checkpoints and scheduler state.
        self.log("val_loss", mean[0], logger=False, batch_size=len(rows))
        self.log_dict(
            dict(
                zip(
                    loss_names + tuple(f"val_{name}" for name in self.diagnostics), mean
                )
            ),
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            batch_size=len(rows),
        )
        self._log_audition("val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=10,
            min_lr=6.25e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


class JointResynthesisTask(ReconstructionTask):
    diagnostics = DIAGNOSTICS
    training_step_metrics = ("train/loss_step", "train_spec", "train_env")
    training_epoch_metrics = ("train/loss_epoch", "train/spec_epoch", "train/env_epoch")
    validation_metrics = ("validation/loss", "val_spec", "val_env")

    def __init__(
        self,
        bank,
        bank_sha256,
        *,
        seed=42,
        run_seed=None,
        lr=1e-4,
        env_weight=0.1,
        activation_budget_mib=8192,
        log_mss=False,
    ):
        super().__init__(log_mss=log_mss)
        if activation_budget_mib < 0:
            raise ValueError("activation_budget_mib must be nonnegative.")
        self.save_hyperparameters(ignore=["bank"])
        self.objective = JointReconstructionLoss(env_weight)
        self.model = JointResynthesisModel(bank)

    def forward(
        self,
        audio,
        modal_params,
        analysis_num_samples,
        offsets,
        *,
        checkpoint_activations=True,
    ):
        return self.model(
            audio,
            modal_params,
            analysis_num_samples,
            offsets,
            checkpoint_activations=checkpoint_activations,
        )

    def _item(self, item, training, *, checkpoint_activations=True):
        seed = self.hparams.run_seed if training else None
        offsets = clip_offsets(
            item["sample_id"],
            self.hparams.seed if seed is None else seed,
            epoch=self.current_epoch if training else None,
            rank=self.global_rank,
        ).to(self.device)
        audio = item["audio"][None]
        result = self(
            audio,
            item["params"][None],
            item["analysis_num_samples"],
            offsets,
            checkpoint_activations=checkpoint_activations,
        )
        target = audio / result.gain
        return result, target, torch.stack(self.objective(result.waveform, target))

    def _capture_item_audition(self, split, item, result):
        self._capture_audition(
            split,
            item["audio"],
            result.waveform.detach() * result.gain,
            {
                "modal": result.modal_waveform.detach() * result.gain,
                "noise": result.noise_waveform.detach() * result.gain,
                "transient": result.transient_waveform.detach() * result.gain,
            },
        )

    def training_step(self, batch, batch_idx):
        terms = []
        capture = batch_idx == 0 and self._audition_logger() is not None
        plan = checkpoint_plan(batch, self.hparams.activation_budget_mib)
        for item, recompute in zip(batch, plan):
            result, _, losses = self._item(item, True, checkpoint_activations=recompute)
            terms.append(losses)
            if capture:
                self._capture_item_audition("train", item, result)
        terms = torch.stack(terms).mean(0)
        return self.log_training_terms(terms, len(batch), batch_idx)

    def evaluate_item(self, item):
        return self._item(item, False)

    def validation_step(self, batch, batch_idx):
        capture = batch_idx == 0 and self._audition_logger() is not None
        for item in batch:
            result, target, losses = self.evaluate_item(item)
            if capture:
                self._capture_item_audition("val", item, result)
            if self.validation_mss is not None:
                mss = self.validation_mss(
                    result.waveform * result.gain, item["audio"][None]
                )
                losses = torch.cat((losses, mss[None]))
            diagnostics = reconstruction_diagnostics(
                result, target, item["analysis_num_samples"]
            )
            self.validation_rows.append(
                (item["sample_id"], torch.cat((losses, diagnostics)).detach())
            )

    def on_load_checkpoint(self, checkpoint):
        if (
            checkpoint.get("hyper_parameters", {}).get("bank_sha256")
            != self.hparams.bank_sha256
        ):
            raise ValueError("Checkpoint requires the original noise bank asset.")
