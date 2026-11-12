"""Train the original all-parallel model under STAMP's reconstruction protocol."""

import torch
import torch.distributed as dist
import yaml

from stamp.tasks.joint_resynthesis import (
    TAIL_DIAGNOSTICS,
    ReconstructionTask,
    reconstruction_seed,
    tail_diagnostics,
)

BRANCHES = ("modal", "noise", "transient")
SIGNALS = (*BRANCHES, "recon", "target")
RMS_NAMES = tuple(f"{name}_rms" for name in SIGNALS)
PEAK_NAMES = tuple(f"{name}_peak" for name in SIGNALS)


@torch.no_grad()
def waveform_levels(signals, *, include_peaks=False):
    rms = torch.stack([signal.square().mean().sqrt() for signal in signals])
    if not include_peaks:
        return rms
    peaks = torch.stack([signal.abs().amax() for signal in signals])
    return torch.cat((rms, peaks))


def load_og_model(config_path):
    from drumblender.baseline import VariableLengthDrumBlender
    from jsonargparse import ArgumentParser

    model = yaml.safe_load(config_path.read_text(encoding="utf-8"))["model"]
    args = model["init_args"]
    for name, value in args.items():
        if isinstance(value, str) and value.endswith(".yaml"):
            args[name] = yaml.safe_load(
                (config_path.parent / value).read_text(encoding="utf-8")
            )
    args["float32_matmul_precision"] = None
    parser = ArgumentParser()
    parser.add_subclass_arguments(VariableLengthDrumBlender, "model")
    config = parser.parse_object({"model": model})
    return parser.instantiate_classes(config).model


def collate_og(items):
    from drumblender.baseline import collate_baseline

    return collate_baseline(items), [item[2] for item in items]


class DrumBlenderOGTask(ReconstructionTask):
    diagnostics = (*RMS_NAMES, *PEAK_NAMES, *TAIL_DIAGNOSTICS)
    training_step_metrics = ("train/mss_step",)
    training_epoch_metrics = ("train/mss_epoch", *(f"train/{key}" for key in RMS_NAMES))
    validation_metrics = ("validation/og_mss",)

    def __init__(self, model, *, seed, run_seed, lr):
        super().__init__(log_mss=True)
        self.save_hyperparameters(ignore=["model"])
        self.model = model

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        items, ids = batch
        return self.model.transfer_batch_to_device(items, device, dataloader_idx), ids

    def training_step(self, batch, batch_idx):
        (audio, params, lengths, frames), _ = batch
        seed = reconstruction_seed(
            batch_idx,
            self.hparams.run_seed,
            epoch=self.current_epoch,
            rank=self.global_rank,
        )
        # The upstream noise generator draws on the CPU before moving to the device.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            prediction, branches = self.model(
                audio, params, lengths, frames, return_branches=True
            )
        loss = self.model.loss_fn(prediction, audio, lengths=lengths)
        capture = batch_idx == 0 and self._audition_logger() is not None
        levels = []
        for i, length in enumerate(lengths.tolist()):
            target = audio[i : i + 1, :, :length]
            recon = prediction[i : i + 1, :, :length].detach()
            signals = (*(branches[i][name] for name in BRANCHES), recon, target)
            levels.append(waveform_levels(signals))
            if capture:
                self._capture_audition("train", target, recon, branches[i])
        terms = torch.cat((loss[None], torch.stack(levels).mean(dim=0)))
        return self.log_training_terms(terms, len(audio), batch_idx)

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        # Evaluate the rank-zero BatchNorm state that will be saved in the checkpoint.
        if dist.is_available() and dist.is_initialized():
            for module in self.model.modules():
                if (
                    isinstance(module, torch.nn.BatchNorm1d)
                    and module.track_running_stats
                ):
                    for name in ("running_mean", "running_var", "num_batches_tracked"):
                        dist.broadcast(getattr(module, name), src=0)

    def evaluate_item(self, target, features, sample_id, *, return_branches=False):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(
                reconstruction_seed(sample_id, self.hparams.seed)
            )
            if return_branches:
                prediction, branches = self.model(
                    target, features, return_branches=True
                )
            else:
                prediction = self.model(target, features)
        loss = self.model.loss_fn(prediction, target)
        return (
            (prediction, loss, branches[0]) if return_branches else (prediction, loss)
        )

    def validation_step(self, batch, batch_idx):
        (audio, params, lengths, frames), ids = batch
        capture = batch_idx == 0 and self._audition_logger() is not None
        for i, (length, frame_count, sample_id) in enumerate(
            zip(lengths.tolist(), frames.tolist(), ids)
        ):
            target = audio[i : i + 1, :, :length]
            features = params[i : i + 1, ..., :frame_count]
            prediction, loss, branches = self.evaluate_item(
                target, features, sample_id, return_branches=True
            )
            mss = self.validation_mss(prediction, target)
            signals = (*(branches[name] for name in BRANCHES), prediction, target)
            tail = tail_diagnostics(torch.cat((*signals[:3], target), dim=1)[0])
            values = torch.cat(
                (
                    torch.stack((loss, mss)),
                    waveform_levels(signals, include_peaks=True),
                    tail,
                )
            )
            self.validation_rows.append((sample_id, values.detach()))
            if capture:
                self._capture_audition("val", target, prediction, branches)
