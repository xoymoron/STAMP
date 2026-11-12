"""Train the original all-parallel model under STAMP's reconstruction protocol."""

import torch
import torch.distributed as dist
import yaml

from stamp.tasks.joint_resynthesis import ReconstructionTask, reconstruction_seed


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
    training_step_metrics = ("train/mss_step",)
    training_epoch_metrics = ("train/mss_epoch",)
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
            prediction = self.model(audio, params, lengths, frames)
        loss = self.model.loss_fn(prediction, audio, lengths=lengths)
        return self.log_training_terms(loss[None], len(audio), batch_idx)

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

    def evaluate_item(self, target, features, sample_id):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(
                reconstruction_seed(sample_id, self.hparams.seed)
            )
            prediction = self.model(target, features)
        return prediction, self.model.loss_fn(prediction, target)

    def validation_step(self, batch, batch_idx):
        (audio, params, lengths, frames), ids = batch
        for i, (length, frame_count, sample_id) in enumerate(
            zip(lengths.tolist(), frames.tolist(), ids)
        ):
            target = audio[i : i + 1, :, :length]
            features = params[i : i + 1, ..., :frame_count]
            prediction, loss = self.evaluate_item(target, features, sample_id)
            mss = self.validation_mss(prediction, target)
            self.validation_rows.append((sample_id, torch.stack((loss, mss)).detach()))
