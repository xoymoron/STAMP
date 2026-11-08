"""Train the joint model with variable-length data and one DDP rank per GPU."""

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from stamp.configs import TRAIN_CONFIG
from stamp.data import JointResynthesisDataset
from stamp.synths.noise_band import SAMPLE_RATE, load_noise_bank
from stamp.tasks.joint_resynthesis import JointResynthesisTask


def _items(batch):
    # Item-wise computation needs no padded storage or extra copies.
    return batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=TRAIN_CONFIG)
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Modal feature cache root."
    )
    parser.add_argument("--audio-dir", type=Path, help="Optional processed audio root.")
    parser.add_argument("--noise-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume", type=Path, help="Resume a checkpoint from this joint model."
    )
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"))
    parser.add_argument("--devices", type=int, help="Number of training devices.")
    parser.add_argument("--num-workers", type=int, help="DataLoader workers.")
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    settings = dict(config["training"])
    data = dict(config["data"])
    for key in ("accelerator", "devices"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    if args.num_workers is not None:
        data["num_workers"] = args.num_workers
    if settings["devices"] < 1 or data["num_workers"] < 0:
        parser.error("Devices must be positive and workers nonnegative.")
    seed = settings["seed"]
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False

    if (
        settings["accelerator"] == "gpu"
        and torch.cuda.device_count() < settings["devices"]
    ):
        parser.error("The configured number of CUDA devices is not visible.")

    def loader(split):
        dataset = JointResynthesisDataset(
            args.data_dir,
            data["meta_file"],
            SAMPLE_RATE,
            None,
            split=split,
            seed=seed,
            audio_dir=args.audio_dir,
        )
        if len(dataset) == 0:
            raise ValueError(f"Empty {split} split.")
        return DataLoader(
            dataset,
            batch_size=data["batch_size"],
            shuffle=split == "train",
            num_workers=data["num_workers"],
            persistent_workers=data["num_workers"] > 0,
            pin_memory=settings["accelerator"] != "cpu" and torch.cuda.is_available(),
            collate_fn=_items,
        )

    bank, checksum = load_noise_bank(args.noise_bank)
    task = JointResynthesisTask(
        bank,
        checksum,
        seed=seed,
        lr=settings["lr"],
        env_weight=settings["env_weight"],
        activation_budget_mib=settings.get("activation_budget_mib", 8192),
    )
    train, val = loader("train"), loader("val")
    strategy = (
        DDPStrategy(
            find_unused_parameters=False,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
        if settings["devices"] > 1
        else "auto"
    )
    logger = CSVLogger(str(args.output), name="joint")
    callbacks = [
        LearningRateMonitor(),
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_last=True,
            save_top_k=1,
            filename="{epoch:03d}-{val_loss:.6f}",
        ),
        EarlyStopping(monitor="val_loss", mode="min", patience=20),
    ]
    trainer = pl.Trainer(
        accelerator=settings["accelerator"],
        devices=settings["devices"],
        strategy=strategy,
        precision=32,
        accumulate_grad_batches=settings["accumulate_grad_batches"],
        max_epochs=settings["max_epochs"],
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        logger=logger,
        callbacks=callbacks,
        replace_sampler_ddp=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(
        task, train_dataloaders=train, val_dataloaders=val, ckpt_path=args.resume
    )


if __name__ == "__main__":
    main()
