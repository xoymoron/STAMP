"""Train the joint model with variable-length data and one DDP rank per GPU."""

import argparse
import hashlib
import json
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from stamp.configs import TRAIN_CONFIG
from stamp.data import JointResynthesisDataset
from stamp.data.paired import PairedDataModule, paired_rows
from stamp.synths.noise_band import SAMPLE_RATE, load_noise_bank
from stamp.tasks.joint_resynthesis import JointResynthesisTask


def _items(batch):
    # Item-wise computation needs no padded storage or extra copies.
    return batch


def _paired_data(args, data, settings):
    stamp_dataset = JointResynthesisDataset(
        args.data_dir,
        data["meta_file"],
        SAMPLE_RATE,
        None,
        seed=data["split_seed"],
        audio_dir=args.audio_dir,
        strict_audio_dir=True,
    )
    og_metadata = (args.og_data_dir / "metadata.json").read_bytes()
    rows = paired_rows(stamp_dataset.metadata, json.loads(og_metadata), args.audio_dir)
    for split, members in rows.items():
        expected = stamp_dataset._split_within_pack(stamp_dataset.file_list, split)
        if set(expected) != {stamp_key for _, stamp_key, _ in members}:
            raise ValueError(
                f"Frozen OG {split} membership differs from STAMP with seed "
                f"{data['split_seed']}. Set --split-seed to the seed used to build the OG cache."
            )
    provenance = {
        "og_metadata_sha256": hashlib.sha256(og_metadata).hexdigest(),
        "split_counts": {split: len(items) for split, items in rows.items()},
        "sampler": "global_epoch_shuffle_full_optimizer_batches",
    }
    if args.model == "stamp":
        module = PairedDataModule(
            stamp_dataset, stamp_dataset.file_list, rows, 1, _items, data, settings
        )
    else:
        try:
            from drumblender.baseline import BaselineDataset
        except ModuleNotFoundError as error:
            if error.name not in {"drumblender", "drumblender.baseline"}:
                raise
            raise ValueError(
                "Install the variable-length drumblender_og checkout in this environment."
            ) from error

        from stamp.tasks.drumblender_og import collate_og

        dataset = BaselineDataset(args.og_data_dir, audio_dir=args.audio_dir)
        module = PairedDataModule(
            dataset, dataset.keys, rows, 2, collate_og, data, settings
        )
    return module, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=TRAIN_CONFIG)
    parser.add_argument("--model", choices=("stamp", "drumblender_og"), default="stamp")
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Modal feature cache root."
    )
    parser.add_argument("--audio-dir", type=Path, help="Optional processed audio root.")
    parser.add_argument(
        "--og-data-dir",
        type=Path,
        help="Use the paired protocol with this OG modal cache.",
    )
    parser.add_argument("--og-config", type=Path, help="OG variable-length model YAML.")
    parser.add_argument("--noise-bank", type=Path, help="Required for the STAMP model.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume", type=Path, help="Resume a checkpoint from this training protocol."
    )
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"))
    parser.add_argument("--devices", type=int, help="Number of training devices.")
    parser.add_argument("--seed", type=int, help="Shared run seed.")
    parser.add_argument(
        "--split-seed",
        type=int,
        help="Split and validation noise seed. Defaults to the run seed.",
    )
    parser.add_argument("--num-workers", type=int, help="DataLoader workers.")
    parser.add_argument("--wandb-project", help="Also log to this W&B project.")
    parser.add_argument("--wandb-group", help="W&B group for paired runs.")
    parser.add_argument(
        "--wandb-name", help="W&B run name. Defaults to the output name."
    )
    args = parser.parse_args(argv)
    paired = args.og_data_dir is not None
    if args.model == "stamp" and args.noise_bank is None:
        parser.error("--noise-bank is required for STAMP.")
    if args.model == "drumblender_og" and (not paired or args.og_config is None):
        parser.error("drumblender_og requires --og-data-dir and --og-config.")
    if paired and args.audio_dir is None:
        parser.error("Paired training requires a shared --audio-dir.")
    if (args.wandb_group or args.wandb_name) and not args.wandb_project:
        parser.error("Set --wandb-project to use W&B run options.")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    settings = dict(config["training"])
    data = dict(config["data"])
    for key in ("accelerator", "devices", "seed"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    if args.num_workers is not None:
        data["num_workers"] = args.num_workers
    if settings["devices"] < 1 or data["num_workers"] < 0:
        parser.error("Devices must be positive and workers nonnegative.")
    if data["batch_size"] < 1 or settings["accumulate_grad_batches"] < 1:
        parser.error("Batch size and gradient accumulation must be positive.")
    seed = settings["seed"]
    if args.split_seed is not None:
        data["split_seed"] = args.split_seed
    split_seed = data.setdefault("split_seed", seed)
    if any(type(s) is not int or not 0 <= s < 2**32 for s in (seed, split_seed)):
        parser.error("Seeds must be integers in [0, 2**32).")
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
            seed=split_seed,
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

    if paired:
        try:
            module, provenance = _paired_data(args, data, settings)
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))
        fit_kwargs = {"datamodule": module}
    else:
        provenance = {"sampler": "lightning_shuffle"}
        fit_kwargs = {
            "train_dataloaders": loader("train"),
            "val_dataloaders": loader("val"),
        }
    loss_settings = dict(
        seed=split_seed,
        run_seed=seed,
        lr=settings["lr"],
    )
    if args.model == "stamp":
        bank, checksum = load_noise_bank(args.noise_bank)
        task = JointResynthesisTask(
            bank,
            checksum,
            env_weight=settings["env_weight"],
            activation_budget_mib=settings.get("activation_budget_mib", 8192),
            log_mss=paired,
            **loss_settings,
        )
    else:
        from stamp.tasks.drumblender_og import DrumBlenderOGTask, load_og_model

        task = DrumBlenderOGTask(
            load_og_model(args.og_config),
            env_weight=settings.get("env_weight", 0.1),
            **loss_settings,
        )
        provenance["og_config_sha256"] = hashlib.sha256(
            args.og_config.read_bytes()
        ).hexdigest()
    task.hparams["training_protocol"] = {
        **provenance,
        "model": args.model,
        "objective": (
            "stamp_spectral_envelope" if args.model == "stamp" else "og_per_item_mss"
        ),
        "paired": paired,
        "data": data,
        "training": settings,
        "stamp_metadata_sha256": hashlib.sha256(
            (args.data_dir / data["meta_file"]).read_bytes()
        ).hexdigest(),
    }
    strategy = (
        DDPStrategy(
            find_unused_parameters=False,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
        if settings["devices"] > 1
        else "auto"
    )
    loggers = [
        CSVLogger(
            str(args.output), name="joint" if args.model == "stamp" else args.model
        )
    ]
    if args.wandb_project:
        args.output.mkdir(parents=True, exist_ok=True)
        loggers.append(
            WandbLogger(
                project=args.wandb_project,
                group=args.wandb_group,
                name=args.wandb_name or args.output.name,
                save_dir=str(args.output),
                log_model=False,
            )
        )
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
        logger=loggers,
        callbacks=callbacks,
        replace_sampler_ddp=not paired,
        log_every_n_steps=settings.get("log_every_n_steps", 50),
        num_sanity_val_steps=0,
    )
    trainer.fit(task, ckpt_path=args.resume, **fit_kwargs)


if __name__ == "__main__":
    main()
