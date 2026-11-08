"""The training task runs on CPU without an external logging service."""

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from stamp.tasks.joint_resynthesis import JointResynthesisTask


def test_one_training_and_validation_batch_on_cpu(tmp_path):
    torch.set_num_threads(2)
    length = 4097
    params = torch.zeros(3, 2, 33)
    params[0] = 0.01
    params[1] = 0.1
    item = dict(
        sample_id="cpu-smoke",
        audio=(0.1 * torch.sin(torch.arange(length) * 0.1))[None],
        params=params,
        length=length,
        mode_count=2,
        frame_length=33,
        analysis_num_samples=8192,
    )
    loader = DataLoader([item], batch_size=1, collate_fn=list, num_workers=0)
    bank = torch.zeros(1, 65536).expand(256, -1)
    task = JointResynthesisTask(bank, "cpu-smoke-bank", activation_budget_mib=0)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        fast_dev_run=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
        num_sanity_val_steps=0,
    )
    trainer.fit(task, train_dataloaders=loader, val_dataloaders=loader)
    assert trainer.global_step == 1
    assert torch.isfinite(trainer.callback_metrics["val_loss"])
