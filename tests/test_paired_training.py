from itertools import chain

import pytest
import torch

from stamp.data.paired import EpochSampler, paired_rows
from stamp.loss import JointReconstructionLoss, LengthAwareMRSTFTLoss
from stamp.tasks.drumblender_og import DrumBlenderOGTask


def test_epoch_sampler_shards_full_optimizer_batches_without_repeats():
    samplers = [
        EpochSampler(43, batch_size=3, accumulate=2, seed=42, replicas=2, rank=rank)
        for rank in range(2)
    ]
    first = [list(sampler) for sampler in samplers]
    assert len(first[0]) == len(first[1]) == 18
    assert len(set(chain.from_iterable(first))) == 36
    assert first == [list(sampler) for sampler in samplers]
    for sampler in samplers:
        sampler.set_epoch(1)
    second = [list(sampler) for sampler in samplers]
    assert first != second
    assert len(set(chain.from_iterable(second))) == 36
    resumed = EpochSampler(43, batch_size=3, accumulate=2, seed=42, replicas=2, rank=0)
    resumed.set_epoch(1)
    assert list(resumed) == second[0]


def test_pair_uses_frozen_membership_and_path_order(tmp_path):
    stamp = {
        "1": {"orig_relpath": "b.wav", "num_samples": 8},
        "2": {"orig_relpath": "a.wav", "num_samples": 12},
    }
    og = {"A": {**stamp["2"], "split": "train"}, "B": {**stamp["1"], "split": "train"}}
    for name in ("a.wav", "b.wav"):
        (tmp_path / name).touch()
    rows = paired_rows(stamp, og, tmp_path)
    assert rows["train"] == [("a.wav", "2", "A"), ("b.wav", "1", "B")]
    with pytest.raises(ValueError, match="waveform paths"):
        paired_rows(stamp, {"A": og["A"]}, tmp_path)
    og["B"]["num_samples"] += 1
    with pytest.raises(ValueError, match="num_samples"):
        paired_rows(stamp, og, tmp_path)
    og["B"]["num_samples"] -= 1
    (tmp_path / "b.wav").unlink()
    with pytest.raises(FileNotFoundError, match="b.wav"):
        paired_rows(stamp, og, tmp_path)


def test_og_validation_noise_is_clip_keyed_and_preserves_the_training_rng():
    class NoisyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_fn = LengthAwareMRSTFTLoss()

        def forward(self, audio, _, *, return_branches=False):
            noise = torch.rand_like(audio)
            prediction = audio + noise
            if return_branches:
                return prediction, [
                    dict(modal=audio, noise=noise, transient=torch.zeros_like(audio))
                ]
            return prediction

    task = DrumBlenderOGTask(NoisyModel(), seed=42, run_seed=73, lr=1e-4)
    audio = torch.full((2, 1, 257), 0.5)
    params = torch.zeros(2, 3, 1, 1)
    lengths, frames = torch.tensor([257, 257]), torch.tensor([1, 1])
    state = torch.random.get_rng_state()
    task.validation_step(((audio, params, lengths, frames), ["a", "b"]), 0)
    first = dict(task.validation_rows)
    task.on_validation_epoch_start()
    task.validation_step(((audio, params, lengths, frames), ["b", "a"]), 1)
    second = dict(task.validation_rows)
    assert torch.equal(state, torch.random.get_rng_state())
    assert not torch.equal(first["a"], first["b"])
    for key in first:
        torch.testing.assert_close(
            first[key], second[key], rtol=0, atol=0, equal_nan=True
        )


def test_og_backpropagates_the_configured_mss_on_raw_valid_clips(monkeypatch):
    class Baseline(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(0.7))
            self.loss_fn = LengthAwareMRSTFTLoss()

        def forward(self, audio, *args, return_branches=False):
            prediction = self.gain * audio
            if return_branches:
                return prediction, [
                    dict(
                        modal=prediction[i : i + 1, :, :length].detach(),
                        noise=torch.zeros_like(prediction[i : i + 1, :, :length]),
                        transient=torch.zeros_like(prediction[i : i + 1, :, :length]),
                    )
                    for i, length in enumerate(args[1])
                ]
            return prediction

    model = Baseline()
    task = DrumBlenderOGTask(model, seed=42, run_seed=73, lr=1e-4)
    monkeypatch.setattr(task, "log_training_terms", lambda terms, *_: terms[0])
    audio = torch.linspace(-0.02, 0.03, 514).view(2, 1, 257)
    params = torch.zeros(2, 3, 1, 1)
    lengths, frames = torch.tensor([191, 257]), torch.tensor([1, 1])
    expected = model.loss_fn(model.gain * audio, audio, lengths=lengths)
    (expected_gradient,) = torch.autograd.grad(expected, model.gain)
    actual = task.training_step(((audio, params, lengths, frames), ["a", "b"]), 0)
    actual.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model.gain.grad, expected_gradient)


def test_og_epoch_logs_branch_levels_and_wandb_audio(tmp_path):
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import WandbLogger
    from torch.utils.data import DataLoader

    wandb = pytest.importorskip("wandb")

    class RecordingLogger(WandbLogger):
        def log_metrics(self, metrics, step=None):
            recorded.update(metrics)
            super().log_metrics(metrics, step=step)

        def log_hyperparams(self, params):
            pass

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.level = torch.nn.Parameter(torch.tensor(0.5))
            self.loss_fn = LengthAwareMRSTFTLoss()

        def transfer_batch_to_device(self, batch, device, dataloader_idx):
            return batch

        def forward(
            self, target, params, lengths=None, frames=None, *, return_branches=False
        ):
            modal = self.level * target
            noise = torch.full_like(target, 0.1)
            transient = torch.full_like(target, -0.05)
            result = (modal + transient) + noise
            if not return_branches:
                return result
            sizes = lengths if lengths is not None else [target.shape[-1]]
            branches = [
                dict(
                    modal=modal[i : i + 1, :, :size].detach(),
                    noise=noise[i : i + 1, :, :size],
                    transient=transient[i : i + 1, :, :size],
                )
                for i, size in enumerate(sizes)
            ]
            return result, branches

    recorded = {}
    logger = RecordingLogger(project="test", offline=True, save_dir=tmp_path)
    task = DrumBlenderOGTask(Model(), seed=42, run_seed=42, lr=0.0)
    batch = (
        (
            torch.full((1, 1, 4096), 0.4),
            torch.zeros(1, 3, 1, 1),
            torch.tensor([4096]),
            torch.tensor([1]),
        ),
        ["hit"],
    )
    loader = DataLoader([batch], batch_size=None, num_workers=0)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )
    try:
        trainer.fit(task, train_dataloaders=loader, val_dataloaders=loader)
    finally:
        wandb.finish()

    import soundfile as sf

    files = list(tmp_path.glob("wandb/offline-run-*/files/media/audio/**/*.wav"))
    assert len(files) == 2
    for path in files:
        waveform, _ = sf.read(path)
        assert waveform.shape == (8192,)
        assert waveform[:4096].mean() == pytest.approx(0.4, abs=1e-4)
        assert waveform[4096:].mean() == pytest.approx(0.25, abs=1e-4)
    assert trainer.global_step == 1
    for key, value in {
        "train/modal_rms": 0.5,
        "train/noise_rms": 0.25,
        "val_modal_rms": 0.5,
        "val_transient_peak": 0.125,
    }.items():
        assert float(trainer.callback_metrics[key]) == pytest.approx(value)
    target = torch.ones(1, 1, 4096)
    prediction = torch.full_like(target, 0.625)
    expected = JointReconstructionLoss()(prediction, target)
    for key, value in zip(("validation/loss", "val_spec", "val_env"), expected):
        assert float(trainer.callback_metrics[key]) == pytest.approx(float(value))
    assert float(trainer.callback_metrics["val_loss"]) == pytest.approx(
        float(trainer.callback_metrics["validation/og_mss"])
    )
    assert {
        key for key, value in recorded.items() if isinstance(value, wandb.Audio)
    } == {
        "train/audio",
        "val/audio",
    }
