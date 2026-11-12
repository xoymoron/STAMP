from itertools import chain

import pytest
import torch

from stamp.data.paired import EpochSampler, paired_rows
from stamp.loss import LengthAwareMRSTFTLoss
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

        def forward(self, audio, _):
            return audio + torch.rand_like(audio)

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
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


def test_og_backpropagates_the_configured_mss_on_raw_valid_clips(monkeypatch):
    class Baseline(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(0.7))
            self.loss_fn = LengthAwareMRSTFTLoss()

        def forward(self, audio, *_):
            return self.gain * audio

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
