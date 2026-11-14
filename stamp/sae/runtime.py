"""Connect SAE experiments to the existing frozen STAMP inference path."""

from dataclasses import dataclass
from pathlib import Path

import torch

from stamp.data.joint_resynthesis import JointResynthesisDataset
from stamp.sae.experiment import onset_frame
from stamp.synths.noise_band import SAMPLE_RATE, load_noise_bank
from stamp.utils.io import sha256


@dataclass
class Shot:
    reference: object
    h: torch.Tensor
    code: torch.Tensor
    onset: int
    neutral: torch.Tensor


class Runtime:
    def __init__(self, checkpoint, noise_bank, data_dir, audio_dir, meta_file, device):
        from stamp.tasks.joint_resynthesis import JointResynthesisTask

        bank, bank_hash = load_noise_bank(noise_bank)
        task = JointResynthesisTask.load_from_checkpoint(
            str(checkpoint), map_location="cpu", bank=bank, bank_sha256=bank_hash
        )
        self.device = device
        self.model = task.model.to(device).eval().requires_grad_(False)
        self.seed = int(task.hparams.seed)
        self.data_dir, self.audio_dir, self.meta_file = data_dir, audio_dir, meta_file
        self.source = {
            "checkpoint_sha256": sha256(Path(checkpoint)),
            "bank_sha256": bank_hash,
            "metadata_sha256": sha256(Path(data_dir) / meta_file),
            "modal_config_sha256": sha256(Path(data_dir) / "modal_config.json"),
            "split_seed": self.seed,
        }

    def dataset(self, split):
        return JointResynthesisDataset(
            self.data_dir,
            self.meta_file,
            SAMPLE_RATE,
            None,
            split=split,
            audio_dir=self.audio_dir,
            strict_audio_dir=self.audio_dir is not None,
            seed=self.seed,
        )

    @torch.no_grad()
    def encode(self, item):
        from stamp.tasks.joint_resynthesis import clip_offsets

        audio = item["audio"]
        reference = self.model.encode_reference(
            audio[None].to(self.device),
            item["params"][None].to(self.device),
            item["analysis_num_samples"],
            clip_offsets(item["sample_id"], self.seed).to(self.device),
            checkpoint_activations=False,
        )
        return reference, onset_frame(audio)

    def cache_records(self, dataset):
        for index in range(len(dataset)):
            item = dataset[index]
            ref, onset = self.encode(item)
            yield item["sample_id"], ref.h_ref[0].cpu(), item["length"], onset

    @torch.no_grad()
    def shot(self, item, model, corpus):
        reference, onset = self.encode(item)
        sample_id = item["sample_id"]
        row = corpus.by_id[sample_id]
        h = reference.h_ref[0]
        cached = corpus.get(sample_id).to(h.device)
        if (
            reference.length != row["samples"]
            or onset != row["onset"]
            or h.shape != cached.shape
            or not torch.allclose(h, cached, rtol=1e-4, atol=1e-5)
        ):
            raise ValueError(
                f"Inputs or recon latents differ from the SAE cache for {sample_id!r}."
            )
        result = self.model.decode(reference, checkpoint_activations=False)
        neutral = (result.waveform * result.gain).reshape(-1)
        return Shot(reference, h, model.encode(h), onset, neutral)

    @torch.no_grad()
    def render(self, shot, h):
        result = self.model.decode(
            shot.reference, h[None], checkpoint_activations=False
        )
        return (result.waveform * result.gain).reshape(-1)
