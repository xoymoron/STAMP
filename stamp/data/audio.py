"""
Audio datasets
"""

import json
import logging
import os
from collections import defaultdict
from numbers import Integral
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torchaudio
from torch.utils.data import Dataset

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def _sample_pack_key(item: dict) -> str:
    """Keep the complete recorded pack name, including nested components."""
    return str(item.get("sample_pack_key") or "__root__").replace("\\", "/").strip("/")


def _split_within_sample_packs(
    metadata: dict,
    keys: List[str],
    split: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> List[str]:
    """Split files independently inside each pack before applying any filters."""
    if split not in ("train", "val", "test"):
        raise ValueError("Invalid split. Must be one of 'train', 'val', or 'test'.")
    if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio > 1:
        raise ValueError("Require train >= 0, val >= 0, and train+val <= 1.")

    by_pack = defaultdict(list)
    for key in sorted(keys):
        by_pack[_sample_pack_key(metadata[key])].append(key)

    generator = torch.Generator().manual_seed(seed)
    selected = []
    for pack in sorted(by_pack):
        items = by_pack[pack]
        permutation = torch.randperm(len(items), generator=generator).tolist()
        shuffled = [items[index] for index in permutation]
        count = len(shuffled)
        n_train = max(1, int(count * train_ratio)) if train_ratio > 0 else 0
        n_val = min(int(count * val_ratio), count - n_train)
        if split == "train":
            selected.extend(shuffled[:n_train])
        elif split == "val":
            selected.extend(shuffled[n_train : n_train + n_val])
        else:
            selected.extend(shuffled[n_train + n_val :])
    return selected


class AudioWithParametersDataset(Dataset):
    """
    Loads (waveform, params, length).

    waveform: [1, T]
    params:   [P, M, F] where P is typically 3 (freq, amp, phase)
    length:   int (T)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        meta_file: str,
        sample_rate: int,
        num_samples: Optional[int],
        split: Optional[str] = None,
        seed: int = 42,
        normalize: bool = False,
        sample_types: Optional[List[str]] = None,
        instruments: Optional[List[str]] = None,
        sample_pack_keys: Optional[List[str]] = None,
        parameter_key: str = "feature_file",
        audio_dir: Optional[Union[str, Path]] = None,
        expected_num_modes: Optional[int] = None,
        split_train_ratio: float = 0.8,
        split_val_ratio: float = 0.1,
        modal_config_file: str = "modal_config.json",
        preserve_analysis_clock: bool = False,
    ):
        super().__init__()
        if num_samples is not None and num_samples > 0:
            raise ValueError(
                "AudioWithParametersDataset requires num_samples=None: fixed-length "
                "cropping/padding changes the cached modal time grid."
            )
        if normalize:
            raise ValueError(
                "AudioWithParametersDataset requires normalize=False: audio-only "
                "normalization would leave cached modal amplitudes unchanged."
            )
        self.data_dir = Path(data_dir)
        self.meta_file = meta_file
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.seed = seed
        self.split = split
        self.sample_types = sample_types
        self.instruments = instruments
        self.sample_pack_keys = sample_pack_keys
        self.parameter_key = parameter_key
        self.audio_dir = Path(audio_dir) if audio_dir is not None else None
        self._audio_fallback_warned = False
        self.expected_num_modes = expected_num_modes
        self.split_train_ratio = split_train_ratio
        self.split_val_ratio = split_val_ratio
        # Explicit-clock consumers carry analysis timestamps into their renderer.
        # Consumers without explicit timing reject a longer analysis extent.
        self.preserve_analysis_clock = preserve_analysis_clock

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset dir not found: {self.data_dir}")

        config_path = self.data_dir / modal_config_file
        if not config_path.is_file():
            raise ValueError(f"Missing modal_config timing provenance: {config_path}")
        self.modal_config = json.loads(config_path.read_text(encoding="utf-8"))
        self.hop_length = self._positive_integer(self.modal_config, "hop_length")
        cached_sample_rate = self._positive_integer(self.modal_config, "sample_rate")
        if cached_sample_rate != self.sample_rate:
            raise ValueError(
                f"modal_config sample_rate {cached_sample_rate} != {self.sample_rate}"
            )
        self.modal_backend = self.modal_config.get("modal_backend")
        if self.modal_backend not in ("hybrid", "stft", "cqt"):
            raise ValueError("Missing or unsupported modal_config modal_backend.")

        with open(self.data_dir.joinpath(self.meta_file), "r") as f:
            self.metadata = json.load(f)

        # Keep deterministic ordering before filtering/splitting.
        self.file_list = sorted(self.metadata.keys())

        # Assign splits before filtering to preserve membership across subsets.
        if split is not None:
            self.file_list = self._split_within_pack(
                self.file_list, split, self.split_train_ratio, self.split_val_ratio
            )

        # Filters select a subset of the already fixed split.
        if sample_types is not None:
            self.file_list = [
                k
                for k in self.file_list
                if (
                    "type" in self.metadata[k]
                    and self.metadata[k]["type"] in sample_types
                )
            ]
        if instruments is not None:
            self.file_list = [
                k
                for k in self.file_list
                if (
                    "instrument" in self.metadata[k]
                    and self.metadata[k]["instrument"] in instruments
                )
            ]
        if sample_pack_keys is not None:
            allowed_pack_keys = set(sample_pack_keys)
            self.file_list = [
                k
                for k in self.file_list
                if (
                    self.sample_pack_key(self.metadata[k]) in allowed_pack_keys
                    or self.top_level_pack(self.metadata[k]) in allowed_pack_keys
                )
            ]

        # Cache lengths for bucketing.
        self.lengths = []
        for k in self.file_list:
            item = self.metadata[k]
            self.lengths.append(self._positive_integer(item, "num_samples"))

    def __len__(self):
        return len(self.file_list)

    @staticmethod
    def sample_pack_key(item: dict) -> str:
        """Return the recorded pack identifier used for split membership."""
        return _sample_pack_key(item)

    @staticmethod
    def top_level_pack(item: dict) -> str:
        """Select an entire collection when an existing config names its root."""
        relative = str(item.get("orig_relpath", "")).replace("\\", "/").strip("/")
        if "/" in relative:
            return relative.split("/", 1)[0]
        key = str(item.get("sample_pack_key") or "__root__").replace("\\", "/")
        return key.strip("/").split("/", 1)[0]

    def _split_within_pack(
        self,
        keys: List[str],
        split: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> List[str]:
        selected = _split_within_sample_packs(
            self.metadata, keys, split, self.seed, train_ratio, val_ratio
        )

        log.info(
            "AudioWithParametersDataset split='%s' strategy='sample_pack' size=%d",
            split,
            len(selected),
        )
        return selected

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, int]:
        key = self.file_list[idx]
        item = self.metadata[key]

        wav_path = self._audio_path(item)
        waveform, sr = torchaudio.load(str(wav_path))
        assert sr == self.sample_rate, (
            f"Sample rate mismatch: {sr} != {self.sample_rate}"
        )

        # Mono-only policy: always use channel 0.
        waveform = waveform[:1, :]
        length = int(waveform.shape[-1])

        if self.parameter_key not in item:
            raise KeyError(f"metadata missing '{self.parameter_key}' for key={key}")

        p_path = self.data_dir.joinpath(item[self.parameter_key])
        params = torch.load(p_path, map_location="cpu", weights_only=True)
        if params.ndim != 3:
            raise ValueError(
                f"Expected params [P,M,F], got {tuple(params.shape)} from {p_path}"
            )
        self._validate_cache_item(key, item, length, params)

        if self.expected_num_modes is not None:
            m_expected = int(self.expected_num_modes)
            p_dim, m_dim, f_dim = params.shape
            if m_dim > m_expected:
                params = params[:, :m_expected, :]
            elif m_dim < m_expected:
                pad = params.new_zeros((p_dim, m_expected - m_dim, f_dim))
                params = torch.cat([params, pad], dim=1)

        return waveform, params, length

    @staticmethod
    def _positive_integer(record: dict, field: str) -> int:
        """Reject missing or coerced timing values instead of inventing a clock."""
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(
                f"{field} must be a recorded positive integer; got {value!r}"
            )
        return int(value)

    def _validate_cache_item(
        self, key: str, item: dict, audio_length: int, params: torch.Tensor
    ) -> None:
        """Validate stored clocks before any mode-count adaptation or rendering."""
        recorded_length = self._positive_integer(item, "num_samples")
        analysis_length = self._positive_integer(item, "analysis_num_samples")
        if audio_length != recorded_length:
            raise ValueError(
                f"{key}: audio num_samples={audio_length}, cache records {recorded_length}"
            )
        if analysis_length < recorded_length:
            raise ValueError(
                f"{key}: analysis_num_samples={analysis_length} < num_samples={recorded_length}"
            )
        if analysis_length != recorded_length and not self.preserve_analysis_clock:
            raise ValueError(
                f"{key}: analysis_num_samples={analysis_length} != num_samples="
                f"{recorded_length}; re-extract this cache with the current "
                "extractor to preserve the modal time grid."
            )
        if item.get("modal_backend") != self.modal_backend:
            raise ValueError(f"{key}: modal_backend differs from modal_config.")
        expected_frames = analysis_length // self.hop_length + 1
        frames = params.shape[-1]
        if frames != expected_frames:
            raise ValueError(
                f"{key}: modal frame count={frames}, expected {expected_frames} "
                f"from analysis_num_samples={analysis_length}, hop_length={self.hop_length}"
            )

    def _audio_path(self, item: dict) -> Path:
        if self.audio_dir is not None and item.get("orig_relpath"):
            audio_path = self.audio_dir.joinpath(item["orig_relpath"])
            if audio_path.exists():
                return audio_path
            if not self._audio_fallback_warned:
                log.warning(
                    "audio_dir/orig_relpath is missing; falling back to dataset audio. "
                    "Missing example: %s",
                    audio_path,
                )
                self._audio_fallback_warned = True
        return self.data_dir.joinpath(item["filename"])
