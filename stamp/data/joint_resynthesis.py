"""Load modal caches into the joint model's FP32 cycles/sample contract."""

import math

import torch

from stamp.data.audio import AudioWithParametersDataset


def cache_params_to_cycles(params: torch.Tensor) -> torch.Tensor:
    """Convert cache radians/sample once on CPU, before building a model graph.

    Amplitude and initial sine phase (radians) keep their cache meanings.
    No cache files are changed. FP64 is restricted to this loading conversion.
    """
    if params.device.type != "cpu" or params.requires_grad:
        raise ValueError("Cache conversion requires detached CPU parameters.")
    if params.ndim != 3 or params.shape[0] != 3:
        raise ValueError("Expected cache parameters [3,modes,frames].")
    result = params.to(torch.float32).clone()
    result[0] = (params[0].to(torch.float64) / (2 * math.pi)).to(torch.float32)
    return result


class JointResynthesisDataset(AudioWithParametersDataset):
    """Preserve the cache clock and return explicit timing with cycle frequencies."""

    def __init__(self, *args, **kwargs):
        kwargs["preserve_analysis_clock"] = True
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        audio, params, length = super().__getitem__(index)
        key = self.file_list[index]
        extent = self._positive_integer(self.metadata[key], "analysis_num_samples")
        # Drop only modes with exactly zero amplitude on the entire native grid.
        params = params[:, params[1].ne(0).any(dim=-1)]
        return {
            "sample_id": key,
            "audio": audio.float(),
            "params": cache_params_to_cycles(params),
            "length": length,
            "analysis_num_samples": extent,
            "frame_length": params.shape[-1],
            "mode_count": params.shape[-2],
        }
