"""Exercise cache timing and valid-length contracts with real WAV/PT files."""

import json

import numpy as np
import pytest
import soundfile as sf
import torch

from stamp.data import AudioWithParametersDataset


@pytest.fixture
def modal_cache(tmp_path):
    cache = tmp_path / "cache"
    source = tmp_path / "source"
    cache.mkdir()
    source.mkdir()
    sf.write(source / "hit.wav", np.full(100, 0.25), 8000, subtype="PCM_16")
    sf.write(cache / "hit.wav", np.full(100, 0.25), 8000, subtype="PCM_16")
    params = torch.zeros(3, 2, 7)
    params[0] = 100
    params[1] = 0.25
    torch.save(params, cache / "hit.pt")
    item = {
        "filename": "hit.wav",
        "feature_file": "hit.pt",
        "orig_relpath": "hit.wav",
        "num_samples": 100,
        "analysis_num_samples": 100,
        "modal_backend": "hybrid",
    }
    (cache / "metadata.json").write_text(json.dumps({"hit": item}))
    (cache / "modal_config.json").write_text(
        json.dumps({"sample_rate": 8000, "hop_length": 16, "modal_backend": "hybrid"})
    )
    return cache, source, item


def open_cache(cache, source, **kwargs):
    return AudioWithParametersDataset(
        cache, "metadata.json", 8000, None, audio_dir=source, **kwargs
    )


def test_cache_keeps_the_waveform_and_analysis_on_the_same_clock(modal_cache):
    cache, source, _ = modal_cache
    waveform, params, length = open_cache(cache, source)[0]
    assert waveform.shape == (1, 100)
    assert params.shape == (3, 2, 7)
    assert length == 100


def test_old_analysis_padding_requires_reextraction(modal_cache):
    cache, source, item = modal_cache
    item["analysis_num_samples"] = 128
    (cache / "metadata.json").write_text(json.dumps({"hit": item}))
    torch.save(torch.ones(3, 2, 9), cache / "hit.pt")
    with pytest.raises(ValueError, match="[Rr]e-extract"):
        open_cache(cache, source)[0]


def test_replaced_source_audio_length_is_rejected(modal_cache):
    cache, source, _ = modal_cache
    dataset = open_cache(cache, source)
    sf.write(source / "hit.wav", np.ones(101), 8000)
    with pytest.raises(ValueError, match="num_samples"):
        dataset[0]


def test_wrong_cached_frame_count_is_rejected(modal_cache):
    cache, source, _ = modal_cache
    torch.save(torch.ones(3, 2, 8), cache / "hit.pt")
    with pytest.raises(ValueError, match="frame"):
        open_cache(cache, source)[0]


def test_frame_validation_uses_cache_hop_instead_of_a_default(modal_cache):
    cache, source, _ = modal_cache
    (cache / "modal_config.json").write_text(
        json.dumps({"sample_rate": 8000, "hop_length": 32, "modal_backend": "hybrid"})
    )
    torch.save(torch.ones(3, 2, 4), cache / "hit.pt")
    assert open_cache(cache, source)[0][1].shape[-1] == 4


@pytest.mark.parametrize("field", ["num_samples", "analysis_num_samples"])
def test_missing_timing_metadata_is_rejected(modal_cache, field):
    cache, source, item = modal_cache
    del item[field]
    (cache / "metadata.json").write_text(json.dumps({"hit": item}))
    with pytest.raises(ValueError, match=field):
        open_cache(cache, source)[0]


def test_missing_modal_config_is_rejected(modal_cache):
    cache, source, _ = modal_cache
    (cache / "modal_config.json").unlink()
    with pytest.raises(ValueError, match="modal_config"):
        open_cache(cache, source)[0]


@pytest.mark.parametrize("hop", [0, -1, 1.5, True])
def test_invalid_recorded_hop_is_rejected(modal_cache, hop):
    cache, source, _ = modal_cache
    (cache / "modal_config.json").write_text(
        json.dumps({"sample_rate": 8000, "hop_length": hop, "modal_backend": "hybrid"})
    )
    with pytest.raises(ValueError, match="hop_length"):
        open_cache(cache, source)[0]


def test_backend_mismatch_is_rejected(modal_cache):
    cache, source, item = modal_cache
    item["modal_backend"] = "unsupported"
    (cache / "metadata.json").write_text(json.dumps({"hit": item}))
    with pytest.raises(ValueError, match="backend"):
        open_cache(cache, source)[0]


def test_analysis_length_cannot_be_shorter_than_cached_audio(modal_cache):
    cache, source, item = modal_cache
    item["analysis_num_samples"] = 99
    (cache / "metadata.json").write_text(json.dumps({"hit": item}))
    with pytest.raises(ValueError, match="analysis_num_samples"):
        open_cache(cache, source)[0]


@pytest.mark.parametrize("target", [50, 200])
def test_modal_dataset_rejects_fixed_length_requests(modal_cache, target):
    cache, source, _ = modal_cache
    with pytest.raises(ValueError, match="num_samples"):
        AudioWithParametersDataset(
            cache, "metadata.json", 8000, target, audio_dir=source
        )


def test_modal_dataset_rejects_audio_only_normalization(modal_cache):
    cache, source, _ = modal_cache
    with pytest.raises(ValueError, match="normalize"):
        open_cache(cache, source, normalize=True)
