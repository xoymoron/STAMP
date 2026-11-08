import numpy as np
import pytest
import torch
import torchaudio

import stamp.utils.audio as audio_utils


def preprocess_audio_file(
    path_factory,
    in_sr=16000,
    out_sr=16000,
    in_dur=1.0,
    out_dur=1.0,
    in_stereo=False,
    amp=1.0,
):
    # Generate a test audio file and save it
    n = int(in_dur * in_sr)
    audio = (amp * torch.sin(2 * torch.pi * 440 * torch.arange(n) / in_sr))[None]
    if in_stereo:
        audio = audio.repeat(2, 1)
    input_file = path_factory.mktemp("data") / "test_input.wav"
    torchaudio.save(input_file, audio, in_sr)

    assert audio.shape[0] == 2 if in_stereo else 1

    # Preprocess the audio file
    output_file = path_factory.mktemp("data") / "test_preprocessed.wav"
    audio_utils.preprocess_audio_file(
        input_file=input_file,
        output_file=output_file,
        sample_rate=out_sr,
        num_samples=int(out_dur * out_sr),
    )
    assert output_file.exists()
    return output_file


def test_preprocess_audio_file_noresample(tmp_path_factory):
    input_sample_rate = 16000
    target_sample_rate = 16000
    output_file = preprocess_audio_file(
        tmp_path_factory, in_sr=input_sample_rate, out_sr=target_sample_rate
    )

    # Load the preprocessed audio file
    waveform, sample_rate = torchaudio.load(output_file)
    assert sample_rate == target_sample_rate


def test_preprocess_audio_file_resample(tmp_path_factory):
    input_sample_rate = 16000
    target_sample_rate = 48000
    output_file = preprocess_audio_file(
        tmp_path_factory, in_sr=input_sample_rate, out_sr=target_sample_rate
    )

    # Load the preprocessed audio file
    waveform, sample_rate = torchaudio.load(output_file)
    assert sample_rate == target_sample_rate


def test_preprocess_audio_file_resample_stereo(tmp_path_factory):
    input_sample_rate = 16000
    target_sample_rate = 48000
    output_file = preprocess_audio_file(
        tmp_path_factory,
        in_sr=input_sample_rate,
        out_sr=target_sample_rate,
        in_stereo=True,
    )

    # Load the preprocessed audio file
    waveform, sample_rate = torchaudio.load(output_file)
    assert sample_rate == target_sample_rate
    assert waveform.shape[0] == 1


def test_preprocess_audio_file_passes_string_paths_to_torchaudio(tmp_path, monkeypatch):
    """Keep preprocessing compatible with SoX backends that reject Path objects."""
    input_file = tmp_path / "input.wav"
    output_file = tmp_path / "output.wav"
    calls = {}

    def fake_load(path):
        calls["load_path"] = path
        return torch.ones(1, 512), 16000

    def fake_save(path, waveform, sample_rate):
        calls["save_path"] = path
        calls["waveform"] = waveform
        calls["sample_rate"] = sample_rate

    monkeypatch.setattr(audio_utils.torchaudio, "load", fake_load)
    monkeypatch.setattr(audio_utils.torchaudio, "save", fake_save)

    audio_utils.preprocess_audio_file(
        input_file=input_file,
        output_file=output_file,
        sample_rate=16000,
        num_samples=512,
    )

    assert calls["load_path"] == str(input_file)
    assert calls["save_path"] == str(output_file)


def test_preprocess_audio_file_selects_the_channel_with_higher_rms(
    tmp_path_factory,
):
    sample_rate = 16000
    num_frames = sample_rate
    tone = torch.sin(2 * torch.pi * 440 * torch.arange(num_frames) / sample_rate)
    stereo = torch.stack((0.1 * tone, 0.8 * tone))
    input_file = tmp_path_factory.mktemp("data") / "stereo_input.wav"
    output_file = tmp_path_factory.mktemp("data") / "selected_channel.wav"
    torchaudio.save(input_file, stereo, sample_rate)

    audio_utils.preprocess_audio_file(
        input_file=input_file,
        output_file=output_file,
        sample_rate=sample_rate,
        num_samples=num_frames,
    )

    output, _ = torchaudio.load(output_file)
    assert output.shape == (1, num_frames)
    assert torch.sqrt(torch.mean(output.square())).item() == pytest.approx(
        0.8 / np.sqrt(2), abs=1e-3
    )


def test_preprocess_audio_file_does_not_fade_the_retained_tail(tmp_path_factory):
    sample_rate = 1000
    input_file = tmp_path_factory.mktemp("data") / "tail_input.wav"
    output_file = tmp_path_factory.mktemp("data") / "tail_output.wav"
    waveform = torch.tensor([[1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0]])
    torchaudio.save(input_file, waveform, sample_rate)

    audio_utils.preprocess_audio_file(
        input_file=input_file,
        output_file=output_file,
        sample_rate=sample_rate,
        frame_size=4,
        hop_size=4,
        min_tail_silence_ms=1.0,
        remove_end_silence=True,
    )

    output, _ = torchaudio.load(output_file)
    assert output.shape == (1, 4)
    assert output[0, -1].item() == pytest.approx(0.5, abs=1e-3)


def test_preprocess_audio_file_resample_pad(tmp_path_factory):
    input_sample_rate = 16000
    target_sample_rate = 48000
    input_duration = 1.0
    target_duration = 2.0
    output_file = preprocess_audio_file(
        tmp_path_factory,
        in_sr=input_sample_rate,
        out_sr=target_sample_rate,
        in_dur=input_duration,
        out_dur=target_duration,
    )

    # Load the preprocessed audio file
    waveform, sample_rate = torchaudio.load(output_file)
    assert sample_rate == target_sample_rate
    assert waveform.shape[1] == int(target_duration * target_sample_rate)


def test_preprocess_audio_file_resample_trim(tmp_path_factory):
    input_sample_rate = 16000
    target_sample_rate = 48000
    input_duration = 1.0
    target_duration = 0.5
    output_file = preprocess_audio_file(
        tmp_path_factory,
        in_sr=input_sample_rate,
        out_sr=target_sample_rate,
        in_dur=input_duration,
        out_dur=target_duration,
    )

    # Load the preprocessed audio file
    waveform, sample_rate = torchaudio.load(output_file)
    assert sample_rate == target_sample_rate
    assert waveform.shape[1] == int(target_duration * target_sample_rate)


def test_preprocess_audio_file_rejects_silent_sound(tmp_path_factory):
    with pytest.raises(ValueError, match="silent_all"):
        preprocess_audio_file(
            tmp_path_factory,
            amp=1e-6,
        )


def test_first_non_silent_sample_returns_correct_sample():
    # Create a fake signal with a silent start and energy in the second half
    waveform = torch.zeros(1000)
    waveform[500:] = 1.0

    # Find the first non-silent sample
    first_non_silent_sample = audio_utils.first_non_silent_sample(
        waveform, frame_size=100, hop_size=100
    )

    assert first_non_silent_sample == 500


def test_first_non_silent_sample_thresholding_works_correctly():
    # Create a fake signal with a start below the threshold
    # and second half above the threshold
    threshold_db = -20.0
    waveform = torch.zeros(1000)
    waveform[:500] = np.power(10.0, threshold_db / 20.0) * 0.99
    waveform[500:] = np.power(10.0, threshold_db / 20.0) * 1.01

    # Find the first non-silent sample
    first_non_silent_sample = audio_utils.first_non_silent_sample(
        waveform, frame_size=100, hop_size=100, threshold_db=threshold_db
    )

    assert first_non_silent_sample == 500


def test_first_non_silent_sample_below_threshold_returns_none():
    # Create a fake signal entirely below the threshold
    threshold_db = -20.0
    waveform = torch.ones(1000) * np.power(10.0, threshold_db / 20.0) * 0.99

    # Find the first non-silent sample
    first_non_silent_sample = audio_utils.first_non_silent_sample(
        waveform, frame_size=100, hop_size=100, threshold_db=threshold_db
    )

    assert first_non_silent_sample is None
