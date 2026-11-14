import pytest
import torch


def test_jumprelu_threshold_and_count_gradients():
    from stamp.sae.model import activation

    z = torch.tensor([[0.75, 1.0, 1.25, 2.0]], requires_grad=True)
    threshold = torch.ones(4, requires_grad=True)
    c, active = activation(z, threshold, 1.0)
    torch.testing.assert_close(c, torch.tensor([[0.0, 0.0, 1.25, 2.0]]))
    torch.testing.assert_close(active, torch.tensor([[0.0, 0.0, 1.0, 1.0]]))
    (c.sum() + 2 * active.sum()).backward()
    torch.testing.assert_close(z.grad, torch.tensor([[0.0, 0.0, 1.0, 1.0]]))
    torch.testing.assert_close(threshold.grad, torch.tensor([-3.0, -3.0, -3.0, 0.0]))


def test_clip_weighting_and_checkpoint_round_trip(tmp_path):
    from stamp.sae.corpus import Corpus, write_split
    from stamp.sae.model import fit, measure, read_sae

    path = tmp_path / "cache"
    records = [
        ("a", torch.full((1, 256), 1.0), 64, 0),
        ("b", torch.full((2, 256), 3.0), 160, 1),
    ]
    write_split(path, "train", records)
    write_split(path, "val", records)
    train, val = Corpus(path, "train"), Corpus(path, "val")
    mean, scale = train.normalization()
    torch.testing.assert_close(mean, torch.full((256,), 2.0))
    torch.testing.assert_close(scale, torch.tensor(16.0))
    model, log = fit(train, val, width=8, steps=3, batch_size=7, seed=21, penalty=0.003)
    from stamp.sae.model import save_sae

    save_sae(tmp_path / "sae.pt", model, {"seed": 21})
    loaded, metadata = read_sae(tmp_path / "sae.pt", "cpu")
    assert metadata == {"seed": 21}
    assert log[-1]["step"] == 3
    result = measure(loaded, val)
    assert 0 <= result["mean_l0"] <= 8
    assert result["clips"] == 2
    assert len(result["feature_clip_counts"]) == 8
    h = records[1][1]
    assert torch.equal(model.encode(h), loaded.encode(h))
    assert torch.equal(loaded.edit(h, torch.zeros(2, 8)), h)


def test_alignment_and_morph_use_original_clocks():
    from stamp.sae.experiment import aligned_delta, mix_aligned, onset_frame

    a = torch.tensor([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]])
    b = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
    torch.testing.assert_close(
        aligned_delta(a, b, 1, 0)[:, 0], torch.tensor([0.0, 8.0, 17.0, 0.0])
    )
    assert onset_frame(torch.zeros(17)) == 0
    assert onset_frame(torch.cat([torch.zeros(128), torch.ones(3)])) == 1
    raw_a = torch.ones(129)
    raw_b = torch.full((2,), 2.0)
    expected_a = raw_a
    expected_b = torch.nn.functional.pad(raw_b, (128, 0))
    torch.testing.assert_close(mix_aligned(raw_a, raw_b, 1, 0, 1.0), expected_b)
    torch.testing.assert_close(
        mix_aligned(raw_a, raw_b, 1, 0, 0.0),
        torch.nn.functional.pad(expected_a, (0, 1)),
    )


def test_sae_workflow_available_in_root_cli(capsys):
    from stamp.cli import main

    with pytest.raises(SystemExit) as error:
        main(["sae", "--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    for name in ("cache", "train", "sparsity", "audition", "evaluate"):
        assert name in help_text


def test_sae_cli_from_frozen_checkpoint_to_held_out_audio(tmp_path):
    import hashlib
    from pathlib import Path

    import pytorch_lightning as pl
    import soundfile as sf

    from stamp.cli import main
    from stamp.sae.corpus import Corpus
    from stamp.tasks.joint_resynthesis import JointResynthesisTask
    from stamp.utils.io import read_json, write_json

    torch.set_num_threads(2)
    torch.manual_seed(20261106)
    data = tmp_path / "modal"
    data.mkdir()
    write_json(
        data / "modal_config.json",
        {
            "sample_rate": 48000,
            "hop_length": 128,
            "modal_backend": "hybrid",
        },
    )
    metadata = {}
    for i in range(8):
        sample_id = f"hit-{i}"
        length = 129 + i * 39
        audio = 0.22 * (torch.arange(length) * (i + 2) * 0.013).sin()
        audio[: (i % 2) * 128] = 0
        sf.write(data / f"{sample_id}.wav", audio.numpy(), 48000, subtype="FLOAT")
        frames = length // 128 + 1
        params = torch.zeros(3, 1, frames)
        params[0] = (i + 2) * 0.013
        params[1] = 0.2
        torch.save(params, data / f"{sample_id}.pt")
        metadata[sample_id] = {
            "filename": f"{sample_id}.wav",
            "feature_file": f"{sample_id}.pt",
            "orig_relpath": f"{sample_id}.wav",
            "sample_pack_key": "test-pack",
            "num_samples": length,
            "analysis_num_samples": length,
            "modal_backend": "hybrid",
        }
    # Use a synthetic checkpoint with the real frozen model and the existing split code.
    bank = torch.zeros(1, 65536).expand(256, -1)
    bank_path = tmp_path / "noise.pt"
    checksum = hashlib.sha256(bank.contiguous().numpy()).hexdigest()
    torch.save(
        {"bank": bank, "metadata": {"version": 1, "sample_rate": 48000}}, bank_path
    )
    reconstruction = JointResynthesisTask(bank, checksum, seed=20261106)
    checkpoint = tmp_path / "recon.ckpt"
    torch.save(
        {
            "state_dict": reconstruction.state_dict(),
            "hyper_parameters": dict(reconstruction.hparams),
            "pytorch-lightning_version": pl.__version__,
        },
        checkpoint,
    )
    # 30 clips yield nonempty train, validation and test splits, including two donors.
    template = list(metadata.values())
    for i in range(8, 30):
        metadata[f"hit-{i}"] = template[i % 8]
    write_json(data / "metadata.json", metadata)

    cache = tmp_path / "cache"
    base = ["--cache", str(cache), "--device", "cpu"]
    assets = [
        "--checkpoint",
        str(checkpoint),
        "--noise-bank",
        str(bank_path),
        "--data-dir",
        str(data),
    ]
    run = tmp_path / "run"
    assert main(["sae", "cache", *assets, *base]) == 0
    assert (
        main(
            [
                "sae",
                "train",
                *base,
                "--output",
                str(run),
                "--width",
                "8",
                "--steps",
                "4",
                "--batch-size",
                "32",
                "--penalties",
                "0.003",
            ]
        )
        == 0
    )
    checkpoint_path = run / "lambda-0.003" / "sae.pt"
    model = ["--sae", str(checkpoint_path)]
    report = tmp_path / "sparsity.json"
    assert main(["sae", "sparsity", *base, *model, "--output", str(report)]) == 0
    assert read_json(report)["runs"][0]["clips"] == 3
    listening = tmp_path / "audition"
    assert (
        main(
            [
                "sae",
                "audition",
                *assets,
                *base,
                *model,
                "--output",
                str(listening),
                "--candidates",
                "2",
            ]
        )
        == 0
    )
    selection = read_json(listening / "protocol.json")["candidates"][0]["id"]
    for action in ("compare", "evaluate"):
        out = tmp_path / action
        assert (
            main(
                [
                    "sae",
                    action,
                    *assets,
                    *base,
                    *model,
                    "--features",
                    str(selection),
                    "--triples",
                    "1",
                    "--sweep-clips",
                    "2",
                    "--output",
                    str(out),
                ]
            )
            == 0
        )
        protocol = read_json(out / "protocol.json")
        assert protocol["features"] == [selection]
        assert protocol["split"] == ("test" if action == "evaluate" else "val")
        draw = protocol["triples"][0]
        assert set(draw) == set(Corpus(cache, protocol["split"]).ids)
        groups = read_json(out / "renders.json")
        assert len(groups) == 5
        assert (out / "index.html").is_file()
        for group in groups[:2]:
            entries = {row["label"]: row for row in group["clips"]}
            for fraction in (0, 1):
                raw = Path(entries[f"SAE {fraction}"]["raw"])
                expected = Path(entries[f"Crossfade {fraction}"]["raw"])
                assert sf.info(out / raw).subtype == "FLOAT"
                assert (sf.read(out / raw)[0] == sf.read(out / expected)[0]).all()


def test_temporal_audio_diagnostics_use_reference_onset():
    from stamp.sae.experiment import waveform_stats

    audio = torch.zeros(5000)
    audio[128:256] = 1
    audio[1128:1256] = 1
    stats = waveform_stats(audio, 1)
    assert stats["early_20ms_energy_fraction"] == 0.5
    assert stats["energy_95_ms"] >= 20
    assert waveform_stats(torch.zeros_like(audio), 1)["energy_95_ms"] is None
