"""Installed entry points and resources must work outside a source checkout."""

import importlib
import sys

import pytest
import yaml


def test_root_help_does_not_import_training_modules(monkeypatch, capsys):
    cli = importlib.import_module("stamp.cli")

    def unexpected_import(name):
        raise AssertionError(f"Help imported a command: {name}")

    monkeypatch.setattr(cli.importlib, "import_module", unexpected_import)
    with pytest.raises(SystemExit) as error:
        cli.main(["--help"])
    assert error.value.code == 0
    assert "reconstruct" in capsys.readouterr().out


def test_command_dispatch_preserves_process_arguments(monkeypatch):
    cli = importlib.import_module("stamp.cli")
    original = sys.argv
    received = []

    class Command:
        @staticmethod
        def main(argv):
            received.append(argv)
            assert sys.argv is original
            return 7

    monkeypatch.setattr(cli.importlib, "import_module", lambda name: Command)
    assert cli.main(["train", "--devices", "1"]) == 7
    assert received == [["--devices", "1"]]


def test_config_resources_do_not_depend_on_current_directory(tmp_path, monkeypatch):
    from stamp.configs import METRICS_CONFIG, TRAIN_CONFIG
    from stamp.metrics import load_evaluation_metrics

    monkeypatch.chdir(tmp_path)
    config = yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8"))
    assert config["training"]["devices"] == 1
    assert config["training"]["accelerator"] == "auto"
    assert "wandb" not in config
    assert METRICS_CONFIG.is_file()
    metrics, checksum = load_evaluation_metrics()
    assert "mr_stft" in metrics and len(checksum) == 64


def test_parallel_extraction_uses_caller_paths(tmp_path, monkeypatch, capsys):
    from stamp.cli.extract_parallel import main

    monkeypatch.chdir(tmp_path)
    main(
        [
            "--cpu_workers",
            "2",
            "--processed_root",
            "input",
            "--out_dir",
            "cache",
            "--dry_run",
        ]
    )
    output = capsys.readouterr().out
    assert str(tmp_path / "input") in output
    assert str(tmp_path / "cache") in output
    assert "stamp.cli.extract" in output


def test_reconstruction_help_documents_auto_device(capsys):
    from stamp.cli.reconstruct import main

    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    assert "auto" in capsys.readouterr().out


def test_parallel_extraction_defaults_to_one_cpu_worker(tmp_path, monkeypatch, capsys):
    from stamp.cli.extract_parallel import main

    monkeypatch.chdir(tmp_path)
    main(["--processed_root", "input", "--out_dir", "cache", "--dry_run"])
    output = capsys.readouterr().out
    assert "CPU 0:" in output
    assert "CUDA_VISIBLE_DEVICES" not in output
