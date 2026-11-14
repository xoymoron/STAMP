"""Command-line entry points, loaded only when selected."""

import argparse
import importlib
import sys

COMMANDS = {
    "preprocess": "stamp.cli.preprocess",
    "extract": "stamp.cli.extract",
    "extract-parallel": "stamp.cli.extract_parallel",
    "merge-shards": "stamp.cli.merge_shards",
    "noise-bank": "stamp.cli.build_noise_bank",
    "train": "stamp.cli.train",
    "reconstruct": "stamp.cli.reconstruct",
    "evaluate": "stamp.cli.evaluate",
    "sae": "stamp.cli.sae",
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="stamp",
        description="Percussion reconstruction and SAE experiments.",
    )
    parser.add_argument("command", choices=COMMANDS)
    if not argv:
        parser.print_help()
        return 0
    command = parser.parse_args(argv[:1]).command
    return importlib.import_module(COMMANDS[command]).main(argv[1:])
