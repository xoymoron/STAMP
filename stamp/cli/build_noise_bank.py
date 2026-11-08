"""Build the fixed noise bank once before launching reconstruction training."""

import argparse
from pathlib import Path

import torch

from stamp.synths.noise_band import build_noise_bank


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as file:
        torch.save(build_noise_bank(args.seed), file)


if __name__ == "__main__":
    main()
