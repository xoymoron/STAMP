# STAMP

Percussion audio reconstruction and sparse timbre control in PyTorch.

STAMP reconstructs variable-length, mono percussion recordings at 48 kHz using
joint modal, noise-band and transient synthesis. A post-hoc temporal sparse
autoencoder (SAE) learns edit directions in the frozen model's shared
representation.

## Model

- A hybrid CQT/STFT analyzer extracts reference frequency, amplitude and phase tracks.
- A waveform encoder and pooled modal features feed a shared temporal representation.
- Bounded frequency and amplitude corrections refine the modal tracks.
- A 256-band noise synthesizer uses the shared representation and corrected modal context.
- A small temporal convolutional network produces a modal-conditioned transient.

The reconstruction objective combines multi-resolution spectral and envelope
losses. SAE training is separate: the reconstruction model is frozen, and paired
SAEs share a latent-only warm-up before intervention-aware continuation.
Controls edit the temporal representation while retaining neutral strike
conditioning.

## Installation

Python 3.10–3.12:

```sh
python -m pip install -e ".[train,analysis]"
```

The base package contains the model, synthesis, preprocessing and metric code.
The `train` extra adds Lightning for training and checkpoint loading; `analysis`
adds plotting. No tracking account or machine-specific launcher is required.

```sh
stamp --help
python -m stamp --help
```

## Data preparation

Arrange single-strike WAV files by sample pack, then preprocess and analyze them:

```sh
stamp preprocess --input data/raw --output data/processed --rejected data/rejected --logs outputs/preprocess
stamp extract --processed_root data/processed --out_dir data/modal --num_modes 64
stamp noise-bank assets/noise_bank_48k.pt
```

Preprocessing defaults to mono 48 kHz, a 20-second maximum duration, leading
silence removal and retained trailing decay. Relative pack/subfolder paths are
preserved. The analyzer records physical waveform lengths and analysis extents
separately. Dataset splits are deterministic within each pack.

Parallel analysis supports CPU processes or explicitly selected CUDA devices:

```sh
stamp extract-parallel --cpu_workers 2 --processed_root data/processed --out_dir data/modal --num_modes 64
stamp extract-parallel --gpus 0,1 --processed_root data/processed --out_dir data/modal --num_modes 64
```

## Training and reconstruction

```sh
stamp train --data-dir data/modal --audio-dir data/processed --noise-bank assets/noise_bank_48k.pt --output outputs/model
stamp reconstruct --checkpoint CHECKPOINT.ckpt --noise-bank assets/noise_bank_48k.pt --data-dir data/modal --audio-dir data/processed --output outputs/test
stamp evaluate outputs/test
```

Training defaults to one automatically selected CPU/CUDA device and local CSV
logging. Override hardware with `--accelerator`, `--devices` and `--num-workers`,
or pass a custom `--config`. Reconstruction and evaluation accept
`--device auto|cpu|cuda|cuda:N`. Relative paths resolve from the current working
directory.

Reconstruction requires the checkpoint's original noise bank and matching modal
analysis configuration. Exports contain target, reconstruction and analyzed
sinusoidal WAVs; `--branches` adds component signals. Evaluation reports per-file
and aggregate spectral, onset, envelope and MFCC errors over valid clip lengths.

Default configurations are in [`stamp/configs`](stamp/configs). Data, weights,
rendered audio and reports are not distributed with the package.

## Temporal SAE control

```sh
stamp control cache --checkpoint CHECKPOINT.ckpt --noise-bank assets/noise_bank_48k.pt --data-dir data/modal --audio-dir data/processed --cache outputs/control/cache
stamp control train --cache outputs/control/cache --output outputs/control/warmup
stamp train-control --checkpoint CHECKPOINT.ckpt --noise-bank assets/noise_bank_48k.pt --data-dir data/modal --audio-dir data/processed --cache outputs/control/cache --warmup outputs/control/warmup/best.pt --output outputs/control/paired
```

The SAE uses 512 features, TopK 8 and nine-frame kernels. Calibration uses the
training split; candidate audition uses validation. Open the generated
`outputs/control/paired/audition/index.html` and select two feature indices before
held-out evaluation:

```sh
stamp evaluate-control --checkpoint CHECKPOINT.ckpt --noise-bank assets/noise_bank_48k.pt --data-dir data/modal --audio-dir data/processed --cache outputs/control/cache --run outputs/control/paired --output outputs/control/test --atoms ATOM_A ATOM_B
stamp analyze-control outputs/control/test
```

Replace `ATOM_A ATOM_B` with this run's selected indices. The signed knob maps to
`alpha(k) = sign(k) * (65**abs(k) - 1)`. Listening exports include both raw Float32
WAVs and RMS-matched copies. Strong edits can increase output level. Input and
source identities are recorded in each protocol; resume requires matching
artifacts and code.

## Layout

```text
stamp/
  cli/       data preparation, training, reconstruction and evaluation
  configs/   packaged training and metric defaults
  control/   temporal SAE, interventions, calibration and offline audition
  data/      variable-length datasets and modal metadata
  models/    waveform encoder and shared reconstruction model
  synths/    modal, noise-band and transient synthesis
  tasks/     Lightning reconstruction task
  utils/     audio, device and file helpers
tests/       numerical, data, CLI and CPU training regression tests
```

## Development

```sh
python -m pip install -e ".[dev,train,analysis]"
pytest
ruff check stamp tests
ruff format --check stamp tests
python -m pip wheel . --no-deps --wheel-dir dist
```

## Attribution

The convolutional encoder and temporal convolution components build on
DrumBlender by Shier et al., [Differentiable Modelling of Percussive Audio with
Transient and Spectral Synthesis](https://doi.org/10.61782/fa.2023.1093).
