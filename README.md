# STAMP

Percussion audio reconstruction in PyTorch.

STAMP reconstructs variable-length, mono percussion recordings at 48 kHz using
joint modal, noise-band and transient synthesis.

## Model

- A hybrid CQT/STFT analyzer extracts reference frequency, amplitude and phase tracks.
- A waveform encoder and pooled modal features feed a shared temporal representation.
- Bounded frequency and amplitude corrections refine the modal tracks.
- A 256-band noise synthesizer uses the shared representation and corrected modal context.
- A small temporal convolutional network produces a modal-conditioned transient.

The reconstruction objective combines multi-resolution spectral and envelope
losses.

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

Preprocessing defaults to the highest-RMS channel, 48 kHz, and a 20-second
maximum duration. It trims leading silence at −65 dBFS using 256-sample frames
and keeps the trailing decay. Relative pack/subfolder paths are preserved.
Preprocessing and modal extraction write IEEE float32 WAVs with SoundFile.
The analyzer records physical waveform lengths and analysis extents separately.
Dataset splits are deterministic within each pack.

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
Exports preserve waveform amplitudes in float32. The default evaluator clips
both signals to `[-1, 1]` before scoring, as in the epoch-135 comparison.

### Paired training with DrumBlender OG

Use the STAMP trainer for both models. Install both packages in the same
environment and extract both caches from the same `stamp preprocess` output.
Use STAMP's metadata to freeze the same splits in the OG cache.

```bash
python -m pip install -e ".[train,wandb]" -e "$OG_REPO[modal]"
stamp extract --processed_root "$PROCESSED_ROOT" --out_dir "$STAMP_CACHE" \
  --modal_backend hybrid --num_modes 64 --seed 20261106
python "$OG_REPO/scripts/extract_baseline_modal.py" --processed-root "$PROCESSED_ROOT" \
  --out-dir "$OG_CACHE" --reference-metadata "$STAMP_CACHE/metadata.json" \
  --analysis-method original --num-modes 64 --seed 20261106
COMMON=(--data-dir "$STAMP_CACHE" --og-data-dir "$OG_CACHE" --audio-dir "$PROCESSED_ROOT"
        --split-seed 20261106 --seed 20261106
        --wandb-project percussion --wandb-group paired-20261106)
stamp train "${COMMON[@]}" --noise-bank assets/noise_bank_48k.pt --output outputs/stamp
stamp train "${COMMON[@]}" --model drumblender_og \
  --og-config "$OG_REPO/cfg/05_all_parallel_variable.yaml" --output outputs/drumblender_og
```

Both runs share the Adam schedule, gradient clipping and CSV/W&B setup. OG
optimizes its configured auraloss MSS on valid, unnormalized waveforms. The
variable-length loss crops batch padding and averages independent clip losses.
STAMP optimizes its peak-normalized spectral and envelope objective.
Both models report the STAMP loss, spectral and envelope terms on
target-peak-normalized waveforms in the shared train and validation panels.
OG uses its own MSS for optimization, scheduling and checkpoint selection
and also reports it under `train/mss_epoch` and `validation/og_mss`.
`validation/mss` measures per-clip auraloss MSS at the original waveform
scale for the paired comparison. The step curves record the rank-averaged
final microbatch at Lightning's logging interval.

The sampler shuffles by run seed and epoch, then shards across ranks. It keeps
complete optimizer batches, so a shuffled remainder smaller than one effective
batch is omitted each epoch. Validation uses fixed per-clip noise and counts
each clip once. Keep `--split-seed` fixed when varying `--seed`. For CSV-only
logging, omit the W&B flags. With W&B enabled, both models publish the
first train and validation batches as audition audio at epoch end. Each file
alternates the target and full reconstruction for every clip in the batch.
Both models log train RMS and validation RMS, peak and tail measurements
on target-peak-normalized valid waveform samples.

Export paired checkpoints against the same processed WAV tree and evaluate both
bundles together. Use the matching modal cache for each checkpoint.

```bash
stamp reconstruct --checkpoint "$STAMP_CKPT" --noise-bank assets/noise_bank_48k.pt \
  --data-dir "$STAMP_CACHE" --audio-dir "$PROCESSED_ROOT" --output outputs/test/stamp
stamp reconstruct --model drumblender_og --checkpoint "$OG_CKPT" \
  --og-config "$OG_REPO/cfg/05_all_parallel_variable.yaml" \
  --data-dir "$OG_CACHE" --audio-dir "$PROCESSED_ROOT" --output outputs/test/drumblender_og
stamp evaluate outputs/test
```

For the epoch-135 comparison, use the original `processed` WAV tree, STAMP's
`processed_modal_window_v2_64_2gpu` cache and OG's
`drumblender_og_cqt64_original` cache. The epoch-135 run loaded WAVs from the
STAMP cache itself. On the reference server those WAVs decode to the same
samples as the corresponding files in `processed`.

To replay the archived OG epoch-54 checkpoint, use its test cache and source
from `eval/baseline_og_epoch054`. Its test feature tensors differ from those in
the full OG cache, even though the waveform samples and split agree.

Use `--config stamp/configs/epoch135.yaml` to carry the verified seed, device,
batch, accumulation and optimizer settings into a new run. Paired mode uses
the shared sampler described above. To replay the original training schedule,
use `scripts.train_joint_resynthesis` and `cfg/joint_resynthesis.yaml` from the
original source revision `0bec54b626b46720489668a7785bba22c028585b`.

Default configurations are in [`stamp/configs`](stamp/configs). Data, weights,
rendered audio and reports are not distributed with the package.

## JumpReLU SAE

Train a sparse dictionary on the existing joint model's shared 256-channel
latent. The SAE is trained from unlabeled audio on the original splits. Use
the same checkpoint, modal cache, and noise bank as reconstruction. Install the
`train` extra to load the frozen STAMP checkpoint during cache and rendering.

```sh
stamp sae cache --checkpoint CHECKPOINT.ckpt --data-dir data/modal --noise-bank assets/noise_bank_48k.pt --cache outputs/sae/cache
stamp sae train --cache outputs/sae/cache --output outputs/sae/run
stamp sae sparsity --cache outputs/sae/cache --sae outputs/sae/run/lambda-0.001/sae.pt outputs/sae/run/lambda-0.003/sae.pt outputs/sae/run/lambda-0.01/sae.pt --output outputs/sae/sparsity-val.json
stamp sae audition --checkpoint CHECKPOINT.ckpt --data-dir data/modal --noise-bank assets/noise_bank_48k.pt --cache outputs/sae/cache --sae outputs/sae/run/lambda-0.003/sae.pt --output outputs/sae/audition
```

Add `--audio-dir data/processed` to the cache and render commands if the cache
does not store the WAVs. This explicitly loads WAVs by their original relative
paths. `--device auto` uses CUDA when available. Cache splits are written
individually. Use `--splits val test` to complete remaining splits after a
train-only cache run. Each cache is sized by the number of valid frames
(1 KiB per frame) plus its index.

Training draws uniform clips and weights each frame by its valid sample count.
It uses 512 framewise features, a learned threshold per feature, and latent
reconstruction plus an L0 penalty. The three default coefficients share the
same initialization and 20,000-step budget. Check validation L0 and FVU,
threshold quantiles and activation coverage before choosing the checkpoint.

Open `outputs/sae/audition/index.html`. Every candidate includes high-activation
and random hits. In a sweep, `0x` removes that feature, `1x` is neutral, and
`1.5x` increases its original activation by half. The audition includes direct
SAE reconstructions for checking latent fidelity separately from edit identity.

Choose one or two zero-based feature IDs by listening. Substitute them for
`FEATURE_A FEATURE_B` below. Run `compare` on validation before fixing the IDs,
checkpoint, and any PCA-axis or sweep-gain overrides for `evaluate`.

```sh
stamp sae compare --checkpoint CHECKPOINT.ckpt --data-dir data/modal --noise-bank assets/noise_bank_48k.pt --cache outputs/sae/cache --sae outputs/sae/run/lambda-0.003/sae.pt --features FEATURE_A FEATURE_B --output outputs/sae/compare
stamp sae evaluate --checkpoint CHECKPOINT.ckpt --data-dir data/modal --noise-bank assets/noise_bank_48k.pt --cache outputs/sae/cache --sae outputs/sae/run/lambda-0.003/sae.pt --features FEATURE_A FEATURE_B --output outputs/sae/test
```

The comparison bundles include the following.

- **Transfer**. Two donors per source. SAE, dense latent, and PCA edits use
  the same onset alignment, overlap and interpolation fractions. The PCA
  baseline defaults to as many leading axes as selected features. Try
  `--pca-axes` on validation to audition alternative axes.
- **Interpolation**. A bidirectional edit and aligned waveform mix with exact
  neutral endpoints on the padded timeline. Compare with dense bidirectional
  edits and the same waveform crossfade.
- **Feature sweep**. Independent edits on fixed random clips, including inactive
  clips. Use `--sweep-gains` to choose the common set of gains.
- **Sparsity**. The protocol records full-split L0, FVU and activation counts.
  To report the test sparsity curve, use `stamp sae sparsity --split test`
  with the same checkpoint list.

Open the generated `index.html` for playback. Each audio group has a shared
preview gain for headroom. Raw Float32 WAVs preserve levels and endpoints.
`protocol.json` records selections, input IDs and asset identities.
`renders.json` records raw levels and edit diagnostics.

## Layout

```text
stamp/
  cli/       data preparation, training, reconstruction and evaluation
  configs/   packaged training and metric defaults
  data/      variable-length datasets and modal metadata
  sae/       JumpReLU training and listening experiments
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
