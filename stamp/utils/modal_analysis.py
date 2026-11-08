"""Modal analysis with LF CQT, complementary STFTs, and explicit observations.

This is an offline, centered analysis front end, not a causal audio callback.
Frequencies are in Hz; feature caches store radians/sample.
NumPy/SciPy perform tracking. CUDA mode offloads spectral transforms to PyTorch.
"""

from dataclasses import dataclass, field
from time import perf_counter
from typing import Optional

import numpy as np
from scipy import fft, signal

ANALYZER_VERSION = "hybrid-observed-window-v2"


def _window_gain(window, centers, sample_count):
    """Correct only the missing window mass outside the recorded signal."""
    prefix = np.r_[0.0, np.cumsum(window, dtype=np.float64)]
    left = np.asarray(centers, dtype=np.int64) - len(window) // 2
    begin = np.clip(-left, 0, len(window))
    end = np.clip(sample_count - left, 0, len(window))
    mass = prefix[end] - prefix[begin]
    return prefix[-1] / np.maximum(mass, np.finfo(np.float64).tiny)


# Peak columns. Phase is the cosine phase at the analysis frame center.
FREQUENCY, AMPLITUDE, PHASE, BANDWIDTH, CONFIDENCE, SOURCE = range(6)
SOURCE_NAMES = ("cqt", "long_stft", "short_stft", "long_stft_short_envelope")


@dataclass
class ModalAnalysisResult:
    """Tracks on the frame grid used by the cycles renderer.

    observed records matched peaks, never silent placeholders. Gaps have zero
    amplitude/confidence and carry only the last observed frequency as a safe
    placeholder. phases contains the initial sine phase,
    repeated across frames for feature compatibility, not measured frame phases.
    """

    frequencies: np.ndarray
    amplitudes: np.ndarray
    phases: np.ndarray
    observed: np.ndarray
    confidence: np.ndarray
    source: np.ndarray
    frame_times: np.ndarray
    scores: np.ndarray
    candidates_before_limit: int
    analysis_seconds: float = 0.0
    candidates_before_score_gate: int = 0
    stage_seconds: dict[str, float] = field(default_factory=dict)

    def parameters(self):
        return self.frequencies, self.amplitudes, self.phases


@dataclass
class _Track:
    frames: list = field(default_factory=list)
    peaks: list = field(default_factory=list)

    def append(self, frame, peak):
        self.frames.append(frame)
        self.peaks.append(peak)


def _wrap_phase(phase):
    return (phase + np.pi) % (2 * np.pi) - np.pi


def _hann_response(offset, size):
    """Normalized complex response of a centered, periodic Hann window."""

    def dirichlet(value):
        return (
            np.exp(-1j * np.pi * value / size) * np.sinc(value) / np.sinc(value / size)
        )

    return dirichlet(offset) + 0.5 * (dirichlet(offset - 1) + dirichlet(offset + 1))


class ModalAnalyzer:
    """Sinusoidal tracking without a harmonic or exponential-decay prior.

    Hybrid mode retains CQT below cqt_max_frequency. Its filters run on an
    anti-aliased, downsampled waveform using FFT convolution. STFTs run in frame
    blocks, avoiding several full spectrogram allocations for long samples.

    num_modes is a ceiling, not a target; None retains all valid tracks.
    threshold is an amplitude dBFS floor. The relative floor references the
    input peak, not each quiet frame. diff_threshold is a frequency percentage,
    capped by max_deviation_hz. p_kernel controls contrast scoring only: median
    subtraction never modifies the amplitudes used for synthesis.
    """

    def __init__(
        self,
        sample_rate: int,
        hop_length: int = 256,
        fmin: float = 20.0,
        n_bins: int = 240,
        bins_per_octave: int = 24,
        min_length: int = 4,
        num_modes: Optional[int] = 128,
        threshold: float = -90.0,
        diff_threshold: float = 5.0,
        p_kernel: int = 15,
        *,
        backend: str = "hybrid",
        cqt_max_frequency: float = 700.0,
        filter_scale: float = 1.0,
        short_window: int = 2048,
        long_window: int = 8192,
        relative_threshold_db: float = -80.0,
        min_prominence_db: float = 2.0,
        max_gap: int = 2,
        min_active_frames: Optional[int] = None,
        min_streak: int = 2,
        min_active_ratio: float = 0.5,
        min_track_energy: float = 0.0,
        min_relative_score_db: Optional[float] = -40.0,
        max_deviation_hz: float = 40.0,
        frequency_tolerance_hz: float = 3.0,
        max_peaks: Optional[int] = None,
        frame_block_size: int = 128,
        refine: bool = True,
        compute_device: str = "cpu",
        gpu_cqt_batch_size: int = 16,
    ):
        if sample_rate <= 0 or hop_length < 1 or fmin <= 0 or fmin >= sample_rate / 2:
            raise ValueError(
                "Require a positive sample rate/hop and 0 < fmin < Nyquist."
            )
        if n_bins < 3 or bins_per_octave < 1 or filter_scale <= 0:
            raise ValueError("Invalid CQT resolution or filter scale.")
        if backend not in {"hybrid", "stft", "cqt"}:
            raise ValueError("backend must be hybrid, stft, or cqt.")
        if num_modes is not None and num_modes < 1:
            raise ValueError("num_modes must be positive or None.")
        if (
            short_window < 16
            or long_window < short_window
            or short_window % 2
            or long_window % 2
        ):
            raise ValueError("Use even windows with 16 <= short_window <= long_window.")
        if min_length < 1 or max_gap < 0 or min_streak < 1 or frame_block_size < 1:
            raise ValueError("Invalid track duration, gap, or frame block size.")
        if min_active_frames is not None and min_active_frames < 1:
            raise ValueError("min_active_frames must be positive.")
        if not 0 <= min_active_ratio <= 1 or min_track_energy < 0:
            raise ValueError("Invalid active ratio or track energy.")
        if min_relative_score_db is not None and min_relative_score_db > 0:
            raise ValueError("min_relative_score_db must be nonpositive or None.")
        if diff_threshold <= 0 or max_deviation_hz <= 0 or frequency_tolerance_hz <= 0:
            raise ValueError("Frequency matching tolerances must be positive.")
        if p_kernel < 3 or cqt_max_frequency <= 0:
            raise ValueError("Invalid contrast kernel or CQT crossover.")
        if compute_device not in {"cpu", "cuda"}:
            raise ValueError("compute_device must be cpu or cuda.")
        if gpu_cqt_batch_size < 1:
            raise ValueError("gpu_cqt_batch_size must be positive.")
        if compute_device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested, but PyTorch cannot use a CUDA GPU."
                )
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.fmin = fmin
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.num_modes = num_modes
        self.min_length = min_length
        self.min_active_frames = (
            min_length if min_active_frames is None else min_active_frames
        )
        self.min_streak = min_streak
        self.min_active_ratio = min_active_ratio
        self.min_track_energy = min_track_energy
        self.min_relative_score_db = min_relative_score_db
        self.max_gap = max_gap
        self.threshold = threshold
        self.relative_threshold_db = relative_threshold_db
        self.min_prominence_db = min_prominence_db
        self.diff_threshold = diff_threshold / 100.0
        self.frequency_tolerance_hz = frequency_tolerance_hz
        self.max_deviation_hz = max_deviation_hz
        self.p_kernel = p_kernel | 1
        self.backend = backend
        self.cqt_max_frequency = cqt_max_frequency
        self.filter_scale = filter_scale
        self.short_window = short_window
        self.long_window = long_window
        self.frame_block_size = frame_block_size
        self.refine = refine
        self.compute_device = compute_device
        self.gpu_cqt_batch_size = gpu_cqt_batch_size
        self.max_peaks = (
            max_peaks if max_peaks is not None else max(256, 2 * (num_modes or 128))
        )
        if self.max_peaks < 1 or (num_modes is not None and self.max_peaks < num_modes):
            raise ValueError("max_peaks must be positive and at least num_modes.")
        self.fmax = min(
            fmin * 2 ** ((n_bins - 1) / bins_per_octave), sample_rate * 0.499
        )
        self._cqt_grid = fmin * 2.0 ** (np.arange(n_bins) / bins_per_octave)
        self._cqt_grid = self._cqt_grid[self._cqt_grid <= self.fmax]

    def frequencies(self):
        """CQT bin centers in Hz, including bins outside the LF crossover."""
        return self._cqt_grid.copy()

    def __call__(self, audio):
        """Map [batch, samples] to three [batch, modes, frames] arrays.

        Tensor input returns CPU float32 tensors.
        Unequal mode counts are padded with fully silent slots.
        """
        is_tensor = hasattr(audio, "detach")
        array = audio.detach().cpu().numpy() if is_tensor else np.asarray(audio)
        if array.ndim != 2 or not len(array):
            raise ValueError("Expected a nonempty batch with shape [batch, samples].")
        results = [self.analyze(row) for row in array]
        modes = max(result.frequencies.shape[0] for result in results)
        frames = len(results[0].frame_times)
        parameters = [
            np.zeros((len(results), modes, frames), np.float32) for _ in range(3)
        ]
        for batch, result in enumerate(results):
            for destination, values in zip(parameters, result.parameters()):
                destination[batch, : len(values)] = values
        if is_tensor:
            import torch

            return tuple(torch.from_numpy(values) for values in parameters)
        return tuple(parameters)

    def analyze(self, audio) -> ModalAnalysisResult:
        """Analyze a mono waveform and retain masks, sources, and timings."""
        started = perf_counter()
        stage_seconds = {}
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
            raise ValueError("Expected a nonempty, finite, mono waveform.")
        count = audio.size // self.hop_length + 1
        # align_corners=False maps frame j to this sample position in the renderer.
        # Matching that grid prevents a half-hop phase/onset offset at export.
        centers = (np.arange(count) + 0.5) * audio.size / count - 0.5
        times = centers / self.sample_rate
        frames = [np.empty((0, 6), np.float32) for _ in range(count)]
        floor = max(
            10 ** (self.threshold / 20),
            np.max(np.abs(audio)) * 10 ** (self.relative_threshold_db / 20),
        )
        if np.max(np.abs(audio)) > floor:
            if self.backend != "stft":
                stage_started = perf_counter()
                self._add_cqt_peaks(audio, centers, floor, frames)
                stage_seconds["cqt"] = perf_counter() - stage_started
            if self.backend != "cqt":
                stage_started = perf_counter()
                self._add_stft_peaks(audio, centers, floor, frames, self.long_window, 1)
                stage_seconds["long_stft"] = perf_counter() - stage_started
                stage_started = perf_counter()
                self._add_stft_peaks(
                    audio, centers, floor, frames, self.short_window, 2
                )
                stage_seconds["short_stft"] = perf_counter() - stage_started
        stage_started = perf_counter()
        frames = [self._merge_peaks(peaks) for peaks in frames]
        stage_seconds["merge_peaks"] = perf_counter() - stage_started
        stage_started = perf_counter()
        tracks = self._track_peaks(frames, times)
        stage_seconds["track_peaks"] = perf_counter() - stage_started
        stage_started = perf_counter()
        result = self._pack_tracks(tracks, times)
        stage_seconds["pack_tracks"] = perf_counter() - stage_started
        result.analysis_seconds = perf_counter() - started
        result.stage_seconds = stage_seconds
        return result

    def _add_cqt_peaks(self, audio, centers, floor, frames):
        upper = (
            self.fmax
            if self.backend == "cqt"
            else min(self.cqt_max_frequency, self.fmax)
        )
        # Guard bins give fmin a genuine left neighbor and enough background
        # context for the contrast score. They are never exported as modes.
        grid = self.fmin * 2.0 ** (
            np.arange(-(self.p_kernel // 2), self.n_bins + 1) / self.bins_per_octave
        )
        guard_upper = min(
            upper * 2 ** (1 / self.bins_per_octave), self.sample_rate * 0.499
        )
        frequencies = grid[grid <= guard_upper]
        if len(frequencies) < 3:
            return
        # Decimation leaves an anti-alias transition band above the top CQT bin.
        factor = 1
        while self.sample_rate / (factor * 2) >= 4 * frequencies[-1]:
            factor *= 2
        reduced = (
            signal.resample_poly(audio, 1, factor).astype(np.float32)
            if factor > 1
            else audio
        )
        rate = self.sample_rate / factor
        quality = self.filter_scale / np.expm1(np.log(2) / self.bins_per_octave)
        lengths = (np.ceil(quality * rate / frequencies).astype(int) // 2) * 2 + 1
        width = int(lengths.max())
        transform_size = fft.next_fast_len(len(reduced) + width - 1)
        if self.compute_device == "cuda":
            from stamp.utils.modal_gpu_frontend import cqt_coefficients

            coefficients = cqt_coefficients(
                reduced,
                rate,
                centers,
                frequencies,
                lengths,
                width,
                transform_size,
                factor,
                self.compute_device,
                self.gpu_cqt_batch_size,
            )
        else:
            spectrum = fft.fft(reduced, transform_size, workers=1)
            positions = centers / factor
            left = np.floor(positions).astype(int)
            fraction = positions - left
            coefficients = np.empty((len(centers), len(frequencies)), np.complex64)
            # One kernel at a time bounds workspace memory independently of bin count.
            for index, (frequency, length) in enumerate(zip(frequencies, lengths)):
                offsets = np.arange(length) - length // 2
                window = signal.windows.hann(length, sym=True)
                kernel = np.zeros(width, np.complex64)
                start = (width - length) // 2
                kernel[start : start + length] = (
                    2
                    * window
                    / window.sum()
                    * np.exp(2j * np.pi * frequency * offsets / rate)
                )
                convolved = fft.ifft(
                    spectrum * fft.fft(kernel, transform_size, workers=1), workers=1
                )
                start = width // 2
                # Interpolate the demodulated coefficient, not its rotating carrier.
                first = convolved[start + np.clip(left, 0, len(reduced) - 1)]
                first *= _window_gain(
                    window, np.clip(left, 0, len(reduced) - 1), len(reduced)
                )
                second = convolved[start + np.clip(left + 1, 0, len(reduced) - 1)]
                second *= _window_gain(
                    window, np.clip(left + 1, 0, len(reduced) - 1), len(reduced)
                )
                rotation = np.exp(2j * np.pi * frequency / rate)
                coefficients[:, index] = (
                    (1 - fraction) * first + fraction * second / rotation
                ) * np.exp(2j * np.pi * frequency * fraction / rate)
        for frame, values in enumerate(coefficients):
            peaks = self._peaks(values, frequencies, floor, frequencies / quality, 0)
            peaks = peaks[peaks[:, FREQUENCY] <= upper]
            frames[frame] = np.concatenate((frames[frame], peaks))

    def _add_stft_peaks(self, audio, centers, floor, frames, size, source):
        grid = fft.rfftfreq(size, 1 / self.sample_rate)
        if self.compute_device == "cuda":
            from stamp.utils.modal_gpu_frontend import stft_blocks

            blocks = stft_blocks(
                audio, centers, size, self.frame_block_size, self.compute_device
            )
        else:
            window = signal.windows.hann(size, sym=False).astype(np.float32)
            padded = np.pad(audio, (size // 2, size // 2))
            view = np.lib.stride_tricks.sliding_window_view(padded, size)
            rotation = (1 - 2 * (np.arange(len(grid)) % 2)).astype(np.float32)

            def cpu_blocks():
                for start in range(0, len(centers), self.frame_block_size):
                    stop = min(start + self.frame_block_size, len(centers))
                    locations = np.rint(centers[start:stop]).astype(int)
                    spectra = fft.rfft(view[locations] * window, axis=1, workers=1)
                    spectra *= (2 / window.sum()) * rotation
                    spectra *= _window_gain(window, locations, len(audio))[:, None]
                    yield start, locations, spectra

            blocks = cpu_blocks()
        for start, locations, spectra in blocks:
            for offset, values in enumerate(spectra):
                frame = start + offset
                error = (centers[frame] - locations[offset]) / self.sample_rate
                if source == 2:
                    # Fine frequency estimates need not impose a long envelope
                    # window on an isolated, short high-frequency resonance.
                    anchors = self._merge_peaks(frames[frame])
                    frames[frame] = self._short_envelopes(
                        values, anchors, size, floor, error
                    )
                peaks = self._peaks(
                    values, grid, floor, self.sample_rate / size, source
                )
                if len(peaks) and self.refine:
                    peaks = self._fit_stft_peaks(values, peaks, size)
                if len(peaks):
                    peaks[:, PHASE] += 2 * np.pi * peaks[:, FREQUENCY] * error
                frames[start + offset] = np.concatenate((frames[start + offset], peaks))

    def _short_envelopes(self, spectrum, peaks, size, floor, center_error):
        """Remeasure separated HF anchors without merging a close pair.

        Fit the short spectrum at the fine anchor's frequency. Its amplitude
        can be zero outside a burst even when the long window sees future/past
        energy. Those frames cease to count as observations. Crowded regions
        retain their fine-window amplitude and phase.
        """
        if not len(peaks):
            return peaks
        spacing = np.diff(peaks[:, FREQUENCY])
        nearest = np.minimum(np.r_[np.inf, spacing], np.r_[spacing, np.inf])
        isolated = np.flatnonzero(
            (nearest > 2 * self.sample_rate / size)
            & (peaks[:, FREQUENCY] >= self.cqt_max_frequency)
        )
        if not len(isolated):
            return peaks
        positions = peaks[isolated, FREQUENCY] * size / self.sample_rate
        bins = np.rint(positions).astype(int)[:, None] + np.arange(-2, 3)
        bins = np.clip(bins, 1, len(spectrum) - 2)
        basis = _hann_response(positions[:, None] - bins, size)
        coefficients = np.sum(basis.conj() * spectrum[bins], axis=1) / np.sum(
            np.abs(basis) ** 2, axis=1
        )
        peaks[isolated, AMPLITUDE] = np.abs(coefficients)
        peaks[isolated, PHASE] = np.angle(coefficients) + (
            2 * np.pi * peaks[isolated, FREQUENCY] * center_error
        )
        peaks[isolated, SOURCE] = 3
        return peaks[peaks[:, AMPLITUDE] >= floor]

    def _peaks(self, values, grid, floor, bandwidth, source):
        magnitude = np.abs(values)
        log_magnitude = 20 * np.log10(np.maximum(magnitude, 1e-12))
        indices = (
            np.flatnonzero(
                (magnitude[1:-1] > magnitude[:-2])
                & (magnitude[1:-1] >= magnitude[2:])
                & (magnitude[1:-1] >= floor)
            )
            + 1
        )
        if not len(indices):
            return np.empty((0, 6), np.float32)
        # Only candidate peaks consume the background estimate. Computing the
        # same odd-width median at those locations avoids filtering every bin.
        radius = self.p_kernel // 2
        neighbors = np.clip(
            indices[:, None] + np.arange(-radius, radius + 1),
            0,
            len(log_magnitude) - 1,
        )
        background = np.median(log_magnitude[neighbors], axis=1)
        contrast = log_magnitude[indices] - background
        keep = contrast >= self.min_prominence_db
        indices, contrast = indices[keep], contrast[keep]
        # Bound the work before complex fitting on dense, noise-like spectra.
        if len(indices) > self.max_peaks:
            keep = np.argsort(magnitude[indices])[-self.max_peaks :]
            keep = keep[np.argsort(indices[keep])]
            indices, contrast = indices[keep], contrast[keep]
        left, middle, right = (
            log_magnitude[indices - 1],
            log_magnitude[indices],
            log_magnitude[indices + 1],
        )
        denominator = left - 2 * middle + right
        shift = np.divide(
            0.5 * (left - right),
            denominator,
            out=np.zeros_like(middle),
            where=np.abs(denominator) > 1e-8,
        )
        shift = np.clip(shift, -0.5, 0.5)
        fractional = indices + shift
        frequencies = (
            grid[0] * 2 ** (fractional / self.bins_per_octave)
            if source == 0
            else fractional * (grid[1] - grid[0])
        )
        amplitudes = 10 ** ((middle - 0.25 * (left - right) * shift) / 20)
        phases = np.angle(values[indices])
        widths = np.broadcast_to(bandwidth, grid.shape)[indices]
        confidence = np.clip(contrast / 18, 0, 1)
        peaks = np.column_stack(
            (
                frequencies,
                amplitudes,
                phases,
                widths,
                confidence,
                np.full(len(indices), source),
            )
        ).astype(np.float32)
        # Range checks follow interpolation: a valid edge frequency may have
        # its largest FFT bin just outside the requested interval.
        return peaks[(frequencies >= self.fmin) & (frequencies <= self.fmax)]

    def _fit_stft_peaks(self, spectrum, peaks, size):
        """Fit complex Hann lobes, jointly for groups of up to four close peaks.

        This linear fit has no decay model. Ill-conditioned groups retain their
        observed values rather than producing large, cancelling oscillators.
        """
        positions = peaks[:, FREQUENCY] * size / self.sample_rate
        boundaries = np.r_[0, np.flatnonzero(np.diff(positions) > 4) + 1, len(peaks)]
        lengths = np.diff(boundaries)
        isolated = boundaries[:-1][lengths == 1]
        fitted_coefficients = np.zeros(len(peaks), dtype=np.complex128)
        fitted_mask = np.zeros(len(peaks), dtype=bool)
        if len(isolated):
            # Most resonances are isolated. Fit all their lobes in one array
            # instead of thousands of tiny Python/SciPy calls per sample.
            bins = np.rint(positions[isolated]).astype(int)[:, None] + np.arange(-2, 3)
            bins = np.clip(bins, 1, len(spectrum) - 2)
            basis = _hann_response(positions[isolated, None] - bins, size)
            coefficients = np.sum(basis.conj() * spectrum[bins], axis=1) / np.sum(
                np.abs(basis) ** 2, axis=1
            )
            fitted_coefficients[isolated] = coefficients
            fitted_mask[isolated] = True
        # Evaluate every small group's Hann response in a single vectorized
        # call. The individual lstsq systems and their conditioning checks are
        # unchanged; no normal equations or approximate response lookup is used.
        systems = []
        offsets = []
        for group_index in np.flatnonzero((lengths >= 2) & (lengths <= 4)):
            group = np.arange(boundaries[group_index], boundaries[group_index + 1])
            bins = np.arange(
                max(1, int(positions[group[0]]) - 2),
                min(len(spectrum) - 1, int(np.ceil(positions[group[-1]])) + 3),
            )
            offset = positions[group][None, :] - bins[:, None]
            systems.append((group, bins, offset.shape))
            offsets.append(offset.ravel())
        responses = _hann_response(np.concatenate(offsets), size) if systems else None
        cursor = 0
        for group, bins, shape in systems:
            count = shape[0] * shape[1]
            basis = responses[cursor : cursor + count].reshape(shape)
            cursor += count
            coefficients, _, _, singular = np.linalg.lstsq(
                basis, spectrum[bins], rcond=1e-4
            )
            if singular[-1] < singular[0] * 0.02:
                continue
            fitted_coefficients[group] = coefficients
            fitted_mask[group] = True
        fitted = np.abs(fitted_coefficients)
        valid = (
            fitted_mask
            & (fitted <= 2 * peaks[:, AMPLITUDE])
            & (fitted >= 0.25 * peaks[:, AMPLITUDE])
        )
        peaks[valid, AMPLITUDE] = fitted[valid]
        peaks[valid, PHASE] = np.angle(fitted_coefficients[valid])
        return peaks

    def _merge_peaks(self, peaks):
        if not len(peaks):
            return peaks
        # A coarse peak cannot merge a resolved close pair or average its two
        # frequencies. Prefer the narrower observation when views overlap.
        order = np.lexsort((-peaks[:, AMPLITUDE], peaks[:, BANDWIDTH]))
        selected = []
        suppressed = np.zeros(len(peaks), bool)
        # Bounded by the per-view max_peaks limit. Build this relation once
        # instead of allocating several arrays for every retained peak.
        suppresses = None
        if len(peaks) <= 1024:
            suppresses = (peaks[:, SOURCE, None] != peaks[None, :, SOURCE]) & (
                np.abs(peaks[:, FREQUENCY, None] - peaks[None, :, FREQUENCY])
                < 0.9 * peaks[None, :, BANDWIDTH]
            )
        for index in order:
            if suppressed[index]:
                continue
            selected.append(index)
            if suppresses is not None:
                suppressed |= suppresses[index]
            else:
                # Very large user-selected mode budgets must not allocate an
                # unbounded quadratic matrix. Use the same relation one row at
                # a time in this uncommon case.
                peak = peaks[index]
                suppressed |= (peaks[:, SOURCE] != peak[SOURCE]) & (
                    np.abs(peaks[:, FREQUENCY] - peak[FREQUENCY])
                    < 0.9 * peaks[:, BANDWIDTH]
                )
        selected = np.asarray(selected)
        if len(selected) > self.max_peaks:
            priority = peaks[selected, AMPLITUDE] ** 2 * (
                0.5 + peaks[selected, CONFIDENCE]
            )
            selected = selected[np.argsort(priority)[-self.max_peaks :]]
        return peaks[selected[np.argsort(peaks[selected, FREQUENCY])]]

    def _track_peaks(self, frames, times):
        tracks, active = [], []
        for frame, peaks in enumerate(frames):
            active = [
                index
                for index in active
                if frame - tracks[index].frames[-1] <= self.max_gap + 1
            ]
            used_peaks, used_tracks = set(), set()
            if active and len(peaks):
                last = np.array([tracks[index].peaks[-1] for index in active])
                elapsed = times[frame] - np.array(
                    [times[tracks[index].frames[-1]] for index in active]
                )
                predicted = last[:, FREQUENCY].copy()
                for row, index in enumerate(active):
                    track = tracks[index]
                    if len(track.frames) >= 2:
                        dt = times[track.frames[-1]] - times[track.frames[-2]]
                        slope = (
                            track.peaks[-1][FREQUENCY] - track.peaks[-2][FREQUENCY]
                        ) / dt
                        # Keep the old float64 scalar addition even when the
                        # clamp chooses a Python-float custom bound (NumPy 2
                        # otherwise applies different scalar promotion rules).
                        predicted[row] += np.float64(
                            min(
                                max(slope * elapsed[row], -self.max_deviation_hz),
                                self.max_deviation_hz,
                            )
                        )
                gate = np.minimum(
                    self.max_deviation_hz,
                    np.maximum(
                        self.frequency_tolerance_hz,
                        last[:, FREQUENCY] * self.diff_threshold,
                    ),
                )
                # Frequencies are sorted after _merge_peaks. Search only each
                # track's narrow gate instead of allocating a dense
                # [active tracks, frame peaks] distance matrix.
                frequencies = peaks[:, FREQUENCY]
                lower = np.searchsorted(frequencies, predicted - gate, side="right")
                upper = np.searchsorted(frequencies, predicted + gate, side="left")
                counts = upper - lower
                rows = np.repeat(np.arange(len(active)), counts)
                columns = (
                    np.concatenate(
                        [
                            np.arange(begin, end)
                            for begin, end in zip(lower, upper)
                            if begin < end
                        ]
                    )
                    if rows.size
                    else np.empty(0, dtype=int)
                )
                distance = np.abs(predicted[rows] - frequencies[columns]) / gate[rows]
                phase_error = _wrap_phase(
                    peaks[columns, PHASE]
                    - last[rows, PHASE]
                    - 2
                    * np.pi
                    * 0.5
                    * (last[rows, FREQUENCY] + peaks[columns, FREQUENCY])
                    * elapsed[rows]
                )
                costs = distance + 0.1 * np.abs(phase_error) / np.pi
                # Sort only gated edges. Each observation/track is consumed once.
                for edge in np.argsort(costs, kind="stable"):
                    row, column = int(rows[edge]), int(columns[edge])
                    index = active[row]
                    if index in used_tracks or column in used_peaks:
                        continue
                    tracks[index].append(frame, peaks[column])
                    used_tracks.add(index)
                    used_peaks.add(column)
            for index, peak in enumerate(peaks):
                if index not in used_peaks:
                    track = _Track()
                    track.append(frame, peak)
                    active.append(len(tracks))
                    tracks.append(track)
        return tracks

    def _pack_tracks(self, tracks, times):
        accepted = []
        for track in tracks:
            frames = np.asarray(track.frames)
            peaks = np.asarray(track.peaks)
            runs = np.split(frames, np.flatnonzero(np.diff(frames) != 1) + 1)
            ratio = len(frames) / (frames[-1] - frames[0] + 1)
            energy = (
                np.sum(peaks[:, AMPLITUDE] ** 2) * self.hop_length / self.sample_rate
            )
            if (
                len(frames) < self.min_active_frames
                or max(map(len, runs)) < self.min_streak
                or ratio < self.min_active_ratio
                or energy < self.min_track_energy
            ):
                continue
            score = energy * ratio * (0.5 + np.mean(peaks[:, CONFIDENCE]))
            accepted.append((score, frames, peaks))
        accepted.sort(key=lambda item: item[0], reverse=True)
        before_score_gate = len(accepted)
        if accepted and self.min_relative_score_db is not None:
            # Reject the weakest noise-like fragments relative to this sample's
            # strongest track. The loose default keeps dense cymbal structures;
            # num_modes remains a separate hard ceiling.
            floor = accepted[0][0] * 10 ** (self.min_relative_score_db / 10)
            accepted = [item for item in accepted if item[0] >= floor]
        before_limit = len(accepted)
        if self.num_modes is not None:
            accepted = accepted[: self.num_modes]
        shape = (len(accepted), len(times))
        frequency = np.zeros(shape, np.float32)
        amplitude = np.zeros(shape, np.float32)
        phase = np.zeros(shape, np.float32)
        observed = np.zeros(shape, bool)
        confidence = np.zeros(shape, np.float32)
        source = np.full(shape, -1, np.int8)
        for mode, (_, frames, peaks) in enumerate(accepted):
            measured = peaks[:, FREQUENCY].astype(np.float64)
            if self.refine and len(frames) > 2:
                dt = np.diff(times[frames])
                expected = 2 * np.pi * 0.5 * (measured[:-1] + measured[1:]) * dt
                residual = _wrap_phase(
                    np.diff(peaks[:, PHASE].astype(np.float64)) - expected
                )
                correction = residual / (2 * np.pi * dt)
                limit = np.minimum(
                    10, 0.3 * np.minimum(peaks[:-1, BANDWIDTH], peaks[1:, BANDWIDTH])
                )
                correction[np.abs(correction) > limit] = 0
                correction[dt > 1.5 * self.hop_length / self.sample_rate] = 0
                residual_phase = np.r_[0, np.cumsum(correction * dt)]
                measured += np.gradient(residual_phase, times[frames])
            # A missing observation has no measured frequency. Store the last
            # real frequency there; the synth holds each audible endpoint and
            # never sweeps through a silent placeholder.
            previous = np.searchsorted(frames, np.arange(len(times)), side="right") - 1
            frequency[mode] = measured[np.clip(previous, 0, len(frames) - 1)]
            amplitude[mode, frames] = peaks[:, AMPLITUDE]
            observed[mode, frames] = True
            confidence[mode, frames] = peaks[:, CONFIDENCE]
            source[mode, frames] = peaks[:, SOURCE].astype(np.int8)
            frequencies = frequency[mode].astype(np.float64)
            interval_frequency = np.where(
                observed[mode, :-1] & ~observed[mode, 1:],
                frequencies[:-1],
                np.where(
                    ~observed[mode, :-1] & observed[mode, 1:],
                    frequencies[1:],
                    0.5 * (frequencies[:-1] + frequencies[1:]),
                ),
            )
            integral = np.r_[
                frequencies[0] * times[0],
                np.cumsum(interval_frequency * np.diff(times))
                + frequencies[0] * times[0],
            ]
            # Match inclusive cumsum: sample 0 already contains one increment.
            rendered_phase = (
                2 * np.pi * integral
                + np.pi * (frequencies + frequencies[0]) / self.sample_rate
            )
            offsets = peaks[:, PHASE] + np.pi / 2 - rendered_phase[frames]
            weights = peaks[:, AMPLITUDE] ** 2 * (0.1 + peaks[:, CONFIDENCE])
            phase[mode] = np.angle(np.sum(weights * np.exp(1j * offsets)))
        return ModalAnalysisResult(
            frequency,
            amplitude,
            phase,
            observed,
            confidence,
            source,
            times,
            np.array([item[0] for item in accepted]),
            before_limit,
            candidates_before_score_gate=before_score_gate,
        )
