"""Small, resumable summaries for offline modal extraction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_json_atomic(path: Path, value: dict, *, indent: int | None = None) -> None:
    """Keep the previous readable snapshot until the replacement is complete."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=indent), encoding="utf-8"
    )
    temporary.replace(path)


def result_details(result, crossover_hz: float) -> dict:
    """Count observed modes and candidates before the score and slot gates."""
    mode_count = len(result.frequencies)
    observed = result.observed
    medians = [
        float(np.median(result.frequencies[mode, observed[mode]]))
        for mode in range(mode_count)
    ]
    sources = np.zeros(4, dtype=int)
    for mode in range(mode_count):
        values = result.source[mode, observed[mode]]
        if values.size:
            sources[int(np.bincount(values, minlength=4).argmax())] += 1
    return {
        "modal_candidates_pre_score": int(result.candidates_before_score_gate),
        "modal_candidates_pre_limit": int(result.candidates_before_limit),
        "modal_score_rejected": int(
            result.candidates_before_score_gate - result.candidates_before_limit
        ),
        "modal_limit_rejected": int(result.candidates_before_limit - mode_count),
        "modal_lf_modes": int(sum(frequency < crossover_hz for frequency in medians)),
        "modal_hf_modes": int(sum(frequency >= crossover_hz for frequency in medians)),
        "modal_observed_frames": int(observed.sum()),
        "modal_mean_confidence": (
            float(result.confidence[observed].mean()) if observed.any() else 0.0
        ),
        "modal_source_modes": {
            name: int(count)
            for name, count in zip(
                ("cqt", "long_stft", "short_stft", "long_stft_short_envelope"),
                sources,
            )
        },
        "analysis_seconds": round(result.analysis_seconds, 6),
        "analysis_stage_seconds": {
            name: round(seconds, 6) for name, seconds in result.stage_seconds.items()
        },
    }


def summarize_metadata(
    metadata: dict, mode_ceiling: int | None = None, sample_rate: int = 48000
) -> dict:
    """Summarize completed files, including files loaded by --resume."""
    items = list(metadata.values())
    modes = sorted(int(item.get("active_modes", 0)) for item in items)

    def percentile(percent: float) -> int | None:
        if not modes:
            return None
        return modes[min(len(modes) - 1, int((len(modes) - 1) * percent))]

    stage_seconds: dict[str, float] = {}
    source_modes: dict[str, int] = {}
    for item in items:
        for name, seconds in item.get("analysis_stage_seconds", {}).items():
            stage_seconds[name] = stage_seconds.get(name, 0.0) + seconds
        for name, count in item.get("modal_source_modes", {}).items():
            source_modes[name] = source_modes.get(name, 0) + count
    return {
        "completed_files": len(items),
        "files_with_stage_timings": sum(
            "analysis_stage_seconds" in item for item in items
        ),
        "files_with_lf_hf_counts": sum("modal_lf_modes" in item for item in items),
        "zero_mode_files": sum(mode == 0 for mode in modes),
        "files_at_mode_ceiling": (
            sum(mode == mode_ceiling for mode in modes)
            if mode_ceiling is not None
            else None
        ),
        "active_modes_total": sum(modes),
        "active_modes_mean": round(sum(modes) / len(modes), 3) if modes else None,
        "active_modes_p50": percentile(0.5),
        "active_modes_p90": percentile(0.9),
        "active_modes_p99": percentile(0.99),
        "lf_modes_total": sum(item.get("modal_lf_modes", 0) for item in items),
        "hf_modes_total": sum(item.get("modal_hf_modes", 0) for item in items),
        "score_rejected_total": sum(
            item.get("modal_score_rejected", 0) for item in items
        ),
        "limit_rejected_total": sum(
            item.get("modal_limit_rejected", 0) for item in items
        ),
        "source_modes_total": source_modes,
        "audio_hours": round(
            sum(item.get("num_samples", 0) for item in items) / sample_rate / 3600,
            3,
        ),
        "analysis_hours": round(
            sum(item.get("analysis_seconds", 0.0) for item in items) / 3600, 3
        ),
        "file_processing_hours": round(
            sum(item.get("file_seconds", 0.0) for item in items) / 3600, 3
        ),
        "analysis_stage_hours": {
            name: round(seconds / 3600, 3)
            for name, seconds in sorted(stage_seconds.items())
        },
    }
