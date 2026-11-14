"""Donor edits, two-reference morphs and offline listening bundles."""

import html
import random
import tempfile
from pathlib import Path

import soundfile as sf
import torch
from torch.nn import functional as F

from stamp.synths.noise_band import HOP, SAMPLE_RATE
from stamp.utils.io import write_json

FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
SWEEP_GAINS = (0.0, 0.5, 1.0, 1.25, 1.5)


def onset_frame(audio):
    audio = audio.reshape(-1)
    blocks = F.pad(audio, (0, (-audio.numel()) % HOP)).reshape(-1, HOP)
    energy = blocks.double().square().sum(dim=1)
    return int((energy >= 0.01 * energy.max()).to(torch.int64).argmax())


def aligned_delta(source, donor, source_onset, donor_onset):
    shift = source_onset - donor_onset
    start = max(0, shift)
    stop = min(len(source), len(donor) + shift)
    delta = torch.zeros_like(source)
    if stop > start:
        delta[start:stop] = donor[start - shift : stop - shift] - source[start:stop]
    return delta


def mix_aligned(a, b, onset_a, onset_b, fraction):
    common_onset = max(onset_a, onset_b)
    left_a = (common_onset - onset_a) * HOP
    left_b = (common_onset - onset_b) * HOP
    length = max(len(a) + left_a, len(b) + left_b)
    a = F.pad(a, (left_a, length - left_a - len(a)))
    b = F.pad(b, (left_b, length - left_b - len(b)))
    if fraction == 0:
        return a
    if fraction == 1:
        return b
    return (1 - fraction) * a + fraction * b


def feature_delta(model, a, b, onset_a, onset_b, features):
    dc = aligned_delta(a[:, features], b[:, features], onset_a, onset_b)
    return model.scale * (dc @ model.decoder.weight[:, features].T)


@torch.no_grad()
def candidates(model, corpus, seed, count=10):
    coverage = torch.zeros(model.width, dtype=torch.long)
    scores = torch.zeros((len(corpus), model.width))
    for i, sample_id in enumerate(corpus.ids):
        h = corpus.get(sample_id).to(model.mean.device)
        maximum = model.encode(h).amax(dim=0).cpu()
        scores[i] = maximum
        coverage += maximum.gt(0)
    # Sample across different activation frequencies, using no rendered outputs.
    eligible = [i for i in range(model.width) if coverage[i] >= min(3, len(corpus))]
    if not eligible:
        eligible = coverage.nonzero().flatten().tolist()
    if not eligible:
        raise ValueError("No feature activates in the validation cache.")
    eligible.sort(key=lambda i: (int(coverage[i]), i))
    count = min(count, len(eligible))
    rng = random.Random(seed)
    result = []
    for index in range(count):
        start = index * len(eligible) // count
        stop = (index + 1) * len(eligible) // count
        feature = rng.choice(eligible[start:stop])
        active = [i for i, value in enumerate(scores[:, feature]) if value > 0]
        strongest = sorted(active, key=lambda i: (-float(scores[i, feature]), i))[:3]
        remaining = [i for i in range(len(corpus)) if i not in strongest]
        random_ids = rng.sample(remaining, min(3, len(remaining)))
        ids = [corpus.ids[i] for i in strongest + random_ids]
        result.append((feature, int(coverage[feature]), ids, len(strongest)))
    return result


def sample_protocol(corpus, seed, triples_count, sweep_count):
    rng = random.Random(seed)
    draw = rng.sample(corpus.ids, triples_count * 3)
    triples = [draw[i : i + 3] for i in range(0, len(draw), 3)]
    return triples, rng.sample(corpus.ids, sweep_count)


def paired_edits(runtime, model, a, b, features, axes):
    def directions(source, donor):
        delta = aligned_delta(source.h, donor.h, source.onset, donor.onset)
        edits = {
            "SAE": feature_delta(
                model, source.code, donor.code, source.onset, donor.onset, features
            ),
            "Dense": delta,
        }
        if axes is not None:
            edits["PCA"] = (delta @ axes) @ axes.T
        return edits

    forward, reverse = directions(a, b), directions(b, a)
    transfers = []
    morphs = [
        (
            f"Crossfade {fraction:g}",
            mix_aligned(a.neutral, b.neutral, a.onset, b.onset, fraction),
        )
        for fraction in FRACTIONS
    ]
    diagnostics = {}
    for method, delta in forward.items():
        diagnostics[method] = {
            "delta_h_rms": delta.double().square().mean().sqrt().item(),
            "unchanged": bool(delta.eq(0).all()),
        }
        for fraction in FRACTIONS:
            y_a = (
                a.neutral
                if fraction == 0 or diagnostics[method]["unchanged"]
                else runtime.render(a, a.h + fraction * delta)
            )
            transfers.append((f"{method} {fraction:g}", y_a))
            if method == "PCA":
                continue
            y_b = (
                b.neutral
                if fraction in (0, 1)
                else runtime.render(b, b.h + (1 - fraction) * reverse[method])
            )
            morphs.append(
                (
                    f"{method} {fraction:g}",
                    mix_aligned(y_a, y_b, a.onset, b.onset, fraction),
                )
            )
    return transfers, morphs, diagnostics


def sweep(runtime, model, shot, feature, gains):
    contribution = (
        model.scale * shot.code[:, feature, None] * model.decoder.weight[:, feature]
    )
    return [
        (
            f"{gain:g}x",
            shot.neutral
            if gain == 1 or not contribution.any()
            else runtime.render(shot, shot.h + (gain - 1) * contribution),
        )
        for gain in gains
    ]


def waveform_stats(audio, onset):
    power = audio.detach().cpu().double().square()
    post_onset = power[onset * HOP :]
    total = post_onset.sum()
    early = post_onset[: SAMPLE_RATE // 50].sum()
    end = torch.searchsorted(post_onset.cumsum(0), 0.95 * total)
    return {
        "rms": power.mean().sqrt().item(),
        "peak": power.max().sqrt().item(),
        "early_20ms_energy_fraction": (early / total).item() if total else None,
        "energy_95_ms": 1000 * end.item() / SAMPLE_RATE if total else None,
    }


class ListeningBundle:
    def __init__(self, output):
        self.output = Path(output)
        if self.output.exists():
            raise FileExistsError(f"Output already exists: {self.output}")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.groups = []

    def __enter__(self):
        self._temp = tempfile.TemporaryDirectory(dir=self.output.parent)
        self.path = Path(self._temp.name) / self.output.name
        self.path.mkdir()
        (self.path / "raw").mkdir()
        (self.path / "preview").mkdir()
        return self

    def __exit__(self, *error):
        self._temp.cleanup()

    def add(self, title, clips, details=None, *, onset=0):
        waves = [audio.detach().cpu().numpy() for _, audio in clips]
        if any(not torch.isfinite(torch.as_tensor(wave)).all() for wave in waves):
            raise ValueError(f"Non-finite waveform in {title}.")
        peak = max(float(abs(wave).max()) for wave in waves)
        gain = min(1.0, 0.95 / peak) if peak else 1.0
        rows = []
        for index, ((label, audio), wave) in enumerate(zip(clips, waves)):
            name = f"{len(self.groups):04d}-{index:02d}.wav"
            raw = f"raw/{name}"
            preview = f"preview/{name}"
            sf.write(self.path / raw, wave, SAMPLE_RATE, subtype="FLOAT")
            sf.write(self.path / preview, wave * gain, SAMPLE_RATE, subtype="PCM_16")
            rows.append(
                {
                    "label": label,
                    "raw": raw,
                    "preview": preview,
                    **waveform_stats(audio, onset),
                }
            )
        self.groups.append(
            {
                "title": title,
                "preview_gain": gain,
                "clips": rows,
                "details": details or {},
            }
        )

    def finish(self, protocol):
        write_json(self.path / "protocol.json", protocol)
        write_json(self.path / "renders.json", self.groups)
        sections = []
        for group in self.groups:
            cells = []
            for item in group["clips"]:
                label = html.escape(item["label"])
                preview, raw = item["preview"], item["raw"]
                cells.append(
                    f"<li><strong>{label}</strong> "
                    f'<audio controls preload="none" src="{preview}"></audio> '
                    f'<a href="{raw}">Raw WAV</a></li>'
                )
            sections.append(
                f"<section><h2>{html.escape(group['title'])}</h2>"
                f"<ul>{''.join(cells)}</ul></section>"
            )
        page = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>STAMP SAE listening</title>
<style>
body{font:16px system-ui,sans-serif;line-height:1.5;margin:2rem auto;padding:0 1rem;max-width:70rem}
section{border-top:1px solid #aaa;padding:.5rem 0} h2{font-size:1.1rem;overflow-wrap:anywhere}
ul{list-style:none;padding:0} li{display:flex;align-items:center;gap:1rem;margin:.35rem 0;flex-wrap:wrap}
li strong{min-width:10rem} audio{max-width:100%}
</style>
<h1>STAMP SAE listening</h1>
<p><a href="protocol.json">Protocol</a> · <a href="renders.json">Render details</a></p>
<p>Previews use one shared headroom gain per comparison. Raw WAVs keep original levels.</p>
"""
        (self.path / "index.html").write_text(
            page + "\n".join(sections) + "</html>\n", encoding="utf-8"
        )
        self.path.rename(self.output)
