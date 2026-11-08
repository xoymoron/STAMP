"""Float32 exports and spectral measurements for final control auditions."""

import json
from html import escape

import numpy as np
import soundfile as sf
from scipy.signal import stft


def as_wave(value, ref):
    return (value.detach() * ref.gain).flatten().cpu().numpy()


def difference_score(wave, neutral):
    normalized = matched_waves({"edit": wave, "neutral": neutral})
    scores = []
    for size in (512, 2048, 4096):
        size = min(size, len(wave))
        left = np.abs(stft(normalized["edit"], nperseg=size, noverlap=size * 3 // 4)[2])
        right = np.abs(
            stft(normalized["neutral"], nperseg=size, noverlap=size * 3 // 4)[2]
        )
        scores.append(
            float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1e-12))
        )
    return float(np.mean(scores))


def save_waves(folder, prefix, waves):
    """Keep original floats and separate, level-matched Float32 listening copies."""
    links = {}
    for i, (key, value) in enumerate(matched_waves(waves).items()):
        if not np.isfinite(value).all() or not np.isfinite(waves[key]).all():
            raise FloatingPointError(f"Invalid export: {prefix}/{key}")
        filename = f"{prefix}-{i}.wav"
        sf.write(folder / "raw" / filename, waves[key], 48000, subtype="FLOAT")
        sf.write(folder / "audio" / filename, value, 48000, subtype="FLOAT")
        links[key] = f"audio/{filename}"
    with (folder / "signal_diagnostics.jsonl").open("a", encoding="utf-8") as file:
        for key, value in waves.items():
            rms = float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))
            peak = float(np.max(np.abs(value)))
            file.write(
                json.dumps(
                    dict(
                        prefix=prefix,
                        label=key,
                        rms=rms,
                        peak=peak,
                        dc=float(np.mean(value)),
                        crest=peak / max(rms, 1e-12),
                        first=float(value[0]),
                        last=float(value[-1]),
                        max_sample_step=(
                            float(np.max(np.abs(np.diff(value))))
                            if len(value) > 1
                            else 0
                        ),
                    )
                )
                + "\n"
            )
    return links


def setup_folder(folder):
    (folder / "audio").mkdir(parents=True, exist_ok=True)
    (folder / "raw").mkdir(exist_ok=True)


def matched_waves(waves, target_db=-20):
    """Equal RMS copies with common headroom, leaving render metrics untouched."""
    rms = {k: np.sqrt(np.mean(v.astype(np.float64) ** 2)) for k, v in waves.items()}
    target = 10 ** (target_db / 20)
    ratios = [np.max(np.abs(v)) / rms[k] for k, v in waves.items() if rms[k] > 1e-8]
    if ratios:
        target = min(target, 0.95 / max(ratios))
    return {k: v * (target / rms[k]) if rms[k] > 1e-8 else v for k, v in waves.items()}


def write_html(path, cards):
    blocks = []
    for card in cards:
        atom = card["atom"]
        clips = []
        for clip in card["clips"]:
            buttons = " ".join(
                f'<button type="button" data-audio="{escape(wav, quote=True)}">{escape(amount)}</button>'
                for amount, wav in clip["audio"].items()
            )
            clips.append(
                f'<div class="clip"><span>{escape(clip["name"])}</span><div>{buttons}</div></div>'
            )
        blocks.append(
            f'<section><h2><label><input type="checkbox" value="{atom}"> Atom {atom}</label></h2>'
            f"<p>{card['active']} active validation clips · {card['packs']} packs</p>"
            + "".join(clips)
            + "</section>"
        )
    page = (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>One-shot knob audition</title><style>
*{box-sizing:border-box}body{margin:0;background:#141820;color:#eef2f6;font:16px system-ui}
main{max-width:960px;margin:auto;padding:24px}h1{margin:.4em 0}h2{margin:0;font-size:1.15rem}
.deck{position:sticky;top:0;background:#141820;padding:12px 0;z-index:1}audio{max-width:100%;width:480px}
section{background:#202631;border:1px solid #404c60;border-radius:12px;margin:14px 0;padding:16px}
section p{color:#bfcde0;margin:6px 0 12px}.clip{display:flex;flex-wrap:wrap;align-items:center;
justify-content:space-between;gap:8px;padding:8px 0;border-top:1px solid #404c60}.clip span{overflow-wrap:anywhere}
button{background:#303b4d;color:inherit;border:1px solid #60708a;border-radius:6px;padding:6px 12px;cursor:pointer}
button[aria-pressed=true]{background:#c7e76d;color:#142002}output{display:block;color:#c7e76d;padding:8px 0}
</style><main><h1>One-shot knob audition</h1>
<p>Compare each hit at −1, 0, and +1. Stronger edits are marked +2 and +4. Choose up to two atoms you would use in a rhythm.</p>
<div class="deck"><audio controls></audio><output>Selected atoms: —</output></div>
"""
        + "\n".join(blocks)
        + """</main><script>
const player = document.querySelector('audio');
const output = document.querySelector('output');
document.addEventListener('click', e => {
  const button = e.target.closest('[data-audio]');
  if (!button) return;
  document.querySelector('[aria-pressed=true]')?.setAttribute('aria-pressed', 'false');
  button.setAttribute('aria-pressed', 'true');
  player.src = button.dataset.audio;
  player.play().catch(() => {});
});
document.addEventListener('change', e => {
  if (!e.target.matches('input[type=checkbox]')) return;
  const selected = Array.from(document.querySelectorAll('input:checked'));
  if (selected.length > 2) { e.target.checked = false; return; }
  output.textContent = 'Selected atoms: ' + (selected.map(x => x.value).join(', ') || '—');
});
</script></html>"""
    )
    path.write_text(page, encoding="utf-8")
