"""Summarize complete selected-control inference artifacts and make figures."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def stats(values):
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return dict(
        count=len(a),
        missing=len(values) - len(a),
        median=float(np.median(a)) if len(a) else None,
        mean=float(np.mean(a)) if len(a) else None,
        p10=float(np.quantile(a, 0.1)) if len(a) else None,
        p90=float(np.quantile(a, 0.9)) if len(a) else None,
    )


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def summarize(root):
    protocol = read(root / "protocol.json")
    selected_atoms = protocol["atoms"]
    progress = read(root / "progress.json")
    assert progress["phase"] == "complete"
    paths = sorted((root / "metrics").glob("*.json"))
    assert len(paths) == len(protocol["test_ids"])
    groups = defaultdict(list)
    packs = []
    for index, path in enumerate(paths):
        clip = read(path)
        assert (
            clip["index"] == index and clip["sample_id"] == protocol["test_ids"][index]
        )
        assert clip["zero_exact"] and len(clip["rows"]) == 24
        assert len({(r["model"], r["atom"], r["alpha"]) for r in clip["rows"]}) == 24
        packs.append(clip["pack"])
        for row in clip["rows"]:
            assert row["initial_phase_unchanged"]
            if row["model"] == "main":
                assert row["noise_only_preserves_ST"]
            groups[row["model"], row["atom"], row["alpha"]].append(row)
    fields = [
        f"{s}/{k}"
        for s in ("modal", "noise", "transient", "mix")
        for k in ("spectral_change", "delta_over_mix", "delta_over_branch", "rms_ratio")
    ]
    fields += [
        "cents_abs_mean",
        "amp_db_abs_mean",
        "h_max_frame_rms",
        "restored_output_peak",
        "context_delta_over_noise",
        "context_delta_over_mix",
        "context_delta_over_edited_noise",
        "context_delta_over_noise_edit",
        "context_noise_spectral_change",
    ]
    result = dict(
        atoms=selected_atoms,
        test_clips=len(paths),
        packs=len(set(packs)),
        groups={},
        paired={},
        activity={},
        timing=read(root / "timing.json"),
    )
    for (model, atom, alpha), rows in groups.items():
        result["groups"][f"{model}/{atom}/{alpha}"] = {
            key: stats([r.get(key) for r in rows]) for key in fields if key in rows[0]
        }
        if alpha == 4:
            result["activity"][f"{model}/{atom}"] = dict(
                active_clips=sum(r["active"] for r in rows),
                inactive_clips=sum(not r["active"] for r in rows),
                floor_hits=sum(r["floor_hit"] for r in rows),
            )
    for atom in selected_atoms:
        for alpha in (-64, -16, -4, 4, 16, 64):
            for scope in ("modal", "noise", "mix"):
                field = scope + "/spectral_change"
                delta = [
                    a[field] - b[field]
                    if a[field] is not None and b[field] is not None
                    else None
                    for a, b in zip(
                        groups["main", atom, alpha], groups["comparison", atom, alpha]
                    )
                ]
                valid = [d for d in delta if d is not None]
                per_pack = defaultdict(list)
                for pack, d in zip(packs, delta):
                    if d is not None:
                        per_pack[pack].append(d)
                result["paired"][f"{atom}/{alpha}/{scope}"] = dict(
                    **stats(delta),
                    main_larger_fraction=float(np.mean(np.asarray(valid) > 1e-6))
                    if valid
                    else None,
                    numerical_tie_fraction=float(np.mean(np.abs(valid) <= 1e-6))
                    if valid
                    else None,
                    pack_macro=stats([float(np.mean(v)) for v in per_pack.values()]),
                )
    data = read(root / "redundancy.json")
    atoms = data["atoms"]
    result["redundancy"] = dict(atoms=atoms, scopes={})
    clip_scope = []
    for scope in ("modal", "noise", "mix"):
        arrays = {name: [] for name in ("same", "reversed")}
        for rec in data["records"]:
            row = rec["scopes"][scope]
            valid = np.asarray(row["valid"])
            for name in arrays:
                arrays[name].append(
                    np.where(valid[:, None] & valid[None, :], row[name], np.nan)
                )
        clip_scope.append({key: np.asarray(value) for key, value in arrays.items()})
    aggregate = {}
    for name in ("same", "reversed"):
        values = np.stack([a[name] for a in clip_scope])
        count = np.isfinite(values).sum(0)
        aggregate[name] = np.divide(
            np.nansum(values, 0),
            count,
            out=np.full_like(values[0], np.nan),
            where=count > 0,
        )
    for scope, arrays in zip(
        ("modal", "noise", "mix", "balanced"), clip_scope + [aggregate]
    ):
        out = {}
        for name, values in arrays.items():
            counts = np.isfinite(values).sum(0)
            average = np.divide(
                np.nansum(values, 0),
                counts,
                out=np.zeros_like(values[0]),
                where=counts > 0,
            )
            out[name] = [
                [float(x) if counts[i, j] else None for j, x in enumerate(row)]
                for i, row in enumerate(average)
            ]
            out["counts"] = counts.tolist()
        result["redundancy"]["scopes"][scope] = out
    balanced = result["redundancy"]["scopes"]["balanced"]
    pairs = []
    for i, a in enumerate(atoms):
        for j in range(i + 1, len(atoms)):
            same, reverse = balanced["same"][i][j], balanced["reversed"][i][j]
            if same is not None:
                pairs.append(
                    dict(
                        a=a,
                        b=atoms[j],
                        same=same,
                        reversed=reverse,
                        clips=balanced["counts"][i][j],
                    )
                )
    result["redundancy"]["pairs"] = sorted(
        pairs, key=lambda r: -max(r["same"], r["reversed"])
    )
    result["redundancy"]["selected_pair"] = next(
        (r for r in pairs if {r["a"], r["b"]} == set(selected_atoms)),
        dict(
            a=selected_atoms[0], b=selected_atoms[1], same=None, reversed=None, clips=0
        ),
    )
    curves = read(root / "curves.json")
    result["sweep"] = {}
    for atom in selected_atoms:
        static = [r for r in curves if r["atom"] == atom and not r.get("dynamic")]
        dynamic = [r for r in curves if r["atom"] == atom and r.get("dynamic")]
        assert len(static) == 6 * 33 and len(dynamic) == 6
        result["sweep"][str(atom)] = dict(
            static_points=len(static),
            dynamic_clips=len(dynamic),
            adjacent_delta=stats(
                [
                    r["adjacent_delta_over_neutral"]
                    for r in static
                    if "adjacent_delta_over_neutral" in r
                ]
            ),
            dynamic_max_step=stats([r["maximum_sample_step"] for r in dynamic]),
            neutral_max_step=stats([r["neutral_maximum_sample_step"] for r in dynamic]),
        )
    grid = read(root / "interactions.json")
    assert len(grid) == 6 * 25
    result["grid"] = {}
    for j, scope in enumerate(("modal", "noise", "transient", "mix")):
        result["grid"][scope] = stats(
            [
                r["nonadditivity_over_neutral"][j]
                for r in grid
                if r["a"] != 0 and r["b"] != 0
            ]
        )
    save(root / "summary.json", result)
    return result, groups


def figures(root, summary, groups):
    destination = root / "figures"
    destination.mkdir(exist_ok=True)
    gains = [-64, -16, -4, 0, 4, 16, 64]
    positions = np.sign(gains) * np.log1p(np.abs(gains)) / np.log(65)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    for i, atom in enumerate(summary["atoms"]):
        for j, scope in enumerate(("modal", "noise", "mix")):
            ax = axes[i, j]
            for model, color in (("main", "#007a8a"), ("comparison", "#c36b26")):
                y = [
                    0
                    if a == 0
                    else summary["groups"][f"{model}/{atom}/{a}"][
                        scope + "/spectral_change"
                    ]["median"]
                    for a in gains
                ]
                ax.plot(positions, y, marker="o", label=model, color=color)
            ax.set(
                title=f"Atom {atom}: {scope}",
                xlabel="Latent edit gain (exponential knob)",
                ylabel="Median spectral change",
            )
            ax.set_xticks(positions, list(map(str, gains)))
            ax.grid(alpha=0.2)
            ax.legend()
    for suffix in ("png", "pdf"):
        fig.savefig(destination / ("test_controls." + suffix), dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    red = summary["redundancy"]
    for ax, name in zip(axes, ("same", "reversed")):
        matrix = np.asarray(red["scopes"]["balanced"][name], dtype=float)
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set(
            title="Same sign" if name == "same" else "Opposite sign responses",
            xlabel="Atom",
            ylabel="Atom",
        )
        ax.set_xticks(range(len(red["atoms"])), red["atoms"], rotation=90)
        ax.set_yticks(range(len(red["atoms"])), red["atoms"])
        fig.colorbar(im, ax=ax, label="Mean acoustic-effect cosine")
    fig.savefig(destination / "redundancy.png", dpi=170)
    fig.savefig(destination / "redundancy.pdf")
    plt.close(fig)


def report(root, s):
    selected_atoms = s["atoms"]
    first, second = selected_atoms
    lines = [
        f"# Selected controls {first} and {second}: held-out evaluation",
        "",
        f"This report evaluates atoms {first} and {second} from the locked protocol. "
        "No training, atom replacement or test-driven range adjustment was performed. "
        "Primary strengths are±4 and±16;±64 is a stress condition.",
        "",
        f"Test coverage: **{s['test_clips']} clips, {s['packs']} packs**. "
        "This is the original sample-held-out reconstruction split within packs, not an unseen-pack test. "
        "Inactive clips remain in the denominators. Silent-reference spectral metrics are null and their counts are retained in summary.json.",
        "",
        "[Audition](audition/index.html) · [Within-hit sweep](audition/sweep.html) · "
        "[Two-knob grid](audition/grid.html) · [Joint/noise-only](audition/coupling.html)",
        "",
        "## Coverage and paired change measurements",
        "",
        "| Atom | Main active clips | Comparison active clips |",
        "|---|---:|---:|",
    ]
    for atom in selected_atoms:
        lines.append(
            f"| {atom} | {s['activity'][f'main/{atom}']['active_clips']} | "
            f"{s['activity'][f'comparison/{atom}']['active_clips']} |"
        )
    lines += [
        "",
        "Values below are median RMS-matched spectral changes, **main / comparison**. "
        "More change is not a quality score. Both SAEs share the latent-only warm-up initialization; "
        "this is a conditional warm-start comparison, not independently trained from-scratch baselines.",
        "",
        "| Atom | Gain | Modal | Noise | Mix |",
        "|---|---:|---:|---:|---:|",
    ]
    for atom in selected_atoms:
        for alpha in (-64, -16, -4, 4, 16, 64):
            cells = []
            for scope in ("modal", "noise", "mix"):
                a = s["groups"][f"main/{atom}/{alpha}"][scope + "/spectral_change"][
                    "median"
                ]
                b = s["groups"][f"comparison/{atom}/{alpha}"][
                    scope + "/spectral_change"
                ]["median"]
                cells.append(
                    f"{a:.3f} / {b:.3f}" if a is not None and b is not None else "null"
                )
            lines.append(f"| {atom} | {alpha:+d} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "![Held-out curves](figures/test_controls.png)",
        "",
        "Raw level changes are kept separately. The following ratios are relative to neutral mix RMS; "
        "the listening pages normalize RMS, while raw/ does not.",
        "",
        "| Atom | Gain | Mix RMS ratio, median | Mix RMS ratio, p90 | Modal mean absolute cents, median | Modal mean absolute dB, median |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for atom in selected_atoms:
        for alpha in (-64, -16, 16, 64):
            row = s["groups"][f"main/{atom}/{alpha}"]
            vals = [
                row["mix/rms_ratio"]["median"],
                row["mix/rms_ratio"]["p90"],
                row["cents_abs_mean"]["median"],
                row["amp_db_abs_mean"]["median"],
            ]
            lines.append(
                f"| {atom} | {alpha:+d} | "
                + " | ".join("null" if v is None else f"{v:.4f}" for v in vals)
                + " |"
            )
    lines += [
        "",
        "## Modal-conditioned noise path",
        "",
        "Joint and noise-only receive exactly the same edited h and carrier. Noise-only holds the "
        "neutral corrected-modal context, S and T fixed. Nonzero N differences demonstrate use of "
        "this computational path; they do not establish its necessity or reconstruction benefit.",
        "",
        "| Atom | Gain | Context delta / edited N RMS, median | Context delta / entire N edit RMS, median |",
        "|---|---:|---:|---:|",
    ]
    for atom in selected_atoms:
        for alpha in (-16, 16):
            row = s["groups"][f"main/{atom}/{alpha}"]
            vals = [
                row["context_delta_over_edited_noise"]["median"],
                row["context_delta_over_noise_edit"]["median"],
            ]
            lines.append(
                f"| {atom} | {alpha:+d} | "
                + " | ".join("null" if x is None else f"{x:.6g}" for x in vals)
                + " |"
            )
    selected = s["redundancy"]["selected_pair"]
    lines += [
        "",
        "## Similar controls",
        "",
        "Fixed acoustic-effect cosine was measured on32 seeded validation clips at±16. "
        "The sign-reversed version swaps the two endpoint responses. Comparisons involving "
        "observer normalization-floor signals are excluded with explicit support counts. "
        "Balanced scores average available S/N/mix cosines per clip; no threshold merges features.",
        "",
        f"For {first} versus {second}: same-sign cosine **{selected['same']}**, "
        f"reversed-sign cosine **{selected['reversed']}**, support **{selected['clips']} clips**. "
        "Null values mean no shared valid support in the fixed redundancy subset. "
        "This measures acoustic overlap in these views, not identity of the learned controls.",
        "",
        "| Pair | Same sign | Reversed sign | Valid clips |",
        "|---|---:|---:|---:|",
    ]
    for row in s["redundancy"]["pairs"][:5]:
        lines.append(
            f"| {row['a']}, {row['b']} | {row['same']:.3f} | {row['reversed']:.3f} | {row['clips']} |"
        )
    lines += [
        "",
        "![Redundancy](figures/redundancy.png)",
        "",
        "## Continuous and simultaneous edits",
        "",
        "Test examples are selected by seed before rendering. Each atom has33 static "
        "positions spanning±16 plus an offline within-hit sinusoidal knob trajectory. "
        "The5×5 two-control grid uses[-16,-4,0,4,16] for each atom. These are fresh full-clip "
        "decodes of edited h, not WAV crossfades. h directions are computed once from neutral.",
        "",
        "Static whole-clip zero reproduces neutral exactly. Returning a dynamic knob to zero "
        "does not reset accumulated oscillator phase; phase can retain the earlier intervention. "
        "Adjacent waveform changes and maximum sample steps are diagnostics, not click/percussiveness "
        "scores. No streaming-state, CPU latency, VST or listener-quality guarantee was tested.",
        "",
        "Joint-grid nonadditivity uses y(a,b)-y(a,0)-y(0,b)+y(0,0), divided by neutral branch norm. "
        "It is expected for a nonlinear decoder and is not an independence or failure test.",
        "",
        "| Branch | Median nonadditivity ratio, both knobs nonzero |",
        "|---|---:|",
    ]
    for scope in ("modal", "noise", "mix"):
        value = s["grid"][scope]["median"]
        lines.append(
            f"| {scope} | {value:.4f} |" if value is not None else f"| {scope} | null |"
        )
    lines += [
        "",
        "## Provenance and limits",
        "",
        "- All test clips passed explicit-zero and noise-only S/T invariants; initial modal phases stayed unchanged. "
        "Checkpoint/cache and imported source hashes are locked in protocol.json.",
        "- Main and comparison use the same clips, atoms, alpha values and normalization rule. "
        "Actual h displacement and floor hits are recorded; no claim of exactly matched actual displacement is made when a floor is active.",
        "- Evaluation uses frozen weights. Raw and RMS-matched Float32 WAVs are exported separately.",
        "- Acoustic-change statistics measure edit magnitude. Perceptual quality and feature "
        "disentanglement require separate evaluation.",
        "- Per-clip records, silent counts, paired difference summaries and pack-macro summaries remain available in metrics/ and summary.json. "
        "Demo listening cannot replace the complete test distributions.",
        "",
        f"Current inference invocation: {s['timing']['current_invocation_seconds']:.2f}s; "
        f"training time: {s['timing']['training_seconds']}s.",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    summary, groups = summarize(args.root)
    figures(args.root, summary, groups)
    report(args.root, summary)
    print(
        json.dumps(
            dict(
                test_clips=summary["test_clips"],
                activity=summary["activity"],
                selected_pair=summary["redundancy"]["selected_pair"],
                strongest_pairs=summary["redundancy"]["pairs"][:5],
                timing=summary["timing"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
