"""Locked held-out evaluation of the user-selected frozen SAE controls."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from stamp.control.audio import as_wave, save_waves, setup_folder, write_html
from stamp.control.cache import dataset_for, reference
from stamp.control.editing import (
    BETA,
    AcousticView,
    InterventionSAE,
    edit_direction,
)
from stamp.control.evaluation import (
    EPS,
    SpectralReference,
    branches,
    compare_branches,
    redundancy_valid,
    relative_norms,
    runtime_sources,
)
from stamp.control.objective import gain_from_knob
from stamp.control.protocol import validate_atoms
from stamp.control.runtime import RunData
from stamp.utils.device import resolve_device
from stamp.utils.io import sha256, write_json

GAINS = (-64, -16, -4, 4, 16, 64)
GRID = (-16, -4, 0, 4, 16)


def directions(run, net, ref, floors, atoms):
    codes = run.codes(net, ref)
    values, info = {}, {}
    for atom in atoms:
        direction, radius = edit_direction(net, codes, atom, floors[atom])
        values[atom] = direction
        info[atom] = dict(
            active=bool(codes[..., atom].any()),
            radius=float(radius),
            floor_hit=bool(radius < floors[atom]),
            active_frame_fraction=float(
                ((codes[..., atom] > 0) * ref.rho).sum() / ref.rho.sum()
            ),
        )
    return values, info


def edited_h(run, ref, values, amounts):
    delta = torch.zeros_like(ref.h_ref)
    for atom, amount in amounts.items():
        delta = delta + values[atom] * torch.as_tensor(amount, device=delta.device)
    return ref.h_ref + run.std * BETA * delta


def modal_changes(result, neutral, ref):
    assert torch.equal(result.refined_params[:, 2], neutral.refined_params[:, 2])
    mask = ref.observed
    delta = result.log_corrections - neutral.log_corrections
    return dict(
        cents_abs_mean=(
            float(delta[..., 0][mask].abs().mean() * 1200 / math.log(2))
            if mask.any()
            else None
        ),
        amp_db_abs_mean=(
            float(delta[..., 1][mask].abs().mean() * 20 / math.log(10))
            if mask.any()
            else None
        ),
        initial_phase_unchanged=True,
    )


def atomic_json(path, value):
    pending = path.with_suffix(".pending")
    write_json(pending, value)
    pending.replace(path)


def lock_protocol(run, data, args):
    existing = {
        row["id"] for split in ("train", "val") for row in run.manifest["splits"][split]
    }
    assert not existing.intersection(data.file_list), (
        "Test overlaps SAE train/validation"
    )
    rng = np.random.default_rng(run.seed + 1301)
    demos = rng.permutation(len(data)).tolist()[:6]
    redundancy = rng.permutation(len(run.datasets["val"])).tolist()[:32]
    protocol = dict(
        atoms=args.atoms,
        primary_gains=[-16, -4, 4, 16],
        stress_gains=[-64, 64],
        identity=run.identity,
        seed=run.seed,
        sae_sha256=sha256(args.run / "final.pt"),
        cache_sha256=sha256(args.cache / "cache.json"),
        sources=runtime_sources(Path(__file__)),
        test_ids=data.file_list,
        demo_indices=demos,
        demo_ids=[data.file_list[i] for i in demos],
        redundancy_val_indices=redundancy,
        redundancy_atoms=json.loads((args.run / "screening.json").read_text())[
            "intervention_top"
        ],
        static_knob_positions=np.linspace(
            -math.log(17) / math.log(65), math.log(17) / math.log(65), 33
        ).tolist(),
        grid=list(GRID),
        transient="neutral strike code; original T follows edited S",
        scope="all sample-held-out test clips, within original pack-stratified split",
        no_training=True,
    )
    protocol = json.loads(json.dumps(protocol))
    if args.resume:
        if protocol != json.loads((args.output / "protocol.json").read_text()):
            raise ValueError("Locked protocol changed")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "protocol.json", protocol)
        (args.output / "metrics").mkdir()
    return protocol


def evaluate_test(run, nets, floors, data, args):
    start = time.monotonic()
    for index in tqdm(range(len(data)), desc="Held-out selected controls"):
        path = args.output / "metrics" / f"{index:05d}.json"
        if path.is_file():
            previous = json.loads(path.read_text())
            assert previous["sample_id"] == data.file_list[index]
            continue
        item = data[index]
        ref = reference(run.model, item, run.seed, args.device)
        neutral = run.model.decode(ref, checkpoint_activations=False)
        zero = run.model.decode(ref, ref.h_ref, checkpoint_activations=False)
        assert torch.equal(branches(zero), branches(neutral)), "Non-exact zero edit"
        spectral = SpectralReference(branches(neutral))
        rows = []
        for label, net in nets.items():
            values, info = directions(run, net, ref, floors, args.atoms)
            for atom in args.atoms:
                for alpha in GAINS:
                    h = edited_h(run, ref, values, {atom: alpha})
                    result = run.model.decode(ref, h, checkpoint_activations=False)
                    row = dict(
                        model=label,
                        atom=atom,
                        alpha=alpha,
                        **info[atom],
                        **compare_branches(result, neutral, spectral),
                        **modal_changes(result, neutral, ref),
                    )
                    row["h_max_frame_rms"] = float(
                        ((h - ref.h_ref) / run.std).square().mean(-1).sqrt().amax()
                    )
                    row["restored_output_peak"] = float(
                        result.waveform.abs().max() * ref.gain
                    )
                    if label == "main":
                        noise_only = run.model.decode(
                            ref, h, control_scope="noise", checkpoint_activations=False
                        )
                        assert torch.equal(
                            noise_only.modal_waveform, neutral.modal_waveform
                        )
                        assert torch.equal(
                            noise_only.transient_waveform, neutral.transient_waveform
                        )
                        a, b = (
                            result.noise_waveform.flatten(),
                            noise_only.noise_waveform.flatten(),
                        )
                        delta = float((a - b).square().mean().sqrt())
                        nrms = float(neutral.noise_waveform.square().mean().sqrt())
                        xrms = float(neutral.waveform.square().mean().sqrt())
                        edited_nrms = float(a.square().mean().sqrt())
                        noise_edit = float(
                            (a - neutral.noise_waveform.flatten())
                            .square()
                            .mean()
                            .sqrt()
                        )
                        row.update(
                            context_delta_over_noise=(
                                delta / nrms if nrms > EPS else None
                            ),
                            context_delta_over_mix=delta / xrms if xrms > EPS else None,
                            context_delta_over_edited_noise=(
                                delta / edited_nrms if edited_nrms > EPS else None
                            ),
                            context_delta_over_noise_edit=(
                                delta / noise_edit if noise_edit > EPS else None
                            ),
                            context_noise_spectral_change=SpectralReference(
                                b[None]
                            ).change(a[None])[0],
                            noise_only_preserves_ST=True,
                        )
                    rows.append(row)
        atomic_json(
            path,
            dict(
                index=index,
                sample_id=item["sample_id"],
                pack=data.sample_pack_key(data.metadata[item["sample_id"]]),
                samples=ref.length,
                zero_exact=True,
                rows=rows,
            ),
        )
        if (index + 1) % 50 == 0 or index == 0:
            write_json(
                args.output / "progress.json",
                dict(
                    phase="test",
                    clips=index + 1,
                    total=len(data),
                    current_invocation_seconds=time.monotonic() - start,
                ),
            )
        del ref, neutral, zero, result, spectral, values, noise_only
    return time.monotonic() - start


def redundancy(run, net, floors, payload, protocol, args):
    target = args.output / "redundancy.json"
    if args.resume and target.exists():
        return
    observers = {}
    for scope in ("modal", "noise", "mix"):
        observer = AcousticView().to(args.device)
        prefix = scope + "."
        observer.load_state_dict(
            {
                k[len(prefix) :]: v
                for k, v in payload["observer"].items()
                if k.startswith(prefix)
            }
        )
        observers[scope] = observer.eval()
    atoms = protocol["redundancy_atoms"]
    records = []
    for index in tqdm(protocol["redundancy_val_indices"], desc="Validation redundancy"):
        ref = run.reference("val", index)
        neutral = run.model.decode(ref, checkpoint_activations=False)
        values, info = directions(run, net, ref, floors, atoms)
        base = branches(neutral)
        origin = {
            s: obs(base[j : j + 1])
            for s, obs, j in (
                (s, observers[s], j)
                for s, j in (("modal", 0), ("noise", 1), ("mix", 3))
            )
        }
        effects = {s: [] for s in observers}
        levels = {s: [] for s in observers}
        for atom in atoms:
            pair = {s: [] for s in observers}
            pair_levels = {s: [] for s in observers}
            for alpha in (-16, 16):
                result = run.model.decode(
                    ref,
                    edited_h(run, ref, values, {atom: alpha}),
                    checkpoint_activations=False,
                )
                waves = branches(result)
                for scope, j in (("modal", 0), ("noise", 1), ("mix", 3)):
                    pair_levels[scope].append(waves[j].square().mean().sqrt())
                    pair[scope].append(
                        (observers[scope](waves[j : j + 1]) - origin[scope]).flatten()
                    )
            for scope in observers:
                effects[scope].append(torch.stack(pair[scope]))
                levels[scope].append(torch.stack(pair_levels[scope]))
        row = dict(
            index=index,
            sample_id=run.caches["val"].rows[index]["id"],
            active=[info[a]["active"] for a in atoms],
            scopes={},
        )
        for scope in observers:
            neutral_rms = float(
                base[{"modal": 0, "noise": 1, "mix": 3}[scope]].square().mean().sqrt()
            )
            edited_rms = torch.stack(levels[scope])
            matrix = torch.stack(effects[scope])
            flat = matrix.flatten(1)
            norm = flat.norm(dim=1)
            normalized = flat / norm[:, None].clamp_min(1e-8)
            reversed_sign = matrix.flip(1).flatten(1) / norm[:, None].clamp_min(1e-8)
            row["scopes"][scope] = dict(
                same=(normalized @ normalized.T).cpu().tolist(),
                reversed=(normalized @ reversed_sign.T).cpu().tolist(),
                neutral_rms=neutral_rms,
                edited_rms=edited_rms.cpu().tolist(),
                valid=redundancy_valid(neutral_rms, edited_rms, norm).tolist(),
            )
        records.append(row)
    write_json(
        target,
        dict(
            atoms=atoms,
            records=records,
            interpretation="Signed cosine of actual fixed-observer changes at +/-16; reversed swaps the two sign responses. No threshold or automatic merging.",
        ),
    )


def page(folder, name, cards, title, explanation):
    path = folder / (name + ".html")
    write_html(path, cards)
    text = path.read_text(encoding="utf-8").replace("One-shot knob audition", title)
    old = "Compare each hit at −1, 0, and +1. Stronger edits are marked +2 and +4. Choose up to two atoms you would use in a rhythm."
    nav = " · ".join(
        f'<a href="{key}.html">{value}</a>'
        for key, value in (
            ("index", "Held-out controls"),
            ("modal", "Modal solo"),
            ("noise", "Noise solo"),
            ("baseline", "Comparison SAE"),
            ("coupling", "Joint / noise-only"),
            ("sweep", "Within-hit sweep"),
            ("grid", "Two-knob grid"),
        )
    )
    text = text.replace(old, explanation + "<br>" + nav)
    text = text.replace("active validation clips", "seed-selected test examples")
    path.write_text(text, encoding="utf-8")


def demonstrations(run, nets, floors, data, protocol, args):
    folder = args.output / "audition"
    if args.resume and (folder / "complete.json").exists():
        return
    setup_folder(folder)
    (folder / "signal_diagnostics.jsonl").unlink(missing_ok=True)
    pages = {
        s: []
        for s in ("index", "modal", "noise", "baseline", "coupling", "sweep", "grid")
    }
    curves, interactions = [], []
    cards = {
        s: {a: dict(atom=a, active=6, packs=0, clips=[]) for a in args.atoms}
        for s in pages
    }
    for index in tqdm(protocol["demo_indices"], desc="Selected control demos"):
        item = data[index]
        ref = reference(run.model, item, run.seed, args.device)
        neutral = run.model.decode(ref, checkpoint_activations=False)
        values, _ = directions(run, nets["main"], ref, floors, args.atoms)
        base_values, _ = directions(run, nets["comparison"], ref, floors, args.atoms)
        name = data.metadata[item["sample_id"]].get("orig_relpath", item["sample_id"])
        spectral = SpectralReference(branches(neutral))
        for atom in args.atoms:
            saved = {
                s: {
                    "0": as_wave(
                        getattr(
                            neutral,
                            {"modal": "modal_waveform", "noise": "noise_waveform"}.get(
                                s, "waveform"
                            ),
                        ),
                        ref,
                    )
                }
                for s in ("index", "modal", "noise", "baseline", "coupling")
            }
            previous = None
            for j, knob in enumerate(protocol["static_knob_positions"]):
                alpha = gain_from_knob(knob)
                result = run.model.decode(
                    ref,
                    edited_h(run, ref, values, {atom: alpha}),
                    checkpoint_activations=False,
                )
                measured = compare_branches(result, neutral, spectral)
                wave = result.waveform.flatten()
                if previous is not None:
                    measured["adjacent_delta_over_neutral"] = float(
                        (wave - previous).norm()
                        / neutral.waveform.norm().clamp_min(1e-12)
                    )
                curves.append(
                    dict(
                        index=index,
                        sample_id=item["sample_id"],
                        atom=atom,
                        knob=knob,
                        alpha=alpha,
                        **measured,
                    )
                )
                previous = wave.clone()
                if j % 4 == 0:
                    label = "0" if j == 16 else f"{alpha:+.2f}x"
                    for scope, field in (
                        ("index", "waveform"),
                        ("modal", "modal_waveform"),
                        ("noise", "noise_waveform"),
                    ):
                        saved[scope][label] = as_wave(getattr(result, field), ref)
                    baseline = run.model.decode(
                        ref,
                        edited_h(run, ref, base_values, {atom: alpha}),
                        checkpoint_activations=False,
                    )
                    saved["baseline"][label] = as_wave(baseline.waveform, ref)
                    if j in (0, 32):
                        noise_only = run.model.decode(
                            ref,
                            edited_h(run, ref, values, {atom: alpha}),
                            control_scope="noise",
                            checkpoint_activations=False,
                        )
                        saved["coupling"]["Joint " + label] = as_wave(
                            result.waveform, ref
                        )
                        saved["coupling"]["Noise-only " + label] = as_wave(
                            noise_only.waveform, ref
                        )
                        saved["coupling"]["N joint " + label] = as_wave(
                            result.noise_waveform, ref
                        )
                        saved["coupling"]["N fixed context " + label] = as_wave(
                            noise_only.noise_waveform, ref
                        )
            for scope in saved:
                links = save_waves(folder, f"{scope}-{atom}-{index}", saved[scope])
                cards[scope][atom]["clips"].append(dict(name=name, audio=links))
            t = torch.linspace(0, 1, ref.h_ref.shape[1], device=args.device)
            k = math.log(17) / math.log(65) * torch.sin(2 * torch.pi * t)
            amount = k.sign() * torch.expm1(k.abs() * math.log(65))
            amount[0] = amount[-1] = 0
            sweep = run.model.decode(
                ref,
                edited_h(run, ref, values, {atom: amount[None, :, None]}),
                checkpoint_activations=False,
            )
            links = save_waves(
                folder,
                f"sweep-{atom}-{index}",
                {
                    "0": as_wave(neutral.waveform, ref),
                    "Within-hit 0 / +16 / 0 / -16 / 0": as_wave(sweep.waveform, ref),
                },
            )
            cards["sweep"][atom]["clips"].append(dict(name=name, audio=links))
            curves.append(
                dict(
                    index=index,
                    sample_id=item["sample_id"],
                    atom=atom,
                    dynamic=True,
                    **compare_branches(sweep, neutral, spectral),
                    maximum_sample_step=float(sweep.waveform.diff().abs().max()),
                    neutral_maximum_sample_step=float(
                        neutral.waveform.diff().abs().max()
                    ),
                )
            )
        grid_waves, grid_branches = {}, {}
        for a in GRID:
            for b in GRID:
                result = run.model.decode(
                    ref,
                    edited_h(run, ref, values, dict(zip(args.atoms, (a, b)))),
                    checkpoint_activations=False,
                )
                grid_waves[f"{args.atoms[0]} {a:+d} / {args.atoms[1]} {b:+d}"] = (
                    as_wave(result.waveform, ref)
                )
                grid_branches[a, b] = branches(result).cpu()
        for a in GRID:
            for b in GRID:
                value = grid_branches[a, b]
                error = (
                    value
                    - grid_branches[a, 0]
                    - grid_branches[0, b]
                    + grid_branches[0, 0]
                )
                interactions.append(
                    dict(
                        index=index,
                        sample_id=item["sample_id"],
                        a=a,
                        b=b,
                        nonadditivity_over_neutral=relative_norms(
                            error, grid_branches[0, 0]
                        ),
                        rms_ratio=relative_norms(value, grid_branches[0, 0]),
                    )
                )
        links = save_waves(folder, f"grid-{index}", grid_waves)
        cards["grid"][args.atoms[0]]["clips"].append(dict(name=name, audio=links))
    for scope in pages:
        pages[scope] = (
            list(cards[scope].values())
            if scope != "grid"
            else [cards["grid"][args.atoms[0]]]
        )
        for card in pages[scope]:
            card["active"] = len(card["clips"])
            card["packs"] = len(
                {
                    data.sample_pack_key(data.metadata[data.file_list[i]])
                    for i in protocol["demo_indices"]
                }
            )
        explanation = f"Controls {args.atoms[0]} and {args.atoms[1]} are fixed. Test examples were selected by seed before rendering. RMS-matched FLOAT listening copies; original levels in raw/. "
        explanation += (
            "The waveform is decoded once from a time-varying h; this is offline and not a realtime VST test. "
            if scope == "sweep"
            else ""
        )
        explanation += (
            "Each button jointly edits both latent directions; this is not a crossfade of rendered WAVs. "
            if scope == "grid"
            else ""
        )
        page(folder, scope, pages[scope], "Selected controls: " + scope, explanation)
    write_json(folder / "cards.json", pages)
    write_json(args.output / "curves.json", curves)
    write_json(args.output / "interactions.json", interactions)
    write_json(
        folder / "complete.json",
        dict(demo_ids=protocol["demo_ids"], format="FLOAT48k mono"),
    )


@torch.inference_mode()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "noise-bank", "data-dir", "cache", "run", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--meta-file", default="metadata.json")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--atoms",
        type=int,
        nargs=2,
        required=True,
        help="Two IDs selected from this run's validation audition.",
    )
    args = parser.parse_args(argv)
    try:
        args.device = resolve_device(args.device)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    run = RunData(args)
    start = time.monotonic()
    payload = torch.load(
        args.run / "final.pt", map_location=args.device, weights_only=True
    )
    if payload["identity"] != run.identity or payload["cache_sha256"] != sha256(
        args.cache / "cache.json"
    ):
        raise ValueError("SAE or cache identity mismatch")
    args.atoms = validate_atoms(args.atoms, payload["cohort"].tolist())
    selected = json.loads((args.run / "screening.json").read_text())["intervention_top"]
    if not set(args.atoms).issubset(selected):
        raise ValueError(
            "Selected atoms must come from this run's validation audition."
        )
    data = dataset_for(args, run.seed, "test")
    protocol = lock_protocol(run, data, args)
    nets = {}
    for name, key in (("main", "state_dict"), ("comparison", "baseline_state_dict")):
        net = InterventionSAE().to(args.device).eval().requires_grad_(False)
        net.load_state_dict(payload[key])
        nets[name] = net
    print("LOCKED", len(data), "test clips", protocol["demo_indices"], flush=True)
    test_seconds = evaluate_test(run, nets, payload["floors"], data, args)
    redundancy(run, nets["main"], payload["floors"], payload, protocol, args)
    demonstrations(run, nets, payload["floors"], data, protocol, args)
    assert len(list((args.output / "metrics").glob("*.json"))) == len(data)
    assert sha256(args.checkpoint) == run.identity["checkpoint"]
    assert sha256(args.run / "final.pt") == protocol["sae_sha256"]
    write_json(
        args.output / "timing.json",
        dict(
            current_invocation_seconds=time.monotonic() - start,
            test_seconds=test_seconds,
            training_seconds=0,
            resumed=args.resume,
        ),
    )
    write_json(
        args.output / "progress.json",
        dict(phase="complete", clips=len(data), total=len(data)),
    )
    print("COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
