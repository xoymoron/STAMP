"""Train paired post-hoc SAEs on finite modal/noise/mix interventions."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from stamp.control import BATCH, CORE_SLICE, FEATURES, training_loss
from stamp.control.audio import difference_score
from stamp.control.cache import clip_activity, prepare_batch, save_best, shortlist
from stamp.control.editing import BETA, atom_delta, edit_direction
from stamp.control.objective import (
    SCOPES,
    JointObjective,
    gain_from_knob,
    joint_prefix,
)
from stamp.control.protocol import (
    fork_warmup,
    paired_budget,
    resumed_seconds,
    select_cohort,
)
from stamp.control.runtime import RunData, choose_group, crop_bounds
from stamp.utils.device import resolve_device
from stamp.utils.io import sha256, write_json


def excerpt(run, ref, h, left, end):
    return {k: v[:, left:end] for k, v in joint_prefix(run.model, ref, h, end).items()}


def means(rows):
    return {
        k: float(np.mean([r[k] for r in rows if k in r]))
        for k in sorted({k for row in rows for k in row})
    }


def restore_trace(output, checkpoint_step):
    """Archive rolled-back attempts before resuming the retained optimizer state."""
    discarded = {}
    for name in ("training.json", "sampling.jsonl", "validation.jsonl"):
        path = output / name
        if not path.exists():
            continue
        rows = (
            json.loads(path.read_text())
            if name.endswith(".json")
            else [json.loads(line) for line in path.read_text().splitlines()]
        )
        kept = [row for row in rows if row["step"] <= checkpoint_step]
        tail = [row for row in rows if row["step"] > checkpoint_step]
        if tail:
            discarded[name] = tail
            if name.endswith(".json"):
                write_json(path, kept)
            else:
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in kept), encoding="utf-8"
                )
    if discarded:
        write_json(output / f"interrupted-{time.time_ns()}.json", discarded)


def selection_rows(activity, rows, cohort, seed):
    supported = {
        atom: (active, packs, examples)
        for active, packs, _, atom, examples in shortlist(activity, rows, FEATURES)
    }
    rng = np.random.default_rng(seed)
    result = []
    for atom in cohort:
        if atom in supported:
            active, packs, examples = supported[atom]
            result.append(
                dict(
                    atom=atom,
                    active=active,
                    packs=packs,
                    examples=examples,
                    eligible=True,
                    reason=None,
                )
            )
        else:
            active = np.flatnonzero(activity[:, atom] > 0)
            chosen = active[
                np.argsort(-activity[active, atom], kind="stable")
            ].tolist()[:3]
            for i in rng.permutation(len(rows)):
                if len(chosen) == 3:
                    break
                if int(i) not in chosen:
                    chosen.append(int(i))
            result.append(
                dict(
                    atom=atom,
                    active=len(active),
                    packs=len({rows[i]["pack"] for i in active}),
                    examples=chosen,
                    eligible=False,
                    reason="Initialization has fewer than3 active clips or fewer than2 packs; fallback examples scored but excluded from shortlist.",
                )
            )
    return result


@torch.no_grad()
def calibrate_dictionary(run, sae, output):
    """Recompute train-only edit units from the new latent-only warm-up."""
    start = time.monotonic()
    cache = run.caches["train"]
    activity = clip_activity(sae, cache, run.mean, run.std)
    np.save(output / "train_activity.npy", activity)
    cohort = select_cohort(activity, cache.rows)
    radii = {i: [] for i in cohort}
    trace = []
    rng = np.random.default_rng(run.seed + 1200)
    for index in tqdm(rng.permutation(len(cache.rows))[:2048], desc="Train edit units"):
        h = run.hidden("train", int(index))
        mask = torch.ones(h.shape[:2], device=h.device, dtype=torch.bool)
        q = (h - run.mean) / run.std
        codes = sae.select(sae.preactivations(q, mask), mask, sae.top_k)[0]
        for atom in cohort:
            if len(radii[atom]) < 128 and codes[..., atom].any():
                radius = float(atom_delta(sae, codes, atom).norm(dim=-1).amax() / 16)
                if radius > 0:
                    radii[atom].append(radius)
        trace.append(cache.rows[index]["id"])
        if len(trace) >= 128 and min(map(len, radii.values())) >= 16:
            break
    cohort = [i for i in cohort if len(radii[i]) >= 16]
    if len(cohort) < 4:
        raise ValueError("Insufficient positive train calibration support.")
    floors = torch.full((FEATURES,), 1e-6, device=run.args.device)
    for atom in cohort:
        floors[atom] = max(float(np.quantile(radii[atom], 0.1)), 1e-6)
    write_json(
        output / "dictionary_calibration.json",
        dict(
            cohort=cohort,
            clip_ids=trace,
            support_counts={str(i): len(radii[i]) for i in cohort},
            floors={str(i): float(floors[i]) for i in cohort},
        ),
    )
    return dict(
        cohort=torch.tensor(cohort, device=run.args.device),
        floors=floors,
        seconds=time.monotonic() - start,
    )


@torch.no_grad()
def prepare(run, sae, objective, warm, output):
    start = time.monotonic()
    rng = np.random.default_rng(run.seed + 1201)
    calibration = {k: [] for k in SCOPES}
    trace = []
    for index in tqdm(
        rng.permutation(len(run.caches["train"].rows))[:512],
        desc="Joint observer calibration",
    ):
        ref = run.reference("train", int(index))
        frame = int(rng.integers(ref.h_ref.shape[1]))
        left, end = crop_bounds(ref, frame)
        waves = excerpt(run, ref, ref.h_ref, left, end)
        trace.append(
            dict(
                index=int(index),
                sample_id=run.caches["train"].rows[index]["id"],
                frame=frame,
            )
        )
        for scope, wave in waves.items():
            if (
                len(calibration[scope]) < 32
                and float(wave.square().mean().sqrt()) >= 1e-5
            ):
                calibration[scope].append(wave)
        if all(len(values) >= 32 for values in calibration.values()):
            break
    for scope, waves in calibration.items():
        if not waves:
            raise ValueError(f"No train calibration examples for {scope}.")
        objective.observers[scope].calibrate(waves)
    cohort = warm["cohort"]
    validation = []
    rng = np.random.default_rng(run.seed + 1202)
    for index in rng.permutation(len(run.caches["val"].rows))[:128]:
        ref = run.reference("val", int(index))
        chosen = choose_group(run.codes(sae, ref), cohort, np.zeros(len(cohort)), rng)
        if chosen is None:
            continue
        frame, group = chosen
        alpha = (4, -4, 16, -16, 64, -64)[len(validation) % 6]
        validation.append(
            dict(
                index=int(index),
                frame=frame,
                group=group.tolist(),
                knob=float(np.sign(alpha) * np.log1p(abs(alpha)) / np.log(65)),
            )
        )
        if len(validation) == 12:
            break
    if not validation:
        raise ValueError("No fixed validation intervention groups.")
    activity = clip_activity(sae, run.caches["val"], run.mean, run.std)
    selection = selection_rows(
        activity, run.caches["val"].rows, cohort.tolist(), run.seed + 1205
    )
    if sum(row["eligible"] for row in selection) < 10:
        raise ValueError(
            "Fewer than ten supported validation candidates; inspect initial activity before training."
        )
    payload = dict(
        observer=objective.observers.state_dict(),
        cohort=cohort,
        floors=warm["floors"],
        validation=validation,
        selection=selection,
        calibration=trace,
        scope_counts={k: len(v) for k, v in calibration.items()},
        seconds=time.monotonic() - start + warm["seconds"],
    )
    save_best(output / "calibration.pt", payload)
    write_json(output / "fixed_validation.json", validation)
    write_json(output / "initial_selection.json", selection)
    write_json(
        output / "calibration.json",
        {
            k: v
            for k, v in payload.items()
            if k in ("calibration", "scope_counts", "seconds")
        },
    )
    return payload


@torch.no_grad()
def validate(run, sae, baseline, objective, calibration, deadline):
    cache = run.caches["val"]
    rng = np.random.default_rng(run.seed + 1203)
    ids = rng.choice(len(cache.starts), min(256, len(cache.starts)), replace=False)
    errors = [0.0, 0.0]
    weight = 0.0
    for subset in np.array_split(ids, 16):
        if time.monotonic() >= deadline:
            break
        q, rho = prepare_batch(cache.batch(subset), run.mean, run.std)
        w = rho[:, CORE_SLICE]
        for n, net in enumerate((sae, baseline)):
            pred, _ = net.reconstruct(q, rho > 0)
            errors[n] += float(
                ((pred[:, CORE_SLICE] - q[:, CORE_SLICE]).square() * w[..., None]).sum()
            )
        weight += float(w.sum()) * 256
    rows = []
    for job in calibration["validation"]:
        if time.monotonic() >= deadline:
            break
        ref = run.reference("val", job["index"])
        codes = run.codes(sae, ref)
        left, end = crop_bounds(ref, job["frame"])
        neutral = excerpt(run, ref, ref.h_ref, left, end)
        views = objective.views(neutral, rng)
        if not views:
            continue
        group = torch.tensor(job["group"], device=run.args.device)
        for position, ci in enumerate(group):
            atom = int(calibration["cohort"][ci])
            direction, _ = edit_direction(sae, codes, atom, calibration["floors"][atom])
            h = ref.h_ref + run.std * BETA * gain_from_knob(job["knob"]) * direction
            waves = excerpt(run, ref, h, left, end)
            loss, metrics = objective.loss(
                objective.effects(waves, views), group, position, job["knob"]
            )
            rows.append({"loss": float(loss), **metrics})
    return dict(
        latent_mse=[v / max(weight, 1) for v in errors],
        reader_examples=len(rows),
        reader=means(rows),
    )


def fit(run, sae, baseline, objective, calibration, output):
    cohort, floors = calibration["cohort"], calibration["floors"]
    optim = torch.optim.Adam(sae.parameters(), lr=1e-4, foreach=True)
    shadow = torch.optim.Adam(baseline.parameters(), lr=1e-4, foreach=True)
    read_optim = torch.optim.Adam(objective.readers.parameters(), lr=3e-4, foreach=True)
    seen = [
        torch.full((FEATURES,), -1, dtype=torch.long, device=run.args.device)
        for _ in range(2)
    ]
    rng = np.random.default_rng(run.seed + 1204)
    cache = run.caches["train"]
    counts = np.zeros(len(cohort), dtype=np.int64)
    step = groups = frames = 0
    started = time.monotonic()
    if run.args.resume:
        saved = torch.load(
            output / "latest.pt", map_location=run.args.device, weights_only=True
        )
        sae.load_state_dict(saved["state_dict"])
        baseline.load_state_dict(saved["baseline_state_dict"])
        objective.readers.load_state_dict(saved["reader_state_dict"])
        state = saved["optimization"]
        for optimizer, values in zip((optim, shadow, read_optim), state["optimizers"]):
            optimizer.load_state_dict(values)
        seen, frames = state["seen"], state["frames"]
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if run.args.device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        step, groups = saved["steps"], saved["groups"]
        restore_trace(output, step)
        counts = np.asarray(saved["feature_exposure"], dtype=np.int64)
        clock = json.loads((output / "clock.json").read_text())
        started = time.monotonic() - resumed_seconds(
            clock, saved["seconds"], time.time()
        )
    else:
        write_json(
            output / "clock.json",
            dict(started_monotonic=started, utc_started=time.time()),
        )
    target_seconds, cap_seconds = paired_budget(
        run.args.warmup_seconds, run.args.minutes, run.args.max_minutes
    )
    hard_end = started + cap_seconds
    if time.monotonic() >= hard_end:
        raise ValueError("The combined warm-up/paired budget has already expired.")
    update_end = min(started + target_seconds - 300, hard_end - 600)
    history = (
        json.loads((output / "training.json").read_text())
        if run.args.resume and (output / "training.json").exists()
        else []
    )
    rolling = []
    trace_rows = []
    last_seconds = 1.0

    def snapshot(name, resumable=False):
        payload = dict(
            state_dict=sae.state_dict(),
            baseline_state_dict=baseline.state_dict(),
            reader_state_dict=objective.readers.state_dict(),
            observer=objective.observers.state_dict(),
            cohort=cohort,
            floors=floors,
            identity=run.identity,
            cache_sha256=sha256(run.args.cache / "cache.json"),
            seed=run.seed,
            steps=step,
            groups=groups,
            seconds=time.monotonic() - started,
            feature_exposure=counts.tolist(),
            beta=BETA,
            version=3,
            initialization="shared_latent_only_warmup",
            warmup_seconds=run.args.warmup_seconds,
        )
        if resumable:
            payload["optimization"] = dict(
                optimizers=[o.state_dict() for o in (optim, shadow, read_optim)],
                seen=seen,
                frames=frames,
                numpy_rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),
                cuda_rng=(
                    torch.cuda.get_rng_state_all()
                    if run.args.device.type == "cuda"
                    else []
                ),
            )
        save_best(output / name, payload)
        return payload

    def flush_trace():
        with (output / "sampling.jsonl").open("a", encoding="utf-8") as file:
            file.write("".join(json.dumps(row) + "\n" for row in trace_rows))
        trace_rows.clear()

    if not run.args.resume:
        print("Initial paired validation", flush=True)
        initial = validate(run, sae, baseline, objective, calibration, hard_end - 600)
        write_json(output / "initial_validation.json", initial)
        print("INITIAL", json.dumps(initial), flush=True)
        snapshot("latest.pt", True)
    progress = tqdm(
        total=max(0, update_end - started),
        initial=min(time.monotonic() - started, max(0, update_end - started)),
        desc="Paired SAE",
        unit="s",
        dynamic_ncols=True,
        mininterval=1,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}{postfix}]",
    )
    while (
        groups < run.args.groups
        and time.monotonic() + max(2 * last_seconds, 10) < update_end
    ):
        tick = time.monotonic()
        step += 1
        ids = rng.integers(len(cache.starts), size=BATCH)
        q, rho = prepare_batch(cache.batch(ids), run.mean, run.std)
        count = int(rho[:, CORE_SLICE].count_nonzero())
        for optimizer in (optim, shadow, read_optim):
            optimizer.zero_grad(set_to_none=True)
        loss_h = training_loss(sae, q, rho, seen[0], count, frames)
        loss_h.backward()
        recon_grad = float(
            torch.stack(
                [
                    p.grad.detach().square().sum()
                    for p in sae.parameters()
                    if p.grad is not None
                ]
            )
            .sum()
            .sqrt()
        )
        loss_b = training_loss(baseline, q, rho, seen[1], count, frames)
        loss_b.backward()
        frames += count
        index = int(rng.integers(len(cache.rows)))
        ref = run.reference("train", index)
        codes = run.codes(sae, ref)
        chosen = choose_group(codes, cohort, counts, rng)
        trace = dict(
            step=step,
            index=index,
            sample_id=cache.rows[index]["id"],
            pack=cache.rows[index]["pack"],
            crop_ids=ids.tolist(),
            intervention=False,
        )
        batch_metrics = []
        if chosen is not None:
            frame, group = chosen
            left, end = crop_bounds(ref, frame)
            with torch.no_grad():
                neutral = excerpt(run, ref, ref.h_ref, left, end)
            views = objective.views(neutral, rng)
            if views:
                knob = float(rng.uniform(-1, 1))
                alpha = gain_from_knob(knob)
                trace.update(
                    intervention=True,
                    frame=frame,
                    crop=[left, end],
                    knob=knob,
                    alpha=alpha,
                    atoms=cohort[group].tolist(),
                    eligible=cohort[codes[0, frame, cohort].detach() > 0].tolist(),
                    scopes=list(views),
                )
                for position, ci in enumerate(group):
                    atom = int(cohort[ci])
                    direction, radius = edit_direction(sae, codes, atom, floors[atom])
                    edited = ref.h_ref + run.std * BETA * alpha * direction
                    leaf = edited.detach().requires_grad_(True)
                    waves = excerpt(run, ref, leaf, left, end)
                    effect = objective.effects(waves, views)
                    loss, measured = objective.loss(
                        effect, group, position, knob, frozen=True
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite joint intervention loss.")
                    gradient = torch.autograd.grad(0.1 * loss / 4, leaf)[0]
                    edited.backward(gradient, retain_graph=position < 3)
                    read_loss, _ = objective.loss(
                        {k: v.detach() for k, v in effect.items()},
                        group,
                        position,
                        knob,
                    )
                    (read_loss / 4).backward()
                    measured.update(
                        intervention=float(loss.detach()),
                        h_gradient=float(gradient.norm()),
                        floor_hit=float(radius.detach() < floors[atom]),
                    )
                    for scope, wave in waves.items():
                        measured[f"{scope}/rms"] = float(
                            wave.detach().square().mean().sqrt()
                        )
                        measured[f"{scope}/below_floor"] = float(
                            measured[f"{scope}/rms"] < 1e-5
                        )
                    batch_metrics.append(measured)
                    counts[int(ci)] += 1
                objective.zero_loss(views, group).backward()
                groups += 1
            else:
                trace["skip_reason"] = "all_neutral_scopes_silent"
        else:
            trace["skip_reason"] = "fewer_than_four_cohort_features"
        trace_rows.append(trace)
        norm = torch.nn.utils.clip_grad_norm_(
            sae.parameters(), 1.0, error_if_nonfinite=True
        )
        torch.nn.utils.clip_grad_norm_(
            baseline.parameters(), 1.0, error_if_nonfinite=True
        )
        optim.step()
        shadow.step()
        if batch_metrics:
            read_optim.step()
        last_seconds = time.monotonic() - tick
        progress.set_postfix(
            step=step,
            groups=groups,
            main=f"{float(loss_h.detach()):.4f}",
            comparison=f"{float(loss_b.detach()):.4f}",
            int=f"{means(batch_metrics).get('intervention', 0):.3f}",
            refresh=False,
        )
        progress.update(min(time.monotonic() - started, progress.total) - progress.n)
        rolling.append(
            dict(
                latent_loss=float(loss_h.detach()),
                baseline_loss=float(loss_b.detach()),
                reconstruction_gradient=recon_grad,
                gradient_norm=float(norm),
                clipped=float(norm > 1),
                step_seconds=last_seconds,
                **means(batch_metrics),
            )
        )
        if step == 1 or step % 100 == 0:
            entry = dict(
                step=step,
                groups=groups,
                skipped=step - groups,
                seconds=time.monotonic() - started,
                **means(rolling),
            )
            history.append(entry)
            rolling.clear()
            write_json(output / "training.json", history)
            tqdm.write("TRAIN " + json.dumps(entry))
            flush_trace()
        if step % 1000 == 0:
            if time.monotonic() + 120 < update_end:
                progress.set_description("Paired validation")
                result = validate(
                    run, sae, baseline, objective, calibration, update_end - 30
                )
                result.update(step=step, groups=groups)
                with (output / "validation.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(result) + "\n")
                tqdm.write("VALIDATION " + json.dumps(result))
                progress.set_description("Paired SAE")
            snapshot("latest.pt", True)
        del codes, ref, loss_h, loss_b
    progress.close()
    print("Final paired validation and checkpoint", flush=True)
    flush_trace()
    snapshot("latest.pt", True)
    final = snapshot("final.pt")
    final_validation = validate(
        run, sae, baseline, objective, calibration, hard_end - 30
    )
    write_json(output / "final_validation.json", final_validation)
    write_json(
        output / "timing.json",
        dict(
            training_validation_seconds=time.monotonic() - started,
            warmup_seconds=run.args.warmup_seconds,
            total_training_validation_seconds=run.args.warmup_seconds
            + time.monotonic()
            - started,
            preparation_seconds=calibration["seconds"],
            hard_cap_seconds=run.args.max_minutes * 60,
        ),
    )
    print("FINAL_VALIDATION", json.dumps(final_validation), flush=True)
    return final


@torch.no_grad()
def screen(run, sae, baseline, calibration, output):
    start = time.monotonic()
    metadata = copy.deepcopy(calibration["selection"])
    jobs = {}
    for row in metadata:
        row["scores"] = [[], []]
        for index in row["examples"]:
            jobs.setdefault(index, []).append(row)
    for index, rows in tqdm(jobs.items(), desc="Paired joint screening"):
        ref = run.reference("val", index)
        neutral = joint_prefix(run.model, ref, ref.h_ref, ref.length)
        neutral_np = {k: v.flatten().cpu().numpy() for k, v in neutral.items()}
        for n, net in enumerate((sae, baseline)):
            codes = run.codes(net, ref)
            for row in rows:
                atom = row["atom"]
                direction, _ = edit_direction(
                    net, codes, atom, calibration["floors"][atom]
                )
                for alpha in (-16, -4, 4, 16):
                    h = ref.h_ref + run.std * BETA * alpha * direction
                    waves = joint_prefix(run.model, ref, h, ref.length)
                    measured = {
                        scope: difference_score(
                            wave.flatten().cpu().numpy(), neutral_np[scope]
                        )
                        for scope, wave in waves.items()
                    }
                    row["scores"][n].append(dict(index=index, alpha=alpha, **measured))
    for row in metadata:
        row["rank_scores"] = [
            float(
                np.median(
                    [
                        np.mean([r[s] for r in rows if r["index"] == i for s in SCOPES])
                        for i in row["examples"]
                    ]
                )
            )
            for rows in row["scores"]
        ]
    ranking = [
        sorted(
            (r for r in metadata if r["eligible"]),
            key=lambda r: (-r["rank_scores"][n], r["atom"]),
        )[:10]
        for n in (0, 1)
    ]
    result = dict(
        candidates=metadata,
        intervention_top=[r["atom"] for r in ranking[0]],
        baseline_top=[r["atom"] for r in ranking[1]],
        selection="Same initialization-selected validation examples; mean S/N/mix change over +/-4,+/-16; no test data",
        seconds=time.monotonic() - start,
    )
    write_json(output / "screening.json", result)
    activity = {}
    for label, net in (("intervention", sae), ("baseline", baseline)):
        values = clip_activity(net, run.caches["val"], run.mean, run.std)
        activity[label] = dict(
            never_active_fraction=float((values.max(0) == 0).mean()),
            clip_support=(values > 0).sum(0).tolist(),
        )
    write_json(output / "validation_activity.json", activity)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "noise-bank", "data-dir", "cache", "warmup", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--meta-file", default="metadata.json")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--groups", type=int, default=20000)
    parser.add_argument("--minutes", type=float, default=120)
    parser.add_argument("--max-minutes", type=float, default=180)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--audition-dir", type=Path)
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Defer screening and WAV/HTML rendering.",
    )
    args = parser.parse_args(argv)
    try:
        args.device = resolve_device(args.device)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    if not 10 < args.minutes <= args.max_minutes <= 180 or args.groups < 1:
        parser.error("Invalid time/update budget.")
    if args.render_only and (args.resume or args.train_only):
        parser.error("--render-only cannot be combined with --resume or --train-only.")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    run = RunData(args)
    torch.manual_seed(run.seed)
    warm = torch.load(args.warmup, map_location=args.device, weights_only=True)
    sae, baseline = fork_warmup(
        warm, run.identity, sha256(args.cache / "cache.json"), args.device
    )
    args.warmup_seconds = warm.get("warmup_seconds", float("nan"))
    paired_budget(args.warmup_seconds, args.minutes, args.max_minutes)
    if args.resume or args.render_only:
        protocol = json.loads((args.output / "protocol.json").read_text())
        if (
            protocol.get("initialization") != "shared_latent_only_warmup"
            or protocol["warm_start_sha256"] != sha256(args.warmup)
            or protocol["groups_target"] != args.groups
            or protocol["minutes_target"] != args.minutes
            or protocol["hard_cap_minutes"] != args.max_minutes
        ):
            raise ValueError("Resume protocol mismatch.")
        calibration = torch.load(
            args.output / "calibration.pt", map_location=args.device, weights_only=True
        )
        objective = JointObjective(len(calibration["cohort"])).to(args.device)
        objective.observers.load_state_dict(calibration["observer"])
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(
            args.output / "protocol.json",
            dict(
                identity=run.identity,
                initialization="shared_latent_only_warmup",
                warm_start_sha256=sha256(args.warmup),
                cache_sha256=warm["cache_sha256"],
                warmup_seconds=args.warmup_seconds,
                warmup_selected_step=warm["step"],
                optimizer_initialization="both SAE Adam states reset; identical initial weights",
                budget_scope="warm-up plus paired fit/validation; cache/calibration/audition excluded",
                groups_target=args.groups,
                minutes_target=args.minutes,
                hard_cap_minutes=args.max_minutes,
                seed=run.seed,
                transient_training=False,
                knob_map="sign(k)*(65**abs(k)-1), k uniform[-1,1]",
                scopes=SCOPES,
                readout_tau=0.1,
                loss_weight=0.1,
                sae_lr=1e-4,
                reader_lr=3e-4,
            ),
        )
        units = calibrate_dictionary(run, sae, args.output)
        objective = JointObjective(len(units["cohort"])).to(args.device)
        calibration = prepare(run, sae, objective, units, args.output)
    if args.render_only:
        final = torch.load(
            args.output / "final.pt", map_location=args.device, weights_only=True
        )
        sae.load_state_dict(final["state_dict"])
        baseline.load_state_dict(final["baseline_state_dict"])
    else:
        fit(run, sae, baseline, objective, calibration, args.output)
    if not args.train_only:
        screen(run, sae, baseline, calibration, args.output)
        from stamp.control.audition import render

        audition_args = copy.copy(args)
        audition_args.run = args.output
        audition_args.selection = args.output / "screening.json"
        audition_args.output = args.audition_dir or args.output / "audition"
        render(audition_args)
    if sha256(args.checkpoint) != run.identity["checkpoint"]:
        raise RuntimeError("Recon checkpoint changed.")
    print("COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
