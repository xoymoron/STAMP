"""SAE initialization, feature selection and training budgets."""

import copy
import math

import numpy as np

from stamp.control import FEATURES
from stamp.control.editing import InterventionSAE

WARMUP_KIND = "latent_reconstruction_warmup_v1"


def fork_warmup(artifact, identity, cache_hash, device):
    if (
        artifact.get("kind") != WARMUP_KIND
        or artifact.get("identity") != identity
        or artifact.get("cache_sha256") != cache_hash
        or any(
            k in artifact
            for k in ("reader_state_dict", "baseline_state_dict", "cohort")
        )
        or artifact.get("step", 0) < 1
    ):
        raise ValueError(
            "Expected a fresh latent-only warm-up from this checkpoint/cache."
        )
    main = InterventionSAE().to(device)
    main.load_state_dict(artifact["state_dict"])
    return main, copy.deepcopy(main)


def paired_budget(warmup_seconds, minutes, max_minutes):
    """Charge the full warm-up invocation, including its validation passes."""
    if not math.isfinite(warmup_seconds) or warmup_seconds < 0:
        raise ValueError("Missing or invalid warm-up duration.")
    if not 10 < minutes <= max_minutes <= 180:
        raise ValueError("Require 10 < total target <= total cap <= 180 minutes.")
    target, cap = minutes * 60 - warmup_seconds, max_minutes * 60 - warmup_seconds
    if target <= 600 or cap <= 600:
        raise ValueError("Warm-up leaves insufficient paired-training budget.")
    return target, cap


def resumed_seconds(clock, retained_seconds, utc_now):
    """Charge interruption time too; monotonic timestamps do not survive reboot."""
    elapsed = utc_now - clock["utc_started"]
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Wall clock moved backwards; cannot resume budget safely.")
    return max(elapsed, retained_seconds)


def select_cohort(activity, rows):
    support = (activity > 0).sum(0)
    order = sorted(range(activity.shape[1]), key=lambda i: (-int(support[i]), i))
    cohort = [
        i
        for i in order
        if support[i] >= 16
        and len({rows[j]["pack"] for j in np.flatnonzero(activity[:, i] > 0)}) >= 2
    ][:64]
    if len(cohort) < 4:
        raise ValueError("Fewer than four train-supported intervention features.")
    return cohort


def validate_atoms(atoms, cohort):
    atoms = list(atoms)
    if (
        len(atoms) != 2
        or len(set(atoms)) != 2
        or any(not 0 <= i < FEATURES or i not in cohort for i in atoms)
    ):
        raise ValueError(
            "Choose two distinct validation-audition atoms from this run's cohort."
        )
    return atoms
