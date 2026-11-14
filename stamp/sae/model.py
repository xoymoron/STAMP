"""Framewise JumpReLU dictionary learning for cached recon latents."""

import math
from pathlib import Path

import torch
from torch import nn

from stamp.sae.corpus import CHANNELS


class _JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, threshold, bandwidth):
        active = z > threshold
        ctx.save_for_backward(z, threshold, active)
        ctx.bandwidth = bandwidth
        return z * active, active.to(z.dtype)

    @staticmethod
    def backward(ctx, magnitude_grad, count_grad):
        z, threshold, active = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        # Eq. (11)-(12): only the threshold uses a surrogate derivative.
        window = ((z - threshold).abs() < bandwidth / 2) / bandwidth
        threshold_grad = -window * (threshold * magnitude_grad + count_grad)
        return (
            magnitude_grad * active,
            threshold_grad.sum_to_size(threshold.shape),
            None,
        )


def activation(z, threshold, bandwidth):
    return _JumpReLU.apply(z, threshold, bandwidth)


class JumpReLUSAE(nn.Module):
    def __init__(self, mean, scale, *, width=512, bandwidth=0.001):
        super().__init__()
        self.width = width
        self.bandwidth = bandwidth
        self.register_buffer("mean", mean.clone())
        self.register_buffer("scale", scale.clone())
        self.encoder = nn.Linear(CHANNELS, width)
        self.decoder = nn.Linear(width, CHANNELS)
        self.log_threshold = nn.Parameter(torch.full((width,), math.log(0.001)))
        with torch.no_grad():
            self.decoder.weight.normal_()
            self.normalize_decoder()
            self.encoder.weight.copy_(self.decoder.weight.T)
            self.encoder.bias.zero_()
            self.decoder.bias.zero_()

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.div_(self.decoder.weight.norm(dim=0).clamp_min(1e-12))

    def forward(self, x):
        z = self.encoder(x - self.decoder.bias).relu()
        c, active = activation(z, self.log_threshold.exp(), self.bandwidth)
        return self.decoder(c), c, active

    def encode(self, h):
        return self((h - self.mean) / self.scale)[1]

    def approximate(self, h):
        return self.mean + self.scale * self((h - self.mean) / self.scale)[0]

    def edit(self, h, delta):
        return h + self.scale * (delta @ self.decoder.weight.T)


@torch.no_grad()
def measure(model, corpus):
    device = model.mean.device
    first = torch.zeros(CHANNELS, dtype=torch.float64, device=device)
    second = torch.zeros((), dtype=torch.float64, device=device)
    err = torch.zeros_like(second)
    weighted_l0 = torch.zeros_like(second)
    counts = torch.zeros(model.width, dtype=torch.long, device=device)
    histogram = torch.zeros(model.width + 1, dtype=torch.long, device=device)
    frame_counts = torch.zeros_like(counts)
    for h, weights in corpus.weighted_clips():
        weights = weights.to(device)
        x = (h.to(device) - model.mean) / model.scale
        predicted, _, active = model(x)
        framed_l0 = active.sum(dim=1).long()
        histogram.scatter_add_(0, framed_l0, torch.ones_like(framed_l0))
        used = active.bool()
        counts += used.any(dim=0)
        frame_counts += used.sum(dim=0)
        weighted_l0 += weights @ framed_l0.double()
        x = x.double()
        first += weights @ x
        second += weights @ x.square().sum(dim=1)
        err += weights @ (predicted - x).square().sum(dim=1)
    variance = second - first.square().sum() / len(corpus)
    return {
        "clips": len(corpus),
        "frames": corpus.frames,
        "mean_l0": float(weighted_l0 / len(corpus)),
        "frame_l0_histogram": histogram.tolist(),
        "latent_mse_per_frame": float(err / len(corpus)),
        "fvu": float(err / variance) if variance > 0 else None,
        "feature_clip_counts": counts.tolist(),
        "feature_frame_counts": frame_counts.tolist(),
        "inactive_features": int(counts.eq(0).sum()),
    }


def fit(
    train,
    val,
    *,
    width,
    steps,
    batch_size,
    seed,
    penalty,
    device="cpu",
    lr=7e-5,
    lr_warmup=1000,
    penalty_warmup=2000,
):
    if train.source != val.source:
        raise ValueError(
            "Training and validation caches must use the same recon assets."
        )
    if min(width, steps, batch_size) < 1 or lr <= 0 or penalty < 0:
        raise ValueError(
            "Use positive training dimensions/rate and a nonnegative penalty."
        )
    generator = torch.Generator().manual_seed(seed)
    mean, scale = train.normalization()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = JumpReLUSAE(mean, scale, width=width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.0, 0.999))
    log = []
    for step in range(1, steps + 1):
        x = (train.sample(batch_size, generator).to(device) - model.mean) / model.scale
        lr_progress = min(step / lr_warmup, 1) if lr_warmup else 1
        optimizer.param_groups[0]["lr"] = lr * (
            0.55 - 0.45 * math.cos(math.pi * lr_progress)
        )
        weight = penalty * (min(step / penalty_warmup, 1) if penalty_warmup else 1)
        predicted, _, active = model(x)
        error = (predicted - x).square().sum(dim=1).mean()
        mean_l0 = active.sum(dim=1).mean()
        loss = error + weight * mean_l0
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.normalize_decoder()
        if step == 1 or step % 1000 == 0 or step == steps:
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite SAE loss at step {step}.")
            log.append(
                {
                    "step": step,
                    "batch_mse": error.item(),
                    "batch_l0": mean_l0.item(),
                    "penalty": weight,
                }
            )
    model.eval().requires_grad_(False)
    log[-1]["validation"] = measure(model, val)
    log[-1]["threshold_quantiles"] = (
        model.log_threshold.exp()
        .quantile(model.mean.new_tensor([0.1, 0.5, 0.9]))
        .tolist()
    )
    return model, log


def save_sae(path, model, metadata):
    with Path(path).open("xb") as stream:
        torch.save(
            {
                "version": 1,
                "width": model.width,
                "bandwidth": model.bandwidth,
                "metadata": metadata,
                "state": model.state_dict(),
            },
            stream,
        )


def read_sae(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["version"] != 1:
        raise ValueError("Unsupported JumpReLU SAE checkpoint version.")
    state = saved["state"]
    model = JumpReLUSAE(
        state["mean"],
        state["scale"],
        width=saved["width"],
        bandwidth=saved["bandwidth"],
    )
    model.load_state_dict(state)
    return model.to(device).eval().requires_grad_(False), saved["metadata"]
