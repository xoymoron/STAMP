import numpy as np
import torch

from stamp.control import ControlSAE, FrameCache


def test_crop_halo_matches_full_clip_and_masks_neighbor(tmp_path):
    rng = np.random.default_rng(21)
    values = rng.standard_normal((281, 256), dtype=np.float32)
    np.save(tmp_path / "train.npy", values)
    cache = FrameCache(
        tmp_path, "train", [{"samples": 279 * 128 + 37}, {"samples": 128}]
    )
    sae = ControlSAE(features=12)
    full = torch.from_numpy(values[:280].copy())[None]
    full_mask = torch.ones((1, 280), dtype=torch.bool)
    full_out, _ = sae.reconstruct(full, full_mask)
    x, weight = cache.batch(np.array([1]))
    out, _ = sae.reconstruct(x, weight > 0)
    torch.testing.assert_close(out[0, 8:32], full_out[0, 256:])
    assert weight[0, 31].item() == 37 / 128
    assert not weight[0, 32:].any()
