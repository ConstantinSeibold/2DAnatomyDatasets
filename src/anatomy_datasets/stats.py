"""Channel-wise image statistics for per-dataset normalization."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
from PIL import Image


def compute_image_stats(
    image_paths: Iterable[str],
    sample_cap: int = 200,
    rng_seed: int = 0,
) -> dict:
    """Channel-wise mean/std over (up to ``sample_cap``) images, in [0,1] space.

    Returns ``{"mean": [r, g, b], "std": [r, g, b]}`` suitable for embedding
    directly in the splits JSON. Reads images as RGB regardless of source
    format so the stats are always 3-channel.

    The sample cap keeps this cheap for large datasets (BS80k, JumpBroadcast)
    where iterating every training image during prepare would dominate runtime.
    Override ``sample_cap`` (or pass ``sample_cap=len(image_paths)``) to use
    every image.
    """
    paths: List[str] = list(image_paths)
    if not paths:
        raise ValueError("compute_image_stats() received an empty image list")

    if len(paths) > sample_cap:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(paths), size=sample_cap, replace=False)
        paths = [paths[i] for i in idx]

    sums = np.zeros(3, dtype=np.float64)
    sqs = np.zeros(3, dtype=np.float64)
    n_pixels = 0

    for p in paths:
        img = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        flat = img.reshape(-1, 3)
        sums += flat.sum(axis=0)
        sqs += (flat ** 2).sum(axis=0)
        n_pixels += flat.shape[0]

    mean = sums / n_pixels
    var = sqs / n_pixels - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))

    return {"mean": mean.tolist(), "std": std.tolist()}
