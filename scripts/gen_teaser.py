"""Generate the README teaser image: one image+overlay tile per prepared dataset.

Scans ``./datasets/<name>/`` for splits JSONs (including the multilabel
``.npy`` mask variants), picks the first train sample with a target,
renders an image / mask overlay, and composes the tiles into a grid
written to ``./assets/teaser.png``.

Run from repo root:

    python scripts/gen_teaser.py [--tile-size 256] [--cols 4]

Datasets that have no on-disk prep are silently skipped. Re-run after
preparing more datasets to refresh the teaser.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from visualization.visualize import visualize_multiclass, get_colors  # noqa: E402


# Map of dataset directory name -> (splits JSON filename, pretty label).
DATASETS = [
    ("drive",          "drive_splits.json",          "DRIVE\nfundus"),
    ("stare",          "stare_splits.json",          "STARE\nfundus"),
    ("chasedb1",       "chasedb1_splits.json",       "CHASE_DB1\nfundus"),
    ("hrf",            "hrf_splits.json",            "HRF\nfundus"),
    ("ravir",          "ravir_splits.json",          "RAVIR\nIR retina"),
    ("duke",           "duke_splits.json",           "DUKE\nOCT"),
    ("jsrt",           "jsrt_splits.json",           "JSRT\nchest X-ray"),
    ("paxray",         "paxray.json",                "PAXRay\nchest X-ray"),
    ("paxraypp",       "paxraypp.json",              "PAXRay++\nchest X-ray"),
    ("montgomery",     "montgomery_splits.json",     "Montgomery\nchest X-ray"),
    ("teeth",          "train.json",                 "Teeth\npanoramic X-ray"),
    ("bs80k",          "bs80k_splits.json",          "BS80k\nscintigraphy"),
    ("medaka_heart",   "medaka_heart_splits.json",   "MedakaHeart\nmicroscopy"),
    ("jump_broadcast", "jump_broadcast_splits.json", "JumpBroadcast\nRGB video"),
]


def load_sample(root: str, splits_json: str) -> Optional[tuple]:
    """Return (image_array, mask_array, label_dict) for the first usable sample.

    Returns None if the dataset is not prepared, or no train entry has a
    decodable image+target. Handles both the multilabel ``.npy`` stack
    format (collapses to argmax-style mask for display) and the
    multiclass PNG format.
    """
    if not os.path.isfile(splits_json):
        return None
    try:
        with open(splits_json) as f:
            splits = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    label_dict = splits.get("label_dict")
    if not label_dict:
        return None

    for split in ("train", "val", "test"):
        for entry in splits.get(split, []):
            target = entry.get("target")
            if not target:
                continue
            image_path = os.path.join(root, entry["image"])
            target_path = os.path.join(root, target)
            if not (os.path.isfile(image_path) and os.path.isfile(target_path)):
                continue
            try:
                image = np.array(Image.open(image_path).convert("RGB"))
            except Exception:
                continue

            if target_path.endswith(".npy"):
                try:
                    stack = np.load(target_path)
                except Exception:
                    continue
                if stack.ndim == 2:
                    mask = stack.astype(np.int32)
                else:
                    # Collapse multilabel stack to a single-channel display mask:
                    # pixel = (first non-zero channel index + 1), else 0.
                    binary = stack > 0
                    mask = np.zeros(binary.shape[1:], dtype=np.int32)
                    # iterate from last to first so lower channels win on overlap
                    for ch in range(binary.shape[0] - 1, -1, -1):
                        mask[binary[ch]] = ch + 1
            else:
                try:
                    mask = np.array(Image.open(target_path))
                except Exception:
                    continue
                if mask.ndim == 3:
                    mask = mask[..., 0]
                mask = mask.astype(np.int32)

            # Build a display label_dict sized to mask values for color count.
            n_classes = max(mask.max() + 1, len(label_dict))
            display_label_dict = {i: f"c{i}" for i in range(n_classes)}
            return image, mask, display_label_dict

    return None


def make_overlay_tile(
    image: np.ndarray,
    mask: np.ndarray,
    label_dict: dict,
    title: str,
    tile_size: int,
) -> Image.Image:
    """Build one (image | overlay) tile with caption."""
    overlay_img = visualize_multiclass(image, mask, label_dict)
    src_img = Image.fromarray(image)

    # Resize both to tile_size x tile_size (square), preserving aspect by
    # padding on a black canvas.
    def fit(im):
        im = im.copy()
        im.thumbnail((tile_size, tile_size), Image.LANCZOS)
        canvas = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
        canvas.paste(im, ((tile_size - im.width) // 2, (tile_size - im.height) // 2))
        return canvas

    left = fit(src_img)
    right = fit(overlay_img)

    pair = Image.new("RGB", (tile_size * 2 + 4, tile_size + 36), (12, 12, 12))
    pair.paste(left, (0, 36))
    pair.paste(right, (tile_size + 4, 36))

    draw = ImageDraw.Draw(pair)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
    draw.text((6, 4), title.replace("\n", " — "), fill=(230, 230, 230), font=font)
    return pair


def compose_grid(tiles: list, cols: int) -> Image.Image:
    if not tiles:
        raise RuntimeError("No prepared datasets found to render.")
    tile_w, tile_h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    pad = 8
    canvas_w = cols * tile_w + (cols + 1) * pad
    canvas_h = rows * tile_h + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (24, 24, 24))
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, cols)
        x = pad + c * (tile_w + pad)
        y = pad + r * (tile_h + pad)
        canvas.paste(tile, (x, y))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-size", type=int, default=256,
                        help="Per-image thumbnail edge length in pixels (default: 256).")
    parser.add_argument("--cols", type=int, default=4,
                        help="Number of tile columns in the grid (default: 4).")
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "assets", "teaser.png"),
                        help="Output path (default: ./assets/teaser.png).")
    args = parser.parse_args()

    datasets_dir = os.path.join(REPO_ROOT, "datasets")
    tiles = []
    skipped = []
    for dir_name, splits_name, title in DATASETS:
        root = os.path.join(datasets_dir, dir_name)
        splits_json = os.path.join(root, splits_name)
        sample = load_sample(root, splits_json)
        if sample is None:
            skipped.append(dir_name)
            continue
        image, mask, label_dict = sample
        tile = make_overlay_tile(image, mask, label_dict, title, args.tile_size)
        tiles.append(tile)
        print(f"rendered {dir_name}")

    if skipped:
        print(f"skipped (no prepared data): {', '.join(skipped)}")

    canvas = compose_grid(tiles, args.cols)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    canvas.save(args.out)
    print(f"wrote {args.out} ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    main()
