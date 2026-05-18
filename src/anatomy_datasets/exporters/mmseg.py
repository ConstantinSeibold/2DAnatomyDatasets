"""mmsegmentation-format exporter.

Layout produced
---------------
Multiclass mode (default for multiclass-source datasets):
::

    <out_dir>/
    ├── img_dir/
    │   ├── train/<stem>.png
    │   ├── val/<stem>.png
    │   └── test/<stem>.png
    ├── ann_dir/
    │   ├── train/<stem>.png            # palette PNG, mode='P', class IDs as pixel values
    │   ├── val/<stem>.png
    │   └── test/<stem>.png
    ├── palette.json                     # {class_id: [r,g,b]}
    └── meta.json                        # license/citation/source/seed/normalization

Per-channel mode (``mode="per_channel"``) for multilabel ``.npy`` sources:
emits one mmseg dataset per non-empty class channel under
``<out_dir>/<class_name>/`` with the same layout as multiclass. Each
per-class dataset is a 2-class (background/foreground) problem. This
multiplies disk use but is the only path to consuming multilabel data
with stock mmseg without a custom Dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image

from anatomy_datasets.registry import get_dataset_info


# --------------------------------------------------------------------------- #
# Palette generation
# --------------------------------------------------------------------------- #

def _default_palette(n_classes: int) -> List[List[int]]:
    """Deterministic, visually-distinct palette for ``n_classes`` indices.

    Index 0 is always black (background convention). Other indices use a
    simple HSV ramp converted to RGB so colors stay distinguishable up to
    ~256 classes. For more, colors will start to clash but indices remain
    unique - palette is only for visualization, not training.
    """
    import colorsys

    palette = [[0, 0, 0]]
    for i in range(1, n_classes):
        h = (i / max(1, n_classes - 1)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        palette.append([int(r * 255), int(g * 255), int(b * 255)])
    return palette


def _save_palette_png(mask: np.ndarray, palette: List[List[int]], out_path: str) -> None:
    """Save a 2D integer mask as a palette PNG (mmseg convention)."""
    flat_palette = []
    for rgb in palette:
        flat_palette.extend(rgb)
    # Pillow requires exactly 768 entries
    while len(flat_palette) < 768:
        flat_palette.extend([0, 0, 0])
    flat_palette = flat_palette[:768]

    img = Image.fromarray(mask.astype(np.uint8), mode="P")
    img.putpalette(flat_palette)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, format="PNG")


def _save_image_copy(src_path: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    # Always re-encode as PNG so the mmseg loader sees a uniform extension
    with Image.open(src_path) as im:
        im.convert("RGB").save(dst_path, format="PNG")


# --------------------------------------------------------------------------- #
# Multiclass: single integer mask per image
# --------------------------------------------------------------------------- #

def _load_multiclass_mask(target_path: str) -> np.ndarray:
    """Load a multiclass mask. PNG → integer pixel values; .npy → first
    channel if 3D, else 2D array."""
    if target_path.endswith(".npy"):
        arr = np.load(target_path)
        if arr.ndim == 3:
            # Multilabel input wrongly fed to multiclass path; pick argmax
            # so the export is still defined.
            arr = arr.argmax(axis=0)
        return arr.astype(np.int32)
    with Image.open(target_path) as im:
        return np.array(im.convert("L"), dtype=np.int32)


def _export_multiclass(
    splits_json_dict: dict,
    root_dir: str,
    out_dir: str,
    splits: Iterable[str],
    palette: List[List[int]],
) -> dict:
    label_dict = splits_json_dict["label_dict"]
    counts = {}
    for split in splits:
        entries = splits_json_dict.get(split, [])
        n_with_target = 0
        for entry in entries:
            stem = os.path.splitext(os.path.basename(entry["image"]))[0]
            image_src = os.path.join(root_dir, entry["image"])
            img_dst = os.path.join(out_dir, "img_dir", split, f"{stem}.png")
            _save_image_copy(image_src, img_dst)

            target = entry.get("target")
            if target is None:
                continue  # skip ann_dir write; mmseg test set without GT
            mask = _load_multiclass_mask(os.path.join(root_dir, target))
            ann_dst = os.path.join(out_dir, "ann_dir", split, f"{stem}.png")
            _save_palette_png(mask, palette, ann_dst)
            n_with_target += 1
        counts[split] = {"images": len(entries), "with_target": n_with_target}

    with open(os.path.join(out_dir, "palette.json"), "w") as f:
        json.dump(
            {str(i): rgb for i, rgb in enumerate(palette)},
            f,
            indent=2,
        )

    return counts


# --------------------------------------------------------------------------- #
# Per-channel: explode multilabel .npy stack into N binary mmseg datasets
# --------------------------------------------------------------------------- #

def _export_per_channel(
    splits_json_dict: dict,
    root_dir: str,
    out_dir: str,
    splits: Iterable[str],
) -> dict:
    label_dict = {int(k): v for k, v in splits_json_dict["label_dict"].items()}
    binary_palette = [[0, 0, 0], [255, 255, 255]]

    counts_by_class: dict = {}
    for channel_idx, class_name in label_dict.items():
        sub_out = os.path.join(out_dir, _safe_dirname(class_name))
        counts = {}
        for split in splits:
            entries = splits_json_dict.get(split, [])
            n_with_target = 0
            for entry in entries:
                stem = os.path.splitext(os.path.basename(entry["image"]))[0]
                image_src = os.path.join(root_dir, entry["image"])
                img_dst = os.path.join(sub_out, "img_dir", split, f"{stem}.png")
                _save_image_copy(image_src, img_dst)

                target = entry.get("target")
                if target is None:
                    continue
                target_path = os.path.join(root_dir, target)
                if not target_path.endswith(".npy"):
                    raise NotImplementedError(
                        "per_channel mmseg export only supports multilabel "
                        f".npy mask sources; got {target_path!r}"
                    )
                stack = np.load(target_path)
                channel = (stack[channel_idx] > 0).astype(np.uint8)
                ann_dst = os.path.join(
                    sub_out, "ann_dir", split, f"{stem}.png"
                )
                _save_palette_png(channel, binary_palette, ann_dst)
                n_with_target += 1
            counts[split] = {"images": len(entries), "with_target": n_with_target}

        with open(os.path.join(sub_out, "palette.json"), "w") as f:
            json.dump({"0": [0, 0, 0], "1": [255, 255, 255]}, f, indent=2)
        counts_by_class[class_name] = counts

    return counts_by_class


def _safe_dirname(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).lower()


# --------------------------------------------------------------------------- #
# Top-level entrypoint
# --------------------------------------------------------------------------- #

def to_mmseg(
    splits_json: str,
    root_dir: str,
    out_dir: str,
    mode: str = "auto",
    splits: Iterable[str] = ("train", "val", "test"),
    dataset_name: Optional[str] = None,
) -> dict:
    """Export a splits-JSON dataset to mmsegmentation layout.

    Parameters
    ----------
    mode : ``"multiclass"``, ``"per_channel"``, or ``"auto"`` (pick based
        on whether targets are PNG masks or ``.npy`` stacks).
    """
    with open(splits_json) as f:
        source = json.load(f)

    label_dict = source["label_dict"]
    dataset_name = dataset_name or source.get(
        "name", os.path.splitext(os.path.basename(splits_json))[0]
    )

    if mode == "auto":
        mode = _auto_mode(source, root_dir, splits)

    os.makedirs(out_dir, exist_ok=True)

    if mode == "multiclass":
        n_classes = max(int(k) for k in label_dict) + 1 if label_dict else 1
        palette = _default_palette(max(n_classes, 2))
        counts = _export_multiclass(source, root_dir, out_dir, splits, palette)
    elif mode == "per_channel":
        counts = _export_per_channel(source, root_dir, out_dir, splits)
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    meta = _build_meta(dataset_name, source, mode, counts)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {"mode": mode, "meta_path": os.path.join(out_dir, "meta.json"), "counts": counts}


def _auto_mode(source: dict, root_dir: str, splits: Iterable[str]) -> str:
    """Sniff first entry's target file extension to choose mode."""
    for split in splits:
        for entry in source.get(split, []):
            target = entry.get("target")
            if target is None:
                continue
            return "per_channel" if target.endswith(".npy") else "multiclass"
    return "multiclass"  # default if no targets at all


def _build_meta(dataset_name: str, source: dict, mode: str, counts: dict) -> dict:
    try:
        reg = get_dataset_info(dataset_name)
        reg_info = {
            "modality": reg.modality,
            "license": reg.license,
            "source_url": reg.source_url,
            "citation": reg.citation,
        }
    except KeyError:
        reg_info = {}

    return {
        "dataset": dataset_name,
        "format": "mmseg_v1",
        "mode": mode,
        "label_dict": source.get("label_dict"),
        "seed": source.get("seed"),
        "normalization": source.get("normalization"),
        "version": source.get("version", date.today().isoformat()),
        "counts": counts,
        "source_splits_json": os.path.basename(
            "_unknown_"
        ),
        **reg_info,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--mode", default="auto",
        choices=("auto", "multiclass", "per_channel"),
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--splits-only", nargs="+", default=None)
    args = parser.parse_args()

    result = to_mmseg(
        splits_json=args.splits,
        root_dir=args.root,
        out_dir=args.out,
        mode=args.mode,
        splits=tuple(args.splits_only) if args.splits_only else ("train", "val", "test"),
        dataset_name=args.dataset,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
