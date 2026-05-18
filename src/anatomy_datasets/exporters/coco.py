"""COCO-format exporter (for mmdetection, detectron2, generic COCO consumers).

Layout produced
---------------
::

    <out_dir>/
    ├── images/                            # symlinks (default) or copies
    │   └── <stem>.<ext>
    ├── annotations/
    │   ├── train.json                     # monolithic COCO
    │   ├── val.json
    │   └── test.json
    └── meta.json                          # license/citation/source/seed/normalization

Source-format handling
----------------------

- **Multilabel ``.npy`` mask stacks** (PAXRay, PAXRay++, BS80k): each
  non-empty channel becomes one COCO annotation with an RLE
  ``segmentation`` field. ``category_id`` = channel index.
- **Multiclass PNG masks** (DRIVE, STARE, CHASE_DB1, RAVIR, ...): each
  unique non-zero pixel value becomes one COCO annotation (semantic-
  segmentation semantics — no connected-component splitting).
  ``category_id`` = pixel value.
- **Monolithic COCO** input (already-COCO datasets, e.g. Teeth output of
  ``prepare_teeth.py``): each split JSON gets metadata merged into its
  ``info`` block. Use ``anatomy_datasets.add_metadata_to_splits_json``
  directly for that path - this exporter is for the splits-JSON → COCO
  conversion direction.

Big-dataset auto-shard
----------------------
If the projected total annotations exceed ``shard_threshold`` (default
50_000), the exporter delegates to ``anatomy_datasets.formats.sharded_coco``
to avoid producing a multi-GB JSON. Override with ``shard_threshold=0``
to force sharded output or ``shard_threshold=float("inf")`` to force
monolithic.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import date
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image

from anatomy_datasets.formats.sharded_coco import (
    _link_image,
    binary_mask_to_rle,
    write_sharded_coco,
)
from anatomy_datasets.registry import get_dataset_info


DEFAULT_SHARD_THRESHOLD = 50_000


def _categories_from_label_dict(label_dict: dict) -> List[dict]:
    cats = []
    for key, name in label_dict.items():
        cats.append(
            {"id": int(key), "name": str(name), "supercategory": ""}
        )
    cats.sort(key=lambda c: c["id"])
    return cats


def _coco_info(dataset_name: str, source_splits: dict) -> dict:
    """Build a COCO ``info`` block from the registry + source splits metadata."""
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
        "description": dataset_name,
        "version": source_splits.get("version", date.today().isoformat()),
        "year": date.today().year,
        "contributor": "",
        "date_created": date.today().isoformat(),
        "seed": source_splits.get("seed"),
        "normalization": source_splits.get("normalization"),
        **reg_info,
    }


def _project_annotation_count(source_splits: dict) -> int:
    """Cheap upper-bound on total annotations across all splits.

    For multilabel .npy sources we don't open the .npy files (too slow);
    we assume worst case = (#images) * (#categories). This intentionally
    over-estimates - safer to err toward sharding than to blow up RAM.
    """
    n_cats = len(source_splits.get("label_dict", {}))
    n_imgs = sum(
        len(source_splits.get(s, []))
        for s in ("train", "val", "test")
    )
    return n_cats * n_imgs


def _entry_to_coco_annotations(
    entry: dict,
    root_dir: str,
    image_id: int,
    base_ann_id: int,
    valid_cat_ids: set,
) -> tuple:
    """Returns (annotations_list, next_ann_id)."""
    target = entry.get("target")
    if target is None:
        return [], base_ann_id

    target_path = os.path.join(root_dir, target)
    annotations = []
    ann_id = base_ann_id

    if target_path.endswith(".npy"):
        stack = np.load(target_path)
        if stack.ndim != 3:
            raise ValueError(
                f"Expected mask stack [C,H,W], got shape {stack.shape}"
            )
        binary = stack > 0
        for channel_idx in range(binary.shape[0]):
            if channel_idx not in valid_cat_ids:
                continue
            m = binary[channel_idx]
            if not m.any():
                continue
            rle = binary_mask_to_rle(m)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(channel_idx),
                    "segmentation": rle,
                    "area": int(m.sum()),
                    "bbox": _rle_bbox(rle),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
        return annotations, ann_id

    # Multiclass PNG path: one annotation per unique non-zero pixel value.
    mask = np.array(Image.open(target_path).convert("L"))
    for class_id in np.unique(mask):
        cid = int(class_id)
        if cid == 0:
            continue
        if cid not in valid_cat_ids:
            continue
        m = (mask == class_id)
        if not m.any():
            continue
        rle = binary_mask_to_rle(m.astype(np.uint8))
        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": cid,
                "segmentation": rle,
                "area": int(m.sum()),
                "bbox": _rle_bbox(rle),
                "iscrowd": 0,
            }
        )
        ann_id += 1
    return annotations, ann_id


def _rle_bbox(rle: dict) -> list:
    from pycocotools import mask as mask_utils

    return mask_utils.toBbox(rle).tolist()


def _build_split_coco(
    split_entries: List[dict],
    root_dir: str,
    out_dir: str,
    image_link_mode: str,
    info: dict,
    categories: List[dict],
    image_id_start: int,
    ann_id_start: int,
) -> tuple:
    """Build one COCO dict for one split. Returns (coco_dict, next_image_id, next_ann_id)."""
    images = []
    annotations = []
    valid_cat_ids = {c["id"] for c in categories}

    image_id = image_id_start
    ann_id = ann_id_start
    images_subdir = "images"

    for entry in split_entries:
        image_rel = entry["image"]
        image_src = os.path.join(root_dir, image_rel)
        with Image.open(image_src) as im:
            width, height = im.size

        ext = os.path.splitext(image_rel)[1] or ".png"
        file_name = f"{image_id:08d}{ext}"
        if image_link_mode != "reference":
            _link_image(
                image_src,
                os.path.join(out_dir, images_subdir, file_name),
                image_link_mode,
            )
        else:
            file_name = image_rel

        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "height": int(height),
                "width": int(width),
            }
        )

        anns, ann_id = _entry_to_coco_annotations(
            entry, root_dir, image_id, ann_id, valid_cat_ids
        )
        annotations.extend(anns)
        image_id += 1

    coco = {
        "info": info,
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    return coco, image_id, ann_id


def to_coco(
    splits_json: str,
    root_dir: str,
    out_dir: str,
    image_link_mode: str = "symlink",
    splits: Iterable[str] = ("train", "val", "test"),
    dataset_name: Optional[str] = None,
    shard_threshold: int = DEFAULT_SHARD_THRESHOLD,
) -> dict:
    """Export a splits-JSON dataset to COCO layout.

    Returns a small summary dict (paths + counts). The actual JSON files
    are persisted to ``out_dir``. If the projected annotation count
    exceeds ``shard_threshold``, delegates to ``write_sharded_coco`` and
    returns ``{"sharded": True, "index_path": ...}``.
    """
    with open(splits_json) as f:
        source = json.load(f)

    dataset_name = dataset_name or source.get(
        "name", os.path.splitext(os.path.basename(splits_json))[0]
    )

    projected = _project_annotation_count(source)
    if projected > shard_threshold:
        sharded_dir = os.path.join(out_dir, "sharded")
        write_sharded_coco(
            splits_json=splits_json,
            root_dir=root_dir,
            out_dir=sharded_dir,
            image_link_mode=image_link_mode,
            splits=splits,
        )
        return {
            "sharded": True,
            "index_path": os.path.join(sharded_dir, "index.json"),
            "reason": (
                f"projected {projected} annotations exceeds threshold "
                f"{shard_threshold}; emitted sharded format instead"
            ),
        }

    info = _coco_info(dataset_name, source)
    categories = _categories_from_label_dict(source["label_dict"])

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "annotations"), exist_ok=True)
    if image_link_mode != "reference":
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    image_id = 1
    ann_id = 1
    per_split_paths = {}
    counts = {}
    for split in splits:
        entries = source.get(split, [])
        coco, image_id, ann_id = _build_split_coco(
            entries,
            root_dir,
            out_dir,
            image_link_mode,
            info,
            categories,
            image_id_start=image_id,
            ann_id_start=ann_id,
        )
        out_path = os.path.join(out_dir, "annotations", f"{split}.json")
        with open(out_path, "w") as f:
            json.dump(coco, f)
        per_split_paths[split] = out_path
        counts[split] = {
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
        }

    meta = {
        "dataset": dataset_name,
        "format": "coco_v1",
        "image_link_mode": image_link_mode,
        "info": info,
        "categories": categories,
        "counts": counts,
        "splits": per_split_paths,
        "source_splits_json": os.path.basename(splits_json),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {"sharded": False, "meta_path": os.path.join(out_dir, "meta.json"), "counts": counts}


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--image-link-mode", default="symlink",
        choices=("symlink", "copy", "reference"),
    )
    parser.add_argument(
        "--shard-threshold", type=int, default=DEFAULT_SHARD_THRESHOLD,
        help=(
            "Auto-shard when projected annotation count exceeds this. "
            "Pass 0 to force sharding, a very large int to force monolithic."
        ),
    )
    parser.add_argument(
        "--splits-only", nargs="+", default=None,
        help="Restrict to specific split names (default: train val test).",
    )
    args = parser.parse_args()

    result = to_coco(
        splits_json=args.splits,
        root_dir=args.root,
        out_dir=args.out,
        image_link_mode=args.image_link_mode,
        splits=tuple(args.splits_only) if args.splits_only else ("train", "val", "test"),
        dataset_name=args.dataset,
        shard_threshold=args.shard_threshold,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
