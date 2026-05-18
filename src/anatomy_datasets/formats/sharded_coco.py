"""SA-1B-style per-image annotation shards with COCO-compatible RLE masks.

Two writer entrypoints are provided:

- ``write_sharded_coco(splits_json, ...)`` — multilabel ``.npy``-mask
  source (PAXRay-style).
- ``write_sharded_coco_from_coco({split: coco_json}, ...)`` — monolithic
  per-split COCO JSON source (Teeth-style polygon annotations). Polygon
  segmentations are converted to RLE; existing RLE annotations pass
  through unchanged.

Motivation
----------
A single monolithic COCO JSON does not scale to PAXRay++ (14,754 images
× up to 166 binary masks per image). Even RLE-compressed, the file lands
in the multi-GB range and ``pycocotools`` must parse the whole thing
before iterating. The Segment Anything (SA-1B) dataset solved the same
problem by sharding annotations one-file-per-image; this module
implements the same pattern.

Disk layout
-----------
::

    <out_dir>/
    ├── index.json                 # dataset-level: dataset name, splits, registry metadata
    ├── categories.json            # COCO categories list (small, single file)
    ├── annotations/
    │   ├── <image_id>.json        # one per image: {image: {...}, annotations: [{RLE, bbox, ...}, ...]}
    │   └── ...
    └── images/                    # optional: symlinks (default) or copies (image_link_mode=copy);
                                   # omitted when image_link_mode=reference
                                   # filename = <image_id><ext>

``index.json`` schema
~~~~~~~~~~~~~~~~~~~~~
::

    {
      "format": "sharded_coco_v1",
      "dataset": "<DatasetName>",
      "splits": {"train": ["<id>", ...], "val": [...], "test": [...]},
      "image_link_mode": "symlink" | "copy" | "reference",
      "image_dir": "images" | "<absolute path>",     # how to resolve image_id -> file
      "ann_dir": "annotations",
      "metadata": {                                  # passthrough of source splits-JSON metadata
        "version": "...", "seed": ..., "modality": "...", "license": "...",
        "source_url": "...", "citation": "...", "normalization": {...}
      },
      "source_splits_json": "<basename, for provenance>"
    }

``annotations/<image_id>.json`` schema
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    {
      "image": {"id": "<image_id>", "file_name": "<original rel path>", "height": H, "width": W},
      "annotations": [
        {"id": "<image_id>_<cat_id>", "category_id": int, "segmentation": {RLE},
         "bbox": [x, y, w, h], "area": int, "iscrowd": 0},
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from copy import deepcopy
from datetime import date
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

from anatomy_datasets.registry import get_dataset_info


FORMAT_VERSION = "sharded_coco_v1"


# --------------------------------------------------------------------------- #
# RLE helpers (lifted from src/prepare_data/prepare_paxray/prepare_paxray_to_coco.py
# to avoid the prep-script-imports-package coupling)
# --------------------------------------------------------------------------- #

def binary_mask_to_rle(binary_mask: np.ndarray) -> dict:
    """Encode a 2D binary mask as a COCO RLE dict (JSON-safe)."""
    encoded = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return encoded


def rle_to_binary_mask(rle: dict) -> np.ndarray:
    """Inverse of ``binary_mask_to_rle``."""
    rle_copy = deepcopy(rle)
    rle_copy["counts"] = rle_copy["counts"].encode("utf-8")
    return mask_utils.decode(rle_copy)


def _rle_to_bbox(rle: dict) -> list:
    return mask_utils.toBbox(rle).tolist()


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #

def _image_id_from_entry(entry: dict) -> str:
    """Stable string id for an image. Prefer explicit ``image_id`` if present,
    otherwise derive from the image path stem (without extension)."""
    if "image_id" in entry:
        return str(entry["image_id"])
    return os.path.splitext(os.path.basename(entry["image"]))[0]


def _resolve_categories(label_dict: dict) -> List[dict]:
    """label_dict keys may be ints or stringified ints. Emit COCO categories
    starting at category_id = key (no offset), so that channel index in
    a multilabel .npy stack equals category_id."""
    cats = []
    for key, name in label_dict.items():
        cats.append({"id": int(key), "name": str(name), "supercategory": ""})
    cats.sort(key=lambda c: c["id"])
    return cats


def _load_image_size(path: str) -> tuple:
    with Image.open(path) as im:
        return im.size  # (width, height)


def _link_image(
    src_path: str, dst_path: str, mode: str
) -> None:
    if mode == "reference":
        return  # no-op; reader resolves from original file_name
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path) or os.path.islink(dst_path):
        os.remove(dst_path)
    if mode == "symlink":
        os.symlink(os.path.abspath(src_path), dst_path)
    elif mode == "copy":
        shutil.copy2(src_path, dst_path)
    else:
        raise ValueError(f"Unknown image_link_mode={mode!r}")


def _entry_to_annotations(
    entry: dict, root_dir: str, image_id: str, categories: List[dict]
) -> List[dict]:
    """Produce one annotation per non-empty channel of the entry's mask.

    Only the multilabel .npy path is implemented for now (the motivating
    case). Multiclass PNG masks could also be sharded but their natural
    storage is ``ann_dir/*.png`` (mmseg format) - the user should use the
    mmseg exporter for those.
    """
    target = entry.get("target")
    if target is None:
        return []  # missing-GT entry, no annotations

    target_path = os.path.join(root_dir, target)
    if not target_path.endswith(".npy"):
        raise NotImplementedError(
            f"sharded_coco currently only supports multilabel .npy mask "
            f"sources; got target={target_path!r}. Use the mmseg exporter "
            "for single-channel PNG mask datasets."
        )

    stack = np.load(target_path)
    if stack.ndim != 3:
        raise ValueError(
            f"Expected 3D mask stack [C,H,W], got shape {stack.shape}"
        )

    binary_stack = stack > 0
    annotations = []
    valid_ids = {c["id"] for c in categories}
    for channel_idx in range(binary_stack.shape[0]):
        if channel_idx not in valid_ids:
            continue
        channel_mask = binary_stack[channel_idx]
        if not channel_mask.any():
            continue
        rle = binary_mask_to_rle(channel_mask)
        annotations.append(
            {
                "id": f"{image_id}_{channel_idx}",
                "category_id": int(channel_idx),
                "segmentation": rle,
                "bbox": _rle_to_bbox(rle),
                "area": int(channel_mask.sum()),
                "iscrowd": 0,
            }
        )
    return annotations


def write_sharded_coco(
    splits_json: str,
    root_dir: str,
    out_dir: str,
    image_link_mode: str = "symlink",
    splits: Iterable[str] = ("train", "val", "test"),
) -> dict:
    """Convert a multilabel splits JSON to sharded per-image RLE annotations.

    Parameters
    ----------
    splits_json : path to source splits JSON (must contain ``label_dict``,
        ``train``/``val``/``test``, and entries with ``.npy`` ``target`` paths).
    root_dir : root the splits-JSON's image / target paths are relative to.
    out_dir : output directory; created if missing.
    image_link_mode : ``"symlink"`` (default), ``"copy"``, or ``"reference"``
        (don't replicate images; reader resolves via original file_name + ``root_dir``).
    splits : iterable of split names to emit. Splits absent from the source
        JSON are silently skipped.

    Returns the in-memory ``index.json`` dict (also persisted to disk).
    """
    if image_link_mode not in ("symlink", "copy", "reference"):
        raise ValueError(
            f"image_link_mode must be symlink|copy|reference; got {image_link_mode!r}"
        )

    with open(splits_json) as f:
        source = json.load(f)

    label_dict = source["label_dict"]
    dataset_name = source.get("name", os.path.splitext(os.path.basename(splits_json))[0])
    categories = _resolve_categories(label_dict)

    os.makedirs(out_dir, exist_ok=True)
    ann_subdir = "annotations"
    img_subdir = "images"
    os.makedirs(os.path.join(out_dir, ann_subdir), exist_ok=True)
    if image_link_mode != "reference":
        os.makedirs(os.path.join(out_dir, img_subdir), exist_ok=True)

    split_index: dict = {}
    for split_name in splits:
        if split_name not in source:
            continue
        entries = source[split_name]
        ids_for_split: List[str] = []

        for entry in entries:
            image_rel = entry["image"]
            image_src = os.path.join(root_dir, image_rel)
            image_id = _image_id_from_entry(entry)
            ids_for_split.append(image_id)

            width, height = _load_image_size(image_src)
            ext = os.path.splitext(image_rel)[1] or ".png"

            if image_link_mode == "reference":
                file_name = image_rel
            else:
                file_name = os.path.join(img_subdir, f"{image_id}{ext}")
                _link_image(
                    image_src, os.path.join(out_dir, file_name), image_link_mode
                )

            annotations = _entry_to_annotations(
                entry, root_dir, image_id, categories
            )

            shard = {
                "image": {
                    "id": image_id,
                    "file_name": file_name,
                    "height": int(height),
                    "width": int(width),
                },
                "annotations": annotations,
            }
            with open(
                os.path.join(out_dir, ann_subdir, f"{image_id}.json"), "w"
            ) as f:
                json.dump(shard, f)

        split_index[split_name] = ids_for_split

    # Pull metadata fields from source splits JSON so the sharded export
    # stays consistent with the per-dataset metadata round.
    metadata = {
        key: source.get(key)
        for key in (
            "version", "seed", "modality", "license",
            "source_url", "citation", "normalization",
        )
        if key in source
    }
    if not metadata.get("version"):
        metadata["version"] = date.today().isoformat()

    index = {
        "format": FORMAT_VERSION,
        "dataset": dataset_name,
        "splits": split_index,
        "image_link_mode": image_link_mode,
        "image_dir": img_subdir if image_link_mode != "reference" else os.path.abspath(root_dir),
        "ann_dir": ann_subdir,
        "metadata": metadata,
        "source_splits_json": os.path.basename(splits_json),
    }

    with open(os.path.join(out_dir, "categories.json"), "w") as f:
        json.dump({"categories": categories}, f, indent=2)
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    return index


# --------------------------------------------------------------------------- #
# Writer: monolithic COCO -> sharded
# --------------------------------------------------------------------------- #

def _polygon_to_rle(segmentation, height: int, width: int) -> dict:
    """Convert a COCO polygon `segmentation` (list of float lists or RLE dict)
    into the same JSON-safe RLE dict that ``binary_mask_to_rle`` produces."""
    if isinstance(segmentation, dict):
        # Already RLE — could be compressed (str counts) or uncompressed (list).
        counts = segmentation.get("counts")
        if isinstance(counts, list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            rle = dict(segmentation)
        if isinstance(rle.get("counts"), bytes):
            rle["counts"] = rle["counts"].decode("utf-8")
        return rle
    rles = mask_utils.frPyObjects(segmentation, height, width)
    merged = mask_utils.merge(rles)
    if isinstance(merged.get("counts"), bytes):
        merged["counts"] = merged["counts"].decode("utf-8")
    return merged


def write_sharded_coco_from_coco(
    coco_jsons: dict,
    root_dir: str,
    out_dir: str,
    image_link_mode: str = "symlink",
    dataset_name: Optional[str] = None,
) -> dict:
    """Convert per-split monolithic COCO JSONs into sharded per-image RLE annotations.

    Parameters
    ----------
    coco_jsons : mapping of split name -> path to a COCO JSON for that split
        (e.g. ``{"train": "...train.json", "val": "...val.json", "test": "...test.json"}``).
        Any split absent from the mapping (or with a falsy path) is skipped.
    root_dir : image root the COCO ``file_name`` fields are relative to.
    out_dir : output directory; created if missing.
    image_link_mode : ``"symlink"`` (default), ``"copy"``, or ``"reference"``.
    dataset_name : optional name; used to look up registry metadata for the
        ``index.json`` metadata block.

    Polygon ``segmentation`` lists are converted to compressed RLE via
    ``pycocotools.mask.frPyObjects`` + ``merge``. Existing RLE annotations
    pass through (decoded ``counts`` -> str). Returns the persisted
    ``index.json`` dict.
    """
    if image_link_mode not in ("symlink", "copy", "reference"):
        raise ValueError(
            f"image_link_mode must be symlink|copy|reference; got {image_link_mode!r}"
        )

    os.makedirs(out_dir, exist_ok=True)
    ann_subdir = "annotations"
    img_subdir = "images"
    os.makedirs(os.path.join(out_dir, ann_subdir), exist_ok=True)
    if image_link_mode != "reference":
        os.makedirs(os.path.join(out_dir, img_subdir), exist_ok=True)

    categories: Optional[list] = None
    split_index: dict = {}
    metadata_seed = None
    metadata_version = None

    for split_name, json_path in coco_jsons.items():
        if not json_path:
            continue
        with open(json_path) as f:
            coco = json.load(f)

        split_cats = coco.get("categories", [])
        if categories is None:
            categories = [
                {"id": int(c["id"]), "name": str(c.get("name", "")),
                 "supercategory": str(c.get("supercategory", ""))}
                for c in split_cats
            ]
            categories.sort(key=lambda c: c["id"])
        else:
            this_set = {(int(c["id"]), str(c.get("name", ""))) for c in split_cats}
            base_set = {(c["id"], c["name"]) for c in categories}
            if this_set != base_set:
                raise ValueError(
                    f"categories mismatch in {json_path!r}: expected {base_set}, got {this_set}"
                )

        info_block = coco.get("info", {}) or {}
        metadata_seed = metadata_seed or info_block.get("seed")
        metadata_version = metadata_version or info_block.get("version")

        anns_by_image: dict = {}
        for ann in coco.get("annotations", []):
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        ids_for_split: list = []
        for image in coco.get("images", []):
            image_id = str(image["id"])
            ids_for_split.append(image_id)

            image_rel = image["file_name"]
            image_src = os.path.join(root_dir, image_rel)
            width = int(image.get("width") or 0)
            height = int(image.get("height") or 0)
            if not width or not height:
                width, height = _load_image_size(image_src)

            ext = os.path.splitext(image_rel)[1] or ".png"
            if image_link_mode == "reference":
                file_name = image_rel
            else:
                file_name = os.path.join(img_subdir, f"{image_id}{ext}")
                _link_image(
                    image_src, os.path.join(out_dir, file_name), image_link_mode
                )

            shard_anns = []
            for ann in anns_by_image.get(image["id"], []):
                rle = _polygon_to_rle(ann["segmentation"], height, width)
                bbox = ann.get("bbox") or _rle_to_bbox(rle)
                area = ann.get("area")
                if area is None:
                    area = int(mask_utils.area(
                        {**rle, "counts": rle["counts"].encode("utf-8")}
                        if isinstance(rle.get("counts"), str) else rle
                    ))
                shard_anns.append(
                    {
                        "id": f"{image_id}_{ann.get('id', len(shard_anns))}",
                        "category_id": int(ann["category_id"]),
                        "segmentation": rle,
                        "bbox": list(bbox),
                        "area": int(area),
                        "iscrowd": int(ann.get("iscrowd", 0)),
                    }
                )

            shard = {
                "image": {
                    "id": image_id,
                    "file_name": file_name,
                    "height": int(height),
                    "width": int(width),
                },
                "annotations": shard_anns,
            }
            with open(
                os.path.join(out_dir, ann_subdir, f"{image_id}.json"), "w"
            ) as f:
                json.dump(shard, f)

        split_index[split_name] = ids_for_split

    if categories is None:
        raise ValueError("no COCO JSONs supplied; nothing to write")

    metadata: dict = {}
    if dataset_name:
        try:
            reg = get_dataset_info(dataset_name)
            metadata = {
                "modality": reg.modality,
                "license": reg.license,
                "source_url": reg.source_url,
                "citation": reg.citation,
            }
        except KeyError:
            pass
    if metadata_seed is not None:
        metadata["seed"] = metadata_seed
    metadata["version"] = metadata_version or date.today().isoformat()

    index = {
        "format": FORMAT_VERSION,
        "dataset": dataset_name or "",
        "splits": split_index,
        "image_link_mode": image_link_mode,
        "image_dir": img_subdir if image_link_mode != "reference" else os.path.abspath(root_dir),
        "ann_dir": ann_subdir,
        "metadata": metadata,
        "source_splits_json": ",".join(
            os.path.basename(p) for p in coco_jsons.values() if p
        ),
    }

    with open(os.path.join(out_dir, "categories.json"), "w") as f:
        json.dump({"categories": categories}, f, indent=2)
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    return index


# --------------------------------------------------------------------------- #
# Reader
# --------------------------------------------------------------------------- #

class ShardedCocoDataset(Dataset):
    """PyTorch Dataset over a sharded-COCO export.

    ``__getitem__(idx)`` returns ``(image, masks)`` where ``image`` is an
    ``np.ndarray`` (H, W, 3) and ``masks`` is an ``np.ndarray`` of shape
    ``(C, H, W)`` with ``C`` = number of categories present in this image
    (variable per sample). The class-ids for each channel are also
    available via ``self[idx]`` returning a dict if ``return_dict=True``::

        ds = ShardedCocoDataset("exports/sharded_paxray", split="train")
        img, masks = ds[0]                        # (H,W,3), (C,H,W)
        img, masks, cat_ids = ds[0, "with_ids"]   # not supported; use return_dict

    For a uniform fixed-channel output (matching ``BaseMultiLabelDataset``),
    pass ``label_dict`` explicitly to align the channel axis with a known
    category set; missing categories become zero masks::

        ds = ShardedCocoDataset(..., label_dict={0: "bg", 10: "lungs", ...})

    Parameters
    ----------
    out_dir : root of the sharded export (the dir containing ``index.json``).
    split : ``"train"`` / ``"val"`` / ``"test"``.
    transform : optional albumentations transform applied as
        ``transform(image=..., mask=...)`` if masks are stacked; skipped if
        not.
    label_dict : optional ``{cat_id: name}`` to enforce a fixed channel
        layout. If omitted, the output's channel axis is variable per item.
    return_dict : if True, ``__getitem__`` returns a dict including
        ``image``, ``masks``, ``category_ids``, ``image_id``,
        ``file_name``. Otherwise returns a tuple ``(image, masks)``.
    image_root_override : when index records ``image_link_mode="reference"``,
        the reader normally uses the recorded ``image_dir``; pass this to
        override (useful if the data was moved).
    """

    def __init__(
        self,
        out_dir: str,
        split: str = "train",
        transform=None,
        label_dict: Optional[dict] = None,
        return_dict: bool = False,
        image_root_override: Optional[str] = None,
    ):
        self.out_dir = out_dir
        self.split = split
        self.transform = transform
        self.return_dict = return_dict

        with open(os.path.join(out_dir, "index.json")) as f:
            self.index = json.load(f)
        if self.index.get("format") != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported sharded-coco format {self.index.get('format')!r}; "
                f"this reader supports {FORMAT_VERSION!r}."
            )
        if split not in self.index["splits"]:
            raise KeyError(
                f"Split {split!r} not in index; available: "
                f"{sorted(self.index['splits'])}"
            )
        self.image_ids: List[str] = list(self.index["splits"][split])

        with open(os.path.join(out_dir, "categories.json")) as f:
            cats = json.load(f)["categories"]
        self.categories = cats
        self.cat_id_to_name = {c["id"]: c["name"] for c in cats}

        # Fixed-channel mode: caller supplies the label_dict that defines
        # which category_ids occupy which channel slots.
        if label_dict is not None:
            self.label_dict = {int(k): v for k, v in label_dict.items()}
            self._fixed_channel_ids = sorted(self.label_dict.keys())
        else:
            self.label_dict = self.cat_id_to_name
            self._fixed_channel_ids = None

        # Image resolution rules:
        self._image_link_mode = self.index["image_link_mode"]
        if image_root_override is not None:
            self._image_root = image_root_override
        elif self._image_link_mode == "reference":
            self._image_root = self.index["image_dir"]
        else:
            self._image_root = self.out_dir

    def __len__(self) -> int:
        return len(self.image_ids)

    def _resolve_image_path(self, file_name: str) -> str:
        if self._image_link_mode == "reference":
            return os.path.join(self._image_root, file_name)
        # Otherwise file_name is already relative to out_dir (e.g. images/<id>.png).
        return os.path.join(self._image_root, file_name)

    def _load_shard(self, image_id: str) -> dict:
        path = os.path.join(self.out_dir, self.index["ann_dir"], f"{image_id}.json")
        with open(path) as f:
            return json.load(f)

    def _build_masks(self, shard: dict) -> tuple:
        """Returns ``(masks[C,H,W], category_ids)``."""
        H = shard["image"]["height"]
        W = shard["image"]["width"]

        if self._fixed_channel_ids is not None:
            channel_ids = self._fixed_channel_ids
            by_cat = {ann["category_id"]: ann for ann in shard["annotations"]}
            masks = np.zeros((len(channel_ids), H, W), dtype=np.uint8)
            for slot, cat_id in enumerate(channel_ids):
                ann = by_cat.get(cat_id)
                if ann is not None:
                    masks[slot] = rle_to_binary_mask(ann["segmentation"])
            return masks, channel_ids

        # Variable-channel mode: one channel per present annotation.
        anns = shard["annotations"]
        if not anns:
            return np.zeros((0, H, W), dtype=np.uint8), []
        masks = np.zeros((len(anns), H, W), dtype=np.uint8)
        ids = []
        for i, ann in enumerate(anns):
            masks[i] = rle_to_binary_mask(ann["segmentation"])
            ids.append(int(ann["category_id"]))
        return masks, ids

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        shard = self._load_shard(image_id)
        image_path = self._resolve_image_path(shard["image"]["file_name"])
        image = np.array(Image.open(image_path).convert("RGB"))
        masks, cat_ids = self._build_masks(shard)

        if self.transform is not None and masks.shape[0] > 0:
            mask_for_tx = masks.transpose([1, 2, 0])
            out = self.transform(image=image, mask=mask_for_tx)
            image = out["image"]
            mask_out = out["mask"]
            if hasattr(mask_out, "permute"):
                masks = mask_out.permute(2, 0, 1)
            else:
                masks = np.transpose(mask_out, (2, 0, 1))

        if self.return_dict:
            return {
                "image": image,
                "masks": masks,
                "category_ids": cat_ids,
                "image_id": image_id,
                "file_name": shard["image"]["file_name"],
            }
        return image, masks


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a splits JSON (with multilabel .npy masks) OR a set of "
            "monolithic per-split COCO JSONs into a sharded SA-1B-style "
            "per-image RLE annotation tree."
        )
    )
    parser.add_argument(
        "--splits", default=None,
        help="Source splits JSON (multilabel .npy path). Mutually exclusive with --from-coco-*.",
    )
    parser.add_argument(
        "--from-coco-train", default=None,
        help="Monolithic COCO JSON for the train split (polygon/RLE passthrough).",
    )
    parser.add_argument("--from-coco-val", default=None)
    parser.add_argument("--from-coco-test", default=None)
    parser.add_argument(
        "--root",
        required=True,
        help="Root dir the image paths are relative to.",
    )
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument(
        "--dataset", default=None,
        help="Dataset name (used to populate metadata from registry).",
    )
    parser.add_argument(
        "--image-link-mode",
        default="symlink",
        choices=("symlink", "copy", "reference"),
        help="How to expose images in the export (default: symlink).",
    )
    parser.add_argument(
        "--splits-only",
        nargs="+",
        default=None,
        help="Restrict to these split names (default: train val test). Splits-JSON path only.",
    )
    args = parser.parse_args()

    from_coco_any = any([args.from_coco_train, args.from_coco_val, args.from_coco_test])
    if args.splits and from_coco_any:
        parser.error("--splits and --from-coco-* are mutually exclusive")
    if not args.splits and not from_coco_any:
        parser.error("must provide --splits OR at least one --from-coco-{train,val,test}")

    if from_coco_any:
        write_sharded_coco_from_coco(
            coco_jsons={
                "train": args.from_coco_train,
                "val": args.from_coco_val,
                "test": args.from_coco_test,
            },
            root_dir=args.root,
            out_dir=args.out,
            image_link_mode=args.image_link_mode,
            dataset_name=args.dataset,
        )
    else:
        write_sharded_coco(
            splits_json=args.splits,
            root_dir=args.root,
            out_dir=args.out,
            image_link_mode=args.image_link_mode,
            splits=tuple(args.splits_only) if args.splits_only else ("train", "val", "test"),
        )
    print(f"Wrote sharded export to {args.out}")


if __name__ == "__main__":
    _cli()
