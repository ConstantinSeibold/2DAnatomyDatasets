"""
Build per-split COCO JSONs for PAXRay++ instance segmentation.

The upstream bundle (Google-Drive zip extracted by ``download_paxraypp.py``)
ships two monolithic COCO files:

    paxray_train_val.json   # 13,278 images / 1.62M anns (frontal + lateral)
    paxray_test.json        #  1,475 images / 180K anns

This script:
1. Prepends ``images_patlas/`` to every ``file_name`` so the JSONs are
   self-contained relative to ``PAXRAYPP_ROOT`` (which now holds the
   extracted ``images_patlas/`` directory next to the JSONs).
2. Splits ``paxray_train_val.json`` into train / val (90 / 10, seed=42)
   by image id, carrying the corresponding annotations with each side.
3. Writes ``annotations/{train,val,test}.json`` under ``PAXRAYPP_ROOT``.
4. Calls ``add_metadata_to_splits_json`` per file to merge license /
   citation / normalization stats into each ``info`` block.
"""

import os
import sys
import json
import random
from copy import deepcopy

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
from anatomy_datasets import add_metadata_to_splits_json


SEED = 42
DATASET_NAME = "PAXRay++"
IMAGES_SUBDIR = "images_patlas"
TRAIN_FRACTION = 0.9


def _retarget_file_names(coco: dict, prefix: str) -> None:
    for img in coco["images"]:
        fn = img["file_name"]
        if not fn.startswith(prefix + "/"):
            img["file_name"] = f"{prefix}/{fn}"


def _split_train_val(coco: dict, train_fraction: float, seed: int) -> tuple:
    images = list(coco["images"])
    rng = random.Random(seed)
    rng.shuffle(images)

    n_train = int(round(train_fraction * len(images)))
    train_imgs = images[:n_train]
    val_imgs = images[n_train:]

    train_ids = {img["id"] for img in train_imgs}
    val_ids = {img["id"] for img in val_imgs}

    train_anns, val_anns = [], []
    for ann in coco["annotations"]:
        if ann["image_id"] in train_ids:
            train_anns.append(ann)
        elif ann["image_id"] in val_ids:
            val_anns.append(ann)

    def _bundle(imgs, anns):
        return {
            "info": deepcopy(coco.get("info", {})) if isinstance(coco.get("info"), dict) else {"description": coco.get("info", "")},
            "licenses": deepcopy(coco.get("licenses", [])),
            "images": imgs,
            "categories": deepcopy(coco["categories"]),
            "annotations": anns,
        }

    return _bundle(train_imgs, train_anns), _bundle(val_imgs, val_anns)


def _as_info_dict(coco: dict) -> None:
    """``add_metadata_to_splits_json`` expects ``info`` to be a dict; the
    upstream JSON ships a string (``"PAX-Ray++, v2"``). Promote it."""
    info = coco.get("info")
    if isinstance(info, dict):
        return
    coco["info"] = {"description": info or ""}


def main() -> None:
    root = os.getenv("PAXRAYPP_ROOT")
    if root is None:
        raise EnvironmentError("PAXRAYPP_ROOT environment variable is not set.")

    src_train_val = os.path.join(root, "paxray_train_val.json")
    src_test = os.path.join(root, "paxray_test.json")
    for p in (src_train_val, src_test):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Expected upstream COCO file missing: {p}. "
                f"Did download_paxraypp.py run?"
            )

    images_root = os.path.join(root, IMAGES_SUBDIR)
    if not os.path.isdir(images_root):
        raise FileNotFoundError(
            f"Expected images dir {images_root} not found. "
            f"Did unpack_files.py run?"
        )

    out_dir = os.path.join(root, "annotations")
    os.makedirs(out_dir, exist_ok=True)

    with open(src_train_val) as f:
        train_val = json.load(f)
    with open(src_test) as f:
        test = json.load(f)

    _retarget_file_names(train_val, IMAGES_SUBDIR)
    _retarget_file_names(test, IMAGES_SUBDIR)
    _as_info_dict(train_val)
    _as_info_dict(test)

    train, val = _split_train_val(train_val, TRAIN_FRACTION, SEED)

    split_paths = {}
    for name, payload in (("train", train), ("val", val), ("test", test)):
        out_path = os.path.join(out_dir, f"{name}.json")
        with open(out_path, "w") as f:
            json.dump(payload, f)
        split_paths[name] = out_path
        print(
            f"{name}: {len(payload['images'])} images, "
            f"{len(payload['annotations'])} annotations -> {out_path}"
        )

    for idx, (name, out_path) in enumerate(split_paths.items()):
        add_metadata_to_splits_json(
            json_path=out_path,
            root_dir=root,
            dataset_name=DATASET_NAME,
            seed=SEED,
            coco_image_dir=root,
            # Normalization stats: compute once on train, reuse for val/test
            # by copying afterwards (cheap and avoids 3x the I/O on ~14k imgs).
            compute_stats=(name == "train"),
        )

    train_info = json.load(open(split_paths["train"]))["info"]
    train_norm = train_info.get("normalization")
    if train_norm:
        for name in ("val", "test"):
            with open(split_paths[name]) as f:
                data = json.load(f)
            data.setdefault("info", {})["normalization"] = train_norm
            with open(split_paths[name], "w") as f:
                json.dump(data, f)


if __name__ == "__main__":
    main()
