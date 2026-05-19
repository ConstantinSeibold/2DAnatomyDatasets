"""
Build train / val / test split files for LES-AV.

LES-AV ships 22 fundus images with no official train/test partition.
We do a single random 14 / 4 / 4 case-level split with ``SEED=42`` and
write two splits JSONs that share both the IDs *and* the underlying
``masks_npy/<id>.npy`` files:

- ``lesav_vessel_splits.json`` - ``label_dict = {0: "vessel"}``
- ``lesav_av_splits.json``     - ``label_dict = {1: "artery",
                                                  2: "vein",
                                                  3: "uncertain"}``

``BaseMultiLabelDataset`` reads the ``.npy`` stack and selects the
channels listed in ``label_dict`` keys, so the two variants share one
on-disk label representation.
"""

import json
import os
import random
import sys


sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
from anatomy_datasets import add_metadata_to_splits_json


SEED = 42
NAME = "LES-AV"

VESSEL_LABEL_DICT = {0: "vessel"}
AV_LABEL_DICT = {1: "artery", 2: "vein", 3: "uncertain"}

TRAIN_N = 14
VAL_N = 4
TEST_N = 4


def _pair(root_folder: str, case_id: str) -> dict:
    img = os.path.join(root_folder, "images", case_id + ".png")
    tgt = os.path.join(root_folder, "masks_npy", case_id + ".npy")
    if not os.path.isfile(img):
        raise FileNotFoundError(f"Missing image for {case_id}: {img}")
    if not os.path.isfile(tgt):
        raise FileNotFoundError(
            f"Missing label stack for {case_id}: {tgt}. Did prepare_lesav_labels.py run?"
        )
    return {
        "image": os.path.relpath(img, root_folder),
        "target": os.path.relpath(tgt, root_folder),
    }


def build_splits(root_folder: str) -> tuple[dict, dict]:
    images_dir = os.path.join(root_folder, "images")
    case_ids = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(images_dir)
        if f.endswith(".png")
    )

    expected = TRAIN_N + VAL_N + TEST_N
    if len(case_ids) != expected:
        raise ValueError(
            f"Expected {expected} LES-AV cases, found {len(case_ids)} under {images_dir}"
        )

    rng = random.Random(SEED)
    shuffled = case_ids[:]
    rng.shuffle(shuffled)

    train_ids = shuffled[:TRAIN_N]
    val_ids = shuffled[TRAIN_N : TRAIN_N + VAL_N]
    test_ids = shuffled[TRAIN_N + VAL_N :]

    def _entries(ids):
        return [_pair(root_folder, cid) for cid in ids]

    train, val, test = _entries(train_ids), _entries(val_ids), _entries(test_ids)

    vessel_doc = {
        "name": NAME,
        "label_dict": VESSEL_LABEL_DICT,
        "train": train,
        "val": val,
        "test": test,
    }
    av_doc = {
        "name": NAME,
        "label_dict": AV_LABEL_DICT,
        "train": train,
        "val": val,
        "test": test,
    }
    return vessel_doc, av_doc


if __name__ == "__main__":
    root_folder = os.getenv("LESAV_ROOT_FOLDER")
    if root_folder is None:
        raise EnvironmentError("LESAV_ROOT_FOLDER environment variable is not set.")

    vessel_doc, av_doc = build_splits(root_folder)

    vessel_path = os.path.join(root_folder, "lesav_vessel_splits.json")
    av_path = os.path.join(root_folder, "lesav_av_splits.json")

    with open(vessel_path, "w") as f:
        json.dump(vessel_doc, f, indent=4)
    print(f"Vessel splits saved to {vessel_path}")

    with open(av_path, "w") as f:
        json.dump(av_doc, f, indent=4)
    print(f"A/V splits saved to {av_path}")

    add_metadata_to_splits_json(
        json_path=vessel_path,
        root_dir=root_folder,
        dataset_name=NAME,
        seed=SEED,
    )
    add_metadata_to_splits_json(
        json_path=av_path,
        root_dir=root_folder,
        dataset_name=NAME,
        seed=SEED,
    )
