"""
Build train / val / test split file for HRF.

HRF has 45 fundus images (15 healthy, 15 glaucoma, 15 diabetic
retinopathy). No official train/test partition. We use a random
70/15/15 case-level split with seed=42 stratified on the condition flag
(h / g / dr) so each split preserves the original case mix
(~31 / 7 / 7 with roughly 5 cases per condition per split).
"""

import os
import sys
import json
import re

from sklearn.model_selection import train_test_split

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
from anatomy_datasets import add_metadata_to_splits_json


SEED = 42
LABEL_DICT = {0: "background", 1: "vessel"}
NAME = "HRF"

CASE_RE = re.compile(r"^(\d{2})_(h|g|dr)\.(JPG|jpg)$")


def _stem_and_condition(fname: str):
    m = CASE_RE.match(fname)
    if not m:
        return None
    return m.group(1) + "_" + m.group(2), m.group(2)


def build_splits(root_folder: str) -> dict:
    images_dir = os.path.join(root_folder, "images")
    masks_dir = os.path.join(root_folder, "manual1_png")

    pairs = []
    conditions = []
    for fname in sorted(os.listdir(images_dir)):
        parsed = _stem_and_condition(fname)
        if parsed is None:
            continue
        stem, cond = parsed
        mask_path = os.path.join(masks_dir, stem + ".png")
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(
                f"Vessel mask missing for {fname}: {mask_path}. "
                f"Did prepare_hrf_labels.py run?"
            )
        pairs.append(
            {
                "image": os.path.relpath(os.path.join(images_dir, fname), root_folder),
                "target": os.path.relpath(mask_path, root_folder),
            }
        )
        conditions.append(cond)

    train_val, test, train_val_y, _ = train_test_split(
        pairs, conditions, test_size=0.15, random_state=SEED, stratify=conditions
    )
    train, val = train_test_split(
        train_val,
        test_size=0.15 / 0.85,
        random_state=SEED,
        stratify=train_val_y,
    )

    return {
        "name": NAME,
        "label_dict": LABEL_DICT,
        "train": train,
        "val": val,
        "test": test,
    }


if __name__ == "__main__":
    root_folder = os.getenv("HRF_ROOT_FOLDER")
    if root_folder is None:
        raise EnvironmentError("HRF_ROOT_FOLDER environment variable is not set.")

    splits = build_splits(root_folder)
    output_file = os.path.join(root_folder, "hrf_splits.json")
    with open(output_file, "w") as f:
        json.dump(splits, f, indent=4)
    print(f"Dataset splits saved to {output_file}")

    add_metadata_to_splits_json(
        json_path=output_file,
        root_dir=root_folder,
        dataset_name=NAME,
        seed=SEED,
    )
