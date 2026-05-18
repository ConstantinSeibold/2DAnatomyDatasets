"""
Build train / val / test split file for FIVES.

FIVES ships an official 600/200 train/test partition balanced across
four disease groups (A=AMD, D=diabetic retinopathy, G=glaucoma,
N=normal). We keep the upstream test split as-is and carve an 80/20
train/val out of the upstream train split with seed=42, stratified on
the disease flag so each split keeps the original case mix.
"""

import os
import re
import sys
import json

from sklearn.model_selection import train_test_split

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
from anatomy_datasets import add_metadata_to_splits_json


SEED = 42
LABEL_DICT = {0: "background", 1: "vessel"}
NAME = "FIVES"

CASE_RE = re.compile(r"^(\d+)_([ADGN])\.png$", re.IGNORECASE)


def _condition(fname: str):
    m = CASE_RE.match(fname)
    return m.group(2).upper() if m else None


def _collect(root_folder: str, split_dir: str):
    images_dir = os.path.join(root_folder, split_dir, "Original")
    masks_dir = os.path.join(root_folder, split_dir, "Ground_truth_png")
    pairs = []
    conditions = []
    for fname in sorted(os.listdir(images_dir)):
        cond = _condition(fname)
        if cond is None:
            continue
        mask_path = os.path.join(masks_dir, fname)
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(
                f"Vessel mask missing for {fname}: {mask_path}. "
                f"Did prepare_fives_labels.py run?"
            )
        pairs.append(
            {
                "image": os.path.relpath(os.path.join(images_dir, fname), root_folder),
                "target": os.path.relpath(mask_path, root_folder),
            }
        )
        conditions.append(cond)
    return pairs, conditions


def build_splits(root_folder: str) -> dict:
    train_full, train_conditions = _collect(root_folder, "train")
    test_pairs, _ = _collect(root_folder, "test")

    train, val = train_test_split(
        train_full,
        test_size=0.20,
        random_state=SEED,
        stratify=train_conditions,
    )

    return {
        "name": NAME,
        "label_dict": LABEL_DICT,
        "train": train,
        "val": val,
        "test": test_pairs,
    }


if __name__ == "__main__":
    root_folder = os.getenv("FIVES_ROOT_FOLDER")
    if root_folder is None:
        raise EnvironmentError("FIVES_ROOT_FOLDER environment variable is not set.")

    splits = build_splits(root_folder)
    output_file = os.path.join(root_folder, "fives_splits.json")
    with open(output_file, "w") as f:
        json.dump(splits, f, indent=4)
    print(f"Dataset splits saved to {output_file}")

    add_metadata_to_splits_json(
        json_path=output_file,
        root_dir=root_folder,
        dataset_name=NAME,
        seed=SEED,
    )
