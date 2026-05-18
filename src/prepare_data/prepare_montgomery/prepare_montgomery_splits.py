"""
Build train / val / test split file for the Montgomery dataset.

Montgomery has no official train/test partition. We use a random
80/10/10 case-level split with seed=42 (yields ~110/14/14 of the 138
cases). Stratification on the TB/normal flag (encoded in the filename
suffix MCUCXR_NNNN_0/_1) is preserved by passing it to
train_test_split, so each split keeps the original ~58/80 TB/normal
ratio.
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
LABEL_DICT = {0: "background", 1: "left_lung", 2: "right_lung"}
NAME = "Montgomery"


def _case_label(fname: str):
    # MCUCXR_NNNN_0.png -> 0 (normal), MCUCXR_NNNN_1.png -> 1 (TB)
    m = re.match(r"MCUCXR_\d+_(\d)\.png$", fname)
    if not m:
        return None
    return int(m.group(1))


def build_splits(root_folder: str) -> dict:
    cxr_dir = os.path.join(root_folder, "MontgomerySet", "CXR_png")
    mask_dir = os.path.join(root_folder, "MontgomerySet", "ManualMask", "CombinedMask")

    images = sorted(f for f in os.listdir(cxr_dir) if _case_label(f) is not None)

    pairs = []
    tb_labels = []
    for fname in images:
        mask_path = os.path.join(mask_dir, fname)
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(
                f"Combined mask missing for {fname}: {mask_path}. "
                f"Did prepare_montgomery_labels.py run?"
            )
        pairs.append(
            {
                "image": os.path.relpath(os.path.join(cxr_dir, fname), root_folder),
                "target": os.path.relpath(mask_path, root_folder),
            }
        )
        tb_labels.append(_case_label(fname))

    train_val, test, train_val_y, _ = train_test_split(
        pairs, tb_labels, test_size=0.10, random_state=SEED, stratify=tb_labels
    )
    train, val = train_test_split(
        train_val,
        test_size=0.111111,  # 0.10 / (1 - 0.10) -> overall ~80/10/10
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
    root_folder = os.getenv("MONTGOMERY_ROOT_FOLDER")
    if root_folder is None:
        raise EnvironmentError(
            "MONTGOMERY_ROOT_FOLDER environment variable is not set."
        )

    splits = build_splits(root_folder)
    output_file = os.path.join(root_folder, "montgomery_splits.json")
    with open(output_file, "w") as f:
        json.dump(splits, f, indent=4)
    print(f"Dataset splits saved to {output_file}")

    add_metadata_to_splits_json(
        json_path=output_file,
        root_dir=root_folder,
        dataset_name=NAME,
        seed=SEED,
    )
