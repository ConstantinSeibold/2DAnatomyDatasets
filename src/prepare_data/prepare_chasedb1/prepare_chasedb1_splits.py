"""
Build train / val / test split file for the CHASE_DB1 dataset.

CHASE_DB1 has 28 retinal images (Image_01L/R ... Image_14L/R). Following
the most common convention in the literature, the first 20 images form
the training partition and the remaining 8 are held out for testing.
We split the training partition into a train/val split (default 20%
validation). The first observer's annotations (*_1stHO) are used as the
ground truth.
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
TRAIN_CASES = 20  # first 20 of 28 images are used for training


def relative(path: str, root_folder: str) -> str:
    return os.path.relpath(path, root_folder)


def case_key(fname: str):
    # "Image_03L.jpg" -> (3, "L")
    match = re.match(r"Image_(\d+)([LR])\.jpg$", fname)
    if not match:
        return None
    return (int(match.group(1)), match.group(2))


def build_splits(root_folder: str, validation_split: float = 0.2) -> dict:
    images = sorted(
        [f for f in os.listdir(root_folder) if case_key(f) is not None],
        key=case_key,
    )

    pairs = []
    for img_name in images:
        mask_name = img_name.replace(".jpg", "_1stHO.png")
        mask_path = os.path.join(root_folder, mask_name)
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Missing mask for {img_name}: {mask_path}")
        pairs.append(
            {
                "image": relative(os.path.join(root_folder, img_name), root_folder),
                "target": relative(mask_path, root_folder),
            }
        )

    train_val = pairs[:TRAIN_CASES]
    test = pairs[TRAIN_CASES:]

    train, val = train_test_split(
        train_val, test_size=validation_split, random_state=SEED
    )

    return {
        "name": "CHASE_DB1",
        "label_dict": LABEL_DICT,
        "train": train,
        "val": val,
        "test": test,
    }


if __name__ == "__main__":
    root_folder = os.getenv("CHASEDB1_ROOT_FOLDER")
    if root_folder is None:
        raise EnvironmentError(
            "CHASEDB1_ROOT_FOLDER environment variable is not set."
        )

    splits = build_splits(root_folder)
    output_file = os.path.join(root_folder, "chasedb1_splits.json")
    with open(output_file, "w") as f:
        json.dump(splits, f, indent=4)
    print(f"Dataset splits saved to {output_file}")

    add_metadata_to_splits_json(
        json_path=output_file,
        root_dir=root_folder,
        dataset_name="CHASE_DB1",
        seed=SEED,
    )
