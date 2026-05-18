"""
Build train / val / test split file for the DRIVE dataset.

DRIVE ships with a fixed train/test partition of 20/20 cases, but the
official test partition does not include vessel annotations. We split the
20 annotated training cases into train/val (default 20% validation) and
record the 20 unlabelled test images under the "test" split without a
target.
"""

import os
import sys
import json
from sklearn.model_selection import train_test_split

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
from anatomy_datasets import add_metadata_to_splits_json


SEED = 42
IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
MASK_EXT = ".png"
LABEL_DICT = {0: "background", 1: "vessel"}


def list_images(folder: str):
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTS)
    )


def matching_mask(image_path: str, mask_folder: str) -> str:
    base = os.path.splitext(os.path.basename(image_path))[0]
    # DRIVE training images are e.g. "21_training.tif" and masks are
    # "21_manual1.png" -> we strip the suffix and match by prefix number.
    prefix = base.split("_")[0]
    for fname in os.listdir(mask_folder):
        if fname.startswith(prefix) and fname.lower().endswith(MASK_EXT):
            return os.path.join(mask_folder, fname)
    raise FileNotFoundError(f"No mask found for {image_path} in {mask_folder}")


def relative(path: str, root_folder: str) -> str:
    return os.path.relpath(path, root_folder)


def build_splits(root_folder: str, validation_split: float = 0.2) -> dict:
    train_images = list_images(os.path.join(root_folder, "training", "images"))
    train_mask_dir = os.path.join(root_folder, "training", "1st_manual_png")
    train_pairs = [
        {
            "image": relative(img, root_folder),
            "target": relative(matching_mask(img, train_mask_dir), root_folder),
        }
        for img in train_images
    ]

    train_split, val_split = train_test_split(
        train_pairs, test_size=validation_split, random_state=SEED
    )

    test_images = list_images(os.path.join(root_folder, "test", "images"))
    test_entries = [
        {"image": relative(img, root_folder)} for img in test_images
    ]

    return {
        "name": "DRIVE",
        "label_dict": LABEL_DICT,
        "train": train_split,
        "val": val_split,
        "test": test_entries,
    }


if __name__ == "__main__":
    root_folder = os.getenv("DRIVE_ROOT_FOLDER")
    if root_folder is None:
        raise EnvironmentError("DRIVE_ROOT_FOLDER environment variable is not set.")

    splits = build_splits(root_folder)
    output_file = os.path.join(root_folder, "drive_splits.json")
    with open(output_file, "w") as f:
        json.dump(splits, f, indent=4)
    print(f"Dataset splits saved to {output_file}")

    add_metadata_to_splits_json(
        json_path=output_file,
        root_dir=root_folder,
        dataset_name="DRIVE",
        seed=SEED,
    )
