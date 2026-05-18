"""
Build train / val / test split file for the Medaka heart dataset.

The dataset ships 565 annotated training frames and two unseen-specimen
test sets (75 N0030_* and 165 color_frame_* / R0004). We split the
training frames into train/val (default 20% validation, fixed seed) and
record both test subsets in the ``test`` list — they share filename
prefixes ("N0030_" vs "color_frame_") so they remain distinguishable.

Images: ``ventral_samples`` (RGB .tif). Train uses the RGB folder; the
test sets ship one RGB folder (``ventral_samples_R0004``) and one grayscale
folder (``ventral_samples_gray``) — both are read as RGB by the dataloader.

Targets: the dense-ID PNGs produced by ``prepare_medaka_heart_labels.py``.
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
LABEL_DICT = {0: "background", 1: "bulbus", 2: "atrium", 3: "heart"}

TRAIN_IMAGE_DIR = os.path.join("train_images", "ventral_samples")
TRAIN_LABEL_DIR = os.path.join("train_images", "ventral_mask_combined_labels")

TEST_SUBSETS = [
    (
        os.path.join("test_images", "ventral_samples_gray"),
        os.path.join("test_images", "ventral_mask_combined_gray_labels"),
        "_mask",
    ),
    (
        os.path.join("test_images", "ventral_samples_R0004"),
        os.path.join("test_images", "ventral_mask_combined_R0004_labels"),
        "_mask",
    ),
]


def list_images(folder: str):
    return sorted(
        f for f in os.listdir(folder) if f.lower().endswith((".tif", ".tiff", ".png"))
    )


def pair(image_dir: str, label_dir: str, mask_suffix: str, root_folder: str):
    pairs = []
    for fname in list_images(os.path.join(root_folder, image_dir)):
        base = os.path.splitext(fname)[0]
        mask_name = base + mask_suffix + ".png"
        mask_path = os.path.join(label_dir, mask_name)
        if not os.path.isfile(os.path.join(root_folder, mask_path)):
            raise FileNotFoundError(
                f"Missing mask for {fname} (expected {mask_path})"
            )
        pairs.append(
            {"image": os.path.join(image_dir, fname), "target": mask_path}
        )
    return pairs


def build_splits(root_folder: str, validation_split: float = 0.2) -> dict:
    train_pairs = pair(TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, "_mask", root_folder)
    train_split, val_split = train_test_split(
        train_pairs, test_size=validation_split, random_state=SEED
    )

    test_entries = []
    for image_dir, label_dir, mask_suffix in TEST_SUBSETS:
        test_entries.extend(pair(image_dir, label_dir, mask_suffix, root_folder))

    return {
        "name": "MedakaHeart",
        "label_dict": LABEL_DICT,
        "train": train_split,
        "val": val_split,
        "test": test_entries,
    }


if __name__ == "__main__":
    root_folder = os.getenv("MEDAKA_HEART_ROOT_FOLDER")
    if root_folder is None:
        raise EnvironmentError(
            "MEDAKA_HEART_ROOT_FOLDER environment variable is not set."
        )

    splits = build_splits(root_folder)
    output_file = os.path.join(root_folder, "medaka_heart_splits.json")
    with open(output_file, "w") as f:
        json.dump(splits, f, indent=4)
    print(f"Dataset splits saved to {output_file}")

    add_metadata_to_splits_json(
        json_path=output_file,
        root_dir=root_folder,
        dataset_name="MedakaHeart",
        seed=SEED,
    )
