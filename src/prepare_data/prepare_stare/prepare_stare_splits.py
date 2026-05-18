"""
Build train / val / test split file for the STARE dataset.

STARE does not ship with an official train/test split. We use a fixed
random split (10 train / 4 val / 6 test on the 20 annotated cases) so that
results are reproducible. The labels used as ground truth are the Adam
Hoover annotations (1st_labels_ah); the second annotator's masks are kept
on disk for inter-rater experiments.
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
IMAGE_EXTS = (".ppm", ".png", ".tif", ".tiff", ".jpg", ".jpeg")
MASK_EXT = ".png"
LABEL_DICT = {0: "background", 1: "vessel"}


def list_images(folder: str):
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTS)
    )


def matching_mask(image_path: str, mask_folder: str) -> str:
    # STARE labels follow the pattern <image_stem>.ah.ppm; we drop everything
    # after the first dot and match by the stem.
    base = os.path.basename(image_path)
    stem = base.split(".")[0]
    for fname in os.listdir(mask_folder):
        if fname.startswith(stem) and fname.lower().endswith(MASK_EXT):
            return os.path.join(mask_folder, fname)
    raise FileNotFoundError(f"No mask found for {image_path} in {mask_folder}")


def relative(path: str, root_folder: str) -> str:
    return os.path.relpath(path, root_folder)


def build_splits(root_folder: str) -> dict:
    image_dir = os.path.join(root_folder, "images")
    mask_dir = os.path.join(root_folder, "1st_labels_ah_png")

    images = list_images(image_dir)
    # Only keep images that have a matching ground-truth mask, since STARE
    # ships 397 raw images but only 20 are manually annotated.
    pairs = []
    for img in images:
        try:
            mask = matching_mask(img, mask_dir)
        except FileNotFoundError:
            continue
        pairs.append(
            {
                "image": relative(img, root_folder),
                "target": relative(mask, root_folder),
            }
        )

    if not pairs:
        raise RuntimeError(
            f"No annotated STARE cases found under {mask_dir}. "
            "Run prepare_stare_labels.py first."
        )

    train_val, test = train_test_split(pairs, test_size=0.3, random_state=SEED)
    train, val = train_test_split(train_val, test_size=0.2857, random_state=SEED)

    return {
        "name": "STARE",
        "label_dict": LABEL_DICT,
        "train": train,
        "val": val,
        "test": test,
    }


if __name__ == "__main__":
    root_folder = os.getenv("STARE_ROOT_FOLDER")
    if root_folder is None:
        raise EnvironmentError("STARE_ROOT_FOLDER environment variable is not set.")

    splits = build_splits(root_folder)
    output_file = os.path.join(root_folder, "stare_splits.json")
    with open(output_file, "w") as f:
        json.dump(splits, f, indent=4)
    print(f"Dataset splits saved to {output_file}")

    add_metadata_to_splits_json(
        json_path=output_file,
        root_dir=root_folder,
        dataset_name="STARE",
        seed=SEED,
    )
