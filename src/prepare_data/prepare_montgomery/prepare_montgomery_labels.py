"""
Build combined lung masks for the Montgomery dataset.

Upstream ships per-side binary masks (values in {0, 255}) under
ManualMask/leftMask and ManualMask/rightMask. The NLM file naming is
image-side (the mask under ``leftMask`` corresponds to the lung on the
LEFT of the radiograph as the viewer sees it), which is the patient's
**anatomical right lung** because PA chest radiographs are mirrored
relative to the patient. The "R" radiographic marker in the upper-left
of each image confirms this.

We merge the two binary masks per case into a single-channel PNG under
ManualMask/CombinedMask/ with class-index pixel values in **anatomical**
convention (matching the convention used by the other chest X-ray
datasets in this repo, e.g. PAXRay):

    0 = background
    1 = left lung   (patient's anatomical left  -> appears on viewer's right)
    2 = right lung  (patient's anatomical right -> appears on viewer's left)

The combined PNGs look near-black in an image viewer (max value 2);
that is correct and is the project-wide convention for
``BaseMultiClassDataset`` inputs.
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm


# NLM file naming is image-side; we re-map to anatomical class IDs.
IMAGE_LEFT_TO_CLASS = 2   # leftMask file  -> anatomical right lung
IMAGE_RIGHT_TO_CLASS = 1  # rightMask file -> anatomical left lung


def combine_masks(root_folder: str) -> None:
    base = os.path.join(root_folder, "MontgomerySet", "ManualMask")
    left_dir = os.path.join(base, "leftMask")
    right_dir = os.path.join(base, "rightMask")
    out_dir = os.path.join(base, "CombinedMask")
    os.makedirs(out_dir, exist_ok=True)

    left_files = sorted(f for f in os.listdir(left_dir) if f.endswith(".png"))

    for fname in tqdm(left_files, desc="Montgomery labels"):
        left_path = os.path.join(left_dir, fname)
        right_path = os.path.join(right_dir, fname)
        if not os.path.isfile(right_path):
            raise FileNotFoundError(
                f"rightMask file missing for {fname}: {right_path}"
            )

        image_left = np.array(Image.open(left_path).convert("L")) > 0
        image_right = np.array(Image.open(right_path).convert("L")) > 0
        if image_left.shape != image_right.shape:
            raise ValueError(
                f"Shape mismatch for {fname}: left {image_left.shape} vs "
                f"right {image_right.shape}"
            )
        if (image_left & image_right).any():
            print(
                f"  warning: overlapping pixels in {fname}; assigning overlap "
                f"to the rightMask side"
            )

        combined = np.zeros_like(image_left, dtype=np.uint8)
        combined[image_left] = IMAGE_LEFT_TO_CLASS
        combined[image_right] = IMAGE_RIGHT_TO_CLASS
        Image.fromarray(combined).save(os.path.join(out_dir, fname))


if __name__ == "__main__":
    root_folder = os.getenv("MONTGOMERY_ROOT_FOLDER")
    if root_folder is None:
        raise EnvironmentError(
            "MONTGOMERY_ROOT_FOLDER environment variable is not set."
        )
    combine_masks(root_folder)
