"""
Convert FIVES vessel annotations to single-channel PNG masks with label
values {0: background, 1: vessel}.

Upstream ships binary vessel masks under ``train/Ground truth/`` and
``test/Ground truth/`` with foreground in {255}. We threshold > 0 and
rewrite each mask into ``train/Ground_truth_png/`` and
``test/Ground_truth_png/`` (underscore avoids the awkward space in the
path) so the splits-JSON can point at it without quoting.
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def convert_masks(root_folder: str) -> None:
    for split in ("train", "test"):
        src_dir = os.path.join(root_folder, split, "Ground truth")
        out_dir = os.path.join(root_folder, split, "Ground_truth_png")
        if not os.path.isdir(src_dir):
            raise FileNotFoundError(f"Missing GT folder: {src_dir}")
        os.makedirs(out_dir, exist_ok=True)

        files = sorted(f for f in os.listdir(src_dir) if f.endswith(".png"))
        for fname in tqdm(files, desc=f"FIVES labels ({split})"):
            mask = np.array(Image.open(os.path.join(src_dir, fname)).convert("L"))
            binary = (mask > 0).astype(np.uint8)
            Image.fromarray(binary).save(os.path.join(out_dir, fname))


if __name__ == "__main__":
    root_folder = os.getenv("FIVES_ROOT_FOLDER")
    if root_folder is None:
        raise EnvironmentError("FIVES_ROOT_FOLDER environment variable is not set.")
    convert_masks(root_folder)
