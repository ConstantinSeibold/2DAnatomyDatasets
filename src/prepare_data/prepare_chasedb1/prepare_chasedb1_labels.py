"""
Convert CHASE_DB1 vessel annotations to single-channel PNG masks with
label values {0: background, 1: vessel}.

The original annotations are *_1stHO.png / *_2ndHO.png with values in
{0, 255}. We rewrite them in-place as binary {0, 1} PNG masks.
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def convert_masks(root_folder: str) -> None:
    files = [
        f
        for f in os.listdir(root_folder)
        if f.endswith(("_1stHO.png", "_2ndHO.png"))
    ]
    for fname in tqdm(files, desc="CHASE_DB1 labels"):
        path = os.path.join(root_folder, fname)
        mask = np.array(Image.open(path).convert("L"))
        binary = (mask > 0).astype(np.uint8)
        Image.fromarray(binary).save(path)


if __name__ == "__main__":
    root_folder = os.getenv("CHASEDB1_ROOT_FOLDER")
    if root_folder is None:
        raise EnvironmentError(
            "CHASEDB1_ROOT_FOLDER environment variable is not set."
        )
    convert_masks(root_folder)
