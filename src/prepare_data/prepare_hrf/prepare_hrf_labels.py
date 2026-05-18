"""
Convert HRF vessel annotations to single-channel PNG masks with label
values {0: background, 1: vessel}.

The upstream manual1/*.tif vessel masks are binary with foreground in
{255} (and a few intermediate values from JPEG-style compression). We
threshold > 0 and rewrite each as a PNG with the same basename
(extension changed to .png) under manual1_png/ so the splits-JSON can
point at it without ambiguity.
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def convert_masks(root_folder: str) -> None:
    src_dir = os.path.join(root_folder, "manual1")
    out_dir = os.path.join(root_folder, "manual1_png")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".tif"))
    if not files:
        raise FileNotFoundError(f"No .tif vessel masks under {src_dir}")

    for fname in tqdm(files, desc="HRF labels"):
        mask = np.array(Image.open(os.path.join(src_dir, fname)).convert("L"))
        binary = (mask > 0).astype(np.uint8)
        out_name = os.path.splitext(fname)[0] + ".png"
        Image.fromarray(binary).save(os.path.join(out_dir, out_name))


if __name__ == "__main__":
    root_folder = os.getenv("HRF_ROOT_FOLDER")
    if root_folder is None:
        raise EnvironmentError("HRF_ROOT_FOLDER environment variable is not set.")
    convert_masks(root_folder)
