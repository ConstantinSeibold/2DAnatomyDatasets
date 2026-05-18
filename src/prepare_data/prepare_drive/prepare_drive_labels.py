"""
Convert DRIVE vessel annotations to single-channel PNG masks with
label values {0: background, 1: vessel}.

The original DRIVE masks are stored as GIF files with values in {0, 255}.
We rewrite them as PNGs alongside the originals so that the splits file can
point to them directly.
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm


SPLITS = ["training", "test"]
MASK_SUBDIRS = ["1st_manual"]


def convert_masks(root_folder: str) -> None:
    for split in SPLITS:
        for mask_subdir in MASK_SUBDIRS:
            src_dir = os.path.join(root_folder, split, mask_subdir)
            if not os.path.isdir(src_dir):
                continue
            dst_dir = os.path.join(root_folder, split, mask_subdir + "_png")
            os.makedirs(dst_dir, exist_ok=True)
            files = [
                f
                for f in os.listdir(src_dir)
                if f.lower().endswith((".gif", ".png", ".tif", ".tiff"))
            ]
            for fname in tqdm(files, desc=f"{split}/{mask_subdir}"):
                src_path = os.path.join(src_dir, fname)
                mask = np.array(Image.open(src_path).convert("L"))
                binary = (mask > 0).astype(np.uint8)
                out_name = os.path.splitext(fname)[0] + ".png"
                Image.fromarray(binary).save(os.path.join(dst_dir, out_name))


if __name__ == "__main__":
    root_folder = os.getenv("DRIVE_ROOT_FOLDER")
    if root_folder is None:
        raise EnvironmentError("DRIVE_ROOT_FOLDER environment variable is not set.")
    convert_masks(root_folder)
