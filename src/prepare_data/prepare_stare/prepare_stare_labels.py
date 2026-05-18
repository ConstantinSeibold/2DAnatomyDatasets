"""
Convert STARE manual annotations to single-channel PNG masks with
label values {0: background, 1: vessel}.

The original labels are .ppm files with values in {0, 255}; we rewrite them
as PNGs alongside the originals.
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm


LABEL_SUBDIRS = ["1st_labels_ah", "snd_label_vk"]


def convert_masks(root_folder: str) -> None:
    for subdir in LABEL_SUBDIRS:
        src_dir = os.path.join(root_folder, subdir)
        if not os.path.isdir(src_dir):
            continue
        dst_dir = os.path.join(root_folder, subdir + "_png")
        os.makedirs(dst_dir, exist_ok=True)
        files = [
            f
            for f in os.listdir(src_dir)
            if f.lower().endswith((".ppm", ".png", ".gif"))
        ]
        for fname in tqdm(files, desc=subdir):
            src_path = os.path.join(src_dir, fname)
            mask = np.array(Image.open(src_path).convert("L"))
            binary = (mask > 0).astype(np.uint8)
            out_name = os.path.splitext(fname)[0] + ".png"
            Image.fromarray(binary).save(os.path.join(dst_dir, out_name))


if __name__ == "__main__":
    root_folder = os.getenv("STARE_ROOT_FOLDER")
    if root_folder is None:
        raise EnvironmentError("STARE_ROOT_FOLDER environment variable is not set.")
    convert_masks(root_folder)
