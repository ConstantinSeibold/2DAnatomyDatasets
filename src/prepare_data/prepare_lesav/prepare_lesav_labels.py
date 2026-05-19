"""
Convert LES-AV vessel + artery/vein annotations to per-case
4-channel boolean ``.npy`` stacks under ``masks_npy/<id>.npy``.

Channel layout (shape ``[4, H, W]``, ``dtype=bool``):

    0: vessel    (from vessel-segmentations/<id>.png > 0)
    1: artery    (A/V RGB pixel is red OR white)
    2: vein      (A/V RGB pixel is blue OR white)
    3: uncertain (A/V RGB pixel is green)

The A/V mask in ``arteries-and-veins/<id>.png`` uses 5 RGB colours
only:
    (0,0,0)       background
    (255,0,0)     artery
    (0,0,255)     vein
    (0,255,0)     uncertain  (annotator could not disambiguate)
    (255,255,255) true A+V crossing  (pixel is BOTH artery AND vein)

We fold white into both the artery and vein channels so multilabel
consumers see the overlap directly. ``BaseMultiLabelDataset`` then
selects channels via the splits-JSON ``label_dict``.
"""

import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def convert_masks(root_folder: str) -> None:
    vessel_dir = os.path.join(root_folder, "vessel-segmentations")
    av_dir = os.path.join(root_folder, "arteries-and-veins")
    out_dir = os.path.join(root_folder, "masks_npy")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(vessel_dir) if f.endswith(".png"))
    if not files:
        raise FileNotFoundError(f"No vessel PNGs under {vessel_dir}")

    for fname in tqdm(files, desc="LES-AV labels"):
        case_id = os.path.splitext(fname)[0]

        vessel = np.array(Image.open(os.path.join(vessel_dir, fname)).convert("L"))
        vessel_bool = vessel > 0

        av_path = os.path.join(av_dir, fname)
        if not os.path.isfile(av_path):
            raise FileNotFoundError(
                f"Missing A/V mask for {case_id}: {av_path}"
            )
        av = np.array(Image.open(av_path).convert("RGB"))
        r, g, b = av[..., 0], av[..., 1], av[..., 2]

        is_red = (r == 255) & (g == 0) & (b == 0)
        is_blue = (r == 0) & (g == 0) & (b == 255)
        is_green = (r == 0) & (g == 255) & (b == 0)
        is_white = (r == 255) & (g == 255) & (b == 255)

        artery = is_red | is_white
        vein = is_blue | is_white
        uncertain = is_green

        if vessel_bool.shape != artery.shape:
            raise ValueError(
                f"Shape mismatch for {case_id}: vessel {vessel_bool.shape} "
                f"vs A/V {artery.shape}"
            )

        stack = np.stack([vessel_bool, artery, vein, uncertain], axis=0)
        np.save(os.path.join(out_dir, case_id + ".npy"), stack)


if __name__ == "__main__":
    root_folder = os.getenv("LESAV_ROOT_FOLDER")
    if root_folder is None:
        raise EnvironmentError("LESAV_ROOT_FOLDER environment variable is not set.")
    convert_masks(root_folder)
