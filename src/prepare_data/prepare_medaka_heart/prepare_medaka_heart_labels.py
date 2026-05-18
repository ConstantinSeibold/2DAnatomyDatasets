"""
Convert the Medaka heart ``ventral_mask_combined`` PNGs to single-channel
PNG masks with dense class IDs.

Source encoding (each combined mask is saved as RGB, single nonzero value
per class region):

    train_images/ventral_mask_combined/         {0, 19, 20, 33}
    test_images/ventral_mask_combined_gray/     {0, 19, 20, 33}
    test_images/ventral_mask_combined_R0004/    {0, 19, 20, 255}

We remap to ``{0: background, 1: bulbus, 2: atrium, 3: heart}``.

The ``ventral_mask_combined`` mask is built by the dataset authors as
heart-painted-first then bulbus/atrium painted on top, so the residual
``heart`` value covers the ventricle (heart minus bulbus minus atrium).
The R0004 test split uses 255 in place of 33 for the heart class — both
are folded to dense ID 3.

Output: ``<combined-folder>_labels/<orig_basename>.png`` (uint8, single channel).
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm


COMBINED_DIRS = [
    os.path.join("train_images", "ventral_mask_combined"),
    os.path.join("test_images", "ventral_mask_combined_gray"),
    os.path.join("test_images", "ventral_mask_combined_R0004"),
]

VALUE_TO_CLASS = {0: 0, 19: 1, 20: 2, 33: 3, 255: 3}


def remap(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    out = np.zeros_like(mask, dtype=np.uint8)
    unknown = []
    for src, dst in VALUE_TO_CLASS.items():
        out[mask == src] = dst
    known = set(VALUE_TO_CLASS.keys())
    for v in np.unique(mask):
        if int(v) not in known:
            unknown.append(int(v))
    if unknown:
        raise ValueError(f"Unexpected mask value(s): {unknown}")
    return out


def convert_dir(root_folder: str, rel_dir: str) -> None:
    src_dir = os.path.join(root_folder, rel_dir)
    if not os.path.isdir(src_dir):
        return
    dst_dir = src_dir + "_labels"
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(".png"))
    for fname in tqdm(files, desc=rel_dir):
        mask = np.array(Image.open(os.path.join(src_dir, fname)))
        Image.fromarray(remap(mask)).save(os.path.join(dst_dir, fname))


def main() -> None:
    root_folder = os.getenv("MEDAKA_HEART_ROOT_FOLDER")
    if root_folder is None:
        raise EnvironmentError(
            "MEDAKA_HEART_ROOT_FOLDER environment variable is not set."
        )
    for rel_dir in COMBINED_DIRS:
        convert_dir(root_folder, rel_dir)


if __name__ == "__main__":
    main()
