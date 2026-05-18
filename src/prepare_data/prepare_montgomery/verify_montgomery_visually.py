"""
Visual-verify pass for the Montgomery dataset.

Loads samples from each split through Montgomery_Dataset, asserts mask
dtype / unique-value invariants, and writes overlay PNGs to
./verify_visually/montgomery/<split>/. Run after get_montgomery_full.sh.

See CLAUDE.md "Adding a dataset from a GitHub issue" -> step 7.4.
"""

import os
import sys

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "training"))

from anatomy_datasets.datasets.montgomery import Montgomery_Dataset
from data.transforms import get_transform
from visualization.visualize import visualize_multiclass


SAMPLES_PER_SPLIT = 4
EXPECTED_CLASSES = {0, 1, 2}


def main() -> None:
    root_path = os.getenv(
        "MONTGOMERY_ROOT_FOLDER",
        os.path.join(REPO_ROOT, "datasets", "montgomery"),
    )
    json_path = os.path.join(root_path, "montgomery_splits.json")
    out_root = os.path.join(REPO_ROOT, "verify_visually", "montgomery")
    os.makedirs(out_root, exist_ok=True)

    for split in ("train", "val", "test"):
        ds_raw = Montgomery_Dataset(
            root_dir=root_path,
            splits_json=json_path,
            split=split,
        )
        print(f"[{split}] {len(ds_raw)} samples")

        split_dir = os.path.join(out_root, split)
        os.makedirs(split_dir, exist_ok=True)

        for i in range(min(SAMPLES_PER_SPLIT, len(ds_raw))):
            image, mask = ds_raw[i]
            unique = set(np.unique(mask.astype(int)).tolist())
            assert unique.issubset(EXPECTED_CLASSES), (
                f"{split}[{i}] mask has unexpected values: {unique}"
            )
            overlay = visualize_multiclass(image, mask.astype(int), ds_raw.label_dict)
            out_path = os.path.join(split_dir, f"sample_{i:02d}.png")
            overlay.save(out_path)
            print(f"  -> {out_path}  (mask classes: {sorted(unique)})")

        ds_t = Montgomery_Dataset(
            root_dir=root_path,
            splits_json=json_path,
            split=split,
            transform=get_transform("train"),
        )
        loader = DataLoader(ds_t, batch_size=2, shuffle=False)
        img_batch, mask_batch = next(iter(loader))
        print(
            f"[{split}] batched tensors: image {tuple(img_batch.shape)} "
            f"mask {tuple(mask_batch.shape)}"
        )

    print("\nMontgomery visual verification complete.")
    print(f"Overlays written to {out_root}/")


if __name__ == "__main__":
    main()
