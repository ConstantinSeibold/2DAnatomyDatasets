"""
Visual-verify pass for HRF.

Renders overlays to ./verify_visually/hrf/<split>/sample_NN.png and
asserts mask dtype / unique-value invariants per CLAUDE.md step 7.4.
"""

import os
import sys

import numpy as np
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "training"))

from anatomy_datasets.datasets.hrf import HRF_Dataset
from data.transforms import get_transform
from visualization.visualize import visualize_multiclass


SAMPLES_PER_SPLIT = 4
EXPECTED_CLASSES = {0, 1}


def main() -> None:
    root_path = os.getenv(
        "HRF_ROOT_FOLDER", os.path.join(REPO_ROOT, "datasets", "hrf")
    )
    json_path = os.path.join(root_path, "hrf_splits.json")
    out_root = os.path.join(REPO_ROOT, "verify_visually", "hrf")
    os.makedirs(out_root, exist_ok=True)

    for split in ("train", "val", "test"):
        ds = HRF_Dataset(root_dir=root_path, splits_json=json_path, split=split)
        print(f"[{split}] {len(ds)} samples")

        split_dir = os.path.join(out_root, split)
        os.makedirs(split_dir, exist_ok=True)

        for i in range(min(SAMPLES_PER_SPLIT, len(ds))):
            image, mask = ds[i]
            unique = set(np.unique(mask.astype(int)).tolist())
            assert unique.issubset(EXPECTED_CLASSES), (
                f"{split}[{i}] mask has unexpected values: {unique}"
            )
            overlay = visualize_multiclass(image, mask.astype(int), ds.label_dict)
            out_path = os.path.join(split_dir, f"sample_{i:02d}.png")
            overlay.save(out_path)
            print(f"  -> {out_path}  (mask classes: {sorted(unique)})")

        ds_t = HRF_Dataset(
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

    print("\nHRF visual verification complete.")
    print(f"Overlays written to {out_root}/")


if __name__ == "__main__":
    main()
