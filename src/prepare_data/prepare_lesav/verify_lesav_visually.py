"""
Visual-verify pass for LES-AV.

Renders overlays to
``./verify_visually/lesav/{vessel,av}/<split>/sample_NN.png`` for both
registered variants and asserts mask shape / value invariants per
CLAUDE.md step 7.4. Because LES-AV uses a multilabel representation,
we use ``visualize_label`` (one alpha-blended color per channel) rather
than ``visualize_multiclass`` (which collapses channels into a single
index map).
"""

import os
import sys

import numpy as np
from torch.utils.data import DataLoader


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "training"))

from anatomy_datasets.datasets.lesav import (  # noqa: E402
    LESAV_AV_Dataset,
    LESAV_Vessel_Dataset,
)
from data.transforms import get_transform  # noqa: E402
from visualization.visualize import visualize_label  # noqa: E402


SAMPLES_PER_SPLIT = 3


def _verify_variant(name, cls, splits_json, root_path, out_root, expected_channels):
    print(f"\n=== {name} ===")
    variant_root = os.path.join(out_root, name)
    os.makedirs(variant_root, exist_ok=True)

    for split in ("train", "val", "test"):
        ds = cls(root_dir=root_path, splits_json=splits_json, split=split)
        print(f"[{split}] {len(ds)} samples")

        split_dir = os.path.join(variant_root, split)
        os.makedirs(split_dir, exist_ok=True)

        for i in range(min(SAMPLES_PER_SPLIT, len(ds))):
            image, mask = ds[i]
            mask_int = np.asarray(mask).astype(np.uint8)
            assert mask_int.shape[0] == expected_channels, (
                f"{name}/{split}[{i}] mask channels: got {mask_int.shape[0]}, "
                f"expected {expected_channels}"
            )
            assert set(np.unique(mask_int).tolist()).issubset({0, 1}), (
                f"{name}/{split}[{i}] mask not binary: {np.unique(mask_int)}"
            )
            overlay = visualize_label(
                label=mask_int,
                img=np.asarray(image).astype(np.uint8),
                label_to_visualize=list(range(expected_channels)),
                concat=True,
                axis=1,
            )
            out_path = os.path.join(split_dir, f"sample_{i:02d}.png")
            overlay.save(out_path)
            print(f"  -> {out_path}")

        ds_t = cls(
            root_dir=root_path,
            splits_json=splits_json,
            split=split,
            transform=get_transform("train"),
        )
        loader = DataLoader(ds_t, batch_size=2, shuffle=False)
        img_batch, mask_batch = next(iter(loader))
        print(
            f"[{split}] batched tensors: image {tuple(img_batch.shape)} "
            f"mask {tuple(mask_batch.shape)}"
        )


def main() -> None:
    root_path = os.getenv(
        "LESAV_ROOT_FOLDER", os.path.join(REPO_ROOT, "datasets", "lesav")
    )
    out_root = os.path.join(REPO_ROOT, "verify_visually", "lesav")
    os.makedirs(out_root, exist_ok=True)

    _verify_variant(
        name="vessel",
        cls=LESAV_Vessel_Dataset,
        splits_json=os.path.join(root_path, "lesav_vessel_splits.json"),
        root_path=root_path,
        out_root=out_root,
        expected_channels=1,
    )
    _verify_variant(
        name="av",
        cls=LESAV_AV_Dataset,
        splits_json=os.path.join(root_path, "lesav_av_splits.json"),
        root_path=root_path,
        out_root=out_root,
        expected_channels=3,
    )

    # Cross-variant sanity: artery + vein channels must overlap at true crossings
    # (the white pixels in the upstream A/V mask). If they never overlap, the
    # white-pixel folding in prepare_lesav_labels.py is broken.
    av_train = LESAV_AV_Dataset(
        root_dir=root_path,
        splits_json=os.path.join(root_path, "lesav_av_splits.json"),
        split="train",
    )
    total_overlap = 0
    for i in range(len(av_train)):
        _, mask = av_train[i]
        total_overlap += int(((mask[0] == 1) & (mask[1] == 1)).sum())
    assert total_overlap > 0, (
        "No artery+vein overlap pixels across train split — "
        "white-crossing handling in prepare_lesav_labels.py is broken."
    )
    print(f"\nartery+vein overlap pixels in train split: {total_overlap}")

    print("\nLES-AV visual verification complete.")
    print(f"Overlays written to {out_root}/")


if __name__ == "__main__":
    main()
