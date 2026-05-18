"""Example: segmentation_models_pytorch U-Net on DRIVE.

Self-contained runnable script. Demonstrates how the dataloaders this
repo ships drop straight into an SMP training loop -- no exporter, no
adapter needed.

Run::

    export DRIVE_ROOT_FOLDER=./datasets/drive
    pip install -e '.[smp]'
    python configs/smp/drive_unet.py

This is an example, not a tuned baseline.
"""

from __future__ import annotations

import json
import os

import torch
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from anatomy_datasets import DRIVE, collate_optional_target, get_transform


def main() -> None:
    splits_json = os.path.join(
        os.environ["DRIVE_ROOT_FOLDER"], "drive_splits.json"
    )

    # Pull per-dataset normalization stats from the splits JSON if present;
    # otherwise fall back to ImageNet defaults.
    meta = json.load(open(splits_json))
    norm = meta.get("normalization", {})
    train_tx = get_transform(
        "train", mean=norm.get("mean"), std=norm.get("std")
    )
    val_tx = get_transform("val", mean=norm.get("mean"), std=norm.get("std"))

    train_ds = DRIVE(split="train", transform=train_tx)
    val_ds = DRIVE(split="val", transform=val_tx)
    test_ds = DRIVE(split="test", transform=val_tx)   # missing-GT in test

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    # collate_optional_target lets us iterate the test split even though
    # entries have no GT.
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=collate_optional_target,
    )

    n_classes = len(meta["label_dict"])
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=n_classes,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loss_fn = smp.losses.DiceLoss(mode="multiclass")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # One short epoch -- example, not tuned.
    model.train()
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.long().to(device)
        opt.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        opt.step()
        print(f"train loss: {loss.item():.4f}")
        break  # remove for actual training

    # Eval with GT (val split).
    model.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            logits = model(images.to(device))
            print(f"val logits shape: {logits.shape}")
            break

    # Inference on test (no GT).
    with torch.no_grad():
        for images, masks in test_loader:
            logits = model(images.to(device))
            print(f"test (no GT) logits shape: {logits.shape}, target is {masks}")
            break


if __name__ == "__main__":
    main()
