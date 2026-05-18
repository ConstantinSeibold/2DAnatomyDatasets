"""Example: segmentation_models_pytorch U-Net on PAXRay (multilabel).

Multilabel head: ``classes`` = number of channels selected via
``label_dict``; loss is per-channel BCE. SMP handles this directly --
no exporter / adapter needed.

Run::

    export PAXRAY_ROOT_PATH=./datasets/paxray
    pip install -e '.[smp]'
    python configs/smp/paxray_unet.py
"""

from __future__ import annotations

import json
import os

import torch
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from anatomy_datasets import PAXRay4, get_transform


def main() -> None:
    splits_json = os.path.join(os.environ["PAXRAY_ROOT_PATH"], "paxray.json")

    meta = json.load(open(splits_json))
    norm = meta.get("normalization", {})
    train_tx = get_transform("train", mean=norm.get("mean"), std=norm.get("std"))
    val_tx = get_transform("val", mean=norm.get("mean"), std=norm.get("std"))

    train_ds = PAXRay4(split="train", transform=train_tx)
    val_ds = PAXRay4(split="val", transform=val_tx)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    n_classes = len(train_ds.label_dict)
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=n_classes,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Multilabel: per-channel BCE.
    loss_fn = torch.nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode="multilabel")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.float().to(device)        # [B, C, H, W]
        opt.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks) + dice(logits, masks)
        loss.backward()
        opt.step()
        print(f"train loss: {loss.item():.4f}")
        break  # remove for actual training

    model.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            logits = model(images.to(device))
            print(f"val logits shape: {logits.shape} (multilabel)")
            break


if __name__ == "__main__":
    main()
