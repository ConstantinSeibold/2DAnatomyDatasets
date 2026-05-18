"""Base dataset classes for the three task families in this repo.

Splits-JSON contract reminder
-----------------------------
Each entry in a split is ``{"image": rel_path, "target": rel_path}``. The
``"target"`` key is optional: when a dataset's test partition ships
without ground truth (e.g. DRIVE), the entry is just ``{"image": ...}``.
In that case ``__getitem__`` returns ``(image, None)``. Downstream eval
code MUST check for ``None`` before computing loss; default PyTorch
collation cannot batch ``(tensor, None)`` mixed with ``(tensor, tensor)``,
so use :func:`collate_optional_target` when batching splits that may
contain target-less entries.
"""

import os
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection
from PIL import Image
import numpy as np
from typing import Any, Callable, Optional, Tuple, List


class BaseMultiLabelDataset(Dataset):
    def __init__(self, root_dir, splits_json, split, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        splits = json.load(open(splits_json))

        self.label_dict = {
            int(key): splits["label_dict"][key] for key in splits["label_dict"]
        }

        self.splits = splits[split]

    def __len__(self):
        return len(self.splits)

    def __getitem__(self, idx):
        current_case = self.splits[idx]

        image_path = os.path.join(self.root_dir, current_case["image"])
        image = np.array(Image.open(image_path).convert("RGB"))

        target = current_case.get("target")
        if target is None:
            if self.transform is not None:
                image = self.transform(image=image)["image"]
            return image, None

        mask_path = os.path.join(self.root_dir, target)
        mask = np.load(mask_path) > 0
        mask = mask[list(self.label_dict.keys())].astype(int)

        if self.transform is not None:
            mask = mask.transpose([1, 2, 0])
            transformed = self.transform(image=image, mask=mask)

            image = transformed["image"]
            mask = transformed["mask"]
            mask = mask.permute(2, 0, 1)

        return image, mask


class BaseMultiClassDataset(Dataset):
    def __init__(self, root_dir, splits_json, split, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        splits = json.load(open(splits_json))

        self.label_dict = splits["label_dict"]

        self.splits = splits[split]

    def __len__(self):
        return len(self.splits)

    def __getitem__(self, idx):
        current_case = self.splits[idx]

        image_path = os.path.join(self.root_dir, current_case["image"])
        image = np.array(Image.open(image_path).convert("RGB"))

        target = current_case.get("target")
        if target is None:
            if self.transform is not None:
                image = self.transform(image=image)["image"]
            return image, None

        mask_path = os.path.join(self.root_dir, target)
        mask = np.array(Image.open(mask_path).convert("F"))

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask


class BaseDetectionDataset(CocoDetection):
    def __init__(
        self, root_dir, annFile, transform=None, target_transform=None, transforms=None
    ):
        super(BaseDetectionDataset, self).__init__(
            root_dir, annFile, transform, target_transform, transforms
        )
        self.label_dict = {
            self.coco.cats[key]["id"]: self.coco.cats[key]["name"]
            for key in self.coco.cats.keys()
        }

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        masks = []
        bboxes = []
        labels = []
        for t in target:
            masks.append(self.coco.annToMask(t))
            bboxes += [t["bbox"]]
            labels += [t["category_id"]]

        img_np = np.array(image)
        masks = (
            np.stack(masks, axis=-1)
            if masks
            else np.zeros((img_np.shape[0], img_np.shape[1], 0), dtype=np.uint8)
        )

        if self.transforms is not None:
            image = image.convert("RGB")
            image = np.array(image)  # [:,:,0]
            masks = masks.transpose([2, 0, 1])

            augmented = self.transforms(
                image=image, bboxes=bboxes, masks=[i for i in masks], labels=labels
            )
            image = augmented["image"]
            bboxes = augmented["bboxes"]
            masks = augmented["masks"]
            labels = augmented["labels"]
            target = {
                "image": torch.tensor(image),
                "boxes": torch.tensor(bboxes, dtype=torch.float32),
                "masks": torch.stack(masks, 0),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "coco": target,
            }
        else:
            target = {
                "image": torch.tensor(np.array(image)),
                "boxes": torch.tensor(bboxes, dtype=torch.float32),
                "masks": torch.tensor(masks),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "coco": target,
            }

        return image, target


def collate_optional_target(batch):
    """Collate function tolerant of ``(image, None)`` entries.

    - If every target in the batch is ``None`` (a homogeneous unlabeled
      batch, e.g. DRIVE test), returns ``(stacked_images, None)``.
    - If every target is a tensor, falls back to default PyTorch stacking.
    - Mixed batches raise: this is almost always a bug in the splits JSON
      (some entries dropped ``target`` and others kept it), and silently
      padding with zeros is what the previous implementation did wrong.
    """
    from torch.utils.data._utils.collate import default_collate

    images, targets = zip(*batch)
    none_count = sum(t is None for t in targets)
    if none_count == len(targets):
        return default_collate(list(images)), None
    if none_count == 0:
        return default_collate(list(images)), default_collate(list(targets))
    raise ValueError(
        f"collate_optional_target received a mixed batch "
        f"({none_count}/{len(targets)} entries had target=None). "
        "All entries in a batch must either have a target or none of them must."
    )
