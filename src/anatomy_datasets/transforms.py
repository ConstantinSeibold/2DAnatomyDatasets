"""Albumentations transform builders for segmentation and detection.

Per-dataset normalization
-------------------------
``mean`` / ``std`` default to ImageNet stats when omitted. To use stats
computed for a specific dataset (recorded in its splits JSON under
``"normalization"``), pass them explicitly::

    import json
    meta = json.load(open(splits_json))
    norm = meta.get("normalization", {})
    tx = get_transform("train", mean=norm.get("mean"), std=norm.get("std"))
"""

from typing import Optional, Sequence

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resolve_norm(
    mean: Optional[Sequence[float]], std: Optional[Sequence[float]]
):
    return (
        list(mean) if mean is not None else list(IMAGENET_MEAN),
        list(std) if std is not None else list(IMAGENET_STD),
    )


def get_transform_det(
    split: str = "train",
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
):
    m, s = _resolve_norm(mean, std)
    if split == "train":
        return A.Compose(
            [
                A.Resize(512, 512),
                # A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.RandomGamma(p=0.2),
                A.ElasticTransform(p=0.1),
                A.GridDistortion(p=0.1),
                A.Normalize(mean=m, std=s),
                ToTensorV2(),
            ],
            additional_targets={
                "mask": "mask",
                "bboxes": "bboxes",
            },
            is_check_shapes=False,
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=[
                    "labels",
                ],
            ),
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=m, std=s),
                ToTensorV2(),
            ]
        )


def get_transform(
    split: str = "train",
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
):
    m, s = _resolve_norm(mean, std)
    if split == "train":
        return A.Compose(
            [
                A.RandomResizedCrop((512, 512), scale=(0.6, 1.)),
                # A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.RandomGamma(p=0.2),
                A.ElasticTransform(p=0.1),
                A.GridDistortion(p=0.1),
                A.Normalize(mean=m, std=s),
                ToTensorV2(),
            ],
            additional_targets={
                "mask": "mask",
            },
            is_check_shapes=False,
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=m, std=s),
                ToTensorV2(),
            ]
        )
