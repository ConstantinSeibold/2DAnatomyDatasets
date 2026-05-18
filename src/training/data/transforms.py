"""Backcompat shim. Real implementation lives in ``anatomy_datasets.transforms``."""

from anatomy_datasets.transforms import (  # noqa: F401
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_transform,
    get_transform_det,
)
