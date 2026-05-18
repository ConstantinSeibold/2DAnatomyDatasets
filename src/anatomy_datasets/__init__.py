"""anatomy_datasets — dataset prep + dataloaders + ecosystem bridges
for 2D anatomical segmentation.

Two surfaces:

1. The **raw classes** (`*_Dataset`) take explicit `root_dir`, `splits_json`,
   `split`, and `transform` arguments. Use these when you want full control
   or you're integrating with a config-driven framework that builds the
   args itself::

       from anatomy_datasets import DRIVE_Dataset
       ds = DRIVE_Dataset("./datasets/drive", "./datasets/drive/drive_splits.json", "train")

2. The **convenience aliases** (`DRIVE`, `STARE`, `PAXRay166`, ...) are
   callables that read the dataset root from a ``<NAME>_ROOT*`` env var
   and auto-locate the default splits JSON::

       export DRIVE_ROOT_FOLDER=./datasets/drive
       from anatomy_datasets import DRIVE
       ds = DRIVE(split="train")

Static metadata (license, modality, citation) is in `anatomy_datasets.registry`.
Computed metadata (seed, normalization stats, version) is written into each
dataset's splits JSON by `anatomy_datasets.add_metadata_to_splits_json`.
"""

from __future__ import annotations

import os
from typing import Optional

# --- Core utilities -----------------------------------------------------------
from .registry import DATASET_REGISTRY, DatasetInfo, get_dataset_info
from .stats import compute_image_stats
from .postprocess import add_metadata_to_splits_json

# --- Base classes -------------------------------------------------------------
from .base import (
    BaseDetectionDataset,
    BaseMultiClassDataset,
    BaseMultiLabelDataset,
    collate_optional_target,
)

# --- Transforms ---------------------------------------------------------------
from .transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_transform,
    get_transform_det,
)

# --- Raw dataset classes ------------------------------------------------------
from .datasets import (
    BS80KAnatomy_detection_Dataset,
    CHASEDB1_Dataset,
    DRIVE_Dataset,
    DUKE_Dataset,
    JSRT_binary_Dataset,
    JSRT_detection_Dataset,
    JumpBroadcast_Dataset,
    MedakaHeart_Dataset,
    PAXRay4_binary_Dataset,
    PAXRay4_detection_Dataset,
    PAXRay166_binary_Dataset,
    PAXRay166_detection_Dataset,
    PAXRayPP_binary_Dataset,
    PAXRayPP_Instance_Dataset,
    RAVIR_Dataset,
    STARE_Dataset,
    Teeth_Dataset,
)

from ._discovery import DISCOVERY, DiscoveryEntry


# --- Convenience constructors -------------------------------------------------

def _resolve_root(entry: DiscoveryEntry, root: Optional[str]) -> str:
    if root is not None:
        return root
    env = os.getenv(entry.env_var)
    if env is None:
        raise EnvironmentError(
            f"Dataset root not provided. Pass root= or set ${entry.env_var}."
        )
    return env


def _resolve_splits_json(
    entry: DiscoveryEntry, root: str, splits_json: Optional[str]
) -> str:
    if splits_json is not None:
        return splits_json
    if entry.default_json_name is None:
        raise ValueError(
            f"{entry.registry_name} has no default splits-JSON name. "
            "Pass splits_json= explicitly."
        )
    return os.path.join(root, entry.default_json_name)


def _make_segmentation_alias(name: str, entry: DiscoveryEntry):
    base_cls = entry.cls

    def factory(
        split: str = "train",
        root: Optional[str] = None,
        splits_json: Optional[str] = None,
        transform=None,
    ):
        resolved_root = _resolve_root(entry, root)
        resolved_json = _resolve_splits_json(entry, resolved_root, splits_json)
        return base_cls(resolved_root, resolved_json, split, transform)

    factory.__name__ = name
    factory.__qualname__ = name
    factory.__doc__ = (
        f"Construct a {name} dataset. "
        f"If ``root`` is omitted, falls back to ${entry.env_var}. "
        f"If ``splits_json`` is omitted, defaults to "
        f"``<root>/{entry.default_json_name}``."
        if entry.default_json_name else
        f"Construct a {name} dataset. ``splits_json`` is required for this dataset."
    )
    factory.dataset_class = base_cls
    factory.discovery = entry
    return factory


def _make_detection_alias(name: str, entry: DiscoveryEntry):
    base_cls = entry.cls

    def factory(
        annFile: str,
        root: Optional[str] = None,
        transform=None,
        target_transform=None,
        transforms=None,
    ):
        resolved_root = _resolve_root(entry, root)
        return base_cls(
            resolved_root, annFile, transform, target_transform, transforms
        )

    factory.__name__ = name
    factory.__qualname__ = name
    factory.__doc__ = (
        f"Construct a {name} detection dataset. ``annFile`` is the COCO "
        f"annotations JSON. If ``root`` is omitted, falls back to "
        f"${entry.env_var}."
    )
    factory.dataset_class = base_cls
    factory.discovery = entry
    return factory


# Build the public aliases. The names are intentionally short so external
# configs can do ``from anatomy_datasets import DRIVE``.
_ALIASES = {}
for _alias, _entry in DISCOVERY.items():
    if _entry.task == "segmentation":
        _ALIASES[_alias] = _make_segmentation_alias(_alias, _entry)
    else:
        _ALIASES[_alias] = _make_detection_alias(_alias, _entry)
globals().update(_ALIASES)


__all__ = [
    # core
    "DATASET_REGISTRY", "DatasetInfo", "get_dataset_info",
    "compute_image_stats", "add_metadata_to_splits_json",
    # base
    "BaseDetectionDataset", "BaseMultiClassDataset", "BaseMultiLabelDataset",
    "collate_optional_target",
    # transforms
    "IMAGENET_MEAN", "IMAGENET_STD", "get_transform", "get_transform_det",
    # raw classes
    "BS80KAnatomy_detection_Dataset",
    "CHASEDB1_Dataset", "DRIVE_Dataset", "DUKE_Dataset",
    "JSRT_binary_Dataset", "JSRT_detection_Dataset",
    "JumpBroadcast_Dataset", "MedakaHeart_Dataset",
    "PAXRay4_binary_Dataset", "PAXRay4_detection_Dataset",
    "PAXRay166_binary_Dataset", "PAXRay166_detection_Dataset",
    "PAXRayPP_binary_Dataset", "PAXRayPP_Instance_Dataset",
    "RAVIR_Dataset", "STARE_Dataset", "Teeth_Dataset",
    # discovery
    "DISCOVERY", "DiscoveryEntry",
] + list(_ALIASES.keys())
