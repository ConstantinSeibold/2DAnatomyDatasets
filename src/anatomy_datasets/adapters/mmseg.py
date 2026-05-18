"""mmsegmentation runtime adapter.

Two registered Dataset classes:

- ``AnatomyMulticlassDataset`` for splits-JSON datasets whose targets
  are single-channel integer-ID PNG masks (DRIVE, STARE, CHASE_DB1,
  RAVIR, DUKE, MedakaHeart, JumpBroadcast).

- ``AnatomyMultilabelDataset`` for splits-JSON datasets whose targets
  are 3D ``.npy`` mask stacks (PAXRay, PAXRay++, BS80k anatomy, JSRT
  binary). The user picks one channel via the ``target_channel``
  config key; mmseg sees a binary mask. To train on all channels at
  once, instantiate one dataset per channel and use mmseg's
  ``ConcatDataset`` or run separate trainings.

Usage (in an mmseg config)::

    from anatomy_datasets.adapters.mmseg import (  # noqa: registers
        AnatomyMulticlassDataset, AnatomyMultilabelDataset,
    )

    train_dataloader = dict(
        dataset=dict(
            type="AnatomyMulticlassDataset",
            splits_json="./datasets/drive/drive_splits.json",
            data_root="./datasets/drive",
            split="train",
            pipeline=[...],
        ),
        ...
    )
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np
from PIL import Image


def _require_mmseg():
    try:
        from mmseg.datasets import BaseSegDataset  # noqa
        from mmseg.registry import DATASETS  # noqa
        return BaseSegDataset, DATASETS
    except ImportError as e:
        raise ImportError(
            "anatomy_datasets.adapters.mmseg requires the optional "
            "'mmseg' extra. Install with: pip install 'anatomy-datasets[mmseg]'"
        ) from e


_BaseSegDataset, _DATASETS = _require_mmseg()


def _resolve_label_dict(raw: dict) -> dict:
    return {int(k): v for k, v in raw.items()}


@_DATASETS.register_module()
class AnatomyMulticlassDataset(_BaseSegDataset):
    """Splits-JSON multiclass dataset for mmseg.

    Config keys consumed (beyond mmseg's standard pipeline / metainfo):

    - ``splits_json`` (required): path to the splits JSON.
    - ``split`` (required): one of ``train`` / ``val`` / ``test``.
    - ``data_root`` (optional): root the splits-JSON paths are relative to;
      defaults to ``dirname(splits_json)``.
    - ``reduce_zero_label`` (optional, default ``False``): mmseg flag.
    - ``ignore_missing_target`` (optional, default ``True``): skip entries
      whose ``target`` is null (DRIVE-style test split) instead of erroring.
    """

    def __init__(
        self,
        splits_json: str,
        split: str,
        data_root: Optional[str] = None,
        reduce_zero_label: bool = False,
        ignore_missing_target: bool = True,
        **kwargs,
    ):
        self._splits_json = splits_json
        self._split = split
        self._anatomy_data_root = data_root or os.path.dirname(splits_json)
        self._ignore_missing_target = ignore_missing_target

        with open(splits_json) as f:
            self._source = json.load(f)
        label_dict = _resolve_label_dict(self._source["label_dict"])
        classes = [label_dict[k] for k in sorted(label_dict)]
        metainfo = kwargs.pop("metainfo", {}) or {}
        metainfo.setdefault("classes", tuple(classes))

        super().__init__(
            data_root=self._anatomy_data_root,
            data_prefix=dict(img_path="", seg_map_path=""),
            reduce_zero_label=reduce_zero_label,
            metainfo=metainfo,
            **kwargs,
        )

    def load_data_list(self) -> List[dict]:
        out = []
        for entry in self._source.get(self._split, []):
            target = entry.get("target")
            if target is None:
                if self._ignore_missing_target:
                    continue
                raise ValueError(
                    f"AnatomyMulticlassDataset entry missing target: {entry!r} "
                    "(pass ignore_missing_target=False to surface this)."
                )
            out.append(
                dict(
                    img_path=os.path.join(self._anatomy_data_root, entry["image"]),
                    seg_map_path=os.path.join(self._anatomy_data_root, target),
                    label_map=None,
                    reduce_zero_label=self.reduce_zero_label,
                    seg_fields=[],
                )
            )
        return out


@_DATASETS.register_module()
class AnatomyMultilabelDataset(_BaseSegDataset):
    """Splits-JSON multilabel (.npy stack) dataset, exposed as a binary
    mask for one selected channel.

    Extra config keys:

    - ``splits_json``, ``split``, ``data_root`` (same as multiclass).
    - ``target_channel`` (required): integer channel index from the .npy
      stack to expose as the foreground.
    - ``cache_decoded_masks`` (optional, default ``False``): write a
      cached binary PNG next to each .npy on first read. Massively
      speeds up subsequent epochs at the cost of disk.

    The decoded mask is written as a palette PNG (0/1) on first access
    when ``cache_decoded_masks=True`` so mmseg's standard PNG loader
    works.
    """

    def __init__(
        self,
        splits_json: str,
        split: str,
        target_channel: int,
        data_root: Optional[str] = None,
        cache_decoded_masks: bool = False,
        reduce_zero_label: bool = False,
        ignore_missing_target: bool = True,
        **kwargs,
    ):
        self._splits_json = splits_json
        self._split = split
        self._anatomy_data_root = data_root or os.path.dirname(splits_json)
        self._target_channel = target_channel
        self._cache_decoded_masks = cache_decoded_masks
        self._ignore_missing_target = ignore_missing_target

        with open(splits_json) as f:
            self._source = json.load(f)
        label_dict = _resolve_label_dict(self._source["label_dict"])
        if target_channel not in label_dict:
            raise KeyError(
                f"target_channel={target_channel} not in label_dict keys "
                f"{sorted(label_dict)}"
            )
        class_name = label_dict[target_channel]
        metainfo = kwargs.pop("metainfo", {}) or {}
        metainfo.setdefault("classes", ("background", class_name))

        super().__init__(
            data_root=self._anatomy_data_root,
            data_prefix=dict(img_path="", seg_map_path=""),
            reduce_zero_label=reduce_zero_label,
            metainfo=metainfo,
            **kwargs,
        )

    def _cached_mask_path(self, npy_rel: str) -> str:
        base, _ = os.path.splitext(npy_rel)
        return os.path.join(
            self._anatomy_data_root,
            f"{base}.ch{self._target_channel}.png",
        )

    def _ensure_cache(self, npy_rel: str) -> str:
        cache_path = self._cached_mask_path(npy_rel)
        if os.path.exists(cache_path):
            return cache_path
        stack = np.load(os.path.join(self._anatomy_data_root, npy_rel))
        if stack.ndim != 3:
            raise ValueError(
                f"Expected 3D mask stack, got shape {stack.shape}"
            )
        if self._target_channel >= stack.shape[0]:
            raise IndexError(
                f"target_channel={self._target_channel} out of range "
                f"(stack has {stack.shape[0]} channels)"
            )
        binary = (stack[self._target_channel] > 0).astype(np.uint8)
        img = Image.fromarray(binary, mode="P")
        img.putpalette([0, 0, 0, 255, 255, 255] + [0] * (768 - 6))
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        img.save(cache_path)
        return cache_path

    def load_data_list(self) -> List[dict]:
        out = []
        for entry in self._source.get(self._split, []):
            target = entry.get("target")
            if target is None:
                if self._ignore_missing_target:
                    continue
                raise ValueError(f"Missing target in entry: {entry!r}")

            if self._cache_decoded_masks:
                seg_map_path = self._ensure_cache(target)
            else:
                # No cache: rely on a custom pipeline transform to decode
                # the .npy at load time. Surface a path mmseg can stat but
                # which downstream code must intercept.
                seg_map_path = os.path.join(self._anatomy_data_root, target)

            out.append(
                dict(
                    img_path=os.path.join(self._anatomy_data_root, entry["image"]),
                    seg_map_path=seg_map_path,
                    label_map=None,
                    reduce_zero_label=self.reduce_zero_label,
                    seg_fields=[],
                    # Channel index travels alongside so a custom transform can use it.
                    anatomy_target_channel=self._target_channel,
                )
            )
        return out
