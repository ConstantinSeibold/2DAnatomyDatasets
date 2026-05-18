"""mmdetection runtime adapter for the sharded-COCO format.

Registers ``ShardedCocoMMDetDataset`` to ``mmdet.registry.DATASETS``.
Reads ``index.json`` once at construction; opens per-image annotation
shards lazily inside ``parse_data_info``.

Usage (mmdet config)::

    from anatomy_datasets.adapters.mmdet import ShardedCocoMMDetDataset  # noqa

    train_dataloader = dict(
        dataset=dict(
            type="ShardedCocoMMDetDataset",
            sharded_root="./exports/sharded_paxray",
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

from anatomy_datasets.formats.sharded_coco import rle_to_binary_mask


def _require_mmdet():
    try:
        from mmdet.datasets import BaseDetDataset  # noqa
        from mmdet.registry import DATASETS  # noqa
        return BaseDetDataset, DATASETS
    except ImportError as e:
        raise ImportError(
            "anatomy_datasets.adapters.mmdet requires the optional "
            "'mmdet' extra. Install with: pip install 'anatomy-datasets[mmdet]'"
        ) from e


_BaseDetDataset, _DATASETS = _require_mmdet()


@_DATASETS.register_module()
class ShardedCocoMMDetDataset(_BaseDetDataset):
    """mmdet Dataset over a sharded-COCO export tree.

    Config keys consumed:

    - ``sharded_root`` (required): dir containing ``index.json``.
    - ``split`` (required): ``train`` / ``val`` / ``test``.
    - ``decode_masks`` (optional, default ``True``): decode RLE to dense
      binary masks at load time. Set ``False`` to keep RLE dicts
      in ``data_info`` (cheaper memory, but downstream pipeline must
      handle RLE explicitly).
    """

    METAINFO: dict = dict(classes=())

    def __init__(
        self,
        sharded_root: str,
        split: str,
        decode_masks: bool = True,
        **kwargs,
    ):
        self._sharded_root = sharded_root
        self._split = split
        self._decode_masks = decode_masks

        with open(os.path.join(sharded_root, "index.json")) as f:
            self._index = json.load(f)
        with open(os.path.join(sharded_root, "categories.json")) as f:
            self._categories = json.load(f)["categories"]
        self._cat_id_to_label = {c["id"]: i for i, c in enumerate(self._categories)}

        if split not in self._index["splits"]:
            raise KeyError(
                f"split {split!r} not in index; have {sorted(self._index['splits'])}"
            )
        self._image_ids: List[str] = list(self._index["splits"][split])

        metainfo = kwargs.pop("metainfo", {}) or {}
        metainfo.setdefault(
            "classes", tuple(c["name"] for c in self._categories)
        )

        super().__init__(metainfo=metainfo, ann_file="", data_root=sharded_root, **kwargs)

    def _resolve_image_path(self, file_name: str) -> str:
        if self._index["image_link_mode"] == "reference":
            return os.path.join(self._index["image_dir"], file_name)
        return os.path.join(self._sharded_root, file_name)

    def _load_shard(self, image_id: str) -> dict:
        path = os.path.join(
            self._sharded_root, self._index["ann_dir"], f"{image_id}.json"
        )
        with open(path) as f:
            return json.load(f)

    def load_data_list(self) -> List[dict]:
        out = []
        for image_id in self._image_ids:
            shard = self._load_shard(image_id)
            img_info = shard["image"]
            instances = []
            for ann in shard["annotations"]:
                label = self._cat_id_to_label[ann["category_id"]]
                seg = ann["segmentation"]
                if self._decode_masks:
                    mask = rle_to_binary_mask(seg)
                else:
                    mask = seg
                x, y, w, h = ann["bbox"]
                instances.append(
                    dict(
                        bbox=[x, y, x + w, y + h],   # mmdet uses xyxy
                        bbox_label=label,
                        mask=mask,
                        ignore_flag=ann.get("iscrowd", 0),
                    )
                )
            out.append(
                dict(
                    img_path=self._resolve_image_path(img_info["file_name"]),
                    img_id=image_id,
                    height=img_info["height"],
                    width=img_info["width"],
                    instances=instances,
                )
            )
        return out
