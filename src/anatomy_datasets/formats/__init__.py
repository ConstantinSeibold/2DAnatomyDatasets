"""Storage formats for dataset annotations.

Currently:

- ``sharded_coco``: SA-1B-style per-image RLE JSON shards. Used to keep
  large instance-segmentation datasets (PAXRay, PAXRay++, BS80k)
  loadable lazily instead of as a single multi-GB COCO JSON.
"""

from .sharded_coco import (
    ShardedCocoDataset,
    binary_mask_to_rle,
    rle_to_binary_mask,
    write_sharded_coco,
)

__all__ = [
    "ShardedCocoDataset",
    "binary_mask_to_rle",
    "rle_to_binary_mask",
    "write_sharded_coco",
]
