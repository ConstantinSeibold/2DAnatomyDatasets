"""One-shot writers that materialize a splits-JSON dataset into the
on-disk layout a specific ecosystem expects.

- ``coco``  : mmdetection / generic COCO (RLE for multilabel sources,
              polygon passthrough for COCO sources, automatic delegation
              to the sharded format for large multilabel datasets).
- ``mmseg`` : mmsegmentation (palette PNGs + ``img_dir``/``ann_dir``).
- ``hf``    : HuggingFace ``datasets.DatasetDict``.

Each exporter exposes a Python function and a ``python -m`` CLI.
"""

from .coco import to_coco
from .mmseg import to_mmseg
from .hf import to_hf_dataset

__all__ = ["to_coco", "to_mmseg", "to_hf_dataset"]
