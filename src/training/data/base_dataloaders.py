"""Backcompat shim. Real implementation lives in ``anatomy_datasets.base``.

This file exists so that existing in-tree imports like
``from data.base_dataloaders import BaseMultiClassDataset`` continue to work
when ``src/training`` is on ``sys.path``. New code should import from
``anatomy_datasets.base`` directly.
"""

from anatomy_datasets.base import (  # noqa: F401
    BaseDetectionDataset,
    BaseMultiClassDataset,
    BaseMultiLabelDataset,
    collate_optional_target,
)
