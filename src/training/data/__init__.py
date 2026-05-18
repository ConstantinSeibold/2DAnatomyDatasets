"""Backcompat shim. Real implementations live in ``anatomy_datasets``.

Existing code that does ``from data import DRIVE_Dataset`` (when
``src/training`` is on ``sys.path``) continues to work because we
re-export the entire ``anatomy_datasets.datasets`` surface here. New
code should ``import anatomy_datasets`` instead.
"""

from anatomy_datasets.datasets import *  # noqa: F401,F403
from anatomy_datasets.datasets import __all__  # noqa: F401
