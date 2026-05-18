"""Runtime adapters - present our splits JSON / sharded format in the
shape that external libraries expect, without duplicating data on disk.

- ``mmseg`` : ``AnatomyMulticlassDataset``, ``AnatomyMultilabelDataset``
  registered to ``mmseg.registry.DATASETS``.
- ``mmdet`` : ``ShardedCocoMMDetDataset`` registered to
  ``mmdet.registry.DATASETS``; wraps the sharded format.

These modules import their target library lazily and raise an
ImportError with install instructions if it is missing.
"""

# Intentionally NOT eager-importing the submodules: importing
# ``anatomy_datasets.adapters.mmseg`` will trigger the optional
# ``mmsegmentation`` import. Users opt in by importing the specific
# adapter they want.

__all__: list = []
