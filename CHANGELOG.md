# Changelog

All notable changes to this project are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-18

Initial public release on PyPI.

### Added
- `anatomy_datasets` package: pip-installable surface for dataset prep,
  dataloaders, and ecosystem bridges.
- Dataset registry (`anatomy_datasets.registry.DATASET_REGISTRY`) with
  license, citation, BibTeX, modality, source URL, and paper URL for
  every shipped dataset.
- Convenience aliases (`DRIVE`, `STARE`, `PAXRay166`, ...) that resolve
  dataset root from `<NAME>_ROOT*` env vars and auto-locate the default
  splits JSON.
- Base dataset classes: `BaseMultiClassDataset`, `BaseMultiLabelDataset`,
  `BaseDetectionDataset`, plus `collate_optional_target` for missing-GT
  batches.
- Splits-JSON metadata postprocessor
  (`anatomy_datasets.add_metadata_to_splits_json`) covering both
  splits-format and COCO-format outputs.
- Sharded SA-1B-style COCO format
  (`anatomy_datasets.formats.sharded_coco`) for PAXRay-scale instance
  segmentation.
- Exporters: `anatomy_datasets.exporters.{coco,mmseg,hf}` with CLIs.
- Runtime adapters: `anatomy_datasets.adapters.{mmseg,mmdet}` (lazy
  imports; require optional extras).
- Optional install extras: `[mmseg]`, `[mmdet]`, `[smp]`, `[hf]`,
  `[all]`.
- Config templates under `configs/{mmseg,mmdet,smp,hf}/`.

### Datasets shipped
DRIVE, STARE, CHASE_DB1, HRF, RAVIR, DUKE, JSRT, PAXRay, PAXRay++,
Montgomery, Teeth, BS80k, MedakaHeart, JumpBroadcast.
