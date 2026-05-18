# anatomy-datasets

Prepare 13 public 2D anatomical segmentation datasets once, plug them into
**SMP**, **mmsegmentation**, **mmdetection**, or **HuggingFace** without
rewriting dataloaders. Single splits-JSON contract with embedded license,
citation, normalization stats, and split seed.

## 60-second demo (CPU)

```bash
git clone https://github.com/ConstantinSeibold/2DAnatomyDatasets.git
cd 2DAnatomyDatasets
pip install -e '.[smp]'
sh src/prepare_data/prepare_drive/get_drive_full.sh
export DRIVE_ROOT_FOLDER=./datasets/drive
python configs/smp/drive_unet.py        # one forward + backward step on CPU
```

## Install

```bash
pip install -e .                # core (dataset prep + raw PyTorch dataloaders)
pip install -e '.[smp]'         # + segmentation_models_pytorch
pip install -e '.[mmseg]'       # + mmsegmentation adapter / exporter
pip install -e '.[mmdet]'       # + mmdetection adapter / exporter
pip install -e '.[hf]'          # + HuggingFace datasets
pip install -e '.[all]'         # everything
```

## Pick a path

| You want… | Start with |
|-----------|------------|
| Fastest end-to-end training | **SMP** — `from anatomy_datasets import DRIVE; DRIVE(split="train")` works with any SMP model. See `configs/smp/`. |
| Already using mmseg | **mmseg runtime adapter** — `AnatomyMulticlassDataset` in your config; no on-disk export. See `configs/mmseg/drive_unet.py`. |
| Training Mask R-CNN / detection | **mmdet** — small datasets use shipped COCO; PAXRay-scale uses sharded export. See `configs/mmdet/`. |
| Training SegFormer / Mask2Former | **HuggingFace** — `to_hf_dataset(splits_json, root)` returns a `DatasetDict`. See `configs/hf/`. |
| Just the raw masks | Use the splits JSON directly. Format spec below. |

## Datasets

Run `sh src/prepare_data/prepare_<name>/get_<name>_full.sh` to download +
prepare. Each emits a splits JSON with full metadata. Bracketed env var =
dataset root override.

| Name | Modality | Task | License | Env var |
|------|----------|------|---------|---------|
| DRIVE | fundus | multiclass | custom-academic | `DRIVE_ROOT_FOLDER` |
| STARE | fundus | multiclass | unknown | `STARE_ROOT_FOLDER` |
| CHASE_DB1 | fundus | multiclass | custom-academic | `CHASEDB1_ROOT_FOLDER` |
| RAVIR | IR retina | multiclass | custom-academic | `RAVIR_ROOT_FOLDER` |
| DUKE_OCT | OCT | multiclass | custom-academic | `DUKE_ROOT_PATH` |
| MedakaHeart | microscopy | multiclass | GPL-3.0 | `MEDAKA_HEART_ROOT_FOLDER` |
| JumpBroadcast | RGB video | multiclass | custom-academic | `JUMP_BROADCAST_ROOT_FOLDER` |
| JSRT | chest X-ray | multilabel | custom-academic | `JSRT_ROOT_PATH` |
| PAXRay (4 / 166) | chest X-ray | multilabel | custom-academic | `PAXRAY_ROOT_PATH` |
| PAXRay++ | chest X-ray | multilabel | custom-academic | `PAXRAYPP_ROOT` |
| Teeth | panoramic X-ray | detection | custom-academic | `TEETH_ROOT` |
| BS80k | scintigraphy | multilabel/detection | custom-academic | `BS80K_ROOT` |
| Montgomery | chest X-ray | multiclass | custom-academic | `MONTGOMERY_ROOT_FOLDER` |
| HRF | fundus | multiclass | CC-BY-4.0 | `HRF_ROOT_FOLDER` |

Programmatic metadata access:
```python
from anatomy_datasets import get_dataset_info
info = get_dataset_info("DRIVE")
print(info.license, info.source_url, info.citation, info.bibtex)
```

## Splits JSON contract

Every prepared dataset writes one file at `<root>/<name>_splits.json`:

```jsonc
{
  "name": "DRIVE",
  "version": "2026-05-18",
  "seed": 42,                                    // null if upstream split is fixed
  "modality": "fundus",
  "license": "custom-academic",
  "license_url": "...",
  "source_url": "...",
  "paper_url": "...",
  "citation": "Staal et al., IEEE TMI 2004.",
  "bibtex": "@article{...}",
  "normalization": {"mean": [r, g, b], "std": [r, g, b]},   // [0,1] float, train images only
  "label_dict": {"0": "background", "1": "vessel"},          // keys = class IDs
  "train": [{"image": "rel/path.png", "target": "rel/path.png"}, ...],
  "val":   [...],
  "test":  [{"image": "..."} , ...]   // target may be missing (e.g. DRIVE test); dataloader returns (image, None)
}
```

Target file conventions:
- **Multilabel** (`.npy`): 3D mask stack `(C, H, W)`. Channel = class ID = `label_dict` key.
- **Multiclass** (PNG): single-channel integer mask. Pixel value = class ID.

Missing-GT entries return `(image, None)`. Use `collate_optional_target` for batching:
```python
from anatomy_datasets import collate_optional_target
DataLoader(ds_test, collate_fn=collate_optional_target)
```

## Exporters (on-disk → other frameworks)

```bash
# mmsegmentation (img_dir / ann_dir / palette PNG; auto multiclass vs per-channel)
python -m anatomy_datasets.exporters.mmseg \
    --splits ./datasets/drive/drive_splits.json --root ./datasets/drive \
    --out ./exports/mmseg_drive

# COCO (mmdet / detectron2 / generic; auto-shards above 50k anns)
python -m anatomy_datasets.exporters.coco \
    --splits ./datasets/paxray/paxray.json --root ./datasets/paxray \
    --out ./exports/coco_paxray

# Sharded SA-1B per-image RLE JSON (lazy load; for PAXRay-scale instance seg)
python -m anatomy_datasets.formats.sharded_coco \
    --splits ./datasets/paxray/paxray.json --root ./datasets/paxray \
    --out ./exports/sharded_paxray --image-link-mode symlink

# HuggingFace DatasetDict (Python only, no CLI)
python -c "from anatomy_datasets.exporters import to_hf_dataset; \
           to_hf_dataset('./datasets/drive/drive_splits.json', './datasets/drive')"
```

## Adapters (runtime, no on-disk export)

```python
# mmseg config side-effect import
custom_imports = dict(imports=["anatomy_datasets.adapters.mmseg"])
train_dataloader = dict(dataset=dict(
    type="AnatomyMulticlassDataset",
    splits_json="./datasets/drive/drive_splits.json",
    split="train",
))

# mmdet config (reads from sharded export)
custom_imports = dict(imports=["anatomy_datasets.adapters.mmdet"])
train_dataloader = dict(dataset=dict(
    type="ShardedCocoMMDetDataset",
    sharded_root="./exports/sharded_paxray",
    split="train",
))
```

## Layout

```
src/anatomy_datasets/        # the importable package
  base.py                    # BaseMultiClassDataset, BaseMultiLabelDataset, BaseDetectionDataset, collate_optional_target
  datasets/                  # per-dataset class aliases
  transforms.py              # get_transform / get_transform_det (albumentations)
  registry.py                # DATASET_REGISTRY (license / citation / modality)
  postprocess.py             # add_metadata_to_splits_json
  stats.py                   # compute_image_stats
  formats/sharded_coco.py    # SA-1B-style writer + reader
  exporters/{coco,mmseg,hf}  # one-shot exporters + CLIs
  adapters/{mmseg,mmdet}     # runtime adapters (lazy lib imports)
src/prepare_data/prepare_<name>/  # per-dataset shell + python prep scripts
configs/{smp,mmseg,mmdet,hf}/     # copy-paste training templates
notebooks/                        # Download + Dataloader examples
src/training/                     # legacy backcompat shim (train.py example, deprecated)
```

## Adding a dataset

1. Add a `DatasetInfo` to `src/anatomy_datasets/registry.py`.
2. Write `src/prepare_data/prepare_<name>/{download_<name>.py, prepare_<name>_splits.py, get_<name>_full.sh}`. Splits script must call `add_metadata_to_splits_json(...)` at the end.
3. Add an entry to `src/anatomy_datasets/_discovery.py` (alias → class + env var + default JSON name).
4. Add a thin alias module under `src/anatomy_datasets/datasets/<name>.py`.
5. Verify by loading via the alias and overlaying masks (see `notebooks/Dataloader_example.ipynb`).

## Caveats

- mmseg has no native multilabel. Use per-channel export (`--mode per_channel`) or `AnatomyMultilabelDataset` with `target_channel`.
- Sharded format currently only writes from `.npy` multilabel sources; polygon/COCO passthrough is TODO.
- COCO exporter is multilabel-only; for multiclass PNG datasets use the mmseg exporter.
- HF exporter has no CLI yet (Python function only).
- Some registry license fields are `"unknown"` — check the source URL before commercial use.

## License

This project: MIT. Each upstream dataset retains its own license — see the table above and the `license` / `license_url` fields in each splits JSON.

## Citation

If you use any dataset prepared by this repo, **cite the upstream paper**. BibTeX is embedded in every splits JSON under the `bibtex` key, accessible via `anatomy_datasets.get_dataset_info(name).bibtex`.
