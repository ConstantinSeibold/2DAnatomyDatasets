"""HuggingFace ``datasets``-format exporter.

``to_hf_dataset(splits_json, root_dir)`` returns a ``DatasetDict`` with
``train`` / ``val`` / ``test`` splits. Each row::

    {
        "image_id":    str,
        "file_name":   str,        # relative to root_dir
        "image":       PIL.Image,
        "mask":        PIL.Image or np.ndarray,   # see below
        "label_names": list[str],
    }

Mask representation
-------------------
- Multiclass PNG-source datasets: ``mask`` is a 2D ``np.ndarray`` of
  ``int32`` class IDs (Arrow handles variable-shape arrays as bytes).
- Multilabel ``.npy``-source datasets: ``mask`` is a 3D ``np.ndarray``
  of shape ``(C, H, W)``, ``np.uint8``.
- Missing-GT entries (e.g. DRIVE test): ``mask`` is ``None``.

This module imports ``datasets`` lazily so it does not pollute the core
``anatomy_datasets`` import for users on the ``[smp]`` / ``[mmseg]``
extras.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image

from anatomy_datasets.registry import get_dataset_info


def _load_target(target_path: Optional[str], root_dir: str):
    if target_path is None:
        return None
    full = os.path.join(root_dir, target_path)
    if full.endswith(".npy"):
        arr = np.load(full)
        return arr.astype(np.uint8)
    return np.array(Image.open(full).convert("L"), dtype=np.int32)


def _row_generator(
    entries: List[dict],
    root_dir: str,
    label_names: List[str],
):
    for entry in entries:
        image_id = os.path.splitext(os.path.basename(entry["image"]))[0]
        image_path = os.path.join(root_dir, entry["image"])
        image = Image.open(image_path).convert("RGB")
        mask = _load_target(entry.get("target"), root_dir)
        yield {
            "image_id": image_id,
            "file_name": entry["image"],
            "image": image,
            "mask": mask,
            "label_names": label_names,
        }


def to_hf_dataset(
    splits_json: str,
    root_dir: str,
    splits: Iterable[str] = ("train", "val", "test"),
    dataset_name: Optional[str] = None,
    push_to_hub: Optional[str] = None,
    private: bool = True,
):
    """Build a ``datasets.DatasetDict`` from a splits JSON.

    Parameters
    ----------
    push_to_hub : if not ``None``, the resulting DatasetDict is pushed
        to ``<hub_repo_id>`` after construction. Defaults to ``None``
        (no network calls).
    private : passed through to ``push_to_hub`` if used.
    """
    try:
        from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value, Sequence
    except ImportError as e:
        raise ImportError(
            "to_hf_dataset requires the optional 'hf' extra. "
            "Install with: pip install anatomy-datasets[hf]"
        ) from e

    with open(splits_json) as f:
        source = json.load(f)

    dataset_name = dataset_name or source.get(
        "name", os.path.splitext(os.path.basename(splits_json))[0]
    )
    label_dict = source.get("label_dict", {})
    label_names = [
        label_dict[k] for k in sorted(label_dict.keys(), key=lambda x: int(x))
    ]

    # Features schema: leave mask untyped so int/uint8/ndarray all work.
    features = Features(
        {
            "image_id": Value("string"),
            "file_name": Value("string"),
            "image": HFImage(),
            "mask": Value("binary"),  # serialize as bytes; consumer reshapes
            "label_names": Sequence(Value("string")),
        }
    )

    def _wrap(entries: List[dict]):
        # Convert mask -> bytes for Arrow safety.
        def gen():
            for row in _row_generator(entries, root_dir, label_names):
                mask = row["mask"]
                if mask is None:
                    mask_bytes = b""
                else:
                    mask_bytes = mask.tobytes()
                yield {
                    "image_id": row["image_id"],
                    "file_name": row["file_name"],
                    "image": row["image"],
                    "mask": mask_bytes,
                    "label_names": row["label_names"],
                }
        return Dataset.from_generator(gen, features=features)

    dsdict = DatasetDict()
    for split in splits:
        entries = source.get(split, [])
        if not entries:
            continue
        dsdict[split] = _wrap(entries)

    # Attach dataset-level metadata as DatasetInfo description fields where possible.
    try:
        reg = get_dataset_info(dataset_name)
        description = (
            f"{dataset_name} ({reg.modality}) - {reg.citation}"
        )
        for split in dsdict:
            dsdict[split].info.description = description
            dsdict[split].info.license = reg.license
            dsdict[split].info.homepage = reg.source_url
            dsdict[split].info.citation = reg.citation
    except KeyError:
        pass

    if push_to_hub is not None:
        dsdict.push_to_hub(push_to_hub, private=private)

    return dsdict


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", required=True, help="Source splits JSON.")
    parser.add_argument(
        "--root", required=True,
        help="Root dir the splits-JSON image / target paths are relative to.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Directory to save_to_disk(). Required unless --push-to-hub is set.",
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--push-to-hub", default=None,
        help="HF Hub repo id (e.g. 'user/anatomy-drive'). Triggers push after build.",
    )
    parser.add_argument(
        "--public", action="store_true",
        help="If pushing, make the repo public (default: private).",
    )
    parser.add_argument(
        "--splits-only", nargs="+", default=None,
        help="Restrict to specific split names (default: train val test).",
    )
    args = parser.parse_args()

    if not args.out and not args.push_to_hub:
        parser.error("must provide --out or --push-to-hub")

    dsdict = to_hf_dataset(
        splits_json=args.splits,
        root_dir=args.root,
        splits=tuple(args.splits_only) if args.splits_only else ("train", "val", "test"),
        dataset_name=args.dataset,
        push_to_hub=args.push_to_hub,
        private=not args.public,
    )
    if args.out:
        dsdict.save_to_disk(args.out)
        print(f"Saved DatasetDict to {args.out}")


if __name__ == "__main__":
    _cli()
