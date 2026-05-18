"""In-place augmentation of an already-written splits JSON with metadata.

Designed to be called at the end of any ``prepare_<name>_splits.py`` after
the JSON file has been written. Idempotent: re-running on the same JSON
just refreshes the metadata fields.

Also exposes a CLI for datasets whose splits JSON ships from upstream
(PAXRay) or whose prepare pipeline doesn't yet write its own splits
(BS80k):

    python -m anatomy_datasets.postprocess \\
        --json datasets/paxray/paxray.json \\
        --root datasets/paxray \\
        --dataset PAXRay --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from typing import Optional

from .registry import get_dataset_info
from .stats import compute_image_stats


def _is_coco_format(data: dict) -> bool:
    return all(k in data for k in ("images", "annotations", "categories"))


def add_metadata_to_splits_json(
    json_path: str,
    root_dir: str,
    dataset_name: str,
    seed: Optional[int] = None,
    stats_split: str = "train",
    stats_sample_cap: int = 200,
    compute_stats: bool = True,
    coco_image_dir: Optional[str] = None,
) -> dict:
    """Read JSON at ``json_path``, add metadata fields, write back.

    Auto-detects format:

    **Splits format** (``{name, label_dict, train, val, test}``) - fields
    added at top level:

    - ``version``       : ISO date of this run.
    - ``seed``          : the int passed in (or null).
    - ``modality``      : pulled from the registry.
    - ``license``       : pulled from the registry.
    - ``source_url``    : pulled from the registry.
    - ``citation``      : pulled from the registry.
    - ``normalization`` : channel mean/std over up to ``stats_sample_cap``
      images from ``stats_split`` (skipped when ``compute_stats=False``
      or the split has no entries).

    **COCO format** (``{images, annotations, categories}``) - the same
    fields are written into ``data["info"]`` (creating it if absent), and
    normalization stats are computed from ``data["images"][*]["file_name"]``
    resolved against ``coco_image_dir`` (falls back to ``root_dir``).

    Returns the modified dict (also persisted to ``json_path``).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    info = get_dataset_info(dataset_name)
    today = date.today().isoformat()

    metadata = {
        "version": today,
        "seed": seed,
        "modality": info.modality,
        "license": info.license,
        "license_url": info.license_url,
        "source_url": info.source_url,
        "paper_url": info.paper_url,
        "citation": info.citation,
        "bibtex": info.bibtex,
    }

    if _is_coco_format(data):
        info_block = data.setdefault("info", {})
        info_block.update(metadata)

        if compute_stats:
            image_root = coco_image_dir or root_dir
            image_paths = [
                os.path.join(image_root, img["file_name"])
                for img in data.get("images", [])
                if "file_name" in img
            ]
            if image_paths:
                info_block["normalization"] = compute_image_stats(
                    image_paths, sample_cap=stats_sample_cap
                )
    else:
        data.update(metadata)

        if compute_stats:
            stats_entries = data.get(stats_split, [])
            image_paths = [
                os.path.join(root_dir, e["image"])
                for e in stats_entries
                if "image" in e
            ]
            if image_paths:
                data["normalization"] = compute_image_stats(
                    image_paths, sample_cap=stats_sample_cap
                )

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    _print_citation_banner(info)

    return data


def _print_citation_banner(info) -> None:
    """Print a citation-required notice for a freshly prepared dataset.

    Surfaced at the tail of every ``prepare_<name>_splits.py`` run so users
    cannot miss attribution requirements. Full BibTeX entries are mirrored
    in the registry under ``DatasetInfo.bibtex``.
    """
    bar = "=" * 72
    lines = [
        "",
        bar,
        f"Dataset prepared: {info.name}",
        f"License: {info.license}"
        + (f"  ({info.license_url})" if info.license_url else ""),
        f"Source:  {info.source_url}",
    ]
    if info.paper_url:
        lines.append(f"Paper:   {info.paper_url}")
    lines.append("")
    lines.append("If you use this dataset, please cite:")
    lines.append(f"  {info.citation}")
    lines.append("")
    lines.append("BibTeX:")
    for raw in info.bibtex.splitlines():
        lines.append(f"  {raw}")
    lines.append(bar)
    print("\n".join(lines))


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Add metadata fields to a splits JSON in place."
    )
    parser.add_argument("--json", required=True, help="Path to the splits JSON.")
    parser.add_argument(
        "--root",
        required=True,
        help="Dataset root directory (image paths in JSON are relative to this).",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (key into anatomy_datasets.registry.DATASET_REGISTRY).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stats-split", default="train")
    parser.add_argument("--stats-sample-cap", type=int, default=200)
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip channel-stats computation (e.g. for detection-only datasets).",
    )
    parser.add_argument(
        "--coco-image-dir",
        default=None,
        help="For COCO-format JSONs only: dir containing the image files. "
        "Defaults to --root.",
    )
    args = parser.parse_args()

    add_metadata_to_splits_json(
        json_path=args.json,
        root_dir=args.root,
        dataset_name=args.dataset,
        seed=args.seed,
        stats_split=args.stats_split,
        stats_sample_cap=args.stats_sample_cap,
        compute_stats=not args.no_stats,
        coco_image_dir=args.coco_image_dir,
    )


if __name__ == "__main__":
    _cli()
