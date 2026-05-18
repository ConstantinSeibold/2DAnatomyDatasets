"""
Build the Jump-Broadcast splits JSON.

Upstream ships athlete-disjoint train / val keypoint annotations (test
annotations are gated behind an e-mail request to the chair). We pool the
released annotations and re-stratify athletes 70/15/15 into train/val/test
with a fixed random seed so the public release ships with a usable test
split. Each athlete appears in exactly one split.

Only segmentation masks for which (a) the source frame was successfully
extracted and (b) the (event, frame_num) row exists in the keypoint CSVs
are emitted in the splits file.
"""

import csv
import json
import os
import random
import re
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
from anatomy_datasets import add_metadata_to_splits_json


SPLIT_FRACTIONS = (0.70, 0.15, 0.15)  # train, val, test
RANDOM_SEED = 42

# Class ids per the upstream Readme.md (values appear as id*10 in the
# shipped PNGs and are divided by 10 during prepare_labels).
LABEL_DICT = {
    "0": "background",
    "1": "torso",
    "2": "head",
    "3": "r_hand",
    "4": "l_hand",
    "5": "l_foot",
    "6": "r_foot",
    "7": "r_thigh",
    "8": "l_thigh",
    "9": "r_lleg",
    "10": "l_lleg",
    "11": "l_uarm",
    "12": "r_uarm",
    "13": "l_farm",
    "14": "r_farm",
}


def load_athletes(csv_path: str) -> dict:
    """Return ``{(event, frame_num): athlete}`` from a keypoint CSV."""
    out = {}
    with open(csv_path, "r") as f:
        # First line is "#<count>"; DictReader picks up the real header next.
        first = f.readline()
        if not first.startswith("#"):
            f.seek(0)
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            out[(row["event"], str(int(row["frame_num"])))] = row["athlete"]
    return out


def stratified_split_by_athlete(athletes: list) -> dict:
    rng = random.Random(RANDOM_SEED)
    shuffled = sorted(set(athletes))
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * SPLIT_FRACTIONS[0]))
    n_val = int(round(n * SPLIT_FRACTIONS[1]))
    return {
        "train": set(shuffled[:n_train]),
        "val": set(shuffled[n_train : n_train + n_val]),
        "test": set(shuffled[n_train + n_val :]),
    }


def main() -> None:
    root = os.getenv("JUMP_BROADCAST_ROOT_FOLDER")
    if root is None:
        raise EnvironmentError(
            "JUMP_BROADCAST_ROOT_FOLDER environment variable is not set."
        )
    ann_dir = os.path.join(root, "annotations")
    images_dir = os.path.join(root, "images")
    masks_dir = os.path.join(root, "masks")

    athlete_lookup = {}
    for split_csv in ("train.csv", "val.csv"):
        path = os.path.join(ann_dir, "keypoints", split_csv)
        if os.path.isfile(path):
            athlete_lookup.update(load_athletes(path))

    fname_re = re.compile(r"^(\d+)_(\d+)\.png$")
    entries = []
    for fname in sorted(os.listdir(images_dir)):
        m = fname_re.match(fname)
        if m is None:
            continue
        vid, frame = m.group(1), str(int(m.group(2)))
        athlete = athlete_lookup.get((vid, frame))
        if athlete is None:
            # Mask exists but no keypoint row -> no athlete known, skip.
            continue
        mask_path = os.path.join(masks_dir, fname)
        if not os.path.isfile(mask_path):
            continue
        entries.append(
            {
                "image": os.path.relpath(
                    os.path.join(images_dir, fname), root
                ),
                "target": os.path.relpath(mask_path, root),
                "_athlete": athlete,
            }
        )

    if not entries:
        raise RuntimeError(
            "No annotated frames found. Run download + prepare_labels first."
        )

    athlete_split = stratified_split_by_athlete([e["_athlete"] for e in entries])
    buckets = {"train": [], "val": [], "test": []}
    for entry in entries:
        athlete = entry.pop("_athlete")
        for split, members in athlete_split.items():
            if athlete in members:
                buckets[split].append(entry)
                break

    splits = {
        "name": "JumpBroadcast",
        "label_dict": LABEL_DICT,
        "train": buckets["train"],
        "val": buckets["val"],
        "test": buckets["test"],
    }

    out_path = os.path.join(root, "jump_broadcast_splits.json")
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=4)
    print(
        f"Saved {out_path} (train={len(buckets['train'])}, "
        f"val={len(buckets['val'])}, test={len(buckets['test'])})."
    )

    add_metadata_to_splits_json(
        json_path=out_path,
        root_dir=root,
        dataset_name="JumpBroadcast",
        seed=RANDOM_SEED,
    )


if __name__ == "__main__":
    main()
