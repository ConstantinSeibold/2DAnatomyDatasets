#!/usr/bin/env bash
# Prepare JSRT.
#
# The Google Drive zip pulled by download_jsrt.py is the fully pre-prepared
# distribution. After extraction the dataset root looks like:
#
#   $JSRT_ROOT_PATH/
#     images/JPCLN*.png + JPCNN*.png
#     masks/JPCLN*.npy  + JPCNN*.npy             (5-channel multilabel stacks)
#     jsrt_splits.json                           (already populated train/val/test)
#     jsons/jsrt_{train,val,test}.json           (pre-built COCO files)
#
# So the only "preparation" we need after download is to add the metadata
# fields (license, citation, normalization, ...) to the shipped splits JSON.
#
# Users who want to regenerate splits from the upstream segmentation02 zip
# (raw JSRT distribution) should instead run prepare_jsrt.py + prepare_jsrt_splits.py
# manually with the env vars those scripts document.

set -e

export JSRT_ROOT_PATH="${JSRT_ROOT_PATH:-./datasets/jsrt}"

python src/prepare_data/prepare_jsrt/download_jsrt.py

# Annotate the shipped splits JSON with metadata + normalization stats.
python -m anatomy_datasets.postprocess \
    --json    "$JSRT_ROOT_PATH/jsrt_splits.json" \
    --root    "$JSRT_ROOT_PATH" \
    --dataset JSRT

# Also annotate the shipped COCO splits with the same metadata block.
for split in train val test; do
    python -m anatomy_datasets.postprocess \
        --json    "$JSRT_ROOT_PATH/jsons/jsrt_${split}.json" \
        --root    "$JSRT_ROOT_PATH" \
        --dataset JSRT \
        --coco-image-dir "$JSRT_ROOT_PATH/images" \
        --no-stats        # stats already on the splits JSON; skip per-COCO recompute
done
