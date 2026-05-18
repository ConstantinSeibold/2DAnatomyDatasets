# Example mmdetection config: Mask R-CNN on PAXRay via the sharded reader.
#
# Single-file COCO does not scale to PAXRay++ (or even full PAXRay with
# 166 classes). This config uses the sharded export and the runtime
# adapter `ShardedCocoMMDetDataset` so mmdet streams per-image annotation
# shards instead of loading a multi-GB JSON.
#
# Prep:
#   sh src/prepare_data/prepare_paxray/get_paxray_full.sh
#   python -m anatomy_datasets.formats.sharded_coco \
#       --splits ./datasets/paxray/paxray.json \
#       --root ./datasets/paxray \
#       --out ./exports/sharded_paxray \
#       --image-link-mode symlink
#
# Train:
#   mim train mmdetection configs/mmdet/paxray_maskrcnn.py --gpus 0

custom_imports = dict(
    imports=["anatomy_datasets.adapters.mmdet"],
    allow_failed_imports=False,
)

_base_ = [
    "mmdet::_base_/models/mask-rcnn_r50_fpn.py",
    "mmdet::_base_/schedules/schedule_1x.py",
    "mmdet::_base_/default_runtime.py",
]

sharded_root = "./exports/sharded_paxray"

# PAXRay4 = 4 anatomy classes; for full PAXRay166 set num_classes=166.
num_classes = 4
classes = ("lungs", "mediastinum", "bones", "diaphragm")

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes),
        mask_head=dict(num_classes=num_classes),
    ),
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataloader = dict(
    batch_size=2, num_workers=2,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="ShardedCocoMMDetDataset",
        sharded_root=sharded_root,
        split="train",
        metainfo=dict(classes=classes),
        decode_masks=True,
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1, num_workers=2,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ShardedCocoMMDetDataset",
        sharded_root=sharded_root,
        split="val",
        metainfo=dict(classes=classes),
        decode_masks=True,
        pipeline=test_pipeline,
        test_mode=True,
    ),
)
test_dataloader = val_dataloader

# CocoMetric needs a single COCO ann file for ground-truth lookup -- for
# validation, run the COCO exporter once to produce a small val.json:
#   python -m anatomy_datasets.exporters.coco \
#       --splits ./datasets/paxray/paxray.json --root ./datasets/paxray \
#       --out ./exports/coco_paxray --splits-only val \
#       --shard-threshold 10000000
val_evaluator = dict(
    type="CocoMetric",
    ann_file="./exports/coco_paxray/annotations/val.json",
    metric=["bbox", "segm"],
)
test_evaluator = val_evaluator

train_cfg = dict(max_epochs=24)
randomness = dict(seed=42)
