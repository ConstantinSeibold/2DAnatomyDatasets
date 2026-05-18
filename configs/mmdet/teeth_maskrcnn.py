# Example mmdetection config: Mask R-CNN on Teeth.
#
# Teeth is small enough that the monolithic COCO output of either
# `prepare_teeth.py` or `python -m anatomy_datasets.exporters.coco --dataset Teeth`
# is fine -- no sharding needed.
#
# Prep:
#   sh src/prepare_data/prepare_teeth_kaggle/get_teeth_full.sh
#   # or, to regenerate via the exporter:
#   python -m anatomy_datasets.exporters.coco \
#       --splits ./datasets/teeth/train.json \
#       --root ./datasets/teeth --out ./exports/coco_teeth
#
# Train:
#   mim train mmdetection configs/mmdet/teeth_maskrcnn.py --gpus 0

_base_ = [
    "mmdet::_base_/models/mask-rcnn_r50_fpn.py",
    "mmdet::_base_/datasets/coco_instance.py",
    "mmdet::_base_/schedules/schedule_1x.py",
    "mmdet::_base_/default_runtime.py",
]

data_root = "./datasets/teeth/"            # has train.json/val.json/test.json + img/
classes = ("tooth",)
num_classes = 1

# Override the base model's class count.
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes),
        mask_head=dict(num_classes=num_classes),
    ),
)

train_dataloader = dict(
    batch_size=2, num_workers=2,
    dataset=dict(
        type="CocoDataset",
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file="train.json",
        data_prefix=dict(img="img/"),
        filter_cfg=dict(filter_empty_gt=True, min_size=8),
    ),
)
val_dataloader = dict(
    batch_size=1, num_workers=2,
    dataset=dict(
        type="CocoDataset",
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file="val.json",
        data_prefix=dict(img="img/"),
        test_mode=True,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=f"{data_root}val.json",
    metric=["bbox", "segm"],
)
test_evaluator = val_evaluator

train_cfg = dict(max_epochs=24)
randomness = dict(seed=42)
