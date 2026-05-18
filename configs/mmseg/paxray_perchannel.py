# Example mmsegmentation config: train one binary segmentation per PAXRay
# channel via the per-channel exporter output.
#
# Two ways to use this with mmseg:
#  (A) Pre-export the per-channel data once:
#       python -m anatomy_datasets.exporters.mmseg \
#           --splits ./datasets/paxray/paxray.json \
#           --root ./datasets/paxray \
#           --out ./exports/mmseg_paxray \
#           --mode per_channel
#      Then point ``data_root`` below at one of the per-class subtrees
#      (e.g. ``./exports/mmseg_paxray/lungs``).
#
#  (B) Use the runtime adapter ``AnatomyMultilabelDataset`` (no on-disk
#      duplication) by setting ``custom_imports`` and switching the
#      dataset ``type``. See the commented block at the bottom.
#
# This file shows path (A). For path (B) flip the commented section.

data_root = "./exports/mmseg_paxray/lungs"      # one class at a time
crop_size = (512, 512)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RandomResize", scale=crop_size, ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackSegInputs"),
]
val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=crop_size, keep_ratio=True),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="BaseSegDataset",
        data_root=data_root,
        data_prefix=dict(img_path="img_dir/train", seg_map_path="ann_dir/train"),
        metainfo=dict(classes=("background", "lungs")),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="BaseSegDataset",
        data_root=data_root,
        data_prefix=dict(img_path="img_dir/val", seg_map_path="ann_dir/val"),
        metainfo=dict(classes=("background", "lungs")),
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU", "mDice"])
test_evaluator = val_evaluator

# Plug your own backbone+head. UNet binary head shown below.
model = dict(
    type="EncoderDecoder",
    data_preprocessor=dict(
        type="SegDataPreProcessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size,
    ),
    backbone=dict(type="UNet", in_channels=3, base_channels=64, num_stages=5,
                  strides=(1,1,1,1,1), enc_num_convs=(2,2,2,2,2),
                  dec_num_convs=(2,2,2,2), downsamples=(True,True,True,True),
                  enc_dilations=(1,1,1,1,1), dec_dilations=(1,1,1,1),
                  norm_cfg=dict(type="SyncBN", requires_grad=True),
                  act_cfg=dict(type="ReLU")),
    decode_head=dict(type="FCNHead", in_channels=64, in_index=4, channels=64,
                     num_convs=1, num_classes=2,
                     norm_cfg=dict(type="SyncBN", requires_grad=True),
                     align_corners=False,
                     loss_decode=dict(type="CrossEntropyLoss", loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

train_cfg = dict(type="IterBasedTrainLoop", max_iters=40000, val_interval=4000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=1e-4),
)
param_scheduler = [
    dict(type="PolyLR", power=1.0, begin=0, end=40000, eta_min=0, by_epoch=False),
]
default_scope = "mmseg"
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=4000, save_best="mIoU"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
log_processor = dict(by_epoch=False)
log_level = "INFO"
randomness = dict(seed=42)


# ---------------------------------------------------------------------------
# Alternative (B): runtime adapter, no per-channel export needed.
# Uncomment and remove the path-(A) train_dataloader/val_dataloader above.
# ---------------------------------------------------------------------------
#
# custom_imports = dict(
#     imports=["anatomy_datasets.adapters.mmseg"],
#     allow_failed_imports=False,
# )
# train_dataloader = dict(
#     batch_size=4, num_workers=4,
#     sampler=dict(type="InfiniteSampler", shuffle=True),
#     dataset=dict(
#         type="AnatomyMultilabelDataset",
#         splits_json="./datasets/paxray/paxray.json",
#         data_root="./datasets/paxray",
#         split="train",
#         target_channel=0,                 # 0 = lungs in PAXRay4; consult registry/label_dict
#         cache_decoded_masks=True,         # write binary PNG cache on first read
#         pipeline=train_pipeline,
#     ),
# )
