# Example mmsegmentation config: UNet on DRIVE via the runtime adapter.
#
# This is a starter template -- not authoritative. Copy into your own
# mmseg project and tune. Assumes:
#   - DRIVE prepared on disk (`sh src/prepare_data/prepare_drive/get_drive_full.sh`)
#   - mmseg and anatomy_datasets installed: `pip install -e .[mmseg]`
#
# Train:
#   mim train mmsegmentation configs/mmseg/drive_unet.py --gpus 0
# or:
#   python tools/train.py configs/mmseg/drive_unet.py

# Import for side-effect: registers AnatomyMulticlassDataset to mmseg DATASETS.
custom_imports = dict(
    imports=["anatomy_datasets.adapters.mmseg"],
    allow_failed_imports=False,
)

data_root = "./datasets/drive"
splits_json = f"{data_root}/drive_splits.json"

crop_size = (512, 512)

# Per-dataset normalization is recorded in the splits JSON; defaulting to
# ImageNet stats here for simplicity. Replace with the recorded values for
# a tighter input distribution.
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RandomResize", scale=crop_size, ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
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
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="AnatomyMulticlassDataset",
        splits_json=splits_json,
        data_root=data_root,
        split="train",
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="AnatomyMulticlassDataset",
        splits_json=splits_json,
        data_root=data_root,
        split="val",
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU", "mDice"])
test_evaluator = val_evaluator

# Model -- bring your own. UNet here is a placeholder; swap for whatever
# baseline you want.
model = dict(
    type="EncoderDecoder",
    data_preprocessor=dict(
        type="SegDataPreProcessor",
        mean=img_norm_cfg["mean"],
        std=img_norm_cfg["std"],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size,
    ),
    backbone=dict(
        type="UNet",
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        act_cfg=dict(type="ReLU"),
        upsample_cfg=dict(type="InterpConv"),
        norm_eval=False,
    ),
    decode_head=dict(
        type="FCNHead",
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,            # background + vessel
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

train_cfg = dict(type="IterBasedTrainLoop", max_iters=20000, val_interval=2000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=1e-4),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(type="PolyLR", power=1.0, begin=500, end=20000, eta_min=0, by_epoch=False),
]

default_scope = "mmseg"
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=2000, save_best="mIoU"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
log_processor = dict(by_epoch=False)
log_level = "INFO"
randomness = dict(seed=42)
