"""Example: SegFormer (HuggingFace ``transformers``) on DRIVE.

Uses ``to_hf_dataset`` to materialize the splits JSON as a HuggingFace
``DatasetDict``, then runs through ``AutoImageProcessor`` and
``Trainer``. Single short step -- example, not a tuned baseline.

Run::

    export DRIVE_ROOT_FOLDER=./datasets/drive
    pip install -e '.[hf]'
    pip install transformers accelerate
    python configs/hf/drive_segformer.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from anatomy_datasets.exporters import to_hf_dataset


def main() -> None:
    from transformers import (
        AutoImageProcessor,
        SegformerForSemanticSegmentation,
        Trainer,
        TrainingArguments,
    )

    root = os.environ["DRIVE_ROOT_FOLDER"]
    splits_json = os.path.join(root, "drive_splits.json")

    dsdict = to_hf_dataset(splits_json, root)

    label_dict = json.load(open(splits_json))["label_dict"]
    id2label = {int(k): v for k, v in label_dict.items()}
    label2id = {v: int(k) for k, v in label_dict.items()}

    ckpt = "nvidia/mit-b0"
    processor = AutoImageProcessor.from_pretrained(ckpt, do_reduce_labels=False)

    def transform(batch):
        images = batch["image"]
        # Mask comes back as raw bytes -- reshape per-row.
        masks = []
        for mask_bytes, img in zip(batch["mask"], images):
            if not mask_bytes:
                masks.append(None)
                continue
            arr = np.frombuffer(mask_bytes, dtype=np.int32).reshape(img.height, img.width)
            masks.append(arr.astype(np.int64))

        inputs = processor(images=images, segmentation_maps=masks, return_tensors="pt")
        return inputs

    dsdict.set_transform(transform)

    model = SegformerForSemanticSegmentation.from_pretrained(
        ckpt,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    args = TrainingArguments(
        output_dir="./runs/segformer_drive",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        max_steps=1,                           # example: one step
        logging_steps=1,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dsdict["train"],
        eval_dataset=dsdict.get("val"),
    )
    trainer.train()


if __name__ == "__main__":
    main()
