"""
Extract annotated frames from the Jump-Broadcast source videos and rewrite
the shipped segmentation masks into single-channel PNGs with body-part class
ids in {0, ..., 14}.

Masks shipped with Jump-Broadcast are crops produced by DensePose; we crop
the matching video frame to the same box (from ``bboxes_segmentations.csv``)
so the image and mask align pixel-for-pixel.

The shipped masks use values scaled by 10 (so they're visible when opened in
an image viewer). We divide by 10 to recover the actual class ids.

Outputs under ``$JUMP_BROADCAST_ROOT_FOLDER``:

    images/<video_id>_<frame_number>.png  # cropped to DensePose bbox
    masks/<video_id>_<frame_number>.png   # class ids in [0, 14]
"""

import csv
import os
import re

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


MASK_FILENAME_RE = re.compile(r"^(\d+)_\((\d+)\)\.png$")


def load_bboxes(csv_path: str) -> dict:
    """Return ``{image_id: (min_x, min_y, width, height)}``."""
    bboxes = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            img_id = row["image_id"].strip()
            bboxes[img_id] = (
                float(row["min_x"]),
                float(row["min_y"]),
                float(row["width"]),
                float(row["height"]),
            )
    return bboxes


def scale_bbox(bbox, sx: float, sy: float):
    x, y, w, h = bbox
    return (x * sx, y * sy, w * sx, h * sy)


# DensePose was run on the source broadcasts at 1920x1080, so the shipped
# bbox coordinates use that frame. If a downloaded video is sub-HD, the bbox
# must be scaled by the ratio; for native-HD videos the bbox applies as-is.
SOURCE_W, SOURCE_H = 1920, 1080


def extract_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def crop_to_bbox(frame: np.ndarray, bbox) -> np.ndarray:
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(w, int(round(x + bw)))
    y1 = min(h, int(round(y + bh)))
    return frame[y0:y1, x0:x1]


def main() -> None:
    root = os.getenv("JUMP_BROADCAST_ROOT_FOLDER")
    if root is None:
        raise EnvironmentError(
            "JUMP_BROADCAST_ROOT_FOLDER environment variable is not set."
        )
    ann_dir = os.path.join(root, "annotations")
    videos_dir = os.path.join(root, "videos")
    seg_dir = os.path.join(ann_dir, "segmentations")
    images_dir = os.path.join(root, "images")
    masks_dir = os.path.join(root, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    bboxes = load_bboxes(os.path.join(ann_dir, "bboxes_segmentations.csv"))

    # Group mask files by video id so we open each video only once.
    by_video = {}
    for fname in sorted(os.listdir(seg_dir)):
        m = MASK_FILENAME_RE.match(fname)
        if m is None:
            continue
        vid, frame = m.group(1), m.group(2)
        by_video.setdefault(vid, []).append((frame, fname))

    n_ok = n_missing_video = n_missing_frame = n_missing_bbox = n_empty = 0
    for vid, items in tqdm(sorted(by_video.items()), desc="videos"):
        video_path = os.path.join(videos_dir, f"{vid}.mp4")
        if not os.path.isfile(video_path):
            n_missing_video += len(items)
            continue
        cap = cv2.VideoCapture(video_path)
        try:
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Scale bboxes to the actual frame resolution. On native-HD videos
            # both factors are 1.0 and the bbox is used as-is.
            sx = frame_w / SOURCE_W
            sy = frame_h / SOURCE_H

            for frame_str, fname in items:
                stem = f"{vid}_{frame_str}"
                img_out = os.path.join(images_dir, stem + ".png")
                mask_out = os.path.join(masks_dir, stem + ".png")
                if os.path.isfile(img_out) and os.path.isfile(mask_out):
                    n_ok += 1
                    continue

                bbox_key = f"{vid}_({frame_str})"
                if bbox_key not in bboxes:
                    n_missing_bbox += 1
                    continue

                frame = extract_frame(cap, int(frame_str))
                if frame is None:
                    n_missing_frame += 1
                    continue

                bbox = scale_bbox(bboxes[bbox_key], sx, sy)
                crop = crop_to_bbox(frame, bbox)
                if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                    n_empty += 1
                    continue

                mask = np.array(Image.open(os.path.join(seg_dir, fname)))
                mask = (mask // 10).astype(np.uint8)

                if crop.shape[:2] != mask.shape[:2]:
                    crop = cv2.resize(
                        crop,
                        (mask.shape[1], mask.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )

                Image.fromarray(crop).save(img_out)
                Image.fromarray(mask).save(mask_out)
                n_ok += 1
        finally:
            cap.release()

    print(
        f"Wrote {n_ok} pairs. Skipped: {n_missing_video} (video missing), "
        f"{n_missing_frame} (frame unreadable), {n_missing_bbox} (no bbox), "
        f"{n_empty} (degenerate crop)."
    )


if __name__ == "__main__":
    main()
