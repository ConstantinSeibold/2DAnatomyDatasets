import tarfile
import os

if __name__ == "__main__":
    PAXRAYPP_ROOT = os.getenv("PAXRAYPP_ROOT", "paxraypp")

    tar_file_path = os.path.join(PAXRAYPP_ROOT, "paxray_images_unfiltered.tar.gz")
    images_out = os.path.join(PAXRAYPP_ROOT, "paxray_images_unfiltered")

    # Despite the .tar.gz extension the upstream archive is a plain (uncompressed)
    # tar — use "r:*" so tarfile auto-detects whether it is gzipped.
    with tarfile.open(tar_file_path, "r:*") as tar_ref:
        tar_ref.extractall(images_out)

    os.remove(tar_file_path)

    # labels.zip from the upstream bundle holds unfiltered per-image .npy masks;
    # the paxray_train_val.json / paxray_test.json already encode the filtered
    # labels we use, so the zip is intentionally left untouched here.
