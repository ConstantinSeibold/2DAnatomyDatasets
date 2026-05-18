"""
Set up the HRF (High-Resolution Fundus) retinal vessel dataset.

The dataset is hosted by FAU Erlangen at
https://www5.cs.fau.de/research/data/fundus-images/ . The combined
``all.zip`` (~73 MB) is publicly downloadable without auth and contains
images, vessel annotations, and FOV masks.

Behaviour:
1. Skip if ``HRF_ROOT_FOLDER`` already contains images/manual1/mask.
2. Use ``HRF_ZIP_PATH`` (default ``~/Downloads/hrf_all.zip``) if it exists.
3. Otherwise fetch from FAU and extract.

Expected layout after extraction:
    $HRF_ROOT_FOLDER/
        images/NN_{h,g,dr}.{JPG,jpg}     # 45 fundus images
        manual1/NN_{h,g,dr}.tif          # binary vessel annotations
        mask/NN_{h,g,dr}_mask.tif        # FOV ROI masks (unused for seg)
"""

import os
import sys
import urllib.request
import zipfile


HRF_URL = "https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip"


def has_required_layout(root_folder: str) -> bool:
    images_dir = os.path.join(root_folder, "images")
    manual_dir = os.path.join(root_folder, "manual1")
    if not (os.path.isdir(images_dir) and os.path.isdir(manual_dir)):
        return False
    return any(
        f.lower().endswith(".jpg") for f in os.listdir(images_dir)
    ) and any(f.endswith(".tif") for f in os.listdir(manual_dir))


def download_zip(url: str, dest: str) -> None:
    print(f"Downloading {url} -> {dest} (~73 MB)")
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    def _report(block_num, block_size, total_size):
        done = block_num * block_size
        if total_size > 0:
            pct = min(100, 100 * done / total_size)
            sys.stdout.write(
                f"\r  {done/1e6:.1f} MB / {total_size/1e6:.1f} MB ({pct:.1f}%)"
            )
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_report)
    sys.stdout.write("\n")


def extract_zip(zip_path: str, root_folder: str) -> None:
    print(f"Extracting {zip_path} into {root_folder}")
    os.makedirs(root_folder, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.namelist():
            if member.startswith("__MACOSX/") or member.endswith(".DS_Store"):
                continue
            zip_ref.extract(member, root_folder)


def main() -> None:
    root_folder = os.getenv("HRF_ROOT_FOLDER")
    if root_folder is None:
        raise EnvironmentError("HRF_ROOT_FOLDER environment variable is not set.")

    if has_required_layout(root_folder):
        print(f"HRF already extracted under {root_folder}; skipping download.")
        return

    zip_path = os.getenv(
        "HRF_ZIP_PATH", os.path.expanduser("~/Downloads/hrf_all.zip")
    )

    if not os.path.isfile(zip_path):
        try:
            download_zip(HRF_URL, zip_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch HRF from {HRF_URL}: {exc}.\n"
                f"You may download it manually and place it at {zip_path}, "
                f"then re-run this script."
            )

    extract_zip(zip_path, root_folder)

    if not has_required_layout(root_folder):
        raise RuntimeError(
            f"Extraction completed but expected layout missing under "
            f"{root_folder} (need images/ + manual1/). Check the archive."
        )

    print(f"HRF ready under {root_folder}")


if __name__ == "__main__":
    main()
