"""
Set up the LES-AV retinal vessel + artery/vein dataset.

The dataset is hosted on Figshare at
https://figshare.com/articles/dataset/LES-AV_dataset/11857698 . The
single file is a ~46 MB zip at
https://ndownloader.figshare.com/files/21732282 . Figshare metadata
lists the licence as GPL but the shipped README.txt restricts use to
non-commercial scientific work; downstream code records the dataset as
``custom-academic``.

Behaviour:
1. Skip if ``LESAV_ROOT_FOLDER`` already contains images/ +
   vessel-segmentations/ + arteries-and-veins/.
2. Use ``LESAV_ZIP_PATH`` (default ``~/Downloads/lesav.zip``) if it
   exists.
3. Otherwise fetch from Figshare and extract.

The Figshare archive nests everything under a top-level ``LES-AV/``
folder; we flatten it so the layout under ``$LESAV_ROOT_FOLDER`` lands
directly as:

    $LESAV_ROOT_FOLDER/
        images/<id>.png
        vessel-segmentations/<id>.png
        arteries-and-veins/<id>.png
        arteries/<id>.png
        veins/<id>.png
        masks/<id>_mask.gif
        README.txt
"""

import os
import shutil
import sys
import urllib.request
import zipfile


LESAV_URL = "https://ndownloader.figshare.com/files/21732282"


def has_required_layout(root_folder: str) -> bool:
    for sub in ("images", "vessel-segmentations", "arteries-and-veins"):
        d = os.path.join(root_folder, sub)
        if not os.path.isdir(d):
            return False
        if not any(f.endswith(".png") for f in os.listdir(d)):
            return False
    return True


def download_zip(url: str, dest: str) -> None:
    print(f"Downloading {url} -> {dest} (~46 MB)")
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
    """Extract the figshare zip, stripping the top-level ``LES-AV/`` prefix."""
    print(f"Extracting {zip_path} into {root_folder}")
    os.makedirs(root_folder, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.namelist():
            if member.startswith("__MACOSX/") or member.endswith(".DS_Store"):
                continue
            if member.endswith("/"):
                continue
            rel = member
            if rel.startswith("LES-AV/"):
                rel = rel[len("LES-AV/"):]
            if not rel:
                continue
            out_path = os.path.join(root_folder, rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with zip_ref.open(member) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)


def main() -> None:
    root_folder = os.getenv("LESAV_ROOT_FOLDER")
    if root_folder is None:
        raise EnvironmentError("LESAV_ROOT_FOLDER environment variable is not set.")

    if has_required_layout(root_folder):
        print(f"LES-AV already extracted under {root_folder}; skipping download.")
        return

    zip_path = os.getenv(
        "LESAV_ZIP_PATH", os.path.expanduser("~/Downloads/lesav.zip")
    )

    if not os.path.isfile(zip_path):
        try:
            download_zip(LESAV_URL, zip_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch LES-AV from {LESAV_URL}: {exc}.\n"
                f"You may download it manually and place it at {zip_path}, "
                f"then re-run this script."
            )

    extract_zip(zip_path, root_folder)

    if not has_required_layout(root_folder):
        raise RuntimeError(
            f"Extraction completed but expected layout missing under "
            f"{root_folder} (need images/ + vessel-segmentations/ + "
            f"arteries-and-veins/). Check the archive."
        )

    print(f"LES-AV ready under {root_folder}")


if __name__ == "__main__":
    main()
