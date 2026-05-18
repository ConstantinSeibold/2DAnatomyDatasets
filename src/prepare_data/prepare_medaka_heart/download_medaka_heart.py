"""
Download and extract the Medaka Hatchling Heart dataset from OSF.

The dataset (Gierten et al., PLOS ONE 2022 — https://doi.org/10.1371/journal.pone.0263656)
is publicly hosted on OSF project ``uyk79`` (https://osf.io/uyk79/). The
``osfstorage`` provider exposes a zip-all endpoint that requires no
authentication, so we fetch the whole project archive and unpack the nested
``data.zip`` it contains.

Layout after extraction (under ``MEDAKA_HEART_ROOT_FOLDER``):

    train_images/
        ventral_samples/                565 RGB .tif
        ventral_samples_gray/           565 L   .tif
        ventral_mask_atrium/            565 RGB .png   (values {0, 20})
        ventral_mask_bulbus/            565 RGB .png   (values {0, 19})
        ventral_mask_heart/             565 RGB .png   (values {0, 33})
        ventral_mask_combined/          565 RGB .png   (values {0, 19, 20, 33})
    test_images/
        ventral_samples_gray/            75 L   .tif   (specimen N0030_*)
        ventral_mask_{atrium,bulbus,heart,combined}_gray/   75 each
        ventral_samples_R0004/          165 RGB .tif   (specimen color_frame_*)
        ventral_mask_{atrium,bulbus,heart,combined}_R0004/  165 each
    readme.txt

Override the download URL via ``MEDAKA_HEART_OSF_URL`` if needed, or point
``MEDAKA_HEART_ZIP_PATH`` at a pre-downloaded ``osfstorage.zip`` to skip the
network fetch.
"""

import os
import sys
import urllib.request
import zipfile


OSF_NODE = "uyk79"
DEFAULT_URL = (
    f"https://files.osf.io/v1/resources/{OSF_NODE}/providers/osfstorage/?zip="
)

REQUIRED_SUBDIRS = [
    os.path.join("train_images", "ventral_samples"),
    os.path.join("train_images", "ventral_mask_combined"),
    os.path.join("test_images", "ventral_samples_gray"),
    os.path.join("test_images", "ventral_samples_R0004"),
]


def has_required_layout(root_folder: str) -> bool:
    return all(
        os.path.isdir(os.path.join(root_folder, sub)) for sub in REQUIRED_SUBDIRS
    )


_last_pct = [-1]


def _report(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = int(min(100.0, downloaded * 100.0 / total_size))
        if pct == _last_pct[0]:
            return
        _last_pct[0] = pct
        sys.stdout.write(f"\r  {pct:3d}%  ({downloaded/1e6:7.1f} / {total_size/1e6:.1f} MB)")
    else:
        sys.stdout.write(f"\r  {downloaded/1e6:7.1f} MB")
    sys.stdout.flush()


def download_osf_archive(url: str, dst: str) -> None:
    print(f"Downloading OSF project {OSF_NODE} from {url}")
    urllib.request.urlretrieve(url, dst, reporthook=_report)
    sys.stdout.write("\n")


def extract_outer(zip_path: str, work_dir: str) -> str:
    os.makedirs(work_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as outer:
        names = outer.namelist()
        if "data.zip" not in names:
            raise RuntimeError(
                f"{zip_path} is missing the expected 'data.zip' member "
                f"(got {names[:5]}...)."
            )
        outer.extractall(work_dir)
    return os.path.join(work_dir, "data.zip")


def extract_inner(data_zip_path: str, root_folder: str) -> None:
    with zipfile.ZipFile(data_zip_path, "r") as inner:
        inner.extractall(root_folder)


def main() -> None:
    root_folder = os.getenv("MEDAKA_HEART_ROOT_FOLDER")
    if root_folder is None:
        raise EnvironmentError(
            "MEDAKA_HEART_ROOT_FOLDER environment variable is not set."
        )
    os.makedirs(root_folder, exist_ok=True)

    if has_required_layout(root_folder):
        print(f"Medaka heart dataset already present at {root_folder}.")
        return

    zip_path = os.getenv(
        "MEDAKA_HEART_ZIP_PATH", os.path.join(root_folder, "osfstorage.zip")
    )
    if not os.path.isfile(zip_path):
        url = os.getenv("MEDAKA_HEART_OSF_URL", DEFAULT_URL)
        download_osf_archive(url, zip_path)

    print("Extracting outer archive")
    data_zip_path = extract_outer(zip_path, root_folder)

    print("Extracting inner data.zip")
    extract_inner(data_zip_path, root_folder)
    os.remove(data_zip_path)

    if not has_required_layout(root_folder):
        sys.exit(
            f"Extraction finished but expected folders are missing under {root_folder}."
        )

    # Remove outer archive once we've confirmed extraction.
    if os.path.isfile(zip_path) and zip_path.startswith(root_folder):
        os.remove(zip_path)
    print("Done.")


if __name__ == "__main__":
    main()
