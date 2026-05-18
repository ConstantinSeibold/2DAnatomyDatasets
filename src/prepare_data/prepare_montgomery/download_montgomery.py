"""
Set up the NLM Montgomery County chest X-ray dataset.

The dataset is hosted publicly by the National Library of Medicine at
https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip
(~616 MB). It ships 138 PA chest radiographs (80 normal, 58 TB) with
expert-drawn bilateral lung masks under ManualMask/leftMask and
ManualMask/rightMask. Download is direct (no auth).

Behaviour:
1. If ``MONTGOMERY_ROOT_FOLDER`` already contains the extracted layout
   (MontgomerySet/CXR_png/*.png) we skip the download.
2. If a local zip exists at ``MONTGOMERY_ZIP_PATH`` (default:
   ``~/Downloads/NLM-MontgomeryCXRSet.zip``) we extract it.
3. Otherwise the zip is fetched from NLM and extracted.

Expected layout after extraction:
    $MONTGOMERY_ROOT_FOLDER/
        MontgomerySet/
            CXR_png/MCUCXR_NNNN_X.png
            ManualMask/leftMask/MCUCXR_NNNN_X.png
            ManualMask/rightMask/MCUCXR_NNNN_X.png
            ClinicalReadings/MCUCXR_NNNN_X.txt
            NLM-MontgomeryCXRSet-ReadMe.pdf
"""

import os
import sys
import urllib.request
import zipfile


MONTGOMERY_URL = (
    "https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip"
)


def has_required_layout(root_folder: str) -> bool:
    cxr_dir = os.path.join(root_folder, "MontgomerySet", "CXR_png")
    if not os.path.isdir(cxr_dir):
        return False
    return any(f.endswith(".png") for f in os.listdir(cxr_dir))


def download_zip(url: str, dest: str) -> None:
    print(f"Downloading {url} -> {dest} (~616 MB)")
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
    root_folder = os.getenv("MONTGOMERY_ROOT_FOLDER")
    if root_folder is None:
        raise EnvironmentError(
            "MONTGOMERY_ROOT_FOLDER environment variable is not set."
        )

    if has_required_layout(root_folder):
        print(f"Montgomery already extracted under {root_folder}; skipping download.")
        return

    zip_path = os.getenv(
        "MONTGOMERY_ZIP_PATH",
        os.path.expanduser("~/Downloads/NLM-MontgomeryCXRSet.zip"),
    )

    if not os.path.isfile(zip_path):
        try:
            download_zip(MONTGOMERY_URL, zip_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch Montgomery from {MONTGOMERY_URL}: {exc}.\n"
                f"You may download it manually and place it at {zip_path}, "
                f"then re-run this script."
            )

    extract_zip(zip_path, root_folder)

    if not has_required_layout(root_folder):
        raise RuntimeError(
            f"Extraction completed but expected layout missing under "
            f"{root_folder}/MontgomerySet/CXR_png/. Check the archive."
        )

    print(f"Montgomery ready under {root_folder}")


if __name__ == "__main__":
    main()
