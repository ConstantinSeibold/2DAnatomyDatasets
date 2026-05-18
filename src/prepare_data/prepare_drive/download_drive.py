"""
Set up the DRIVE dataset from a locally downloaded archive.

The official DRIVE dataset
(https://drive.grand-challenge.org/) requires a free account, so it cannot
be downloaded automatically. After downloading ``datasets.zip`` from the
challenge site, point this script at it via the ``DRIVE_ZIP_PATH``
environment variable (default: ``~/Downloads/datasets.zip``).

The shipped archive contains two nested zips, ``training.zip`` and
``test.zip``, both of which we extract into ``DRIVE_ROOT_FOLDER`` to obtain:

    $DRIVE_ROOT_FOLDER/training/{images, 1st_manual, mask}/
    $DRIVE_ROOT_FOLDER/test/{images, mask}/

Note that the official test split does not ship with vessel annotations.
"""

import os
import sys
import zipfile


REQUIRED_SUBDIRS = [
    os.path.join("training", "images"),
    os.path.join("training", "1st_manual"),
    os.path.join("test", "images"),
]


def has_required_layout(root_folder: str) -> bool:
    return all(
        os.path.isdir(os.path.join(root_folder, sub)) for sub in REQUIRED_SUBDIRS
    )


def extract_nested(zip_path: str, root_folder: str) -> None:
    os.makedirs(root_folder, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as outer:
        for inner_name in ("training.zip", "test.zip"):
            if inner_name not in outer.namelist():
                raise RuntimeError(
                    f"{zip_path} is missing the expected '{inner_name}' member."
                )
            with outer.open(inner_name) as inner_file:
                inner_bytes = inner_file.read()
            inner_path = os.path.join(root_folder, inner_name)
            with open(inner_path, "wb") as f:
                f.write(inner_bytes)
            with zipfile.ZipFile(inner_path, "r") as inner_zip:
                inner_zip.extractall(root_folder)
            os.remove(inner_path)


def main() -> None:
    root_folder = os.getenv("DRIVE_ROOT_FOLDER")
    if root_folder is None:
        raise EnvironmentError("DRIVE_ROOT_FOLDER environment variable is not set.")
    os.makedirs(root_folder, exist_ok=True)

    if has_required_layout(root_folder):
        print(f"DRIVE dataset already present at {root_folder}.")
        return

    default_zip = os.path.expanduser("~/Downloads/datasets.zip")
    zip_path = os.getenv("DRIVE_ZIP_PATH", default_zip)
    if not os.path.isfile(zip_path):
        print(__doc__)
        sys.exit(
            f"DRIVE archive not found at {zip_path}. Download datasets.zip from "
            "https://drive.grand-challenge.org/ and set DRIVE_ZIP_PATH."
        )

    print(f"Extracting DRIVE archive from {zip_path}")
    extract_nested(zip_path, root_folder)

    if not has_required_layout(root_folder):
        sys.exit(
            f"Extraction finished but expected folders are missing under {root_folder}."
        )
    print("Done.")


if __name__ == "__main__":
    main()
