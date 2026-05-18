"""
Download the STARE retinal vessel dataset.

The official STARE project is hosted at
https://cecas.clemson.edu/~ahoover/stare/. The image archive and the two
manual vessel annotations (Adam Hoover and Valentina Kouznetsova) are
distributed as separate tar files. Their canonical URLs are kept in the
ARCHIVES list below; if these URLs change you can override them via the
STARE_IMAGES_URL, STARE_LABELS_AH_URL and STARE_LABELS_VK_URL environment
variables.
"""

import gzip
import os
import shutil
import sys
import tarfile
import urllib.request


ARCHIVES = {
    "stare-images": (
        "STARE_IMAGES_URL",
        "https://cecas.clemson.edu/~ahoover/stare/probing/stare-images.tar",
        "images",
    ),
    "labels-ah": (
        "STARE_LABELS_AH_URL",
        "https://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar",
        "1st_labels_ah",
    ),
    "labels-vk": (
        "STARE_LABELS_VK_URL",
        "https://cecas.clemson.edu/~ahoover/stare/probing/labels-vk.tar",
        "snd_label_vk",
    ),
}


def gunzip_in_place(folder: str) -> None:
    for fname in os.listdir(folder):
        if not fname.endswith(".gz"):
            continue
        src = os.path.join(folder, fname)
        dst = os.path.join(folder, fname[:-3])
        with gzip.open(src, "rb") as f_in, open(dst, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(src)


def download_and_extract(url: str, dst_folder: str) -> None:
    os.makedirs(dst_folder, exist_ok=True)
    archive_path = os.path.join(dst_folder, "download.tar")
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, archive_path)
    print(f"Extracting to {dst_folder}")
    with tarfile.open(archive_path, "r") as tar_ref:
        tar_ref.extractall(dst_folder)
    os.remove(archive_path)
    gunzip_in_place(dst_folder)


def main() -> None:
    root_folder = os.getenv("STARE_ROOT_FOLDER")
    if root_folder is None:
        raise EnvironmentError("STARE_ROOT_FOLDER environment variable is not set.")
    os.makedirs(root_folder, exist_ok=True)

    failures = []
    for archive_name, (env_var, default_url, subdir) in ARCHIVES.items():
        url = os.getenv(env_var, default_url)
        dst_folder = os.path.join(root_folder, subdir)
        has_decompressed = os.path.isdir(dst_folder) and any(
            not f.endswith(".gz") for f in os.listdir(dst_folder)
        )
        if has_decompressed:
            print(f"{archive_name} already present at {dst_folder}, skipping.")
            continue
        try:
            if os.path.isdir(dst_folder) and os.listdir(dst_folder):
                # Archive previously extracted but .gz files were not unpacked.
                gunzip_in_place(dst_folder)
            else:
                download_and_extract(url, dst_folder)
        except Exception as exc:
            failures.append((archive_name, url, str(exc)))

    if failures:
        for archive_name, url, msg in failures:
            print(f"Failed to download {archive_name} from {url}: {msg}")
        sys.exit(
            "One or more STARE archives could not be downloaded. Set the "
            "corresponding *_URL environment variable to a working mirror."
        )

    print("STARE download complete.")


if __name__ == "__main__":
    main()
