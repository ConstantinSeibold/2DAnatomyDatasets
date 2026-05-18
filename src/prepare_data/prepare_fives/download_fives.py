"""
Set up the FIVES retinal vessel dataset.

The dataset is hosted on Figshare at
https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169
. The single file is a ~1.76 GB RAR archive at
https://ndownloader.figshare.com/files/34969398 . CC-BY-4.0.

Extraction requires a RAR5-capable extractor. We try in order:
``unar`` (Homebrew: ``brew install unar``), ``unrar`` (Homebrew:
``brew install carlocab/personal/unrar`` or ``apt install unrar``),
then ``7z`` (which works for older RAR3 but fails on RAR5 — Figshare's
archive happens to be RAR5, so it is only a last resort).

Behaviour:
1. Skip if ``FIVES_ROOT_FOLDER`` already contains train/ + test/.
2. Use ``FIVES_RAR_PATH`` (default ``~/Downloads/fives.rar``) if present.
3. Otherwise fetch from Figshare and extract.

Expected layout after extraction (upstream Naming):
    $FIVES_ROOT_FOLDER/
        train/
            Original/<NNN>_{A,D,G,N}.png   # 600 images
            Ground truth/<NNN>_{A,D,G,N}.png
        test/
            Original/<NNN>_{A,D,G,N}.png   # 200 images
            Ground truth/<NNN>_{A,D,G,N}.png
"""

import os
import shutil
import subprocess
import sys
import urllib.request


FIVES_URL = "https://ndownloader.figshare.com/files/34969398"


def has_required_layout(root_folder: str) -> bool:
    for split in ("train", "test"):
        orig = os.path.join(root_folder, split, "Original")
        gt = os.path.join(root_folder, split, "Ground truth")
        if not (os.path.isdir(orig) and os.path.isdir(gt)):
            return False
        if not any(f.endswith(".png") for f in os.listdir(orig)):
            return False
    return True


def download_archive(url: str, dest: str) -> None:
    print(f"Downloading {url} -> {dest} (~1.76 GB)")
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


def extract_rar(rar_path: str, root_folder: str) -> None:
    os.makedirs(root_folder, exist_ok=True)

    unar = shutil.which("unar")
    if unar is not None:
        print(f"Extracting {rar_path} into {root_folder} via {unar}")
        subprocess.run(
            [unar, "-quiet", "-force-overwrite", "-output-directory", root_folder, rar_path],
            check=True,
        )
        return

    unrar = shutil.which("unrar")
    if unrar is not None:
        print(f"Extracting {rar_path} into {root_folder} via {unrar}")
        subprocess.run([unrar, "x", "-y", rar_path, root_folder + "/"], check=True)
        return

    sevenzip = shutil.which("7z") or shutil.which("7zz")
    if sevenzip is not None:
        print(
            f"Falling back to {sevenzip} for {rar_path}; this may fail if the "
            f"archive is RAR5 (the Figshare FIVES archive is)."
        )
        subprocess.run([sevenzip, "x", "-y", f"-o{root_folder}", rar_path], check=True)
        return

    raise RuntimeError(
        "No RAR extractor on PATH. Install one of:\n"
        "  brew install unar                       # The Unarchiver (recommended)\n"
        "  brew install carlocab/personal/unrar    # unrar from RARLAB\n"
        "  apt install unar | unrar | p7zip-full   # Linux\n"
        "Then re-run this script."
    )


def flatten_top_dir(root_folder: str) -> None:
    """Move contents of the single top-level folder up if present."""
    if has_required_layout(root_folder):
        return
    entries = [
        e for e in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, e)) and not e.startswith(".")
    ]
    if len(entries) != 1:
        return
    inner = os.path.join(root_folder, entries[0])
    for child in os.listdir(inner):
        shutil.move(os.path.join(inner, child), os.path.join(root_folder, child))
    os.rmdir(inner)


def main() -> None:
    root_folder = os.getenv("FIVES_ROOT_FOLDER")
    if root_folder is None:
        raise EnvironmentError("FIVES_ROOT_FOLDER environment variable is not set.")

    if has_required_layout(root_folder):
        print(f"FIVES already extracted under {root_folder}; skipping download.")
        return

    rar_path = os.getenv(
        "FIVES_RAR_PATH", os.path.expanduser("~/Downloads/fives.rar")
    )

    if not os.path.isfile(rar_path):
        try:
            download_archive(FIVES_URL, rar_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch FIVES from {FIVES_URL}: {exc}.\n"
                f"You may download it manually and place the .rar at "
                f"{rar_path}, then re-run this script."
            )

    extract_rar(rar_path, root_folder)
    flatten_top_dir(root_folder)

    if not has_required_layout(root_folder):
        raise RuntimeError(
            f"Extraction completed but expected layout missing under "
            f"{root_folder} (need train/Original + train/Ground truth + "
            f"test/Original + test/Ground truth). Check the archive."
        )

    print(f"FIVES ready under {root_folder}")


if __name__ == "__main__":
    main()
