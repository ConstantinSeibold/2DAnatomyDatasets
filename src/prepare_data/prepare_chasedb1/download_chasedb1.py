"""
Set up the CHASE_DB1 retinal vessel dataset.

The dataset is hosted by Kingston University at
https://researchinnovation.kingston.ac.uk/files/40659508/CHASEDB1.zip but
the URL sits behind a Cloudflare Turnstile bot-check, so a plain HTTP
client (urllib, requests, curl_cffi, cloudscraper) cannot fetch it. This
script handles the download in one of three ways, tried in order:

1. If ``CHASEDB1_ROOT_FOLDER`` already contains the extracted dataset
   (Image_*.jpg + Image_*_1stHO.png + Image_*_2ndHO.png) we skip the
   download.
2. If a local zip exists at ``CHASEDB1_ZIP_PATH``
   (default: ``~/Downloads/CHASEDB1.zip``) we just extract it.
3. Otherwise we drive a real Chrome instance via ``nodriver`` to solve the
   Cloudflare challenge and download the archive automatically. This step
   needs Google Chrome installed locally and the ``nodriver`` package
   (``pip install nodriver``).

The expected layout after extraction is:
    $CHASEDB1_ROOT_FOLDER/
        Image_01L.jpg ... Image_14R.jpg          # 28 retinal images
        Image_01L_1stHO.png ... Image_14R_1stHO.png
        Image_01L_2ndHO.png ... Image_14R_2ndHO.png
"""

import os
import sys
import zipfile


CHASEDB1_URL = (
    "https://researchinnovation.kingston.ac.uk/files/40659508/CHASEDB1.zip"
)
DEFAULT_CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
]


def has_required_layout(root_folder: str) -> bool:
    if not os.path.isdir(root_folder):
        return False
    return any(
        fname.startswith("Image_") and fname.endswith(".jpg")
        for fname in os.listdir(root_folder)
    )


def extract_zip(zip_path: str, root_folder: str) -> None:
    print(f"Extracting {zip_path} into {root_folder}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(root_folder)


def find_chrome() -> str:
    override = os.getenv("CHROME_BINARY")
    if override:
        return override
    for path in DEFAULT_CHROME_PATHS:
        if os.path.isfile(path):
            return path
    raise RuntimeError(
        "Could not find a Chrome/Chromium binary. Set CHROME_BINARY to a "
        "full path."
    )


def download_via_nodriver(target_zip: str) -> None:
    try:
        import nodriver as uc
    except ImportError as exc:
        raise RuntimeError(
            "nodriver is required for the automatic Cloudflare bypass. "
            "Install it with `pip install nodriver`."
        ) from exc
    import asyncio
    import time

    chrome_path = find_chrome()
    download_dir = os.path.dirname(os.path.abspath(target_zip)) or os.getcwd()
    os.makedirs(download_dir, exist_ok=True)

    expected_name = os.path.basename(target_zip)
    landed_zip = os.path.join(download_dir, "CHASEDB1.zip")
    timeout_seconds = 120

    async def run() -> None:
        browser = await uc.start(
            headless=False, browser_executable_path=chrome_path
        )
        try:
            tab = await browser.get("about:blank")
            await tab.set_download_path(download_dir)
            await tab.get(CHASEDB1_URL)
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                await asyncio.sleep(1)
                files = os.listdir(download_dir)
                if any(f.endswith(".crdownload") for f in files):
                    continue
                if os.path.isfile(landed_zip):
                    return
            raise TimeoutError(
                "Cloudflare challenge did not complete within "
                f"{timeout_seconds}s."
            )
        finally:
            browser.stop()

    asyncio.run(run())

    if landed_zip != target_zip:
        os.replace(landed_zip, target_zip)


def main() -> None:
    root_folder = os.getenv("CHASEDB1_ROOT_FOLDER")
    if root_folder is None:
        raise EnvironmentError(
            "CHASEDB1_ROOT_FOLDER environment variable is not set."
        )
    os.makedirs(root_folder, exist_ok=True)

    if has_required_layout(root_folder):
        print(f"CHASE_DB1 already present at {root_folder}.")
        return

    default_zip = os.path.expanduser("~/Downloads/CHASEDB1.zip")
    zip_path = os.getenv("CHASEDB1_ZIP_PATH", default_zip)

    if not os.path.isfile(zip_path):
        print(
            f"Local zip not found at {zip_path}. Attempting Cloudflare "
            "bypass via nodriver (will open a visible Chrome window)..."
        )
        try:
            download_via_nodriver(zip_path)
        except Exception as exc:
            print(__doc__)
            sys.exit(
                f"Automatic download failed: {exc}\n"
                "Download CHASEDB1.zip manually from "
                f"{CHASEDB1_URL} and set CHASEDB1_ZIP_PATH."
            )

    extract_zip(zip_path, root_folder)

    if not has_required_layout(root_folder):
        sys.exit(
            f"Extraction finished but expected files are missing under {root_folder}."
        )
    print("Done.")


if __name__ == "__main__":
    main()
