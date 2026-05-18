"""
Download the Jump-Broadcast annotation package and the source YouTube videos.

The annotation zip is small and is fetched directly. The actual image frames
must be extracted from publicly available YouTube videos (see
``video_download_links.txt``).

The video downloader keeps cycling through the URL list until every video is
either successfully downloaded or has been marked as permanently unavailable
(stored in ``videos/failed.txt``). Transient YouTube failures (bot
rate-limit, n-sig challenge, HTTP 429/503) are retried with exponential
backoff; permanent failures (private / removed / copyright takedown / geo
block) are recorded once and skipped on subsequent runs.

Environment variables:

- ``YT_DLP_COOKIES_FROM_BROWSER`` — passed through to ``yt-dlp
  --cookies-from-browser`` to bypass "Sign in to confirm you're not a bot"
  challenges (e.g. ``chrome``, ``firefox``, ``safari``).
- ``YT_DLP_MAX_ROUNDS`` — outer-loop cap (default 10).

Outputs under ``$JUMP_BROADCAST_ROOT_FOLDER``:

    annotations/  # contents of jump-broadcast.zip
    videos/<video_id>.mp4
    videos/failed.txt  # one URL per line for permanently-unavailable videos
"""

import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile


ZIP_URL = (
    "https://assets.uni-augsburg.de/smedia/filer_private/5e/86/"
    "5e867709-49f2-4709-abc4-625ee30ca850/jump-broadcast.zip"
)

# yt-dlp stderr patterns that indicate the video will never be downloadable.
PERMANENT_ERROR_PATTERNS = (
    "video unavailable",
    "this video is private",
    "this video has been removed",
    "copyright",
    "removed by the uploader",
    "account has been terminated",
    "video is not available in your country",
    "video is no longer available",
    "video does not exist",
    "premiere",
    "members-only",
)

# yt-dlp stderr patterns that indicate a retryable / transient failure.
TRANSIENT_ERROR_PATTERNS = (
    "sign in to confirm",
    "http error 429",
    "http error 403",
    "http error 503",
    "n challenge",
    "could not find a working javascript",
    "unable to extract",
    "read timed out",
    "connection reset",
    "temporary failure",
    "throttled",
)


def download_zip(zip_path: str) -> None:
    if os.path.isfile(zip_path):
        print(f"Annotation zip already present at {zip_path}.")
        return
    print(f"Downloading {ZIP_URL} -> {zip_path}")
    urllib.request.urlretrieve(ZIP_URL, zip_path)


def extract_annotations(zip_path: str, ann_dir: str) -> None:
    if os.path.isdir(ann_dir) and os.listdir(ann_dir):
        print(f"Annotations already extracted at {ann_dir}.")
        return
    os.makedirs(ann_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(ann_dir)
    # Flatten the single ``jump-broadcast/`` directory inside the zip so
    # downstream scripts can find files at predictable paths.
    inner = os.path.join(ann_dir, "jump-broadcast")
    if os.path.isdir(inner):
        for name in os.listdir(inner):
            shutil.move(os.path.join(inner, name), os.path.join(ann_dir, name))
        os.rmdir(inner)


def parse_video_urls(links_file: str) -> list:
    urls = []
    with open(links_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("http"):
                urls.append(line)
    return urls


def _classify_yt_dlp_error(stderr_text: str) -> str:
    """Return 'permanent', 'transient', or 'unknown' for a yt-dlp failure."""
    lower = stderr_text.lower()
    for pat in PERMANENT_ERROR_PATTERNS:
        if pat in lower:
            return "permanent"
    for pat in TRANSIENT_ERROR_PATTERNS:
        if pat in lower:
            return "transient"
    return "unknown"


def _yt_dlp_cmd(url: str, target: str) -> list:
    cmd = [
        "yt-dlp",
        "-f",
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        # Polite throttling reduces the chance of triggering bot detection.
        "--sleep-requests",
        "1",
        "--retries",
        "3",
        "--fragment-retries",
        "5",
        "-o",
        target,
        url,
    ]
    cookies_browser = os.getenv("YT_DLP_COOKIES_FROM_BROWSER")
    if cookies_browser:
        cmd.extend(["--cookies-from-browser", cookies_browser])
    return cmd


def _load_failed(failed_path: str) -> set:
    if not os.path.isfile(failed_path):
        return set()
    with open(failed_path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def _persist_failed(failed_path: str, failed: set) -> None:
    with open(failed_path, "w") as f:
        for url in sorted(failed):
            f.write(url + "\n")


def download_videos(links_file: str, videos_dir: str) -> None:
    if shutil.which("yt-dlp") is None:
        print(
            "WARNING: yt-dlp is not installed. Install it with `pip install yt-dlp` "
            "and re-run this script to fetch the source videos."
        )
        return

    os.makedirs(videos_dir, exist_ok=True)
    urls = parse_video_urls(links_file)
    failed_path = os.path.join(videos_dir, "failed.txt")
    permanent_failed = _load_failed(failed_path)
    max_rounds = int(os.getenv("YT_DLP_MAX_ROUNDS", "10"))

    # Map URL -> idx so failed.txt is index-independent (URLs are stable).
    pending = []
    for idx, url in enumerate(urls):
        target = os.path.join(videos_dir, f"{idx}.mp4")
        if os.path.isfile(target):
            print(f"[{idx}] already downloaded.")
            continue
        if url in permanent_failed:
            print(f"[{idx}] previously marked unavailable; skipping.")
            continue
        pending.append((idx, url, target))

    if not pending:
        print("All videos already downloaded (or permanently unavailable).")
        return

    backoff_seconds = 30
    for round_no in range(1, max_rounds + 1):
        if not pending:
            break
        print(
            f"--- Round {round_no}/{max_rounds}: {len(pending)} video(s) remaining ---"
        )
        next_pending = []
        rate_limited_this_round = False
        for idx, url, target in pending:
            print(f"[{idx}] {url}")
            result = subprocess.run(
                _yt_dlp_cmd(url, target),
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and os.path.isfile(target):
                print(f"[{idx}] OK")
                continue
            # Surface yt-dlp output so users can diagnose unknown failures.
            tail = (result.stderr or result.stdout).strip().splitlines()[-5:]
            for line in tail:
                print(f"    {line}")
            kind = _classify_yt_dlp_error(result.stderr + result.stdout)
            if kind == "permanent":
                print(f"[{idx}] permanently unavailable; will not retry.")
                permanent_failed.add(url)
                _persist_failed(failed_path, permanent_failed)
                continue
            # Transient or unknown -> retry next round.
            rate_limited_this_round = rate_limited_this_round or kind == "transient"
            next_pending.append((idx, url, target))
            print(f"[{idx}] transient failure ({kind}); will retry next round.")

        pending = next_pending
        if pending and round_no < max_rounds:
            wait = min(backoff_seconds, 1800)
            if rate_limited_this_round:
                wait = min(backoff_seconds * 2, 1800)
            print(f"Sleeping {wait}s before next round...")
            time.sleep(wait)
            backoff_seconds = min(backoff_seconds * 2, 1800)

    if pending:
        print(
            f"WARNING: {len(pending)} video(s) still missing after {max_rounds} rounds. "
            "Re-run the script later (transient failures reset on a new run), or set "
            "YT_DLP_COOKIES_FROM_BROWSER=<browser> if YouTube is blocking with a bot "
            "challenge."
        )


def main() -> None:
    root_folder = os.getenv("JUMP_BROADCAST_ROOT_FOLDER")
    if root_folder is None:
        raise EnvironmentError(
            "JUMP_BROADCAST_ROOT_FOLDER environment variable is not set."
        )
    os.makedirs(root_folder, exist_ok=True)

    zip_path = os.path.join(root_folder, "jump-broadcast.zip")
    ann_dir = os.path.join(root_folder, "annotations")
    videos_dir = os.path.join(root_folder, "videos")

    download_zip(zip_path)
    extract_annotations(zip_path, ann_dir)

    links_file = os.path.join(ann_dir, "video_download_links.txt")
    if not os.path.isfile(links_file):
        sys.exit(f"Expected {links_file} after extraction.")
    download_videos(links_file, videos_dir)
    print("Done.")


if __name__ == "__main__":
    main()
