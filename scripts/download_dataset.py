"""
Download, verify and extract the RAVDESS speech corpus.

Why this exists
---------------
The archive is ~208 MB and Zenodo occasionally stalls mid-transfer, which leaves
a truncated ZIP that *looks* fine (it still has a valid PK header) and then
fails much later inside ``Expand-Archive``. This script therefore:

* resumes with HTTP ``Range`` requests instead of restarting from zero,
* re-sends ``Range`` whenever the socket dies,
* validates the **byte length** and the **MD5** published by Zenodo,
* only then extracts, and re-checks the file count afterwards.

Usage
-----
    python scripts/download_dataset.py
    python scripts/download_dataset.py --force          # re-download from scratch
    python scripts/download_dataset.py --skip-md5       # length + zip structure only
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
import zipfile
from pathlib import Path

# Allow `python scripts/download_dataset.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests

from ml.config import (
    RAVDESS_ARCHIVE_MD5,
    RAVDESS_ARCHIVE_NAME,
    RAVDESS_DOWNLOAD_URL,
    RAW_DIR,
    resolve_ravdess_root,
)

#: Content URL for the audio-only speech split. The human-facing
#: ``/files/<name>?download=1`` URL redirects and stalls more often than the
#: API content endpoint, so we use the latter.
CONTENT_URL = "https://zenodo.org/api/records/1188976/files/Audio_Speech_Actors_01-24.zip/content"
EXPECTED_BYTES = 208_468_073
EXPECTED_WAV_COUNT = 1440  # 60 trials x 24 actors
CHUNK = 1024 * 256
MAX_ATTEMPTS = 60


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:,.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def md5sum(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def size_ok(path: Path) -> bool:
    return path.is_file() and path.stat().st_size == EXPECTED_BYTES


def download(target: Path, force: bool) -> None:
    """Fetch ``target``, resuming across failures until the length matches."""
    if force and target.exists():
        print("  --force: discarding existing file")
        target.unlink()

    if size_ok(target):
        print(f"  already present and correct size ({human(EXPECTED_BYTES)})")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    start_at = target.stat().st_size if target.is_file() else 0
    if start_at > EXPECTED_BYTES:
        print("  local file larger than expected; restarting")
        target.unlink()
        start_at = 0
    if start_at:
        print(f"  resuming at {human(start_at)} ({start_at / EXPECTED_BYTES:.1%})")

    attempt = 0
    while start_at < EXPECTED_BYTES:
        attempt += 1
        if attempt > MAX_ATTEMPTS:
            raise RuntimeError(
                f"giving up after {MAX_ATTEMPTS} attempts at {start_at}/{EXPECTED_BYTES} bytes"
            )

        headers = {"User-Agent": "ravdess-emotion-recognition/1.0"}
        if start_at:
            headers["Range"] = f"bytes={start_at}-"

        try:
            with requests.get(
                CONTENT_URL, headers=headers, stream=True, timeout=(30, 120)
            ) as response:
                if start_at and response.status_code == 416:
                    # "Range not satisfiable" -> we already have everything.
                    break
                response.raise_for_status()

                # 206 = partial content (append). 200 = server ignored Range and
                # is sending the whole file again (overwrite).
                appending = response.status_code == 206 and start_at > 0
                if start_at and not appending:
                    print("  server ignored Range; restarting from byte 0")
                    start_at = 0

                mode = "ab" if appending else "wb"
                last_report = time.monotonic()
                with target.open(mode) as handle:
                    for block in response.iter_content(chunk_size=CHUNK):
                        if not block:
                            continue
                        handle.write(block)
                        start_at += len(block)
                        now = time.monotonic()
                        if now - last_report > 2:
                            pct = start_at / EXPECTED_BYTES * 100
                            print(
                                f"    {pct:5.1f}%  {human(start_at)} / {human(EXPECTED_BYTES)}",
                                end="\r",
                                flush=True,
                            )
                            last_report = now

            print(f"    {start_at / EXPECTED_BYTES * 100:5.1f}% complete      ")

        except (requests.RequestException, OSError) as exc:
            # Re-open the file to see how much actually landed on disk; the
            # partial bytes are still useful for the next Range request.
            on_disk = target.stat().st_size if target.is_file() else 0
            print(
                f"\n  attempt {attempt}: transfer interrupted at "
                f"{human(on_disk)} ({type(exc).__name__}: {exc})"
            )
            start_at = on_disk
            time.sleep(min(2 + attempt, 15))

    print(f"  received {human(target.stat().st_size)}")


def extract(archive: Path, dest: Path) -> Path:
    print(f"  extracting to {dest}")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt member in archive: {bad}")
        zf.extractall(dest)

    root = resolve_ravdess_root(dest)
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download from scratch")
    parser.add_argument(
        "--skip-md5",
        action="store_true",
        help="only verify byte length and ZIP structure",
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    archive = args.raw_dir / RAVDESS_ARCHIVE_NAME
    dest = args.raw_dir / "RAVDESS"

    print(f"RAVDESS audio-only speech corpus -> {archive}")
    download(archive, args.force)

    print("\n[1/3] size")
    size = archive.stat().st_size
    if size != EXPECTED_BYTES:
        print(f"  FAIL  got {size:,} bytes, expected {EXPECTED_BYTES:,}")
        return 1
    print(f"  OK    {size:,} bytes")

    if args.skip_md5:
        print("\n[2/3] md5  (skipped by request)")
    else:
        print("\n[2/3] md5")
        actual = md5sum(archive)
        if actual == RAVDESS_ARCHIVE_MD5:
            print(f"  OK    {actual}")
        else:
            print(f"  FAIL  got      {actual}")
            print(f"        expected {RAVDESS_ARCHIVE_MD5}")
            print("  Re-run with --skip-md5 to accept a length-verified archive.")
            return 1

    print("\n[3/3] extract + verify")
    try:
        root = extract(archive, dest)
    except Exception as exc:
        print(f"  FAIL  {type(exc).__name__}: {exc}")
        return 1

    wavs = sorted(root.rglob("*.wav"))
    actors = sorted({p.parent.name for p in wavs})
    print(f"  OK    root: {root}")
    print(f"  OK    {len(wavs)} wav files across {len(actors)} actor folders")
    print(f"  OK    actors: {', '.join(actors)}")
    if len(wavs) != EXPECTED_WAV_COUNT:
        print(f"  WARN  expected {EXPECTED_WAV_COUNT} wav files")
        return 1

    print(f"\nDone. Reference URL: {RAVDESS_DOWNLOAD_URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
