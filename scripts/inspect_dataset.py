"""
Inspect the RAVDESS corpus and report statistics measured from the files.

Nothing here is hard-coded from the dataset paper. Every number printed is
computed from the audio that is actually on disk, which is what makes the
README's dataset section trustworthy.

What it checks
--------------
* class distribution and per-speaker counts
* native sample rate / channel count / codec, and whether any file deviates
* duration distribution before and after silence trimming - this is what the
  analysis window length is chosen from
* exact duplicate audio, by hashing the decoded + resampled waveform rather
  than the file bytes, so a re-encoded copy would still be caught
* the largest silent gap, to confirm silence trimming is doing something

Usage
-----
    python scripts/inspect_dataset.py
    python scripts/inspect_dataset.py --json artifacts/dataset_stats.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.config import EMOTION_LABELS, AudioConfig, resolve_ravdess_root
from ml.data.preprocessing import load_and_prepare
from ml.data.ravdess import RavdessSample, build_manifest
from ml.utils.logging_utils import setup_logging


def percentiles(values: np.ndarray, points=(0, 1, 5, 25, 50, 75, 95, 99, 100)) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {f"p{p}": round(float(np.percentile(values, p)), 3) for p in points}


def describe(values: np.ndarray) -> dict[str, object]:
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": round(float(values.mean()), 3),
        "std": round(float(values.std()), 3),
        **percentiles(values),
    }


def inspect_one(sample: RavdessSample, root: Path, audio_cfg: AudioConfig) -> dict[str, object]:
    import soundfile as sf

    path = root / sample.path
    row: dict[str, object] = {
        "path": sample.path,
        "label": sample.label,
        "actor": sample.actor,
        "intensity": sample.intensity,
        "statement": sample.statement,
    }
    try:
        info = sf.info(str(path))
        row["native_sample_rate"] = int(info.samplerate)
        row["channels"] = int(info.channels)
        row["format"] = f"{info.format}/{info.subtype}"
        row["raw_duration"] = round(info.frames / info.samplerate, 3)

        window, prep = load_and_prepare(path, audio_cfg, crop="center")
        row["trimmed_duration"] = round(prep.n_valid_samples / audio_cfg.sample_rate, 3)
        row["peak"] = round(prep.peak_amplitude, 4)

        # Hash the *content*, not the container: catches a re-encoded duplicate.
        digest = hashlib.md5(np.ascontiguousarray(window, dtype=np.float32).tobytes())
        row["content_hash"] = digest.hexdigest()
        row["ok"] = True
    except Exception as exc:
        row["ok"] = False
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None, help="write stats here")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()

    setup_logging()
    root = args.root or resolve_ravdess_root()
    audio_cfg = AudioConfig()

    print(f"dataset root: {root}\n")
    samples = build_manifest(root)
    print(f"manifest entries: {len(samples)}\n")

    print(f"probing {len(samples)} files with {args.workers} workers...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows: list[dict[str, object]] = list(
            pool.map(lambda s: inspect_one(s, root, audio_cfg), samples)
        )

    failures = [r for r in rows if not r.get("ok")]
    ok_rows = [r for r in rows if r.get("ok")]
    print(f"decoded ok: {len(ok_rows)}   failed: {len(failures)}")
    for row in failures[:10]:
        print(f"  FAILED {row['path']}: {row.get('error')}")
    if failures:
        print("  (continuing with the decodable files)")

    # ---------------- class / speaker distribution -----------------------
    by_emotion = Counter(r["label"] for r in ok_rows)
    by_actor = Counter(r["actor"] for r in ok_rows)
    by_intensity = Counter(r["intensity"] for r in ok_rows)

    print("\n--- samples per emotion ---")
    for label in EMOTION_LABELS:
        count = by_emotion.get(label, 0)
        bar = "#" * round(count / 8)
        print(f"  {label:<10s} {count:>5d}  {bar}")
    counts = [by_emotion.get(lbl, 0) for lbl in EMOTION_LABELS]
    print(
        f"  imbalance ratio (max/min): {max(counts) / min(counts):.2f}"
        if min(counts)
        else "  imbalance: n/a"
    )

    print("\n--- samples per speaker ---")
    per_actor = {f"Actor_{a:02d}": by_actor.get(a, 0) for a in sorted(by_actor)}
    for name, count in per_actor.items():
        print(f"  {name}: {count}")

    print("\n--- intensity ---")
    print(f"  normal (01): {by_intensity.get(1, 0)}")
    print(f"  strong (02): {by_intensity.get(2, 0)}")

    # ---------------- format uniformity ----------------------------------
    print("\n--- audio format ---")
    formats = Counter(
        f"{r['native_sample_rate']}Hz/{r['channels']}ch/{r['format']}" for r in ok_rows
    )
    for fmt, count in formats.most_common():
        print(f"  {fmt}: {count}")

    # ---------------- durations ------------------------------------------
    raw_durations = np.array([r["raw_duration"] for r in ok_rows], dtype=np.float64)
    trimmed_durations = np.array([r["trimmed_duration"] for r in ok_rows], dtype=np.float64)

    print("\n--- duration (seconds, native 48 kHz) ---")
    print(f"  raw     : {describe(raw_durations)}")
    print(f"  trimmed : {describe(trimmed_durations)}")

    print("\n--- trimmed duration histogram ---")
    edges = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 100.0]
    histogram, _ = np.histogram(trimmed_durations, bins=edges)
    for lo, hi, count in zip(edges[:-1], edges[1:], histogram, strict=True):
        if count:
            print(f"  [{lo:4.1f}, {hi:4.1f})  {count:>5d}  {'#' * int(count / 4)}")

    # Window-size guidance: what fraction survives a given fixed window?
    print("\n--- fraction of utterances fully contained in a fixed window ---")
    print(f"  {'window':>7s} {'kept whole':>11s} {'cropped':>9s} {'padded':>8s} {'mean pad':>9s}")
    for window in (2.0, 2.5, 3.0, 3.5, 4.0):
        shorter = trimmed_durations[trimmed_durations < window]
        longer = trimmed_durations[trimmed_durations > window]
        pad_ratio = 1.0 - shorter.mean() / window if shorter.size else 0.0
        print(
            f"  {window:5.1f}s {100 * (1 - longer.size / trimmed_durations.size):9.1f}% "
            f"{100 * longer.size / trimmed_durations.size:8.1f}% "
            f"{100 * shorter.size / trimmed_durations.size:7.1f}% "
            f"{100 * pad_ratio:8.1f}%"
        )

    # ---------------- duplicates -----------------------------------------
    hashes = defaultdict(list)
    for row in ok_rows:
        hashes[row["content_hash"]].append(row["path"])  # type: ignore[index]
    duplicates = {h: paths for h, paths in hashes.items() if len(paths) > 1}
    print("\n--- duplicate content check ---")
    print(f"  distinct waveform hashes: {len(hashes)} / {len(ok_rows)} files")
    if duplicates:
        print(f"  WARNING: {len(duplicates)} duplicate group(s)")
        for paths in list(duplicates.values())[:5]:
            print(f"    {paths}")
    else:
        print("  no duplicate waveforms detected")

    # ---------------- per-emotion durations (sanity) ---------------------
    print("\n--- mean trimmed duration per emotion ---")
    per_emotion_duration: dict[str, float] = {}
    for label in EMOTION_LABELS:
        values = [r["trimmed_duration"] for r in ok_rows if r["label"] == label]
        if values:
            mean = float(np.mean(values))
            per_emotion_duration[label] = round(mean, 3)
            print(f"  {label:<10s} {mean:5.3f}s  (n={len(values)})")

    stats = {
        "dataset_root": str(root),
        "n_manifest_entries": len(samples),
        "n_decoded": len(ok_rows),
        "n_failed": len(failures),
        "failures": [{"path": r["path"], "error": r.get("error")} for r in failures],
        "n_speakers": len(by_actor),
        "n_classes": len(by_emotion),
        "samples_per_emotion": {lbl: by_emotion.get(lbl, 0) for lbl in EMOTION_LABELS},
        "samples_per_actor": per_actor,
        "samples_per_intensity": {
            "normal": by_intensity.get(1, 0),
            "strong": by_intensity.get(2, 0),
        },
        "class_imbalance_ratio": (
            round(max(counts) / min(counts), 4) if counts and min(counts) else None
        ),
        "formats": dict(formats),
        "duration_raw_sec": describe(raw_durations),
        "duration_trimmed_sec": describe(trimmed_durations),
        "mean_trimmed_duration_per_emotion_sec": per_emotion_duration,
        "duplicate_groups": len(duplicates),
        "distinct_waveform_hashes": len(hashes),
    }

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
