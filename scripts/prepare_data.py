"""
Stage 1 of the ML pipeline: manifest -> speaker-disjoint split -> feature cache.

Running this is a precondition for training. It is deliberately a separate,
idempotent step so that the expensive part (MFCC extraction over 1440 files) is
done once and can be reused across experiments, and so that a dataset problem
surfaces as a clean failure here rather than as mysterious bad accuracy later.

It also acts as a leakage guard. Two checks are enforced:

1. **No actor may appear in more than one split.** RAVDESS gives each actor the
   same 60 utterances, so a file-level random split would put the same voice on
   both sides of the boundary and let the model score well by recognising *who*
   is speaking rather than *how*.

2. **No identical waveform may appear in more than one split.** Speaker
   separation does not automatically rule out duplicated audio. This corpus
   contains one such pair (two Actor_07 files that differ only in their
   statement/repetition codes but decode to identical audio), so the check is
   real rather than theoretical.

Usage
-----
    python scripts/prepare_data.py
    python scripts/prepare_data.py --rebuild      # ignore an existing cache
    python scripts/prepare_data.py --target-duration 2.5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.config import (
    DATA_DIR,
    EMOTION_LABELS,
    AudioConfig,
    ExperimentConfig,
    FeatureConfig,
    SplitConfig,
    resolve_ravdess_root,
)
from ml.data.dataset import (
    PROCESSED_DIR_NAME,
    build_feature_cache,
    config_fingerprint,
)
from ml.data.ravdess import (
    assign_splits,
    build_manifest,
    save_manifest,
    save_split_manifest,
    summarise,
)
from ml.utils.logging_utils import get_logger, setup_logging

logger = get_logger("prepare_data")

OUTPUT_DIR = DATA_DIR / PROCESSED_DIR_NAME


def check_no_cross_split_duplicates(
    splits: dict[str, list], hashes: dict[str, str]
) -> list[dict[str, object]]:
    """Return any waveform hash that appears in more than one split."""
    by_hash: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for split_name, samples in splits.items():
        for sample in samples:
            digest = hashes.get(sample.path)
            if digest:
                by_hash[digest][split_name].append(sample.path)

    leaks = []
    for digest, placement in by_hash.items():
        if len(placement) > 1:
            leaks.append({"content_hash": digest, "splits": dict(placement)})
    return leaks


def content_hashes(
    samples: list, root: Path, audio_cfg: AudioConfig, workers: int
) -> dict[str, str]:
    """Hash the preprocessed waveform of each file (catches re-encoded copies)."""
    import hashlib
    from concurrent.futures import ThreadPoolExecutor

    from ml.data.preprocessing import load_and_prepare

    def hash_one(sample):
        try:
            window, _ = load_and_prepare(root / sample.path, audio_cfg, crop="center")
        except Exception as exc:
            logger.warning("cannot hash %s: %s", sample.path, exc)
            return sample.path, None
        digest = hashlib.md5(np.ascontiguousarray(window, dtype=np.float32).tobytes())
        return sample.path, digest.hexdigest()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return {path: digest for path, digest in pool.map(hash_one, samples) if digest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="ignore existing cache")
    parser.add_argument("--target-duration", type=float, default=None)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--no-delta2", action="store_true", help="MFCC + delta only")
    parser.add_argument("--no-cmvn", action="store_true")
    parser.add_argument("--no-augment", action="store_true", help="train without SpecAugment")
    parser.add_argument("--test-actors", type=int, nargs="+", default=None)
    parser.add_argument("--val-actors", type=int, nargs="+", default=None)
    args = parser.parse_args()

    setup_logging()

    audio_cfg = AudioConfig()
    feature_cfg = FeatureConfig()
    if args.target_duration is not None:
        audio_cfg.target_duration_sec = args.target_duration
    if args.sample_rate is not None:
        audio_cfg.sample_rate = args.sample_rate
    if args.no_delta2:
        feature_cfg.use_delta2 = False
    if args.no_cmvn:
        feature_cfg.cmvn = False

    split_cfg = SplitConfig()
    if args.test_actors:
        split_cfg.test_actors = list(args.test_actors)
    if args.val_actors:
        split_cfg.val_actors = list(args.val_actors)

    # ---------------- manifest -------------------------------------------
    root = resolve_ravdess_root()
    logger.info("dataset root: %s", root)
    samples = build_manifest(root)
    if not samples:
        logger.error("no samples found")
        return 1

    stats = summarise(samples)
    logger.info(
        "%d utterances, %d speakers, %d classes, imbalance %.2f:1",
        stats["n_samples"],
        stats["n_speakers"],
        stats["n_classes"],
        stats["class_imbalance_ratio"] or 0.0,
    )

    # ---------------- splits ---------------------------------------------
    splits = assign_splits(samples, split_cfg)
    print("\n--- speaker-disjoint split ---")
    for name in ("train", "val", "test"):
        actors = sorted({s.actor for s in splits[name]})
        per_emotion = defaultdict(int)
        for sample in splits[name]:
            per_emotion[sample.label] += 1
        print(f"  {name:<5s} {len(splits[name]):>5d} utterances  actors={actors}")
        print("        " + "  ".join(f"{lbl}={per_emotion.get(lbl, 0)}" for lbl in EMOTION_LABELS))

    # Guard 1: actor disjointness (assign_splits raises, but assert the invariant
    # explicitly so the intent is obvious at the call site).
    actors_by_split = {name: {s.actor for s in items} for name, items in splits.items()}
    names = sorted(actors_by_split)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if actors_by_split[left] & actors_by_split[right]:
                logger.error("actor leakage between %s and %s", left, right)
                return 1

    # Guard 2: identical waveforms must not straddle a split boundary.
    logger.info("hashing waveforms for the duplicate-leakage check...")
    hashes = content_hashes(samples, root, audio_cfg, args.workers)
    within_split = {
        digest: paths
        for digest, paths in (
            (d, [s.path for s in samples if hashes.get(s.path) == d]) for d in set(hashes.values())
        )
        if len(paths) > 1
    }
    if within_split:
        logger.warning("%d group(s) of identical audio exist inside the corpus:", len(within_split))
        for digest, paths in within_split.items():
            actors = sorted({Path(p).parent.name for p in paths})
            logger.warning("  %s -> %s (actors: %s)", digest[:12], paths, actors)

    leaks = check_no_cross_split_duplicates(splits, hashes)
    if leaks:
        logger.error("duplicate audio found across split boundaries - this is data leakage")
        for leak in leaks:
            logger.error("  %s", leak)
        return 1
    logger.info("no identical audio crosses a split boundary")

    # ---------------- persist manifest + splits -------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_manifest(samples, OUTPUT_DIR / "manifest.json")
    save_split_manifest(splits, split_cfg, OUTPUT_DIR / "split_manifest.json")
    logger.info("wrote manifest.json and split_manifest.json")

    # ---------------- feature cache -------------------------------------
    fingerprint = config_fingerprint(audio_cfg, feature_cfg)
    cache_dir = OUTPUT_DIR / f"features_{fingerprint}"
    cache_ready = (cache_dir / "meta.json").is_file() and not args.rebuild

    if cache_ready:
        logger.info("reusing existing feature cache at %s", cache_dir)
    else:
        print(
            f"\nextracting MFCC features -> {cache_dir}\n"
            f"  window   : {audio_cfg.target_duration_sec}s @ {audio_cfg.sample_rate} Hz\n"
            f"  features : {feature_cfg.describe()}\n"
            f"  cmvn     : {feature_cfg.cmvn}"
        )
        cache = build_feature_cache(samples, root, audio_cfg, feature_cfg, num_workers=args.workers)
        cache.save(cache_dir)
        logger.info("wrote feature cache (%s)", cache_dir)

    # ---------------- record the configuration used ---------------------
    experiment = ExperimentConfig(
        audio=audio_cfg,
        features=feature_cfg,
        split=split_cfg,
        notes="prepared by scripts/prepare_data.py",
    )
    experiment.save(OUTPUT_DIR / "prepared_config.json")
    (OUTPUT_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\n" + "=" * 68)
    print("DATA READY")
    print("=" * 68)
    print(f"  manifest      : {OUTPUT_DIR / 'manifest.json'}")
    print(f"  split manifest: {OUTPUT_DIR / 'split_manifest.json'}")
    print(f"  feature cache : {cache_dir}")
    print(f"  fingerprint   : {fingerprint}")
    print(f"  augmentation  : {'SpecAugment off' if args.no_augment else 'SpecAugment on'}")
    print("\nNext:  python scripts/train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
