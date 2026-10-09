"""Torch datasets, feature caching and speaker-aware split handling.

Feature caching
---------------
Because :class:`~ml.config.AudioConfig` fixes ``target_duration_sec`` and
:class:`~ml.config.FeatureConfig` fixes ``hop_length``, the MFCC matrix for
*every* utterance has the identical shape ``(n_features, n_frames)``. That
makes it possible to pre-extract all features once into a single ``.npy`` array
and memory-map it, which turns a CPU-bound epoch into a memory-bandwidth-bound
one.

That trade has a real cost, and it is worth being explicit about it: pre-computed
features are always extracted from the **centre** window, so the random-position
crop that :func:`ml.data.preprocessing.load_and_prepare` supports at waveform
level is not available once cached. Random cropping would mean re-extracting
features every epoch (~10x slower per epoch on this box). Since RAVDESS
utterances are ~2-4 s against a 3 s window, position augmentation would in any
case shift the crop by only a few hundred milliseconds. Instead, augmentation is
done *on the cached features* with SpecAugment-style time/frequency masking
(:mod:`ml.data.augment`), which is both cheaper and a stronger regulariser.

Cache safety
------------
The cache is fingerprinted with a hash of the audio + feature configuration.
Loading a cache built with different settings raises instead of silently
training on features the model will never see at inference time.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.config import AudioConfig, FeatureConfig
from ml.data.preprocessing import load_and_prepare
from ml.data.ravdess import RavdessSample
from ml.features.mfcc import MFCCExtractor

logger = logging.getLogger(__name__)

PROCESSED_DIR_NAME = "processed"
CACHE_VERSION = 2  # bump to invalidate stale caches


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------
def config_fingerprint(
    audio_cfg: AudioConfig, feature_cfg: FeatureConfig
) -> str:
    """Stable hash of the audio and feature configuration a cache was built with.

    Hashes the whole of both dataclasses, including fields that cannot change a
    coefficient (``max_upload_bytes``, the duration guard rails). That is
    deliberately over-strict: the two failure modes are not symmetric. A hash
    that is too broad merely forces a harmless rebuild, while a hash that is too
    narrow would silently serve features that do not match the config and poison
    every metric downstream. So this fails safe.
    """
    from dataclasses import asdict

    payload = json.dumps(
        {
            "cache_version": CACHE_VERSION,
            "audio": asdict(audio_cfg),
            "features": asdict(feature_cfg),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Cache building
# ---------------------------------------------------------------------------
@dataclass
class FeatureCache:
    """Pre-extracted MFCC matrices for the whole corpus.

    Attributes:
        features: ``(n_samples, n_features, n_frames)`` float32.
        labels: ``(n_samples,)`` int64 class indices.
        valid_frames: ``(n_samples,)`` int32 - number of leading frames that
            contain real speech. Everything past this is the zero padding of a
            clip shorter than the analysis window, and is masked out of the
            attention pooling.
        sample_paths: filesystem paths in the same order as the arrays.
        fingerprint: config hash the cache was built with.
    """

    features: np.ndarray
    labels: np.ndarray
    valid_frames: np.ndarray
    sample_paths: list[str]
    fingerprint: str
    root: str

    def __len__(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])

    @property
    def n_frames(self) -> int:
        return int(self.features.shape[2])

    @property
    def mean_padding_ratio(self) -> float:
        """Mean fraction of the window that is padding (0 when nothing is short)."""
        if self.n_frames == 0:
            return 0.0
        return float(1.0 - (self.valid_frames.mean() / self.n_frames))

    # -- persistence ------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "features.npy", self.features)
        np.save(directory / "labels.npy", self.labels)
        np.save(directory / "valid_frames.npy", self.valid_frames)
        meta = {
            "cache_version": CACHE_VERSION,
            "fingerprint": self.fingerprint,
            "root": self.root,
            "sample_paths": self.sample_paths,
            "shape": list(self.features.shape),
            "dtype": str(self.features.dtype),
            "mean_valid_frames": float(np.mean(self.valid_frames)),
        }
        (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, expected_fingerprint: str | None = None) -> FeatureCache:
        directory = Path(directory)
        meta_path = directory / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"No feature cache at '{directory}'.")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("cache_version") != CACHE_VERSION:
            raise ValueError(
                f"Cache at '{directory}' was built by cache version "
                f"{meta.get('cache_version')}, this build is {CACHE_VERSION}. "
                "Delete it and rebuild."
            )
        if expected_fingerprint and meta.get("fingerprint") != expected_fingerprint:
            raise ValueError(
                f"Cache at '{directory}' was built with a different audio/feature "
                f"configuration (fingerprint {meta.get('fingerprint')} != "
                f"{expected_fingerprint}). Refusing to load it - rebuild the cache."
            )

        features = np.load(directory / "features.npy")
        labels = np.load(directory / "labels.npy")
        valid_frames = np.load(directory / "valid_frames.npy")
        return cls(
            features=features,
            labels=labels,
            valid_frames=valid_frames,
            sample_paths=meta["sample_paths"],
            fingerprint=meta.get("fingerprint", ""),
            root=meta.get("root", ""),
        )


def _extract_one(
    args: tuple[int, str, str, int, AudioConfig, FeatureConfig],
) -> tuple[int, str, int, np.ndarray | None, int, str | None]:
    """Worker: decode -> preprocess -> MFCC for a single file.

    Runs in a thread (librosa's STFT/FFT releases the GIL, so threads give real
    speed-up without the pickling and re-import costs of processes).

    ``abs_path`` is read from disk; ``rel_path`` is what identifies the sample in
    the cache, so it must stay POSIX-style and root-relative regardless of the
    host OS - it is the join key between the cache and the manifest.
    """
    index, abs_path, rel_path, label_index, audio_cfg, feature_cfg = args
    try:
        window, info = load_and_prepare(abs_path, audio_cfg, crop="center")
        matrix = MFCCExtractor(feature_cfg).transform(window, audio_cfg.sample_rate)
        # Frames carrying real speech. librosa centres the STFT, so a window of
        # N samples yields 1 + N // hop_length frames.
        valid = min(
            matrix.shape[1], 1 + int(info.n_valid_samples // feature_cfg.hop_length)
        )
        return index, rel_path, label_index, matrix, max(1, valid), None
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return index, rel_path, label_index, None, 0, f"{type(exc).__name__}: {exc}"


def build_feature_cache(
    samples: Sequence[RavdessSample],
    root: Path,
    audio_cfg: AudioConfig,
    feature_cfg: FeatureConfig,
    *,
    num_workers: int = 4,
    show_progress: bool = True,
) -> FeatureCache:
    """Decode + preprocess + MFCC every sample.

    Failures are collected rather than raised: one corrupt file should not abort
    a multi-minute extraction pass. Every failure is logged afterwards, because
    silently shrinking the dataset is exactly the kind of thing that quietly
    invalidates an evaluation.

    Args:
        num_workers: extraction threads. ``1`` runs serially (easiest to debug).

    Returns:
        A :class:`FeatureCache` containing only the successfully processed
        samples, in manifest order.
    """
    root = Path(root)
    tasks = [
        (
            i,
            str(root / sample.path),
            sample.path,
            sample.label_index,
            audio_cfg,
            feature_cfg,
        )
        for i, sample in enumerate(samples)
    ]

    results: list[tuple | None] = [None] * len(tasks)
    failures: list[tuple[str, str]] = []
    step = max(1, len(tasks) // 10)

    def store(outcome) -> None:
        index, path, label_index, matrix, valid, error = outcome
        results[index] = (path, label_index, matrix, valid)
        if error is not None:
            failures.append((path, error))

    workers = max(1, int(num_workers))
    if workers == 1:
        for position, task in enumerate(tasks):
            store(_extract_one(task))
            if show_progress and (position + 1) % step == 0:
                logger.info("  extracted %d/%d", position + 1, len(tasks))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_extract_one, task) for task in tasks]
            for done, future in enumerate(as_completed(futures), start=1):
                store(future.result())
                if show_progress and done % step == 0:
                    logger.info("  extracted %d/%d", done, len(tasks))

    ok = [r for r in results if r is not None]
    if not ok:
        raise RuntimeError(
            f"No samples could be processed out of {len(tasks)}. "
            f"First errors: {failures[:3]}"
        )

    paths = [r[0] for r in ok]
    labels = np.array([r[1] for r in ok], dtype=np.int64)
    valid_frames = np.array([r[3] for r in ok], dtype=np.int32)
    matrices = [r[2] for r in ok]

    expected = matrices[0].shape
    mismatched = [p for p, m in zip(paths, matrices) if m.shape != expected]
    if mismatched:
        raise RuntimeError(
            f"Inconsistent feature shapes: expected {expected}, "
            f"{len(mismatched)} file(s) differ (e.g. {mismatched[:3]}). "
            "This usually means target_duration_sec or hop_length changed."
        )

    features = np.stack(matrices).astype(np.float32)

    if failures:
        logger.warning("%d file(s) failed to process:", len(failures))
        for path, reason in failures[:20]:
            logger.warning("  %s -> %s", path, reason)

    mean_valid = float(valid_frames.mean()) / max(1, features.shape[2])
    fingerprint = config_fingerprint(audio_cfg, feature_cfg)
    logger.info(
        "feature cache: %d/%d samples, shape %s, mean valid-frame fill %.1f%%, "
        "fingerprint %s",
        features.shape[0],
        len(tasks),
        features.shape[1:],
        mean_valid * 100,
        fingerprint,
    )
    return FeatureCache(
        features=features,
        labels=labels,
        valid_frames=valid_frames,
        sample_paths=paths,
        fingerprint=fingerprint,
        root=str(root),
    )



# ---------------------------------------------------------------------------
# Augmentation (operates on cached features)
# ---------------------------------------------------------------------------
@dataclass
class AugmentConfig:
    """SpecAugment-style masking, applied only to the training split."""

    enabled: bool = True
    #: Max fraction of the time axis zeroed per sample (0 disables).
    time_mask_max: int = 20
    time_mask_count: int = 2
    #: Max fraction of the feature axis zeroed per sample.
    freq_mask_max: int = 8
    freq_mask_count: int = 2
    #: Value written into the masked region. 0.0 is the utterance mean after
    #: CMVN, which keeps the mask from shifting the normalisation statistics.
    mask_value: float = 0.0


def augment_features(
    matrix: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator
) -> np.ndarray:
    """Apply time and frequency masking to a ``(n_features, n_frames)`` matrix.

    Rationale: emotion cues are distributed across the spectrogram rather than
    localised in one band, so forcing the classifier to rely on partial evidence
    is a genuinely useful regulariser and reduces overfitting to speaker traits
    in a 960-utterance training set.
    """
    if not cfg.enabled:
        return matrix
    out = matrix.copy()
    n_features, n_frames = out.shape

    for _ in range(cfg.time_mask_count):
        width = int(rng.integers(0, cfg.time_mask_max + 1))
        if width <= 0:
            continue
        start = int(rng.integers(0, max(1, n_frames - width)))
        out[:, start : start + width] = cfg.mask_value

    for _ in range(cfg.freq_mask_count):
        width = int(rng.integers(0, cfg.freq_mask_max + 1))
        if width <= 0:
            continue
        start = int(rng.integers(0, max(1, n_features - width)))
        out[start : start + width, :] = cfg.mask_value

    return out


# ---------------------------------------------------------------------------
# Torch dataset
# ---------------------------------------------------------------------------
class EmotionDataset(Dataset):
    """Serves cached MFCC matrices as ``(1, n_features, n_frames)`` tensors.

    Args:
        cache: the pre-extracted features.
        indices: row indices into ``cache`` - this is how a split selects its
            speakers without duplicating the feature array.
        augment: enabled **only** for the training split.
        seed: base seed; each worker/epoch derives its own RNG so augmentation
            is non-repeating yet still reproducible.
    """

    def __init__(
        self,
        cache: FeatureCache,
        indices: Sequence[int],
        *,
        augment: bool = False,
        augment_cfg: AugmentConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.cache = cache
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = augment
        self.augment_cfg = augment_cfg or AugmentConfig()
        self.seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return int(self.indices.size)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Return ``(features, frame_mask, label, cache_row)``.

        ``features`` is ``(1, n_features, n_frames)`` for ``Conv2d``.
        ``frame_mask`` is ``(n_frames,)`` with 1.0 on real speech frames and 0.0
        on the zero padding of a short clip. It is passed at full frame
        resolution; the model downsamples it to whatever the conv stack
        produces, so no arithmetic here can drift out of sync with the model.
        """
        row = int(self.indices[position])
        matrix = self.cache.features[row]

        if self.augment:
            rng = np.random.default_rng((self.seed, self._epoch, position))
            matrix = augment_features(matrix, self.augment_cfg, rng)

        n_frames = matrix.shape[1]
        valid = int(min(self.cache.valid_frames[row], n_frames))
        mask = np.zeros(n_frames, dtype=np.float32)
        mask[:valid] = 1.0

        # (n_features, n_frames) -> (1, n_features, n_frames) for Conv2d.
        tensor = torch.from_numpy(np.ascontiguousarray(matrix)).unsqueeze(0).float()
        return tensor, torch.from_numpy(mask), int(self.cache.labels[row]), row


def make_dataloaders(
    cache: FeatureCache,
    splits: dict[str, Sequence[RavdessSample]],
    *,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
    augment_cfg: AugmentConfig | None = None,
):
    """Build train/val/test loaders from speaker-disjoint split buckets.

    ``num_workers=0`` is the default on purpose: the data is already a contiguous
    in-memory array, so worker processes would add pickling overhead for no gain
    and can exhaust RAM on a 7.7 GB machine.
    """
    from torch.utils.data import DataLoader

    index_by_path = {path: i for i, path in enumerate(cache.sample_paths)}

    def indices_for(name: str) -> list[int]:
        rows = []
        for sample in splits[name]:
            row = index_by_path.get(sample.path)
            if row is None:
                continue
            rows.append(row)
        return sorted(rows)

    datasets = {
        "train": EmotionDataset(
            cache,
            indices_for("train"),
            augment=True,
            augment_cfg=augment_cfg,
            seed=seed,
        ),
        "val": EmotionDataset(cache, indices_for("val"), augment=False),
        "test": EmotionDataset(cache, indices_for("test"), augment=False),
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=len(datasets["train"]) > batch_size,
            pin_memory=False,
        ),
        "val": DataLoader(
            datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
        "test": DataLoader(
            datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
    }
    return datasets, loaders


__all__ = [
    "PROCESSED_DIR_NAME",
    "AugmentConfig",
    "EmotionDataset",
    "FeatureCache",
    "augment_features",
    "build_feature_cache",
    "config_fingerprint",
    "make_dataloaders",
]
