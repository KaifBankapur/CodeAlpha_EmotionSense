"""One entry point for "give me the prepared data".

``scripts/train.py``, ``scripts/evaluate.py`` and any ablation must all see
exactly the same features, splits and augmentation settings - otherwise an
experiment is not comparable and a reported number means nothing. Rather than
trusting each script to assemble that correctly, they all call
:func:`load_prepared`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from ml.config import (
    DATA_DIR,
    AudioConfig,
    FeatureConfig,
    SplitConfig,
    resolve_ravdess_root,
)
from ml.data.dataset import (
    PROCESSED_DIR_NAME,
    AugmentConfig,
    EmotionDataset,
    FeatureCache,
    config_fingerprint,
    make_dataloaders,
)
from ml.data.ravdess import assign_splits, build_manifest

logger = logging.getLogger(__name__)

PROCESSED_DIR = DATA_DIR / PROCESSED_DIR_NAME


@dataclass
class PreparedData:
    """Everything a training or evaluation run needs, already cross-checked."""

    cache: FeatureCache
    splits: dict[str, list]
    datasets: dict[str, EmotionDataset]
    loaders: dict[str, DataLoader]
    root: Path
    audio_cfg: AudioConfig
    feature_cfg: FeatureConfig
    split_cfg: SplitConfig
    augment_cfg: AugmentConfig

    @property
    def n_features(self) -> int:
        return self.cache.n_features

    @property
    def n_frames(self) -> int:
        return self.cache.n_frames


def cache_dir_for(
    audio_cfg: AudioConfig, feature_cfg: FeatureConfig, processed_dir: Path = PROCESSED_DIR
) -> Path:
    """Where the cache for this configuration lives (may not exist yet)."""
    return processed_dir / f"features_{config_fingerprint(audio_cfg, feature_cfg)}"


def find_existing_cache(processed_dir: Path = PROCESSED_DIR) -> Path | None:
    """Most recently built feature cache, regardless of configuration."""
    candidates = sorted(
        (p for p in processed_dir.glob("features_*") if (p / "meta.json").is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def load_prepared(
    *,
    audio_cfg: AudioConfig | None = None,
    feature_cfg: FeatureConfig | None = None,
    split_cfg: SplitConfig | None = None,
    batch_size: int = 32,
    seed: int = 42,
    augment: bool = True,
    augment_cfg: AugmentConfig | None = None,
    num_workers: int = 0,
    strict_fingerprint: bool = True,
) -> PreparedData:
    """Load the feature cache and build speaker-disjoint splits and loaders.

    Args:
        strict_fingerprint: require the cache to have been built with exactly
            this audio/feature configuration. Leave this on; turning it off is
            how you end up evaluating a model on features it never saw.

    Raises:
        FileNotFoundError: when no feature cache exists - the message names the
            command that creates one.
    """
    audio_cfg = audio_cfg or AudioConfig()
    feature_cfg = feature_cfg or FeatureConfig()
    split_cfg = split_cfg or SplitConfig()
    augment_cfg = augment_cfg or AugmentConfig(enabled=augment)

    directory = cache_dir_for(audio_cfg, feature_cfg)
    if not directory.is_dir():
        fallback = find_existing_cache()
        if fallback is None:
            raise FileNotFoundError(
                "No feature cache found. Build one first:\n"
                "    python scripts/download_dataset.py\n"
                "    python scripts/prepare_data.py"
            )
        if strict_fingerprint:
            raise FileNotFoundError(
                f"No feature cache for this configuration at '{directory}'.\n"
                f"A cache exists at '{fallback.name}' but it was built with "
                "different audio/feature settings.\n"
                "Either use --target-duration/--sample-rate to match it, or "
                "rebuild:\n"
                "    python scripts/prepare_data.py --rebuild"
            )
        logger.warning("falling back to cache %s (fingerprint mismatch)", fallback.name)
        directory = fallback

    expected = config_fingerprint(audio_cfg, feature_cfg) if strict_fingerprint else None
    cache = FeatureCache.load(directory, expected_fingerprint=expected)
    logger.info(
        "loaded feature cache %s: %s, %d features x %d frames",
        directory.name,
        cache.features.shape,
        cache.n_features,
        cache.n_frames,
    )

    root = resolve_ravdess_root()
    manifest = build_manifest(root)
    splits = assign_splits(manifest, split_cfg)

    missing = [
        sample.path
        for name in splits
        for sample in splits[name]
        if sample.path not in set(cache.sample_paths)
    ]
    if missing:
        raise RuntimeError(
            f"{len(missing)} manifest entries are absent from the feature cache "
            f"(e.g. {missing[:3]}). Rebuild it with scripts/prepare_data.py."
        )

    datasets, loaders = make_dataloaders(
        cache,
        splits,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        augment_cfg=augment_cfg,
    )
    logger.info(
        "splits: train=%d val=%d test=%d",
        len(datasets["train"]),
        len(datasets["val"]),
        len(datasets["test"]),
    )

    return PreparedData(
        cache=cache,
        splits=splits,
        datasets=datasets,
        loaders=loaders,
        root=root,
        audio_cfg=audio_cfg,
        feature_cfg=feature_cfg,
        split_cfg=split_cfg,
        augment_cfg=augment_cfg,
    )


__all__ = [
    "PROCESSED_DIR",
    "PreparedData",
    "cache_dir_for",
    "find_existing_cache",
    "load_prepared",
]
