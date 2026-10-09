"""
Figures for development and documentation.

Every function here renders something that explains a result rather than
decorating it: what the front-end sees, how the loss behaved, where the classes
get confused, and how the data is distributed. The PNGs land in
``artifacts/figures/`` and are referenced from the README.

Uses the ``Agg`` backend so this works headless.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ml.config import EMOTION_LABELS, FIGURES_DIR

logger = logging.getLogger(__name__)

# Single calm palette - readable in light mode, no neon.
PALETTE = ["#4C6EF5", "#12B886", "#F59F00", "#E8590C", "#AE3EC9", "#D6336C", "#1C7ED6", "#5C7C99"]
ACCENT = "#4C6EF5"
MUTED = "#868E96"


def _save(figure, path: Path, *, dpi: int = 140) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    logger.info("wrote figure %s", path)
    return path


# ---------------------------------------------------------------------------
# Signal / feature visualisation
# ---------------------------------------------------------------------------
def plot_audio_and_mfcc(
    waveform: np.ndarray,
    sample_rate: int,
    mfcc: np.ndarray,
    *,
    title: str = "",
    label: str | None = None,
    path: Path | None = None,
) -> Path:
    """Waveform + static MFCC heat map for one utterance.

    This is the figure that makes the "MFCC" requirement tangible: it shows the
    preprocessed 3 s mono waveform the model receives, and the cepstral
    representation stacked on top of it.
    """
    path = path or FIGURES_DIR / "example_audio_and_mfcc.png"
    time = np.arange(waveform.size) / sample_rate

    figure, (ax_wave, ax_mfcc) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [1, 2]}
    )

    ax_wave.plot(time, waveform, linewidth=0.6, color=ACCENT)
    ax_wave.set_ylabel("amplitude")
    ax_wave.set_title(
        f"Preprocessed waveform  |  {waveform.size / sample_rate:.2f} s @ {sample_rate} Hz"
        + (f"  |  label: {label}" if label else "")
    )

    frames = mfcc.shape[1]
    extent = [0, waveform.size / sample_rate, mfcc.shape[0], 0]
    ax_mfcc.imshow(
        mfcc,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="magma",
        interpolation="nearest",
    )
    ax_mfcc.set_ylabel("cepstral coefficient")
    ax_mfcc.set_xlabel("time (s)")
    ax_mfcc.set_title(
        f"MFCC ({mfcc.shape[0]} planes = 40 MFCC + 40 delta + 40 delta-delta, "
        f"CMVN applied)  |  {frames} frames"
    )

    if title:
        figure.suptitle(title)
    return _save(figure, path)


def plot_log_mel(
    log_mel: np.ndarray,
    sample_rate: int,
    hop_length: int,
    *,
    label: str | None = None,
    path: Path | None = None,
) -> Path:
    """Log-mel spectrogram - the representation the DCT compresses into MFCCs."""
    path = path or FIGURES_DIR / "example_log_mel.png"
    duration = log_mel.shape[1] * hop_length / sample_rate

    figure, ax = plt.subplots(figsize=(11, 4))
    image = ax.imshow(
        log_mel,
        aspect="auto",
        origin="lower",
        extent=[0, duration, 0, log_mel.shape[0]],
        cmap="magma",
        interpolation="nearest",
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mel band")
    ax.set_title(
        f"Log-mel spectrogram  |  {log_mel.shape[0]} mel bands"
        + (f"  |  label: {label}" if label else "")
    )
    figure.colorbar(image, ax=ax, label="log energy")
    return _save(figure, path)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
def plot_training_curves(
    history_csv: Path, *, path: Path | None = None, run_name: str = ""
) -> Path:
    """Loss and macro-F1 for train and validation, with the best epoch marked."""
    path = path or FIGURES_DIR / "training_curves.png"

    with history_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {history_csv}")

    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_loss = [float(r["val_loss"]) for r in rows]
    train_f1 = [float(r["train_macro_f1"]) for r in rows]
    val_f1 = [float(r["val_macro_f1"]) for r in rows]

    best_index = int(np.argmax(val_f1))
    best_epoch = epochs[best_index]

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(epochs, train_loss, label="train", color=PALETTE[0], linewidth=1.6)
    axes[0].plot(epochs, val_loss, label="validation", color=PALETTE[1], linewidth=1.6)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, train_f1, label="train", color=PALETTE[0], linewidth=1.6)
    axes[1].plot(epochs, val_f1, label="validation", color=PALETTE[1], linewidth=1.6)
    axes[1].axvline(
        best_epoch, color=MUTED, linestyle="--", linewidth=1, label=f"selected epoch {best_epoch}"
    )
    axes[1].scatter([best_epoch], [val_f1[best_index]], color=PALETTE[3], zorder=5, s=45)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("macro F1")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Macro F1  (model-selection metric)")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    if run_name:
        figure.suptitle(f"Training run: {run_name}")
    return _save(figure, path)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    labels: Sequence[str],
    *,
    split: str = "test",
    accuracy: float | None = None,
    macro_f1: float | None = None,
    path: Path | None = None,
) -> Path:
    """Row-normalised confusion matrix.

    Rows are the true class, columns the prediction. Rows are normalised so the
    figure answers "when the model hears X, what does it say?" even though
    ``neutral`` has half the samples of every other class.
    """
    path = path or FIGURES_DIR / f"confusion_matrix_{split}.png"
    array = np.asarray(matrix, dtype=np.float64)
    totals = array.sum(axis=1, keepdims=True)
    normalised = np.divide(array, totals, out=np.zeros_like(array), where=totals > 0)

    figure, ax = plt.subplots(figsize=(8.5, 7))
    image = ax.imshow(normalised, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")

    threshold = normalised.max() / 2.0 if normalised.size else 0.5
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            value = normalised[i, j]
            ax.text(
                j,
                i,
                f"{value * 100:.0f}%\n({int(array[i, j])})",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > threshold else "#212529",
            )

    subtitle = f"split: {split}"
    if accuracy is not None:
        subtitle += f"   accuracy: {accuracy * 100:.2f}%"
    if macro_f1 is not None:
        subtitle += f"   macro F1: {macro_f1 * 100:.2f}%"
    ax.set_title(f"Confusion matrix  |  {subtitle}")
    figure.colorbar(image, ax=ax, label="share of true class")
    return _save(figure, path)


# ---------------------------------------------------------------------------
# Data distribution
# ---------------------------------------------------------------------------
def plot_class_distribution(
    counts_by_split: dict[str, dict[str, int]], *, path: Path | None = None
) -> Path:
    """Class counts per split - shows the imbalance and the speaker-disjoint sizes."""
    path = path or FIGURES_DIR / "class_distribution.png"
    splits = list(counts_by_split)
    labels = EMOTION_LABELS
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(splits))

    figure, ax = plt.subplots(figsize=(11, 4.5))
    for index, split in enumerate(splits):
        values = [counts_by_split[split].get(label, 0) for label in labels]
        bars = ax.bar(
            x + index * width - 0.4 + width / 2,
            values,
            width,
            label=split,
            color=PALETTE[index % len(PALETTE)],
        )
        ax.bar_label(bars, padding=2, fontsize=8, color=MUTED)

    ax.set_xticks(x, labels)
    ax.set_ylabel("utterances")
    ax.set_title(
        "Class distribution per split  (neutral has half the samples: RAVDESS "
        "has no 'strong neutral' condition)"
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    return _save(figure, path)


# ---------------------------------------------------------------------------
# Duration distribution
# ---------------------------------------------------------------------------
def plot_duration_distribution(
    durations: Sequence[float],
    *,
    window: float | None = None,
    path: Path | None = None,
) -> Path:
    """Histogram of post-trim durations, with the analysis window marked."""
    path = path or FIGURES_DIR / "duration_distribution.png"
    array = np.asarray(durations, dtype=np.float64)

    figure, ax = plt.subplots(figsize=(10, 4))
    ax.hist(array, bins=40, color=ACCENT, alpha=0.85, edgecolor="white", linewidth=0.4)
    if window:
        ax.axvline(
            window,
            color=PALETTE[3],
            linestyle="--",
            linewidth=1.6,
            label=f"analysis window = {window:g}s",
        )
        ax.legend(frameon=False)
    ax.set_xlabel("duration after silence trimming (s)")
    ax.set_ylabel("utterances")
    ax.set_title(
        f"Utterance duration (n={array.size}, mean {array.mean():.2f}s, "
        f"median {np.median(array):.2f}s)"
    )
    ax.grid(axis="y", alpha=0.25)
    return _save(figure, path)


def plot_per_class_f1(
    report_dict: dict[str, object],
    *,
    split: str = "test",
    path: Path | None = None,
) -> Path:
    """Per-class F1/precision/recall bars - makes the weakest classes obvious."""
    path = path or FIGURES_DIR / f"per_class_f1_{split}.png"
    per_class: dict[str, dict[str, float]] = report_dict["per_class"]  # type: ignore[assignment]
    labels = list(per_class)

    precision = [per_class[name]["precision"] for name in labels]
    recall = [per_class[name]["recall"] for name in labels]
    f1 = [per_class[name]["f1"] for name in labels]

    x = np.arange(len(labels))
    width = 0.26
    figure, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - width, precision, width, label="precision", color=PALETTE[0])
    ax.bar(x, recall, width, label="recall", color=PALETTE[1])
    ax.bar(x + width, f1, width, label="F1", color=PALETTE[2])
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(
        f"Per-class performance ({split})  |  accuracy "
        f"{report_dict['accuracy'] * 100:.2f}%   macro F1 "
        f"{report_dict['macro_f1'] * 100:.2f}%"
    )
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    return _save(figure, path)


def generate_run_figures(
    run_dir: Path,
    metrics: dict,
    *,
    split: str = "test",
    include_dataset: bool = True,
) -> list[Path]:
    """Render every figure the documentation needs for one evaluated run.

    Args:
        run_dir: the run directory, which holds ``history.csv``.
        metrics: the ``metrics.json`` payload written by
            :mod:`ml.training.evaluate`. Split actors, audio and feature
            settings are read from it, so the illustrations always describe the
            run being documented rather than the current defaults.
        split: which split to draw the performance figures for.
        include_dataset: also draw the corpus-level figures (class balance,
            duration distribution, one waveform with its MFCC and log-mel).

    Returns the paths written. Anything that cannot be rendered - a missing
    history file, an absent feature cache - is logged and skipped rather than
    aborting: losing one illustration is no reason to throw away an evaluation
    that has already been computed.
    """
    written: list[Path] = []

    history = run_dir / "history.csv"
    if history.is_file():
        written.append(_attempt(plot_training_curves, history, run_name=run_dir.name))
    else:
        logger.warning("%s has no history.csv; skipping training curves", run_dir.name)

    report = (metrics.get("reports") or {}).get(split)
    if report:
        written += _performance_figures(metrics, report, split)
    else:
        logger.warning("metrics have no '%s' report; skipping performance figures", split)

    if include_dataset:
        written += _corpus_figures(metrics)

    return [path for path in written if path is not None]


def _performance_figures(metrics: dict, report: dict, split: str) -> list[Path | None]:
    return [
        _attempt(
            plot_confusion_matrix,
            report["confusion_matrix"],
            metrics["labels"],
            split=split,
            accuracy=report["accuracy"],
            macro_f1=report["macro_f1"],
        ),
        _attempt(plot_per_class_f1, report, split=split),
    ]


def _corpus_figures(metrics: dict) -> list[Path | None]:
    figures: list[Path | None] = [
        _attempt(plot_class_distribution, _split_class_counts(metrics)),
    ]
    hop_length = _int(metrics, "features", "hop_length", default=160)
    sample_rate = _int(metrics, "audio", "sample_rate", default=16000)
    window = _float(metrics, "audio", "target_duration_sec")

    durations = _trimmed_durations(hop_length, sample_rate)
    if durations:
        figures.append(_attempt(plot_duration_distribution, durations, window=window))
    else:
        logger.warning("no valid-frame counts available; skipping the duration figure")

    example = _example_utterance(metrics)
    if example is None:
        logger.warning("no readable test utterance; skipping the signal illustrations")
        return figures

    waveform, rate, label, mfcc, log_mel = example
    figures.append(_attempt(plot_audio_and_mfcc, waveform, rate, mfcc, label=label))
    figures.append(_attempt(plot_log_mel, log_mel, rate, hop_length, label=label))
    return figures


def _attempt(fn, *args, **kwargs) -> Path | None:
    """Call a plot function, logging rather than propagating a failure.

    A backend problem or a single unreadable file should cost one figure, not
    the whole set.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - depends on the matplotlib backend
        logger.warning("could not render %s: %s", getattr(fn, "__name__", fn), exc)
        return None


def _int(metrics: dict, section: str, key: str, *, default: int) -> int:
    return int((metrics.get(section) or {}).get(key, default))


def _float(metrics: dict, section: str, key: str) -> float | None:
    value = (metrics.get(section) or {}).get(key)
    return None if value is None else float(value)


def _split_class_counts(metrics: dict) -> dict[str, dict[str, int]]:
    """Per-emotion counts per split, read from the prepared manifest.

    The manifest is the same artefact the training pipeline split on, so the
    figure cannot drift from the split that was actually used.
    """
    from ml.data.loading import PROCESSED_DIR

    manifest_path = PROCESSED_DIR / "manifest.json"
    if not manifest_path.is_file():
        logger.warning("no manifest at %s; skipping the class figure", manifest_path)
        return {}

    try:
        samples = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("cannot read %s: %s", manifest_path, exc)
        return {}

    split = metrics.get("split") or {}
    test_actors = set(split.get("test_actors") or ())
    val_actors = set(split.get("val_actors") or ())
    if not (test_actors and val_actors):
        logger.warning("metrics record no split actors; skipping the class figure")
        return {}

    counts: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    for sample in samples:
        actor = sample.get("actor")
        if actor is None:
            continue
        bucket = "test" if actor in test_actors else "val" if actor in val_actors else "train"
        label = sample["label"]
        counts[bucket][label] = counts[bucket].get(label, 0) + 1

    return {name: per_class for name, per_class in counts.items() if per_class}


def _trimmed_durations(hop_length: int, sample_rate: int) -> list[float]:
    """Post-trim duration of every utterance, from the feature cache.

    The cache stores ``valid_frames`` - the count of leading frames holding real
    signal rather than padding - and one frame lasts ``hop_length /
    sample_rate`` seconds. That is exactly the trimmed duration, so this reads a
    6 kB array instead of re-decoding 1440 files.
    """
    from ml.data.loading import find_existing_cache

    cache_dir = find_existing_cache()
    if cache_dir is None:
        return []

    try:
        valid_frames = np.load(cache_dir / "valid_frames.npy")
    except (OSError, ValueError) as exc:
        logger.warning("cannot read valid_frames from %s: %s", cache_dir, exc)
        return []

    seconds_per_frame = hop_length / float(sample_rate)
    return (valid_frames.astype(np.float64) * seconds_per_frame).tolist()


def _example_utterance(
    metrics: dict,
) -> tuple[np.ndarray, int, str, np.ndarray, np.ndarray] | None:
    """One real test-split utterance with its MFCC and log-mel views.

    Returns ``(waveform, sample_rate, label, mfcc, log_mel)``, or ``None`` when
    the corpus is not on disk. A *test* utterance is used deliberately: the
    documentation should illustrate held-out audio, not audio the model has
    already fitted.
    """
    from ml.config import ExperimentConfig
    from ml.data.preprocessing import load_and_prepare
    from ml.data.ravdess import build_manifest, resolve_ravdess_root
    from ml.features.mfcc import MFCCExtractor, log_mel_spectrogram

    try:
        root = resolve_ravdess_root()
    except Exception as exc:  # pragma: no cover - depends on the corpus being present
        logger.warning("cannot locate the RAVDESS corpus: %s", exc)
        return None

    test_actors = set((metrics.get("split") or {}).get("test_actors") or ())
    test = [s for s in build_manifest(root) if s.actor in test_actors]
    if not test:
        logger.warning("no test-split utterance found; metrics list no usable test actors")
        return None

    config_path = Path(metrics.get("run_dir", "")) / "config.json"
    experiment = ExperimentConfig.load(config_path) if config_path.is_file() else ExperimentConfig()
    sample = test[0]
    try:
        waveform, info = load_and_prepare(root / sample.path, experiment.audio)
        extractor = MFCCExtractor(experiment.features)
        mfcc = extractor.transform(waveform, info.sample_rate)
        log_mel = log_mel_spectrogram(waveform, info.sample_rate, experiment.features)
    except Exception as exc:  # pragma: no cover - depends on the corpus being present
        logger.warning("cannot process %s: %s", sample.path, exc)
        return None

    return waveform, info.sample_rate, sample.label, mfcc, log_mel


__all__ = [
    "FIGURES_DIR",
    "generate_run_figures",
    "plot_audio_and_mfcc",
    "plot_class_distribution",
    "plot_confusion_matrix",
    "plot_duration_distribution",
    "plot_log_mel",
    "plot_per_class_f1",
    "plot_training_curves",
]
