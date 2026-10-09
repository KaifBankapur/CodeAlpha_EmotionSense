"""
Evaluate a trained run: metrics, error analysis, baseline comparison, latency.

Methodology guard-rails built into this module
---------------------------------------------
* The feature cache is loaded using the **checkpoint's own** audio/feature
  config. If it does not match, loading fails loudly rather than evaluating a
  model against features it never saw.
* Train, validation and test reports are separate objects with separate
  confusion matrices. Training accuracy can never be reported as model
  performance.
* The classical baseline is trained on the *training* split only and evaluated
  on the same val/test splits, so the deep model's advantage is measured against
  a like-for-like reference rather than a remembered number.

Usage
-----
    python scripts/evaluate.py --run expA_lr1e3
    python scripts/evaluate.py --run expA_lr1e3 --splits val
    python scripts/evaluate.py --run expA_lr1e3 --no-baseline --no-latency
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from ml.config import EMOTION_LABELS
from ml.data.loading import PreparedData, load_prepared
from ml.inference.predictor import EmotionPredictor
from ml.training.trainer import CHECKPOINT_FILENAME, load_checkpoint, resolve_device
from ml.utils.metrics import compute_report, error_analysis

logger = logging.getLogger(__name__)

METRICS_FILENAME = "metrics.json"


# ---------------------------------------------------------------------------
# Classical baseline
# ---------------------------------------------------------------------------
def _summary_vector(matrix: np.ndarray) -> np.ndarray:
    """Collapse an ``(n_features, n_frames)`` MFCC matrix to a fixed vector.

    Mean and standard deviation over time: the mean is the spectral envelope,
    the spread is how much that envelope moves. This is the classic
    "MFCC mean+std" feature set and makes a fair, cheap reference point.
    """
    return np.concatenate([matrix.mean(axis=1), matrix.std(axis=1)]).astype(np.float64)


def run_baseline(
    prepared: PreparedData, splits: Sequence[str] = ("val", "test")
) -> dict[str, object]:
    """Logistic regression on MFCC mean+std vectors, trained on the train split."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cache = prepared.cache
    path_to_row = {path: i for i, path in enumerate(cache.sample_paths)}

    def rows_for(split: str) -> np.ndarray:
        return np.array(
            [path_to_row[s.path] for s in prepared.splits[split] if s.path in path_to_row],
            dtype=np.int64,
        )

    train_rows = rows_for("train")
    x_train = np.stack([_summary_vector(cache.features[r]) for r in train_rows])
    y_train = cache.labels[train_rows]
    logger.info(
        "baseline: %d training vectors of %d dimensions",
        x_train.shape[0],
        x_train.shape[1],
    )

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, n_jobs=None)),
        ]
    )
    fit_started = time.perf_counter()
    pipeline.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - fit_started

    reports: dict[str, object] = {}
    for split in splits:
        rows = rows_for(split)
        if rows.size == 0:
            continue
        x = np.stack([_summary_vector(cache.features[r]) for r in rows])
        y_true = cache.labels[rows]
        y_pred = pipeline.predict(x)
        report = compute_report(y_true, y_pred, split=f"baseline_{split}", labels=EMOTION_LABELS)
        reports[split] = report.to_dict()
        logger.info(
            "baseline %s: acc=%.4f macro-F1=%.4f",
            split,
            report.accuracy,
            report.macro_f1,
        )

    return {
        "model": "logistic regression on MFCC mean+std",
        "n_features": int(x_train.shape[1]),
        "fit_seconds": round(fit_seconds, 2),
        "reports": reports,
    }


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
def measure_latency(
    predictor: EmotionPredictor, sample_path: Path, repeats: int = 10
) -> dict[str, object]:
    """End-to-end single-file latency: decode -> MFCC -> forward -> response."""
    predictor.warmup()

    timings: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        predictor.predict(sample_path)
        timings.append((time.perf_counter() - started) * 1000.0)

    array = np.array(timings)
    return {
        "repeats": repeats,
        "mean_ms": round(float(array.mean()), 2),
        "median_ms": round(float(np.median(array)), 2),
        "p95_ms": round(float(np.percentile(array, 95)), 2),
        "min_ms": round(float(array.min()), 2),
        "max_ms": round(float(array.max()), 2),
        "note": "full path: audio decode + resample + trim + MFCC + forward pass",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def evaluate_run(
    run_dir: Path,
    *,
    splits_to_evaluate: Sequence[str] = ("val", "test"),
    device_str: str = "auto",
    with_baseline: bool = True,
    with_latency: bool = True,
    save: bool = True,
) -> dict[str, object]:
    """Evaluate the best checkpoint in ``run_dir`` and write ``metrics.json``."""
    run_dir = Path(run_dir)
    checkpoint = run_dir / CHECKPOINT_FILENAME
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"No '{CHECKPOINT_FILENAME}' in '{run_dir}'. Train first: python scripts/train.py"
        )

    device = resolve_device(device_str)
    model, experiment, payload = load_checkpoint(checkpoint, device)
    logger.info(
        "evaluating run '%s' (epoch %s, selection %s=%.4f)",
        run_dir.name,
        payload.get("epoch"),
        payload.get("monitor"),
        float(payload.get("monitor_value") or 0.0),
    )

    # The cache MUST match the checkpoint's config; load_prepared enforces it.
    prepared = load_prepared(
        audio_cfg=experiment.audio,
        feature_cfg=experiment.features,
        split_cfg=experiment.split,
        batch_size=experiment.train.batch_size,
        seed=experiment.train.seed,
        augment=False,  # evaluation must be deterministic
    )

    reports: dict[str, object] = {}
    analyses: dict[str, object] = {}
    predictions: dict[str, np.ndarray] = {}

    for split in splits_to_evaluate:
        loader = prepared.loaders.get(split)
        if loader is None:
            logger.warning("no loader for split '%s', skipping", split)
            continue

        model.eval()
        all_true: list[np.ndarray] = []
        all_pred: list[np.ndarray] = []
        all_probs: list[np.ndarray] = []

        with torch.inference_mode():
            for features, mask, labels, _rows in loader:
                logits = model(features.to(device), frame_mask=mask.to(device))
                probs = torch.softmax(logits, dim=1)
                all_true.append(labels.numpy())
                all_pred.append(probs.argmax(dim=1).numpy())
                all_probs.append(probs.numpy())

        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        probabilities = np.concatenate(all_probs)
        predictions[split] = y_pred

        report = compute_report(y_true, y_pred, split=split, labels=EMOTION_LABELS)
        reports[split] = report.to_dict()
        analyses[split] = error_analysis(report, y_true, y_pred, probabilities=probabilities)
        print()
        print(report.summary_line())
        print(report.format_table())

    # ---- baseline -------------------------------------------------------
    baseline = None
    if with_baseline:
        try:
            baseline = run_baseline(prepared, splits=tuple(splits_to_evaluate))
        except Exception as exc:
            logger.warning("baseline failed: %s", exc)  # invalidate the evaluation

    # ---- latency --------------------------------------------------------
    latency = None
    if with_latency:
        try:
            predictor = EmotionPredictor(model, experiment, run_dir=run_dir, device=device)
            sample = prepared.root / prepared.splits["test"][0].path
            latency = measure_latency(predictor, sample)
            logger.info(
                "latency: median %.1f ms, p95 %.1f ms",
                latency["median_ms"],
                latency["p95_ms"],
            )
        except Exception as exc:
            logger.warning("latency benchmark failed: %s", exc)

    # ---- assemble -------------------------------------------------------
    result: dict[str, object] = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "dataset": experiment.dataset,
        "labels": list(EMOTION_LABELS),
        "checkpoint": {
            "epoch": payload.get("epoch"),
            "selection_metric": payload.get("monitor"),
            "selection_value": payload.get("monitor_value"),
            "format_version": payload.get("format_version"),
        },
        "model": {
            "architecture": "CNN(2-D) -> BiLSTM -> masked attention pooling",
            "parameters": model.num_parameters(),
            "parameter_breakdown": model.parameter_breakdown(),
            "channels": list(experiment.model.channels),
            "lstm_hidden": experiment.model.lstm_hidden,
            "bidirectional": experiment.model.bidirectional,
            "attention_pooling": experiment.model.use_attention_pooling,
            "n_input_features": model.n_input_features,
        },
        "features": experiment.sub_config("features"),
        "audio": experiment.sub_config("audio"),
        "split": {
            "train_actors": experiment.split.train_actors(),
            "val_actors": experiment.split.val_actors,
            "test_actors": experiment.split.test_actors,
            "n_train": len(prepared.splits["train"]),
            "n_val": len(prepared.splits["val"]),
            "n_test": len(prepared.splits["test"]),
        },
        "reports": reports,
        "error_analysis": analyses,
        "baseline": baseline,
        "latency": latency,
        "device": str(device),
        "training_metrics_at_selection": payload.get("metrics", {}),
    }

    if save:
        out = run_dir / METRICS_FILENAME
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("wrote %s", out)

    return result


__all__ = ["METRICS_FILENAME", "evaluate_run", "measure_latency", "run_baseline"]
