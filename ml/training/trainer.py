"""Training loop, checkpointing and best-model selection.

Selection policy
----------------
The checkpoint is chosen on **validation macro-F1**, and the test split is
touched exactly once, after selection is frozen. Two reasons:

* macro-F1 rather than accuracy, because ``neutral`` has half as many
  utterances as every other class (96 vs 192 across the corpus - RAVDESS has no
  "strong neutral" condition), so raw accuracy over-rewards majority classes;
* validation rather than test, because choosing a model by its test score is
  how a reported number quietly stops being an estimate of generalisation.

Class imbalance is handled with frequency-weighted cross-entropy computed from
the *training* split only, plus light label smoothing.
"""

from __future__ import annotations

import contextlib
import csv
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.config import (
    EMOTION_LABELS,
    RUN_FILENAMES,
    ExperimentConfig,
    TrainConfig,
    resolve_num_threads,
)
from ml.data.dataset import AugmentConfig, EmotionDataset, FeatureCache
from ml.models.emotion_cnn_lstm import EmotionCNNBiLSTM
from ml.utils.logging_utils import get_logger
from ml.utils.metrics import compute_report

logger = get_logger("trainer")

CHECKPOINT_FILENAME = RUN_FILENAMES.checkpoint
HISTORY_FILENAME = RUN_FILENAMES.history


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_reproducible(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG that can influence a training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Older torch builds lack this switch; determinism is best-effort.
        with contextlib.suppress(AttributeError, RuntimeError):
            torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_class_weights(
    labels: np.ndarray, num_classes: int, smoothing: float = 1.0
) -> torch.Tensor:
    """Inverse-frequency weights from the training labels.

    ``w_c = (1 / n_c) ** smoothing`` normalised to mean 1, so the loss is not
    rescaled overall and the learning rate keeps its meaning. ``smoothing`` 0.5
    gives a square-root correction - gentler than full inverse frequency, which
    can over-correct a mild 2:1 imbalance into unstable gradients.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = (1.0 / counts) ** smoothing
    weights = weights / weights.mean()
    logger.info(
        "class weights: "
        + "  ".join(f"{EMOTION_LABELS[i]}={weights[i]:.3f}" for i in range(num_classes))
    )
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Epoch-level evaluation
# ---------------------------------------------------------------------------
@dataclass
class EpochResult:
    """Everything one pass over a split produced."""

    loss: float
    accuracy: float
    macro_f1: float
    weighted_f1: float
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray
    seconds: float

    def report(self, split: str, labels: list[str]):
        return compute_report(self.y_true, self.y_pred, split=split, labels=labels)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
    num_classes: int = 8,
) -> EpochResult:
    """One pass over ``loader``.

    With ``optimizer`` given, this is a training pass; without it, an evaluation
    pass. Same code path for both so the loss reported during training is
    computed exactly like the loss reported during validation.
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    start = time.perf_counter()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for features, mask, labels, _rows in loader:
            features = features.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(features, frame_mask=mask)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            probs = torch.softmax(logits.detach(), dim=1)
            predictions = probs.argmax(dim=1)
            total_correct += int((predictions == labels).sum().item())
            total_seen += batch_size

            all_true.append(labels.cpu().numpy())
            all_pred.append(predictions.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    seconds = time.perf_counter() - start
    y_true = np.concatenate(all_true) if all_true else np.zeros(0, dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.zeros(0, dtype=np.int64)
    probabilities = (
        np.concatenate(all_probs) if all_probs else np.zeros((0, num_classes), dtype=np.float32)
    )

    accuracy = total_correct / total_seen if total_seen else 0.0
    report = (
        compute_report(y_true, y_pred, split="epoch", labels=EMOTION_LABELS) if total_seen else None
    )

    return EpochResult(
        loss=total_loss / total_seen if total_seen else float("nan"),
        accuracy=accuracy,
        macro_f1=report.macro_f1 if report else 0.0,
        weighted_f1=report.weighted_f1 if report else 0.0,
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        seconds=seconds,
    )


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
@dataclass
class TrainHistory:
    rows: list[dict[str, object]] = field(default_factory=list)

    def add(self, **row: object) -> None:
        self.rows.append(row)

    def best(self, monitor: str, mode: str = "max") -> dict[str, object] | None:
        if not self.rows:
            return None
        key = lambda r: float(r[monitor])
        return max(self.rows, key=key) if mode == "max" else min(self.rows, key=key)

    def save(self, path: Path) -> None:
        if not self.rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a sibling temp file and replace, so a crash mid-write can
        # never leave a truncated CSV behind.
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)
        tmp.replace(path)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: Path,
    model: nn.Module,
    experiment: ExperimentConfig,
    *,
    epoch: int,
    monitor_value: float,
    metrics: dict[str, object],
    n_input_features: int,
) -> None:
    """Persist weights together with everything needed to rebuild the front-end.

    The saved payload is self-describing on purpose: the inference service
    reconstructs the model from ``config`` and ``n_input_features`` alone, so a
    checkpoint can never be loaded against a mismatched architecture.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "state_dict": model.state_dict(),
            "config": experiment.to_dict(),
            "labels": list(experiment.labels),
            "n_input_features": n_input_features,
            "epoch": epoch,
            "monitor": experiment.train.monitor,
            "monitor_value": monitor_value,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, ExperimentConfig, dict]:
    """Rebuild a model straight from a checkpoint file."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError(
            f"Unsupported checkpoint format {payload.get('format_version')} in '{path}'."
        )
    experiment = ExperimentConfig.from_dict(payload["config"])
    model = EmotionCNNBiLSTM(experiment.model, n_features=int(payload["n_input_features"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, experiment, payload


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def train(
    cache: FeatureCache,
    datasets: dict[str, EmotionDataset],
    loaders: dict[str, DataLoader],
    experiment: ExperimentConfig,
    run_dir: Path,
    *,
    augment_cfg: AugmentConfig | None = None,
    device_str: str = "auto",
    progress_every: int = 1,
) -> dict[str, object]:
    """Train a model and write ``best_model.pt`` + ``history.csv`` to ``run_dir``.

    Returns a summary dict (also embedded in the checkpoint).
    """
    train_cfg = experiment.train
    device = resolve_device(device_str)

    threads = resolve_num_threads(train_cfg.num_threads)
    torch.set_num_threads(threads)

    set_reproducible(train_cfg.seed, train_cfg.deterministic)

    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("run directory: %s", run_dir)
    logger.info("device: %s   torch threads: %d", device, threads)

    model = EmotionCNNBiLSTM(experiment.model, n_features=cache.n_features).to(device)
    logger.info("%s", model.describe())
    logger.info("parameter breakdown: %s", model.parameter_breakdown())

    weights = compute_class_weights(
        cache.labels[datasets["train"].indices],
        experiment.model.num_classes,
        smoothing=0.5,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=train_cfg.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=train_cfg.lr_scheduler_factor,
        patience=train_cfg.lr_scheduler_patience,
        min_lr=train_cfg.lr_scheduler_min_lr,
    )

    logger.info(
        "train=%d  val=%d  test=%d  batch_size=%d  max_epochs=%d  monitor=%s",
        len(datasets["train"]),
        len(datasets["val"]),
        len(datasets["test"]),
        train_cfg.batch_size,
        train_cfg.max_epochs,
        train_cfg.monitor,
    )

    history = TrainHistory()
    best_value = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, train_cfg.max_epochs + 1):
        train_dataset = datasets.get("train")
        if train_dataset is not None and hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)

        train_result = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            grad_clip=train_cfg.grad_clip_norm,
            num_classes=experiment.model.num_classes,
        )
        val_result = run_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            num_classes=experiment.model.num_classes,
        )

        monitored = getattr(val_result, train_cfg.monitor.replace("val_", ""))
        scheduler.step(monitored)

        improved = monitored > best_value
        if improved:
            best_value = monitored
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                run_dir / CHECKPOINT_FILENAME,
                model,
                experiment,
                epoch=epoch,
                monitor_value=float(monitored),
                metrics={
                    "train_loss": round(train_result.loss, 5),
                    "train_accuracy": round(train_result.accuracy, 5),
                    "train_macro_f1": round(train_result.macro_f1, 5),
                    "val_loss": round(val_result.loss, 5),
                    "val_accuracy": round(val_result.accuracy, 5),
                    "val_macro_f1": round(val_result.macro_f1, 5),
                },
                n_input_features=cache.n_features,
            )
        else:
            epochs_without_improvement += 1

        learning_rate = optimizer.param_groups[0]["lr"]
        history.add(
            epoch=epoch,
            lr=round(learning_rate, 8),
            train_loss=round(train_result.loss, 5),
            train_accuracy=round(train_result.accuracy, 5),
            train_macro_f1=round(train_result.macro_f1, 5),
            val_loss=round(val_result.loss, 5),
            val_accuracy=round(val_result.accuracy, 5),
            val_macro_f1=round(val_result.macro_f1, 5),
            val_weighted_f1=round(val_result.weighted_f1, 5),
            lr_group=1,
            improved=int(improved),
            epoch_seconds=round(train_result.seconds + val_result.seconds, 2),
        )

        if epoch % max(1, progress_every) == 0 or improved:
            logger.info(
                "epoch %3d/%d  lr=%.2e  train loss=%.4f acc=%.4f  "
                "val loss=%.4f acc=%.4f macroF1=%.4f  %.1fs%s",
                epoch,
                train_cfg.max_epochs,
                learning_rate,
                train_result.loss,
                train_result.accuracy,
                val_result.loss,
                val_result.accuracy,
                val_result.macro_f1,
                train_result.seconds + val_result.seconds,
                "  *" if improved else "",
            )

        # Persist every epoch. A long CPU run can be interrupted (reboot, an
        # OOM kill, a cancelled job); losing the curve would leave the README
        # and any later diagnosis with nothing to work from. One CSV rewrite of
        # <=100 rows is negligible next to an epoch of training.
        history.save(run_dir / HISTORY_FILENAME)
        _write_summary(
            run_dir,
            history=history,
            best_epoch=best_epoch,
            best_value=best_value,
            device=device,
            threads=threads,
            model=model,
            cache=cache,
            datasets=datasets,
            weights=weights,
            train_cfg=train_cfg,
            elapsed=time.perf_counter() - started,
            complete=False,
        )

        if epochs_without_improvement >= train_cfg.early_stopping_patience:
            logger.info(
                "early stopping at epoch %d: %s did not improve for %d epochs",
                epoch,
                train_cfg.monitor,
                epochs_without_improvement,
            )
            break

    total_seconds = time.perf_counter() - started
    history.save(run_dir / HISTORY_FILENAME)

    summary = _write_summary(
        run_dir,
        history=history,
        best_epoch=best_epoch,
        best_value=best_value,
        device=device,
        threads=threads,
        model=model,
        cache=cache,
        datasets=datasets,
        weights=weights,
        train_cfg=train_cfg,
        elapsed=total_seconds,
        complete=True,
    )

    logger.info(
        "training finished in %.1fs - best epoch %d, val macro-F1 %.4f",
        total_seconds,
        best_epoch,
        best_value,
    )
    logger.info("checkpoint: %s", run_dir / CHECKPOINT_FILENAME)
    return summary


def _write_summary(
    run_dir: Path,
    *,
    history: TrainHistory,
    best_epoch: int,
    best_value: float,
    device: torch.device,
    threads: int,
    model: EmotionCNNBiLSTM,
    cache: FeatureCache,
    datasets: dict[str, EmotionDataset],
    weights: torch.Tensor,
    train_cfg: TrainConfig,
    elapsed: float,
    complete: bool,
) -> dict[str, object]:
    """Write ``training_summary.json`` and return it.

    Called after every epoch as well as at the end, with ``complete=False``, so
    the run's state is always inspectable even if the process is killed.
    """
    summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(float(best_value), 6),
        "epochs_run": len(history.rows),
        "max_epochs": train_cfg.max_epochs,
        "early_stopped": complete and len(history.rows) < train_cfg.max_epochs,
        "complete": complete,
        "total_training_seconds": round(elapsed, 1),
        "seconds_per_epoch_mean": round(elapsed / max(1, len(history.rows)), 2),
        "device": str(device),
        "torch_threads": threads,
        "n_parameters": model.num_parameters(),
        "n_input_features": cache.n_features,
        "n_train": len(datasets["train"]),
        "n_val": len(datasets["val"]),
        "n_test": len(datasets["test"]),
        "class_weights": [round(float(w), 4) for w in weights.cpu().tolist()],
    }
    (run_dir / RUN_FILENAMES.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


__all__ = [
    "CHECKPOINT_FILENAME",
    "HISTORY_FILENAME",
    "EpochResult",
    "TrainHistory",
    "compute_class_weights",
    "load_checkpoint",
    "resolve_device",
    "run_epoch",
    "save_checkpoint",
    "set_reproducible",
    "train",
]
