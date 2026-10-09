"""Classification metrics and the evaluation record they are stored in.

Reporting rules enforced here
-----------------------------
* **Macro and weighted F1 are both always present.** RAVDESS is imbalanced by
  construction - ``neutral`` has 96 utterances while every other emotion has
  192, because the corpus has no "strong neutral" condition. Accuracy alone
  would let a model score 50% by always guessing the majority classes.
* **Train, validation and test are separate objects.** Nothing in this module
  can merge them, so a training score can never be mistaken for a test score.
* **Support counts are always carried alongside the scores**, so a per-class F1
  of 1.00 on 8 samples cannot masquerade as a reliable number.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ml.config import EMOTION_LABELS


def _safe_divide(numerator: float, denominator: float) -> float:
    """Metrics are undefined when a class is absent from y_true; report 0.0.

    sklearn already returns 0.0 with a zero-division warning in that case. We do
    it explicitly so a class with no support is visibly 0 rather than NaN.
    """
    return float(numerator / denominator) if denominator else 0.0


@dataclass
class ClassificationReport:
    """Per-split evaluation result."""

    split: str
    labels: list[str]
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    n_samples: int = 0

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "n_samples": self.n_samples,
            "accuracy": round(self.accuracy, 6),
            "macro_precision": round(self.macro_precision, 6),
            "macro_recall": round(self.macro_recall, 6),
            "macro_f1": round(self.macro_f1, 6),
            "weighted_precision": round(self.weighted_precision, 6),
            "weighted_recall": round(self.weighted_recall, 6),
            "weighted_f1": round(self.weighted_f1, 6),
            "per_class": {
                label: {k: round(float(v), 6) for k, v in stats.items()}
                for label, stats in self.per_class.items()
            },
            "confusion_matrix": self.confusion_matrix,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    # -- human-readable ---------------------------------------------------
    def summary_line(self) -> str:
        return (
            f"{self.split:<5s} n={self.n_samples:<4d} "
            f"acc={self.accuracy * 100:5.2f}%  "
            f"macro-F1={self.macro_f1 * 100:5.2f}%  "
            f"weighted-F1={self.weighted_f1 * 100:5.2f}%"
        )

    def format_table(self) -> str:
        header = f"{'emotion':<12s} {'prec':>7s} {'rec':>7s} {'f1':>7s} {'support':>8s}"
        lines = [header, "-" * len(header)]
        for label in self.labels:
            stats = self.per_class.get(label, {})
            lines.append(
                f"{label:<12s} "
                f"{stats.get('precision', 0.0) * 100:6.2f}% "
                f"{stats.get('recall', 0.0) * 100:6.2f}% "
                f"{stats.get('f1', 0.0) * 100:6.2f}% "
                f"{int(stats.get('support', 0)):>8d}"
            )
        lines.append("-" * len(header))
        lines.append(
            f"{'macro':<12s} {self.macro_precision * 100:6.2f}% "
            f"{self.macro_recall * 100:6.2f}% {self.macro_f1 * 100:6.2f}% "
            f"{self.n_samples:>8d}"
        )
        lines.append(
            f"{'weighted':<12s} {self.weighted_precision * 100:6.2f}% "
            f"{self.weighted_recall * 100:6.2f}% {self.weighted_f1 * 100:6.2f}% "
            f"{self.n_samples:>8d}"
        )
        return "\n".join(lines)


def compute_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    split: str = "test",
    labels: Sequence[str] | None = None,
) -> ClassificationReport:
    """Build a :class:`ClassificationReport` from integer label arrays.

    Args:
        y_true: ground-truth class indices.
        y_pred: predicted class indices.
        split: name recorded in the report ("train" / "val" / "test").
        labels: class names; defaults to the canonical RAVDESS order.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    labels = list(labels or EMOTION_LABELS)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if y_true.size == 0:
        raise ValueError(f"Cannot evaluate split '{split}': it has no samples.")

    accuracy = float(accuracy_score(y_true, y_pred))

    # `labels=` pins the class order and the row order of the confusion matrix,
    # so both are independent of which classes happen to appear in y_true.
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(labels))), zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }

    def avg(values: np.ndarray, weights: np.ndarray | None = None) -> float:
        if values.size == 0:
            return 0.0
        return float(np.average(values, weights=weights))

    return ClassificationReport(
        split=split,
        labels=labels,
        accuracy=accuracy,
        macro_precision=avg(precision),
        macro_recall=avg(recall),
        macro_f1=avg(f1),
        weighted_precision=avg(precision, support),
        weighted_recall=avg(recall, support),
        weighted_f1=avg(f1, support),
        per_class=per_class,
        confusion_matrix=matrix.tolist(),
        n_samples=int(y_true.size),
    )


def error_analysis(
    report: ClassificationReport,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    probabilities: np.ndarray | None = None,
    top_n: int = 10,
) -> dict[str, object]:
    """Dig into the mistakes rather than just reporting an aggregate.

    Produces the three things that actually make an error analysis useful:
    the confusion pairs responsible for most of the loss, the per-class
    difficulty ranking, and the most confident wrong answers (which is where the
    useful, surprising findings usually are).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    matrix = np.asarray(report.confusion_matrix, dtype=np.int64)
    labels = report.labels

    # -- ranked confusion pairs ------------------------------------------
    pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                continue
            count = int(matrix[i, j])
            if count:
                support = int(matrix[i].sum())
                pairs.append(
                    {
                        "true": labels[i],
                        "predicted": labels[j],
                        "count": count,
                        "share_of_true_class": round(count / support, 4) if support else 0.0,
                    }
                )
    pairs.sort(key=lambda p: p["count"], reverse=True)

    # -- hardest classes --------------------------------------------------
    ranked = sorted(
        (
            {
                "label": label,
                "f1": round(report.per_class[label]["f1"], 4),
                "recall": round(report.per_class[label]["recall"], 4),
                "support": report.per_class[label]["support"],
            }
            for label in labels
        ),
        key=lambda item: item["f1"],
    )

    # -- most confident mistakes -----------------------------------------
    confident: list[dict[str, object]] = []
    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        wrong = np.nonzero(y_true != y_pred)[0]
        if wrong.size:
            true_conf = probabilities[wrong, y_true[wrong]]
            order = np.argsort(true_conf)[::-1][:top_n]
            for position in order:
                index = int(wrong[position])
                confident.append(
                    {
                        "index": index,
                        "true_label": labels[int(y_true[index])],
                        "predicted": labels[int(y_pred[index])],
                        "confidence_in_true_class": round(float(true_conf[position]), 4),
                        "confidence_in_predicted": round(
                            float(probabilities[index, y_pred[index]]), 4
                        ),
                    }
                )

    correct_confidence = None
    if probabilities is not None:
        mask = y_true == y_pred
        if mask.any():
            correct_confidence = float(probabilities[mask, y_true[mask]].mean())

    return {
        "n_errors": int((y_true != y_pred).sum()),
        "error_rate": round(float((y_true != y_pred).mean()), 4),
        "top_confusions": pairs[:10],
        "hardest_classes": ranked,
        "easiest_classes": list(reversed(ranked)),
        "most_confident_mistakes": confident,
        "mean_confidence_when_correct": (
            round(correct_confidence, 4) if correct_confidence is not None else None
        ),
        "overconfidence_gap": (
            round(report.accuracy - correct_confidence, 4)
            if correct_confidence is not None
            else None
        ),
    }


__all__ = ["ClassificationReport", "compute_report", "error_analysis"]
