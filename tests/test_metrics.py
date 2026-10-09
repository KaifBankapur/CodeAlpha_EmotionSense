"""Metric computation.

Validated against scikit-learn itself rather than against hand-written
expectations, so a refactor of the averaging logic cannot quietly change what
"macro-F1" means.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from ml.config import EMOTION_LABELS
from ml.utils.metrics import ClassificationReport, compute_report, error_analysis

N_CLASSES = len(EMOTION_LABELS)


@pytest.fixture
def imbalanced_sample():
    """Mirrors RAVDESS: neutral has half the samples of everything else."""
    rng = np.random.default_rng(0)
    y_true = np.concatenate(
        [rng.integers(0, N_CLASSES, size=(16 if i == 0 else 32,)) for i in range(N_CLASSES)]
    )
    y_pred = np.where(
        rng.random(y_true.size) < 0.6, y_true, rng.integers(0, N_CLASSES, y_true.size)
    )
    return y_true, y_pred


class TestComputeReport:
    def test_matches_sklearn(self, imbalanced_sample):
        y_true, y_pred = imbalanced_sample
        report = compute_report(y_true, y_pred)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(N_CLASSES)), zero_division=0
        )

        assert report.accuracy == pytest.approx(accuracy_score(y_true, y_pred))
        assert report.macro_precision == pytest.approx(precision.mean())
        assert report.macro_recall == pytest.approx(recall.mean())
        assert report.macro_f1 == pytest.approx(f1.mean())
        assert report.weighted_f1 == pytest.approx(np.average(f1, weights=support))

    def test_confusion_matrix_matches_sklearn(self, imbalanced_sample):
        y_true, y_pred = imbalanced_sample
        report = compute_report(y_true, y_pred)
        expected = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
        assert np.array_equal(np.asarray(report.confusion_matrix), expected)

    def test_every_class_is_always_present(self):
        """Even a class with zero support must appear, with visible 0 scores.

        Omitting it would let a broken class disappear from the report instead
        of showing up as an explicit failure.
        """
        y_true = np.array([0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 2, 2, 2])
        report = compute_report(y_true, y_pred)

        assert list(report.per_class) == EMOTION_LABELS
        for label in EMOTION_LABELS:
            assert report.per_class[label]["support"] >= 0

        absent = EMOTION_LABELS[5]
        assert report.per_class[absent]["support"] == 0
        assert report.per_class[absent]["f1"] == 0.0  # visible zero, not NaN

    def test_perfect_predictions_score_one(self):
        y_true = np.arange(N_CLASSES)
        report = compute_report(y_true, y_true)
        assert report.accuracy == 1.0
        assert report.macro_f1 == 1.0
        assert report.weighted_f1 == 1.0

    def test_all_wrong_scores_zero(self):
        y_true = np.zeros(8, dtype=int)
        y_pred = np.ones(8, dtype=int)
        report = compute_report(y_true, y_pred)
        assert report.accuracy == 0.0
        assert report.macro_f1 == 0.0

    def test_macro_differs_from_weighted_on_imbalanced_data(self, imbalanced_sample):
        """If these were equal, macro-F1 would be adding nothing."""
        y_true, y_pred = imbalanced_sample
        report = compute_report(y_true, y_pred)
        assert abs(report.macro_f1 - report.weighted_f1) > 1e-6

    def test_empty_split_is_rejected(self):
        with pytest.raises(ValueError, match="no samples"):
            compute_report([], [])

    def test_n_samples_is_recorded(self, imbalanced_sample):
        y_true, y_pred = imbalanced_sample
        assert compute_report(y_true, y_pred).n_samples == y_true.size

    def test_split_name_is_carried(self, imbalanced_sample):
        y_true, y_pred = imbalanced_sample
        assert compute_report(y_true, y_pred, split="val").split == "val"

    def test_custom_labels_are_honoured(self):
        report = compute_report([0, 1], [0, 1], labels=["a", "b"])
        assert report.labels == ["a", "b"]
        assert len(report.confusion_matrix) == 2


class TestSerialisation:
    def test_to_dict_is_json_safe(self, imbalanced_sample, tmp_path):
        import json

        y_true, y_pred = imbalanced_sample
        report = compute_report(y_true, y_pred, split="test")
        path = tmp_path / "metrics.json"
        report.save(path)

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["split"] == "test"
        assert payload["n_samples"] == int(y_true.size)
        assert set(payload) >= {"accuracy", "macro_f1", "weighted_f1", "per_class"}

    def test_summary_line_mentions_all_three_metrics(self, imbalanced_sample):
        y_true, y_pred = imbalanced_sample
        line = compute_report(y_true, y_pred).summary_line()
        assert "acc=" in line and "macro-F1=" in line and "weighted-F1=" in line

    def test_format_table_has_a_row_per_class(self, imbalanced_sample):
        y_true, y_pred = imbalanced_sample
        table = compute_report(y_true, y_pred).format_table()
        for label in EMOTION_LABELS:
            assert label in table
        assert "macro" in table and "weighted" in table


class TestErrorAnalysis:
    def test_counts_errors(self, imbalanced_sample):
        y_true, y_pred = imbalanced_sample
        report = compute_report(y_true, y_pred)
        analysis = error_analysis(report, y_true, y_pred)

        expected = int((y_true != y_pred).sum())
        assert analysis["n_errors"] == expected
        assert analysis["error_rate"] == pytest.approx(expected / y_true.size, abs=1e-4)

    def test_confusion_pairs_are_ranked_and_exclude_the_diagonal(self, imbalanced_sample):
        y_true, y_pred = imbalanced_sample
        report = compute_report(y_true, y_pred)
        pairs = error_analysis(report, y_true, y_pred)["top_confusions"]

        assert pairs == sorted(pairs, key=lambda p: p["count"], reverse=True)
        for pair in pairs:
            assert pair["true"] != pair["predicted"]
            assert pair["count"] > 0

    def test_hardest_classes_ascend_by_f1(self, imbalanced_sample):
        y_true, y_pred = imbalanced_sample
        report = compute_report(y_true, y_pred)
        analysis = error_analysis(report, y_true, y_pred)

        f1s = [item["f1"] for item in analysis["hardest_classes"]]
        assert f1s == sorted(f1s)
        assert analysis["easiest_classes"][0]["f1"] == max(f1s)

    def test_confident_mistakes_are_actually_wrong(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([1, 1, 3, 3])
        # Deliberately overconfident on the wrong classes.
        probabilities = np.array(
            [
                [0.05, 0.95, 0.0, 0.0],
                [0.2, 0.8, 0.0, 0.0],
                [0.0, 0.0, 0.3, 0.7],
                [0.1, 0.0, 0.0, 0.9],
            ]
        )
        report = compute_report(y_true, y_pred, labels=[str(i) for i in range(4)])
        analysis = error_analysis(report, y_true, y_pred, probabilities=probabilities)

        for mistake in analysis["most_confident_mistakes"]:
            assert mistake["true_label"] != mistake["predicted"]
            assert mistake["confidence_in_true_class"] <= mistake["confidence_in_predicted"]

    def test_works_without_probabilities(self, imbalanced_sample):
        y_true, y_pred = imbalanced_sample
        report = compute_report(y_true, y_pred)
        analysis = error_analysis(report, y_true, y_pred)

        assert analysis["most_confident_mistakes"] == []
        assert analysis["mean_confidence_when_correct"] is None
        assert analysis["overconfidence_gap"] is None


class TestReportIsInert:
    def test_reports_are_independent_objects(self):
        """Two evaluations must not be able to alias each other."""
        a = compute_report(np.array([0, 1]), np.array([0, 1]), split="train")
        b = compute_report(np.array([0, 1]), np.array([1, 0]), split="test")

        assert a is not b
        assert a.split != b.split
        assert a.accuracy != b.accuracy

    def test_dataclass_defaults_are_empty(self):
        report = ClassificationReport(
            split="x",
            labels=["a"],
            accuracy=0.5,
            macro_precision=0.5,
            macro_recall=0.5,
            macro_f1=0.5,
            weighted_precision=0.5,
            weighted_recall=0.5,
            weighted_f1=0.5,
        )
        assert report.per_class == {}
        assert report.confusion_matrix == []
