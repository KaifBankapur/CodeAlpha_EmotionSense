"""Figures.

Plotting code is easy to leave broken because nothing depends on it at
runtime. These tests assert the figures are actually written and that the
numbers drawn in them are the numbers in the metrics - a chart that silently
plots the wrong array is worse than no chart.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from ml.utils import plotting
from ml.utils.plotting import (
    generate_run_figures,
    plot_audio_and_mfcc,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_duration_distribution,
    plot_log_mel,
    plot_per_class_f1,
    plot_training_curves,
)

LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


@pytest.fixture(autouse=True)
def figures_dir(tmp_path, monkeypatch):
    """Redirect FIGURES_DIR so tests never touch the real artifacts folder."""
    monkeypatch.setattr(plotting, "FIGURES_DIR", tmp_path / "figures")
    plotting.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return plotting.FIGURES_DIR


@pytest.fixture
def captured(monkeypatch):
    """Collect every figure just before ``_save`` closes it.

    ``_save`` calls ``plt.close``, so ``plt.gcf()`` after a plot returns an
    unrelated empty figure. Wrapping ``_save`` is the only way to inspect what
    was actually drawn.
    """
    figures: list = []
    original = plotting._save

    def spy(figure, path, **kwargs):
        figures.append(figure)
        return original(figure, path, **kwargs)

    monkeypatch.setattr(plotting, "_save", spy)
    return figures


@pytest.fixture
def report() -> dict:
    """A report shaped exactly like one written by ml.training.evaluate."""
    return {
        "split": "test",
        "n_samples": 8,
        "accuracy": 0.5,
        "macro_f1": 0.4,
        "per_class": {
            label: {"precision": 0.4, "recall": 0.3, "f1": 0.35, "support": 1} for label in LABELS
        },
    }


class TestConfusionMatrix:
    def test_writes_a_file(self, tmp_path):
        matrix = np.zeros((8, 8), dtype=int)
        np.fill_diagonal(matrix, 4)
        path = plot_confusion_matrix(matrix, LABELS, path=tmp_path / "cm.png")
        assert path.is_file() and path.stat().st_size > 0

    def test_rows_are_normalised_not_columns(self, tmp_path, captured):
        """Row normalisation is the whole point: 'given the truth, what is said?'"""
        matrix = np.zeros((2, 2), dtype=int)
        matrix[0] = [9, 1]  # heavily over-represented true class
        matrix[1] = [1, 1]
        plot_confusion_matrix(matrix, ["a", "b"], path=tmp_path / "cm.png")

        rendered = captured[-1].axes[0].images[0].get_array()
        # Row 0 sums to 1, so its dominant cell is 0.9 even though it holds 9
        # times as many samples as row 1.
        assert rendered[0].max() == pytest.approx(0.9)
        assert rendered[1].max() == pytest.approx(0.5)

    def test_empty_class_row_does_not_divide_by_zero(self, tmp_path):
        matrix = np.zeros((2, 2), dtype=int)
        matrix[0] = [1, 1]  # class "b" has no samples at all
        path = plot_confusion_matrix(matrix, ["a", "b"], path=tmp_path / "cm.png")
        assert path.is_file()

    def test_default_filename_encodes_the_split(self):
        matrix = np.ones((1, 1), dtype=int)
        written = plot_confusion_matrix(matrix, ["only"])
        assert written.parent == plotting.FIGURES_DIR
        assert written.name == "confusion_matrix_test.png"

    def test_scores_reach_the_subtitle(self, tmp_path, captured):
        plot_confusion_matrix(
            np.eye(2, dtype=int),
            ["a", "b"],
            accuracy=0.4833,
            macro_f1=0.4537,
            path=tmp_path / "cm.png",
        )
        title = captured[-1].axes[0].get_title()
        assert "48.33%" in title and "45.37%" in title


class TestPerClassBars:
    def test_uses_the_report_it_is_given(self, report, tmp_path, captured):
        report["per_class"]["sad"]["f1"] = 0.99
        plot_per_class_f1(report, split="test", path=tmp_path / "bars.png")
        heights = [bar.get_height() for bar in captured[-1].axes[0].containers[2]]
        assert heights[LABELS.index("sad")] == pytest.approx(0.99)

    def test_class_order_follows_the_report(self, report, tmp_path, captured):
        plot_per_class_f1(report, split="val", path=tmp_path / "bars.png")
        axis = captured[-1].axes[0]
        assert [t.get_text() for t in axis.get_xticklabels()] == LABELS
        assert "val" in axis.get_title()


class TestTrainingCurves:
    @pytest.fixture
    def history(self, tmp_path):
        path = tmp_path / "history.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "train_accuracy",
                    "val_accuracy",
                    "train_macro_f1",
                    "val_macro_f1",
                ]
            )
            writer.writerows(
                [
                    [1, 1.5, 1.6, 0.30, 0.28, 0.25, 0.22],
                    [2, 1.1, 1.3, 0.45, 0.40, 0.40, 0.36],
                    [3, 0.9, 1.2, 0.52, 0.55, 0.50, 0.52],  # best
                    [4, 0.8, 1.3, 0.58, 0.50, 0.55, 0.45],
                ]
            )
        return path

    def test_marks_the_selected_epoch(self, history, tmp_path, captured):
        plot_training_curves(history, run_name="demo", path=tmp_path / "curves.png")
        labels = [t.get_text() for t in captured[-1].axes[1].get_legend().get_texts()]
        assert any("selected epoch 3" in text for text in labels)

    def test_draws_both_panels(self, history, tmp_path, captured):
        plot_training_curves(history, path=tmp_path / "curves.png")
        figure = captured[-1]
        assert len(figure.axes) >= 2
        assert figure.axes[0].get_title() == "Loss"
        assert "Macro F1" in figure.axes[1].get_title()

    def test_an_empty_history_is_an_error_not_a_blank_png(self, tmp_path):
        empty = tmp_path / "empty.csv"
        empty.write_text("epoch,train_loss\n", encoding="utf-8")
        with pytest.raises(ValueError, match="No rows"):
            plot_training_curves(empty, path=tmp_path / "curves.png")


class TestCorpusFigures:
    def test_class_distribution_handles_missing_classes(self, tmp_path):
        """A split with no examples of a class must draw a zero, not crash."""
        counts = {"train": {"calm": 128}, "test": {}}
        path = plot_class_distribution(counts, path=tmp_path / "dist.png")
        assert path.is_file()

    def test_duration_distribution_annotates_the_window(self, tmp_path, captured):
        plot_duration_distribution([1.0, 2.0, 4.0], window=3.0, path=tmp_path / "dur.png")
        labels = [t.get_text() for t in captured[-1].axes[0].get_legend().get_texts()]
        assert any("3s" in text for text in labels)

    def test_duration_distribution_without_a_window_still_plots(self, tmp_path):
        assert plot_duration_distribution([1.0, 2.0], path=tmp_path / "dur.png").is_file()

    def test_audio_and_mfcc_labels_both_panes(self, tmp_path, captured):
        waveform = np.sin(np.linspace(0, 40 * np.pi, 40_000)).astype(np.float32)
        mfcc = np.zeros((120, 101), dtype=np.float32)
        path = plot_audio_and_mfcc(
            waveform, 16_000, mfcc, label="calm", title="demo", path=tmp_path / "a.png"
        )
        wave_axis, mfcc_axis = captured[-1].axes[:2]
        assert "calm" in wave_axis.get_title()
        assert "2.50 s" in wave_axis.get_title()  # 40000 / 16000
        assert "120 planes" in mfcc_axis.get_title()
        assert "101 frames" in mfcc_axis.get_title()
        assert path.is_file()

    def test_log_mel_paints_the_matrix_it_is_given(self, tmp_path, captured):
        log_mel = np.arange(128 * 101, dtype=np.float32).reshape(128, 101)
        plot_log_mel(log_mel, 16_000, 160, label="angry", path=tmp_path / "mel.png")
        axis = captured[-1].axes[0]
        image = axis.images[0]
        np.testing.assert_allclose(np.asarray(image.get_array()), log_mel)
        # 101 frames x hop 160 / 16 kHz = 1.01 s of audio on the x axis.
        left, right = image.get_extent()[:2]
        assert left == 0
        assert right == pytest.approx(1.01)
        assert "128 mel bands" in axis.get_title()
        assert "angry" in axis.get_title()


class TestRunFigures:
    @pytest.fixture
    def metrics(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "train_accuracy",
                    "val_accuracy",
                    "train_macro_f1",
                    "val_macro_f1",
                ]
            )
            writer.writerow([1, 1.0, 1.1, 0.4, 0.45, 0.4, 0.44])
        return {
            "run_name": "run",
            "run_dir": str(run_dir),
            "labels": LABELS,
            "reports": {
                "test": {
                    "accuracy": 0.4833,
                    "macro_f1": 0.4537,
                    "confusion_matrix": [[1, 0], [0, 1]],
                    "per_class": {
                        label: {"precision": 0.4, "recall": 0.4, "f1": 0.4, "support": 1}
                        for label in LABELS
                    },
                }
            },
            "features": {"hop_length": 160},
            "audio": {"sample_rate": 16_000, "target_duration_sec": 3.0},
            "split": {"test_actors": [3, 8, 14, 21], "val_actors": [6, 11, 19, 22]},
        }

    def test_produces_the_performance_and_training_figures(self, metrics, tmp_path):
        written = generate_run_figures(
            tmp_path / "run", metrics, split="test", include_dataset=False
        )
        names = {path.name for path in written}
        assert names == {
            "training_curves.png",
            "confusion_matrix_test.png",
            "per_class_f1_test.png",
        }
        assert all(path.is_file() for path in written)

    def test_a_missing_history_is_skipped_not_fatal(self, metrics, tmp_path):
        (tmp_path / "run" / "history.csv").unlink()
        written = generate_run_figures(
            tmp_path / "run", metrics, split="test", include_dataset=False
        )
        assert "training_curves.png" not in {path.name for path in written}
        assert len(written) == 2

    def test_an_unknown_split_still_yields_the_training_curves(self, metrics, tmp_path):
        written = generate_run_figures(
            tmp_path / "run", metrics, split="train", include_dataset=False
        )
        assert [path.name for path in written] == ["training_curves.png"]

    def test_dataset_figures_use_the_run_not_the_defaults(self, metrics, tmp_path, monkeypatch):
        processed = _fake_corpus(tmp_path, monkeypatch)
        assert processed.is_dir()

        written = generate_run_figures(tmp_path / "run", metrics, split="test")
        names = {path.name for path in written}
        assert "class_distribution.png" in names
        assert "duration_distribution.png" in names

        # 100 valid frames at hop 160 / 16 kHz is exactly 1.0 s per frame.
        assert plotting._trimmed_durations(160, 16_000) == [1.0, 2.0, 3.0, 4.0]

    def test_missing_corpus_data_still_yields_the_run_figures(self, metrics, tmp_path, monkeypatch):
        monkeypatch.setattr("ml.data.loading.PROCESSED_DIR", tmp_path / "absent")
        monkeypatch.setattr("ml.data.loading.find_existing_cache", lambda *a, **k: None)
        monkeypatch.setattr(
            "ml.data.ravdess.resolve_ravdess_root",
            lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no corpus")),
        )
        written = generate_run_figures(tmp_path / "run", metrics, split="test")
        assert {path.name for path in written} == {
            "training_curves.png",
            "confusion_matrix_test.png",
            "per_class_f1_test.png",
        }


def _fake_corpus(tmp_path, monkeypatch) -> Path:
    """A manifest and a feature-cache directory, wired into ``ml.data.loading``."""
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "manifest.json").write_text(
        json.dumps(
            [
                {"actor": actor, "label": label}
                for actor in (3, 8, 6, 1)
                for label in ("calm", "sad")
            ]
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    np.save(cache / "valid_frames.npy", np.array([100, 200, 300, 400], dtype=np.int32))
    monkeypatch.setattr("ml.data.loading.PROCESSED_DIR", processed)
    monkeypatch.setattr("ml.data.loading.find_existing_cache", lambda *a, **k: cache)
    return processed


class TestAttempt:
    def test_a_failing_plot_costs_one_figure(self):
        def explode(*args, **kwargs):
            raise RuntimeError("backend on fire")

        assert plotting._attempt(explode) is None

    def test_a_working_plot_is_returned(self, tmp_path):
        path = plotting._attempt(plot_duration_distribution, [1.0], path=tmp_path / "d.png")
        assert path is not None and path.is_file()

    def test_a_failing_plot_does_not_stop_the_set(self, report, tmp_path, monkeypatch):
        """One broken illustration must not cost the whole figure set."""
        original = plotting.plot_per_class_f1

        def explode(*args, **kwargs):
            raise RuntimeError("backend on fire")

        monkeypatch.setattr(plotting, "plot_per_class_f1", explode)
        assert plotting._attempt(plotting.plot_per_class_f1, report) is None
        monkeypatch.setattr(plotting, "plot_per_class_f1", original)
        assert plotting._attempt(plotting.plot_per_class_f1, report, split="test") is not None


class TestSplitClassCounts:
    def test_buckets_actors_into_three_splits(self, tmp_path, monkeypatch):
        processed = tmp_path / "processed"
        processed.mkdir()
        (processed / "manifest.json").write_text(
            json.dumps([{"actor": a, "label": "calm"} for a in (1, 6, 3, 6)]),
            encoding="utf-8",
        )
        monkeypatch.setattr("ml.data.loading.PROCESSED_DIR", processed)
        counts = plotting._split_class_counts({"split": {"test_actors": [3], "val_actors": [6]}})
        assert counts == {"train": {"calm": 1}, "val": {"calm": 2}, "test": {"calm": 1}}

    def test_metrics_without_split_actors_yield_nothing(self, tmp_path, monkeypatch):
        processed = tmp_path / "processed"
        processed.mkdir()
        (processed / "manifest.json").write_text(
            json.dumps([{"actor": 1, "label": "calm"}]), encoding="utf-8"
        )
        monkeypatch.setattr("ml.data.loading.PROCESSED_DIR", processed)
        assert plotting._split_class_counts({"split": {}}) == {}

    def test_a_corrupt_manifest_is_reported_not_raised(self, tmp_path, monkeypatch):
        processed = tmp_path / "processed"
        processed.mkdir()
        (processed / "manifest.json").write_text("{not json", encoding="utf-8")
        monkeypatch.setattr("ml.data.loading.PROCESSED_DIR", processed)
        assert plotting._split_class_counts({"split": {"test_actors": [3]}}) == {}


class TestTrimmedDurations:
    def test_no_cache_yields_no_durations(self, monkeypatch):
        monkeypatch.setattr("ml.data.loading.find_existing_cache", lambda *a, **k: None)
        assert plotting._trimmed_durations(160, 16_000) == []

    def test_an_unreadable_array_yields_no_durations(self, tmp_path, monkeypatch):
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "valid_frames.npy").write_text("garbage", encoding="utf-8")
        monkeypatch.setattr("ml.data.loading.find_existing_cache", lambda *a, **k: cache)
        assert plotting._trimmed_durations(160, 16_000) == []

    def test_hop_length_sets_the_frame_duration(self, tmp_path, monkeypatch):
        cache = tmp_path / "cache"
        cache.mkdir()
        np.save(cache / "valid_frames.npy", np.array([50], dtype=np.int32))
        monkeypatch.setattr("ml.data.loading.find_existing_cache", lambda *a, **k: cache)
        # 50 frames x 512 samples / 16 kHz = 1.6 s
        assert plotting._trimmed_durations(512, 16_000) == [pytest.approx(1.6)]
