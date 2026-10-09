"""Configuration invariants.

These are the assumptions everything else silently depends on: the emotion
mapping, the class order, and the arithmetic that turns MFCC settings into the
tensor shape the network consumes.
"""

from __future__ import annotations

import json

import pytest

from ml.config import (
    EMOTION_LABELS,
    NUM_CLASSES,
    RAVDESS_ARCHIVE_MD5,
    RAVDESS_EMOTION_CODES,
    RAVDESS_N_ACTORS,
    AudioConfig,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    SplitConfig,
    resolve_num_threads,
)


class TestEmotionMapping:
    def test_published_codes_only(self):
        """Guard against the widely circulated *incorrect* RAVDESS mapping.

        The wrong version (03=disgust, 05=happy, 06=surprise, ...) silently
        relabels the whole corpus and produces a model that looks fine and is
        meaningless.
        """
        assert RAVDESS_EMOTION_CODES == {
            "01": "neutral",
            "02": "calm",
            "03": "happy",
            "04": "sad",
            "05": "angry",
            "06": "fearful",
            "07": "disgust",
            "08": "surprised",
        }

    def test_codes_are_unique_and_complete(self):
        assert len(RAVDESS_EMOTION_CODES) == 8
        assert set(RAVDESS_EMOTION_CODES) == {f"{i:02d}" for i in range(1, 9)}
        assert len(set(RAVDESS_EMOTION_CODES.values())) == 8

    def test_label_order_is_code_order(self):
        """Logit index must equal position in the sorted code list."""
        assert EMOTION_LABELS == [RAVDESS_EMOTION_CODES[f"{i:02d}"] for i in range(1, 9)]
        assert NUM_CLASSES == 8 == len(EMOTION_LABELS)

    def test_archive_checksum_is_recorded(self):
        """A pinned checksum makes dataset drift detectable, not silent."""
        assert RAVDESS_ARCHIVE_MD5 == "bc696df654c87fed845eb13823edef8a"
        assert len(RAVDESS_ARCHIVE_MD5) == 32


class TestFeatureConfig:
    def test_default_feature_count(self):
        cfg = FeatureConfig()
        # 40 MFCC + 40 delta + 40 delta-delta
        assert cfg.n_output_features() == 120

    def test_feature_count_scales_with_enabled_planes(self):
        assert FeatureConfig(use_delta=False, use_delta2=False).n_output_features() == 40
        assert FeatureConfig(use_delta=True, use_delta2=False).n_output_features() == 80
        assert FeatureConfig(use_delta=True, use_delta2=True).n_output_features() == 120

    def test_hop_gives_100_frames_per_second(self):
        cfg = FeatureConfig()
        audio = AudioConfig()
        assert audio.sample_rate == 16000
        assert cfg.hop_length == 160
        assert audio.sample_rate / cfg.hop_length == 100.0

    def test_window_covers_full_speech_band(self):
        cfg = FeatureConfig()
        assert cfg.fmin == 0.0
        assert cfg.fmax <= 8000  # Nyquist at 16 kHz

    def test_n_fft_not_smaller_than_window(self):
        cfg = FeatureConfig()
        assert cfg.n_fft >= cfg.win_length

    def test_describe_is_human_readable(self):
        text = FeatureConfig().describe()
        assert "40" in text and "120 features/frame" in text


class TestAudioConfig:
    def test_window_and_guards_are_consistent(self):
        cfg = AudioConfig()
        target = int(cfg.target_duration_sec * cfg.sample_rate)
        assert target == 48000
        assert cfg.min_duration_sec < cfg.target_duration_sec < cfg.max_duration_sec

    def test_allowed_extensions_are_lowercase_dotted(self):
        for ext in AudioConfig().allowed_extensions:
            assert ext.startswith(".")
            assert ext == ext.lower()

    def test_upload_limit_is_bounded(self):
        limit = AudioConfig().max_upload_bytes
        assert 1024 <= limit <= 200 * 1024 * 1024


class TestModelConfig:
    def test_bidirectional_output_dim(self):
        cfg = ModelConfig()
        assert cfg.lstm_output_dim() == cfg.lstm_hidden * 2

        cfg.bidirectional = False
        assert cfg.lstm_output_dim() == cfg.lstm_hidden

    def test_stem_stride_is_two_dimensional(self):
        assert len(ModelConfig().stem_stride) == 2

    def test_channels_increase(self):
        channels = ModelConfig().channels
        assert channels == sorted(channels)
        assert len(channels) >= 2


class TestSplitConfig:
    def test_splits_are_disjoint_and_complete(self):
        split = SplitConfig()
        train = set(split.train_actors())
        val = set(split.val_actors)
        test = set(split.test_actors)

        assert train & val == set()
        assert train & test == set()
        assert val & test == set()
        assert train | val | test == set(range(1, RAVDESS_N_ACTORS + 1))

    def test_test_split_is_the_declared_size(self):
        """A fixed, pre-declared test split keeps evaluation honest."""
        assert len(SplitConfig().test_actors) == 4

    def test_train_split_is_the_majority(self):
        assert len(SplitConfig().train_actors()) == RAVDESS_N_ACTORS - 8


class TestExperimentConfigRoundTrip:
    def test_to_dict_and_back_are_lossless(self, tmp_path):
        original = ExperimentConfig(notes="round-trip check")
        path = tmp_path / "config.json"
        original.save(path)
        restored = ExperimentConfig.load(path)

        assert restored.to_dict() == original.to_dict()
        assert restored.notes == "round-trip check"
        assert restored.labels == EMOTION_LABELS

    def test_unknown_keys_are_ignored(self):
        """An artifact from a newer version must still load."""
        payload = ExperimentConfig().to_dict()
        payload["audio"]["from_the_future"] = 123
        payload["brand_new_section"] = {"x": 1}

        restored = ExperimentConfig.from_dict(payload)
        assert restored.audio.sample_rate == 16000
        assert not hasattr(restored.audio, "from_the_future")


class TestSubConfigSerialisation:
    """Reports serialise individual nested configs, not just the whole thing."""

    def test_every_named_config_can_be_dumped(self):
        import dataclasses

        for name in ("audio", "features", "model", "train", "split"):
            payload = ExperimentConfig().sub_config(name)
            assert isinstance(payload, dict)
            assert payload == dataclasses.asdict(getattr(ExperimentConfig(), name))

    def test_nested_config_dumps_are_json_safe(self):
        import json

        experiment = ExperimentConfig()
        for name in ("audio", "features", "model", "train", "split"):
            json.dumps(experiment.sub_config(name))

    def test_an_unknown_name_is_a_clear_error(self):
        with pytest.raises(AttributeError, match="nonsense"):
            ExperimentConfig().sub_config("nonsense")


class TestThreadResolution:
    @pytest.mark.parametrize("requested,expected", [(1, 1), (4, 4), (16, 16)])
    def test_explicit_request_wins(self, requested, expected):
        assert resolve_num_threads(requested) == expected

    def test_auto_is_capped(self):
        """Uncapped torch threads oversubscribe a laptop and end up slower."""
        assert 1 <= resolve_num_threads(0) <= 8


# ---------------------------------------------------------------------------
# Active model resolution
# ---------------------------------------------------------------------------
class TestActiveModelResolution:
    """Which checkpoint the API is allowed to serve.

    The dangerous failure here is quiet: a training job writes its first
    ``best_model.pt`` within a minute of starting, so any "just use the newest
    one" heuristic will happily serve a 3-epoch model to production traffic.
    """

    @staticmethod
    def _make_run(root, name: str, *, complete: bool) -> None:
        run_dir = root / name
        run_dir.mkdir(parents=True)
        (run_dir / "best_model.pt").write_bytes(b"not a real checkpoint")
        (run_dir / "training_summary.json").write_text(
            json.dumps({"complete": complete, "best_val_macro_f1": 0.5}), encoding="utf-8"
        )

    @pytest.fixture
    def sandbox(self, tmp_path, monkeypatch):
        """A throwaway MODELS_DIR + pointer, so the real ones are untouched."""
        models = tmp_path / "models"
        pointer = tmp_path / "active_model.json"
        models.mkdir()
        monkeypatch.setattr("ml.config.MODELS_DIR", models)
        monkeypatch.setattr("ml.config.ACTIVE_MODEL_POINTER", pointer)
        return models, pointer

    def test_no_models_raises_a_useful_error(self, sandbox):
        from ml.config import get_active_run_dir

        with pytest.raises(FileNotFoundError, match="No completed trained model"):
            get_active_run_dir()

    def test_the_newest_run_is_discovered_when_none_is_selected(self, sandbox):
        from ml.config import get_active_run_dir

        models, _ = sandbox
        self._make_run(models, "older", complete=True)
        self._make_run(models, "newer", complete=True)

        assert get_active_run_dir().name == "newer"

    def test_an_unfinished_run_is_never_discovered(self, sandbox):
        """The regression this test exists for.

        Without the completeness filter, the newest run - i.e. the one training
        right now - would be served the moment a training job starts.
        """
        from ml.config import get_active_run_dir

        models, _ = sandbox
        self._make_run(models, "finished", complete=True)
        self._make_run(models, "training_right_now", complete=False)

        assert get_active_run_dir().name == "finished"

    def test_a_run_with_no_summary_is_never_discovered(self, sandbox):
        """A checkpoint with no summary is a run that never finished writing."""
        from ml.config import get_active_run_dir

        models, _ = sandbox
        self._make_run(models, "finished", complete=True)

        orphan = models / "orphan"
        orphan.mkdir()
        (orphan / "best_model.pt").write_bytes(b"partial")

        assert get_active_run_dir().name == "finished"

    def test_a_corrupt_summary_is_treated_as_incomplete(self, sandbox):
        from ml.config import get_active_run_dir

        models, _ = sandbox
        self._make_run(models, "finished", complete=True)

        broken = models / "broken"
        broken.mkdir()
        (broken / "best_model.pt").write_bytes(b"partial")
        (broken / "training_summary.json").write_text("{not json", encoding="utf-8")

        assert get_active_run_dir().name == "finished"

    def test_the_pointer_wins_over_discovery(self, sandbox):
        from ml.config import get_active_run_dir, set_active_model

        models, _ = sandbox
        self._make_run(models, "pinned", complete=True)
        self._make_run(models, "newer", complete=True)

        (models / "pinned" / "training_summary.json").write_text(
            json.dumps({"complete": False}), encoding="utf-8"
        )
        set_active_model("pinned")

        # An explicit choice is honoured even when the run is not complete:
        # naming a run is a human decision, discovery is automatic.
        assert get_active_run_dir().name == "pinned"

    def test_a_stale_pointer_falls_back_to_discovery(self, sandbox):
        from ml.config import get_active_run_dir

        models, pointer = sandbox
        self._make_run(models, "real", complete=True)
        pointer.write_text(
            json.dumps({"run": "deleted", "run_dir": str(models / "deleted")}),
            encoding="utf-8",
        )

        assert get_active_run_dir().name == "real"

    def test_a_corrupt_pointer_falls_back_to_discovery(self, sandbox):
        from ml.config import get_active_run_dir

        models, pointer = sandbox
        self._make_run(models, "real", complete=True)
        pointer.write_text("{truncated", encoding="utf-8")

        assert get_active_run_dir().name == "real"

    def test_an_explicit_path_overrides_everything(self, sandbox):
        from ml.config import get_active_run_dir, set_active_model

        models, _ = sandbox
        self._make_run(models, "pinned", complete=True)
        self._make_run(models, "newer", complete=True)
        set_active_model("pinned")

        explicit = models / "newer"
        assert get_active_run_dir(explicit).name == "newer"

    def test_an_explicit_path_without_a_checkpoint_is_rejected(self, sandbox):
        from ml.config import get_active_run_dir

        models, _ = sandbox
        empty = models / "empty"
        empty.mkdir()

        with pytest.raises(FileNotFoundError, match=r"best_model\.pt"):
            get_active_run_dir(empty)
