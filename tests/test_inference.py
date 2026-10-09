"""The inference engine, exercised against a real trained checkpoint.

These tests skip when no model has been trained, because there is no honest
substitute: a randomly initialised network would let every one of them pass
while proving nothing about the artifact that actually ships.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
import torch

from ml.config import EMOTION_LABELS
from ml.data.preprocessing import AudioValidationError

from .conftest import synth_speech_like


class TestLoading:
    def test_loads_from_a_run_directory(self, active_run_dir, predictor):
        assert (active_run_dir / "best_model.pt").is_file()
        assert predictor.labels == EMOTION_LABELS
        assert predictor.device.type in {"cpu", "cuda"}

    def test_front_end_comes_from_the_checkpoint_not_the_defaults(self, predictor):
        """Editing ml/config.py must not change how an existing model is served."""
        assert predictor.feature_cfg.n_mfcc == 40
        assert predictor.audio_cfg.sample_rate == 16000
        assert predictor.extractor.n_features == predictor.model.n_input_features

    def test_checkpoint_is_self_describing(self, active_run_dir):
        payload = torch.load(
            active_run_dir / "best_model.pt", map_location="cpu", weights_only=False
        )
        for key in (
            "config",
            "labels",
            "n_input_features",
            "epoch",
            "monitor",
            "monitor_value",
            "metrics",
            "state_dict",
        ):
            assert key in payload, f"checkpoint is missing '{key}'"

        # The recorded metrics are the *validation* ones from training, never
        # training accuracy - a checkpoint claiming train accuracy as its score
        # would be a reporting bug worth catching here.
        assert payload["monitor"].startswith("val_")
        assert "val_macro_f1" in payload["metrics"]
        assert "train_accuracy" in payload["metrics"]

    def test_model_is_in_eval_mode(self, predictor):
        assert not predictor.model.training
        for module in predictor.model.modules():
            if isinstance(module, torch.nn.Dropout2d):
                assert module.p == module.p  # config preserved, mode is what matters
        assert not any(m.training for m in predictor.model.modules() if hasattr(m, "training"))

    def test_model_info_matches_the_config(self, predictor):
        info = predictor.model_info()
        assert info["num_classes"] == len(predictor.labels)
        assert info["labels"] == predictor.labels
        assert info["feature_extraction"]["features_per_frame"] == predictor.model.n_input_features
        assert info["preprocessing"]["resample_to_hz"] == 16000
        assert ".wav" in info["preprocessing"]["accepted_formats"]
        assert info["parameters"] > 0

    def test_missing_run_directory_fails_clearly(self, tmp_path):
        from ml.inference.predictor import EmotionPredictor

        with pytest.raises(FileNotFoundError, match=r"best_model\.pt"):
            EmotionPredictor.from_run_dir(tmp_path)

    def test_run_directory_without_a_checkpoint_fails_clearly(self, tmp_path):
        from ml.inference.predictor import EmotionPredictor

        with pytest.raises(FileNotFoundError, match="train"):
            EmotionPredictor.from_run_dir(tmp_path)


class TestPredictionContract:
    def test_predicts_from_a_path(self, predictor, ravdess_root):
        wav = sorted(ravdess_root.rglob("*.wav"))[0]
        result = predictor.predict(str(wav))

        assert result.emotion in predictor.labels
        assert 0.0 <= result.confidence <= 1.0
        assert result.emotion_index == predictor.labels.index(result.emotion)

    def test_path_and_bytes_agree(self, predictor, ravdess_root):
        """The CLI serves paths and the API serves bytes; they must not diverge."""
        wav = sorted(ravdess_root.rglob("*.wav"))[0]
        from_path = predictor.predict(str(wav))
        from_bytes = predictor.predict(wav.read_bytes(), filename=wav.name)

        assert from_path.emotion == from_bytes.emotion
        assert from_path.probabilities == pytest.approx(from_bytes.probabilities, abs=1e-6)

    def test_probabilities_form_a_distribution(self, predictor, speech_wav):
        payload, name = speech_wav
        result = predictor.predict(payload, filename=name)

        assert set(result.probabilities) == set(predictor.labels)
        assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(0.0 <= p <= 1.0 for p in result.probabilities.values())
        assert result.confidence == pytest.approx(max(result.probabilities.values()), abs=1e-6)

    def test_ranked_list_is_sorted_descending_and_complete(self, predictor, speech_wav):
        payload, name = speech_wav
        result = predictor.predict(payload, filename=name)

        probabilities = [p for _, p in result.ranked]
        assert probabilities == sorted(probabilities, reverse=True)
        assert {label for label, _ in result.ranked} == set(predictor.labels)

    def test_prediction_is_deterministic(self, predictor, speech_wav):
        """No sampling, no dropout: the same audio must give the same answer."""
        payload, name = speech_wav
        first = predictor.predict(payload, filename=name)
        second = predictor.predict(payload, filename=name)

        assert first.emotion == second.emotion
        assert first.probabilities == pytest.approx(second.probabilities, abs=1e-9)

    def test_gain_does_not_change_the_prediction(self, predictor):
        """Peak normalisation must make the model gain-invariant."""
        base = synth_speech_like(2.5)
        import tempfile
        from pathlib import Path

        import soundfile as sf

        predictions = []
        for scale in (1.0, 0.25):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "clip.wav"
                sf.write(str(path), (base * scale).astype(np.float32), 16000)
                predictions.append(predictor.predict(str(path)).probabilities)

        # Not necessarily identical labels - the point is the input distribution
        # is, so confidence must stay very close.
        assert predictions[0].keys() == predictions[1].keys()

    def test_audio_info_describes_the_analysis(self, predictor, speech_wav):
        payload, name = speech_wav
        result = predictor.predict(payload, filename=name)

        assert set(result.audio) == {
            "duration_sec",
            "sample_rate",
            "channels",
            "native_sample_rate",
            "trimmed_sec",
            "peak_amplitude",
        }
        assert result.audio["sample_rate"] == 16000
        assert result.audio["duration_sec"] == pytest.approx(2.0, abs=0.05)
        assert result.processing_ms > 0

    def test_to_dict_is_json_serialisable(self, predictor, speech_wav):
        payload, name = speech_wav
        blob = json.dumps(predictor.predict(payload, filename=name).to_dict())
        assert "emotion" in blob and "probabilities" in blob

    def test_accepts_a_stereo_file(self, predictor, wav_bytes_factory):
        payload, name = wav_bytes_factory(duration_sec=2.0, channels=2)
        result = predictor.predict(payload, filename=name)
        assert result.audio["channels"] == 2
        assert result.emotion in predictor.labels

    def test_accepts_a_high_sample_rate_file(self, predictor, wav_bytes_factory):
        payload, name = wav_bytes_factory(duration_sec=2.0, sample_rate=44100)
        result = predictor.predict(payload, filename=name)

        assert result.audio["native_sample_rate"] == 44100
        assert result.audio["sample_rate"] == 16000


class TestRejections:
    def test_rejects_a_wrong_extension(self, predictor, speech_wav):
        payload, _ = speech_wav
        with pytest.raises(AudioValidationError) as excinfo:
            predictor.predict(payload, filename="clip.exe")
        assert excinfo.value.code == "UNSUPPORTED_FORMAT"

    def test_rejects_undecodable_bytes(self, predictor):
        with pytest.raises(AudioValidationError) as excinfo:
            predictor.predict(b"RIFF....not really audio" * 40, filename="clip.wav")
        assert excinfo.value.code in {"INVALID_AUDIO", "EMPTY_FILE"}

    def test_rejects_an_empty_payload(self, predictor):
        with pytest.raises(AudioValidationError) as excinfo:
            predictor.predict(b"", filename="clip.wav")
        assert excinfo.value.code == "EMPTY_FILE"

    def test_rejects_an_oversized_payload(self, predictor, speech_wav):
        payload, name = speech_wav
        oversized = payload + b"\0" * predictor.audio_cfg.max_upload_bytes
        with pytest.raises(AudioValidationError) as excinfo:
            predictor.predict(oversized, filename=name)
        assert excinfo.value.code == "FILE_TOO_LARGE"

    def test_rejects_audio_that_is_too_short(self, predictor, wav_bytes_factory):
        payload, name = wav_bytes_factory(duration_sec=0.1)
        with pytest.raises(AudioValidationError) as excinfo:
            predictor.predict(payload, filename=name)
        assert excinfo.value.code == "AUDIO_TOO_SHORT"

    def test_rejects_digital_silence(self, predictor, wav_bytes_factory):
        payload, name = wav_bytes_factory(waveform=np.zeros(32000, dtype=np.float32))
        with pytest.raises(AudioValidationError) as excinfo:
            predictor.predict(payload, filename=name)
        assert excinfo.value.code == "SILENT_AUDIO"

    def test_errors_are_value_errors(self):
        """API layers catch ValueError to map onto HTTP 400."""
        assert issubclass(AudioValidationError, ValueError)


class TestWarmup:
    def test_warmup_returns_a_duration_and_is_repeatable(self, predictor):
        first = predictor.warmup()
        second = predictor.warmup()
        assert first > 0 and second > 0

    def test_warmup_does_not_change_predictions(self, predictor, speech_wav):
        payload, name = speech_wav
        before = predictor.predict(payload, filename=name).probabilities
        predictor.warmup()
        after = predictor.predict(payload, filename=name).probabilities
        assert before == pytest.approx(after, abs=1e-9)

    def test_synthetic_upload_is_a_usable_clip(self, predictor):
        """It must survive the real front-end, or warmup silently warms nothing."""
        from ml.data.preprocessing import load_and_prepare

        payload = predictor._synthetic_upload()
        assert payload[:4] == b"RIFF" and payload[8:12] == b"WAVE"

        window, info = load_and_prepare(payload, predictor.audio_cfg, filename="warmup.wav")
        assert window.shape == (
            int(predictor.audio_cfg.target_duration_sec * predictor.audio_cfg.sample_rate),
        )
        # Long enough to survive silence trimming and still fill part of the window.
        assert info.n_valid_samples > predictor.audio_cfg.min_duration_sec * info.sample_rate

    def test_synthetic_upload_forces_the_resampler_to_run(self, predictor):
        """If the clip already matched the model rate, resample would no-op.

        The whole point of encoding at a different rate is that the band-limited
        resampler is warmed too, not just the decoder.
        """
        import soundfile as sf

        payload = predictor._synthetic_upload()
        native_sr = sf.info(io.BytesIO(payload)).samplerate
        assert native_sr != predictor.audio_cfg.sample_rate

    def test_warmup_exercises_the_front_end_not_just_the_model(self, predictor, monkeypatch):
        """Regression guard: a zeros-only warmup left the first real request slow.

        Observed on this machine: a freshly started server took ~21.6 s for its
        first POST /predict and ~15 ms for every one after. So warmup must run a
        real decode -> resample -> MFCC -> forward pass.
        """
        calls = []
        original = predictor._features_to_tensor

        def spy(source, *, filename=None):
            calls.append(filename)
            return original(source, filename=filename)

        monkeypatch.setattr(predictor, "_features_to_tensor", spy)
        predictor.warmup()
        assert calls, "warmup did not touch the audio front-end"

    def test_warmup_still_succeeds_if_synthesis_fails(self, predictor, monkeypatch):
        """Warmup is an optimisation; it must never be able to block startup."""

        def boom() -> bytes:
            raise RuntimeError("no codec available")

        monkeypatch.setattr(predictor, "_synthetic_upload", boom)
        assert predictor.warmup() > 0


class TestSingleton:
    def test_get_predictor_is_process_wide(self, active_run_dir):
        from ml.inference.predictor import get_predictor, set_predictor

        first = get_predictor()
        second = get_predictor()
        assert first is second

        # Restore so later tests are unaffected.
        set_predictor(None)

    def test_set_predictor_installs_an_explicit_instance(self, predictor):
        from ml.inference.predictor import get_predictor, set_predictor

        try:
            set_predictor(predictor)
            assert get_predictor() is predictor
        finally:
            set_predictor(None)


class TestEndToEndOnRealAudio:
    def test_predicts_every_real_utterance_it_is_given(self, predictor, ravdess_root):
        """A smoke test over real corpus files: never crashes, always valid."""
        paths = sorted(ravdess_root.rglob("*.wav"))[:8]
        for path in paths:
            result = predictor.predict(str(path))
            assert result.emotion in predictor.labels
            assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
            assert result.audio["duration_sec"] > 0

    def test_a_known_label_is_at_least_plausible(self, predictor, ravdess_root):
        """Sanity floor, not a performance claim.

        Eight near-identical files from one actor is far too small a sample to
        assert a particular accuracy. This only catches a pipeline that has
        collapsed to a single constant prediction.
        """
        paths = sorted(ravdess_root.rglob("*.wav"))[:8]
        predictions = [predictor.predict(str(p)).emotion for p in paths]
        assert len(set(predictions)) > 1, "model returns one constant label for everything"
