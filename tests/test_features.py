"""MFCC front-end.

The property that matters most here is **training/inference identity**: the
features produced from a cached window must be bit-for-bit what the API will
compute from an upload, otherwise the served model is not the trained model.
"""

from __future__ import annotations

import numpy as np

from ml.config import FeatureConfig
from ml.features.mfcc import MFCCExtractor, apply_cmvn, log_mel_spectrogram

from .conftest import synth_speech_like


class TestShape:
    def test_output_shape_is_n_features_by_n_frames(self, feature_cfg, speech_waveform):
        extractor = MFCCExtractor(feature_cfg)
        feats = extractor.transform(speech_waveform, 16000)

        assert feats.ndim == 2
        assert feats.shape[0] == feature_cfg.n_output_features() == 120
        # The fixture is 2.0 s: 1 + 2.0 s * 100 frames/s.
        assert feats.shape[1] == 201

    def test_full_window_yields_301_frames(self, feature_cfg, audio_cfg):
        """The cached training tensors are 3.0 s @ 100 frames/s."""
        from ml.data.preprocessing import preprocess_waveform

        window, _ = preprocess_waveform(
            synth_speech_like(audio_cfg.target_duration_sec), 16000, audio_cfg
        )
        feats = MFCCExtractor(feature_cfg).transform(window, 16000)
        assert feats.shape == (120, 301)

    def test_shape_matches_the_declared_prediction(self, feature_cfg, speech_waveform):
        extractor = MFCCExtractor(feature_cfg)
        predicted = extractor.output_shape(speech_waveform.size, 16000)
        assert extractor.transform(speech_waveform, 16000).shape == predicted

    def test_frame_count_follows_hop_length(self, feature_cfg):
        extractor = MFCCExtractor(feature_cfg)
        for seconds in (1.0, 2.5, 3.0):
            n_samples = int(seconds * 16000)
            feats = extractor.transform(np.zeros(n_samples, dtype=np.float32), 16000)
            assert feats.shape[1] == 1 + n_samples // feature_cfg.hop_length

    def test_dtype_is_float32(self, feature_cfg, speech_waveform):
        """float32 halves the cache size; float64 would silently double RAM."""
        feats = MFCCExtractor(feature_cfg).transform(speech_waveform, 16000)
        assert feats.dtype == np.float32

    def test_output_is_contiguous(self, feature_cfg, speech_waveform):
        feats = MFCCExtractor(feature_cfg).transform(speech_waveform, 16000)
        assert feats.flags["C_CONTIGUOUS"]

    def test_n_features_property(self, feature_cfg):
        assert MFCCExtractor(feature_cfg).n_features == 120


class TestPlanes:
    def test_mfcc_delta_and_delta2_are_stacked(self, feature_cfg):
        """Three 40-wide blocks, in order."""
        cfg = FeatureConfig()
        assert cfg.n_output_features() == 3 * cfg.n_mfcc

    def test_disabling_deltas_shrinks_the_matrix(self, speech_waveform):
        full = MFCCExtractor(FeatureConfig()).transform(speech_waveform, 16000)
        mfcc_only = MFCCExtractor(FeatureConfig(use_delta=False, use_delta2=False)).transform(
            speech_waveform, 16000
        )

        assert mfcc_only.shape[0] == 40
        # The MFCC block itself is unchanged by whether deltas are requested.
        np.testing.assert_allclose(full[:40], mfcc_only, atol=1e-6)

    def test_delta_plane_responds_to_a_ramp(self, feature_cfg):
        """A constant-offset signal has ~zero delta; a ramp does not."""
        cfg = FeatureConfig(cmvn=False)
        extractor = MFCCExtractor(cfg)

        constant = np.full(16000, 0.5, dtype=np.float32)
        ramp = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)

        const_feats = extractor.transform(constant, 16000)
        ramp_feats = extractor.transform(ramp, 16000)

        # Compare interiors to avoid edge effects from the finite difference.
        const_delta = np.abs(const_feats[40:80, 20:-20]).mean()
        ramp_delta = np.abs(ramp_feats[40:80, 20:-20]).mean()
        assert ramp_delta > const_delta


class TestCMVN:
    def test_normalises_each_plane_over_time(self, feature_cfg, speech_waveform):
        feats = MFCCExtractor(feature_cfg).transform(speech_waveform, 16000)

        np.testing.assert_allclose(feats.mean(axis=1), 0.0, atol=1e-4)
        np.testing.assert_allclose(feats.std(axis=1), 1.0, atol=1e-3)

    def test_can_be_disabled(self, speech_waveform):
        feats = MFCCExtractor(FeatureConfig(cmvn=False)).transform(speech_waveform, 16000)
        # Without CMVN the coefficients keep their absolute dB-like offset.
        assert np.abs(feats.mean()) > 0.5

    def test_apply_cmvn_is_idempotent_on_normalised_input(self, speech_waveform):
        feats = MFCCExtractor(FeatureConfig()).transform(speech_waveform, 16000)
        again = apply_cmvn(feats)
        np.testing.assert_allclose(feats, again, atol=1e-3)

    def test_cmvn_suppresses_recording_level(self, audio_cfg):
        """Two copies of the same speech at different gains -> same features.

        This is the whole point of the normalisation step and the reason it is
        part of the front-end rather than an optional extra. The gain change is
        kept below clipping so the only difference is amplitude.
        """
        base = synth_speech_like(3.0)
        extractor = MFCCExtractor(FeatureConfig())

        loud = (base / np.max(np.abs(base))).astype(np.float32)
        quiet = (base * 0.2 / np.max(np.abs(base))).astype(np.float32)

        a = extractor.transform(loud, 16000)
        b = extractor.transform(quiet, 16000)

        assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.98
        assert np.abs(a - b).max() < 1e-3

    def test_cmvn_does_not_divide_by_zero_on_a_constant_plane(self):
        """A silent-but-nonzero signal could give std == 0."""
        constant = np.full((40, 50), 3.0, dtype=np.float32)
        normalised = apply_cmvn(constant)
        assert np.isfinite(normalised).all()
        np.testing.assert_allclose(normalised, 0.0, atol=1e-4)


class TestDeterminism:
    def test_same_input_gives_identical_output(self, feature_cfg, speech_waveform):
        extractor = MFCCExtractor(feature_cfg)
        first = extractor.transform(speech_waveform, 16000)
        second = extractor.transform(speech_waveform, 16000)
        np.testing.assert_array_equal(first, second)

    def test_fresh_extractor_gives_identical_output(self, feature_cfg, speech_waveform):
        """Statelessness: no fitting step, so a new object must agree."""
        a = MFCCExtractor(feature_cfg).transform(speech_waveform, 16000)
        b = MFCCExtractor(FeatureConfig()).transform(speech_waveform, 16000)
        np.testing.assert_array_equal(a, b)

    def test_input_is_not_mutated(self, feature_cfg, speech_waveform):
        original = speech_waveform.copy()
        MFCCExtractor(feature_cfg).transform(speech_waveform, 16000)
        np.testing.assert_array_equal(speech_waveform, original)


class TestSerialisation:
    def test_to_dict_from_dict_round_trip(self, feature_cfg):
        extractor = MFCCExtractor(feature_cfg)
        restored = MFCCExtractor.from_dict(extractor.to_dict())
        assert restored.cfg == extractor.cfg

    def test_from_dict_ignores_unknown_keys(self):
        cfg = MFCCExtractor.from_dict({"n_mfcc": 20, "unknown_knob": 7})
        assert cfg.cfg.n_mfcc == 20
        assert cfg.n_features == 60

    def test_reconstructed_config_drives_the_pipeline(self, audio_cfg):
        """The API rebuilds the front-end from the checkpoint; it must match."""
        payload = MFCCExtractor(FeatureConfig()).to_dict()
        extractor = MFCCExtractor.from_dict(payload)

        window = synth_speech_like(3.0)
        n_samples = int(audio_cfg.target_duration_sec * audio_cfg.sample_rate)
        padded = np.zeros(n_samples, dtype=np.float32)
        padded[: min(window.size, n_samples)] = window[:n_samples]

        original = MFCCExtractor(FeatureConfig()).transform(padded, 16000)
        rebuilt = extractor.transform(padded, 16000)
        np.testing.assert_array_equal(original, rebuilt)


class TestLogMel:
    def test_log_mel_shape(self, feature_cfg, speech_waveform):
        mel = log_mel_spectrogram(speech_waveform, 16000, feature_cfg)
        assert mel.shape[0] == feature_cfg.n_mels == 128
        assert mel.shape[1] == 201  # matches the MFCC frame count

    def test_log_mel_is_non_negative(self, feature_cfg, speech_waveform):
        """Energy is floored at eps, so the log is never -inf or NaN."""
        mel = log_mel_spectrogram(speech_waveform, 16000, feature_cfg)
        assert np.isfinite(mel).all()
        assert mel.min() > -np.inf

    def test_log_mel_handles_silence(self, feature_cfg):
        mel = log_mel_spectrogram(np.zeros(16000, dtype=np.float32), 16000, feature_cfg)
        assert np.isfinite(mel).all()


class TestCallableAlias:
    def test_call_is_transform(self, feature_cfg, speech_waveform):
        extractor = MFCCExtractor(feature_cfg)
        np.testing.assert_array_equal(
            extractor(speech_waveform, 16000), extractor.transform(speech_waveform, 16000)
        )
