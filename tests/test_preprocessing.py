"""Waveform preprocessing: decoding, normalisation, windowing and every
rejection path.

The rejection tests matter as much as the happy paths. Each of them pins a
stable error *code*, because the HTTP layer maps codes to status codes and a
reworded message must never silently change an API contract.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from ml.config import AudioConfig
from ml.data.preprocessing import (
    AudioValidationError,
    decode_audio,
    load_and_prepare,
    prepare_from_bytes,
    preprocess_waveform,
    redact_for_client,
    resample,
    to_mono,
    validate_extension,
    validate_size,
)

from .conftest import synth_speech_like


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------
class TestDecoding:
    def test_decodes_wav_bytes(self, speech_wav):
        payload, _ = speech_wav
        data, channels, sr = decode_audio(payload)

        assert channels == 1
        assert sr == 16000
        assert data.shape == (32000, 1)
        assert data.dtype == np.float32

    def test_decodes_file_like_object(self, speech_wav, tmp_path):
        payload, name = speech_wav
        path = tmp_path / name
        path.write_bytes(payload)

        data, channels, sr = decode_audio(str(path))
        assert channels == 1 and sr == 16000
        assert data.shape[0] == 32000

    def test_detects_stereo(self, wav_bytes_factory):
        payload, _ = wav_bytes_factory(duration_sec=1.0, channels=2)
        _, channels, _ = decode_audio(payload)
        assert channels == 2

    def test_garbage_bytes_raise_invalid_audio(self):
        with pytest.raises(AudioValidationError) as excinfo:
            decode_audio(b"this is definitely not audio" * 64)
        assert excinfo.value.code == "INVALID_AUDIO"

    def test_truncated_wav_is_rejected_by_the_pipeline(self, speech_wav):
        """A WAV cut off inside its data block must not reach the model.

        ``decode_audio`` itself may or may not refuse the bytes - depending on
        the backend it can hand back the handful of samples that did arrive.
        What matters is that the length guard downstream rejects the result, so
        the test asserts on the full pipeline rather than on one decoder's
        tolerance for a malformed header.
        """
        payload, name = speech_wav

        with pytest.raises(AudioValidationError) as excinfo:
            prepare_from_bytes(payload[:200], filename=name)
        assert excinfo.value.code in {
            "INVALID_AUDIO",
            "EMPTY_FILE",
            "AUDIO_TOO_SHORT",
        }

    def test_a_header_with_no_samples_is_rejected(self):
        """The classic 'truncated in transit' file: valid header, empty data."""
        header = (
            b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
            b"\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00"
            b"\x02\x00\x10\x00data\x00\x00\x00\x00"
        )
        with pytest.raises(AudioValidationError) as excinfo:
            prepare_from_bytes(header, filename="stub.wav")
        assert excinfo.value.code in {"EMPTY_FILE", "INVALID_AUDIO", "AUDIO_TOO_SHORT"}

    def test_the_decode_failure_message_leaks_no_internals(self, tmp_path):
        """The error text is returned to HTTP clients, so it must be scrubbed.

        Decoder exceptions echo whatever they failed on. Two things must never
        reach a response: the heap address of the in-memory upload buffer, and -
        if a file source were ever decoded - the server's own filesystem path.
        """
        path = tmp_path / "not-really-audio.wav"
        path.write_bytes(b"this is definitely not audio" * 64)

        with pytest.raises(AudioValidationError) as excinfo:
            decode_audio(str(path))
        message = excinfo.value.message

        assert str(path) not in message
        assert str(tmp_path) not in message
        assert "0x" not in message
        # The useful part survives: which decoder, which exception type.
        assert "_decode_soundfile" in message
        assert "LibsndfileError" in message or "SoundFileError" in message

    def test_the_byte_path_message_leaks_no_address(self):
        with pytest.raises(AudioValidationError) as excinfo:
            decode_audio(b"not audio" * 100)
        assert "0x" not in excinfo.value.message


class TestRedaction:
    """``redact_for_client`` is the guarantee; these pin down what it removes."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Error opening <_io.BytesIO object at 0x7f8a1c0d5e40>", "Error opening <buffer>"),
            ("at 0x0000020a1b3c4d50", "at 0x..."),
            ("cannot read C:\\secret\\corpus\\x.wav", "cannot read <path>"),
            ("no such file: /srv/private/audio/x.flac", "no such file: <path>"),
            ("opening \\\\build\\secret\\a.wav", "opening <path>"),
        ],
    )
    def test_known_shapes(self, raw, expected):
        assert redact_for_client(raw) == expected

    def test_allowed_extensions_survive(self):
        """The suffix list is the actionable half of the message."""
        text = "Supported formats: .wav, .flac, .ogg, .mp3, .m4a."
        assert redact_for_client(text) == text

    def test_ordinary_prose_is_untouched(self):
        text = "Format not recognised; the data chunk is missing."
        assert redact_for_client(text) == text


# ---------------------------------------------------------------------------
# Mono / resample
# ---------------------------------------------------------------------------
class TestMonoAndResample:
    def test_to_mono_averages_channels(self):
        stereo = np.array([[1.0, -1.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32)
        mono = to_mono(stereo)
        assert mono.ndim == 1
        np.testing.assert_allclose(mono, [0.0, 0.5, 0.5], atol=1e-6)

    def test_to_mono_passes_through_1d(self):
        signal = np.linspace(-1, 1, 10, dtype=np.float32)
        np.testing.assert_allclose(to_mono(signal), signal)

    def test_resample_changes_length_proportionally(self, long_waveform):
        resampled = resample(long_waveform, 16000, 8000)
        assert abs(resampled.size - 40000) < 200  # 5 s at 8 kHz

    def test_resample_to_same_rate_is_identity(self, speech_waveform):
        out = resample(speech_waveform, 16000, 16000)
        np.testing.assert_array_equal(out, speech_waveform)

    def test_resampled_signal_keeps_its_pitch(self):
        """A 220 Hz tone must still measure 220 Hz after resampling.

        Catches an implementation that decimates without filtering (aliasing
        would fold the tone to an inaudible or unrelated frequency).
        """
        sr = 44100
        t = np.arange(sr, dtype=np.float32) / sr
        tone = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)

        out = resample(tone, sr, 16000)
        spectrum = np.abs(np.fft.rfft(out))
        peak_hz = float(np.argmax(spectrum)) * 16000 / out.size

        assert abs(peak_hz - 220.0) < 5.0


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
class TestWindowing:
    def test_output_length_is_exactly_the_window(self, audio_cfg, long_waveform):
        window, _ = load_and_prepare_from_array(long_waveform, audio_cfg)
        expected = int(audio_cfg.target_duration_sec * audio_cfg.sample_rate)
        assert window.size == expected
        assert window.dtype == np.float32

    def test_short_clip_is_padded_to_the_window(self, audio_cfg, speech_waveform):
        window, info = preprocess_waveform(speech_waveform, 16000, audio_cfg)
        expected = int(audio_cfg.target_duration_sec * audio_cfg.sample_rate)

        assert window.size == expected
        # Padding is zeros and the mask tells the model to ignore it.
        assert info.n_valid_samples < expected
        np.testing.assert_allclose(window[info.n_valid_samples :], 0.0, atol=1e-7)

    def test_long_clip_is_cropped_not_padded(self, audio_cfg, long_waveform):
        window, info = preprocess_waveform(long_waveform, 16000, audio_cfg)
        assert info.n_valid_samples == window.size
        assert np.count_nonzero(window[info.n_valid_samples - 1 :]) <= 1

    def test_center_crop_is_deterministic(self, audio_cfg, long_waveform):
        first, _ = preprocess_waveform(long_waveform, 16000, audio_cfg, crop="center")
        second, _ = preprocess_waveform(long_waveform, 16000, audio_cfg, crop="center")
        np.testing.assert_array_equal(first, second)

    def test_random_crop_varies_but_stays_in_range(self, audio_cfg, long_waveform):
        crops = []
        for seed in range(6):
            rng = np.random.default_rng(seed)
            window, _ = preprocess_waveform(long_waveform, 16000, audio_cfg, crop="random", rng=rng)
            crops.append(window)

        # Not every seed picks the same offset (that would mean the RNG is
        # ignored and augmentation is a no-op) ...
        assert any(not np.array_equal(crops[0], crop) for crop in crops[1:])
        # ... but every crop is real audio, never padding.
        for crop in crops:
            assert np.count_nonzero(crop) > crop.size * 0.5

    def test_peak_normalisation_is_applied(self, audio_cfg, wav_bytes_factory):
        """Recording gain must not be able to shift the input distribution."""
        quiet_bytes, quiet_name = wav_bytes_factory(
            waveform=synth_speech_like(2.0) * 0.05, name="q.wav"
        )
        loud_bytes, loud_name = wav_bytes_factory(waveform=synth_speech_like(2.0), name="l.wav")

        quiet_window, _ = load_and_prepare(quiet_bytes, audio_cfg, filename=quiet_name)
        loud_window, _ = load_and_prepare(loud_bytes, audio_cfg, filename=loud_name)

        # An 8 dB gain difference collapses to the same input distribution.
        assert np.isclose(np.max(np.abs(quiet_window)), audio_cfg.peak_target, atol=1e-3)
        assert np.isclose(np.max(np.abs(loud_window)), audio_cfg.peak_target, atol=1e-3)

    def test_silence_trimming_removes_leading_silence(self, audio_cfg):
        """Padding is added, then trimmed away, so it must not reach the model."""
        sr = audio_cfg.sample_rate
        silence = np.zeros(sr, dtype=np.float32)  # 1 s of digital silence
        speech = synth_speech_like(2.0)

        padded = np.concatenate([silence, speech, silence])
        window, info = preprocess_waveform(padded, sr, audio_cfg)

        assert info.trimmed_sec > 1.5  # both silent ends removed
        # The trimmed signal starts at real audio, not at digital silence.
        assert np.count_nonzero(window[: sr // 10]) > 0


def load_and_prepare_from_array(waveform: np.ndarray, cfg: AudioConfig):
    """Resample a float array through the real path (48 kHz -> 16 kHz)."""
    buffer = io.BytesIO()
    import soundfile as sf

    sf.write(buffer, waveform, 48000, format="WAV", subtype="PCM_16")
    return load_and_prepare(buffer.getvalue(), cfg)


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------
class TestRejections:
    def test_too_short_is_rejected(self, audio_cfg):
        short = synth_speech_like(audio_cfg.min_duration_sec / 2)
        with pytest.raises(AudioValidationError) as excinfo:
            preprocess_waveform(short, 16000, audio_cfg)
        assert excinfo.value.code == "AUDIO_TOO_SHORT"

    def test_too_long_is_rejected(self, audio_cfg):
        long = synth_speech_like(audio_cfg.max_duration_sec + 5.0)
        with pytest.raises(AudioValidationError) as excinfo:
            preprocess_waveform(long, 16000, audio_cfg)
        assert excinfo.value.code == "AUDIO_TOO_LONG"

    def test_empty_array_is_rejected(self, audio_cfg):
        with pytest.raises(AudioValidationError) as excinfo:
            preprocess_waveform(np.array([], dtype=np.float32), 16000, audio_cfg)
        assert excinfo.value.code == "EMPTY_FILE"

    def test_pure_silence_is_rejected(self, audio_cfg):
        """Silence carries no emotion; a confident prediction on it is a lie."""
        silence = np.zeros(int(2.0 * audio_cfg.sample_rate), dtype=np.float32)
        with pytest.raises(AudioValidationError) as excinfo:
            preprocess_waveform(silence, 16000, audio_cfg)
        assert excinfo.value.code == "SILENT_AUDIO"

    def test_nan_and_inf_are_neutralised(self, audio_cfg):
        """A corrupt sample must not propagate NaN through the whole network."""
        waveform = synth_speech_like(2.0)
        waveform[100] = np.nan
        waveform[200] = np.inf
        waveform[300] = -np.inf

        window, _ = preprocess_waveform(waveform, 16000, audio_cfg)
        assert np.isfinite(window).all()

    def test_error_codes_are_stable_strings(self):
        """Guards against someone switching to string matching on the message."""
        assert AudioValidationError("x", code="AUDIO_TOO_SHORT").code == "AUDIO_TOO_SHORT"
        assert issubclass(AudioValidationError, ValueError)


# ---------------------------------------------------------------------------
# Upload-level validation
# ---------------------------------------------------------------------------
class TestUploadValidation:
    @pytest.mark.parametrize(
        "name", ["clip.wav", "clip.WAV", "clip.flac", "a.b.c.ogg", "song.mp3", "v.m4a"]
    )
    def test_accepted_extensions(self, name, audio_cfg):
        assert validate_extension(name, audio_cfg) == "." + name.split(".")[-1].lower()

    @pytest.mark.parametrize(
        "name",
        [
            "payload.exe",
            "script.sh",
            "data.csv",
            "no_extension",
            "archive.zip",
            "../../etc/passwd.wav.exe",
        ],
    )
    def test_rejected_extensions(self, name, audio_cfg):
        with pytest.raises(AudioValidationError) as excinfo:
            validate_extension(name, audio_cfg)
        assert excinfo.value.code == "UNSUPPORTED_FORMAT"

    def test_path_traversal_names_cannot_impersonate_an_extension(self, audio_cfg):
        """``..`` in the name must not smuggle a path past the check.

        The check reads only the final suffix, so a traversal name is judged on
        its extension like anything else.
        """
        assert validate_extension("../../../../etc/passwd.wav", audio_cfg) == ".wav"
        with pytest.raises(AudioValidationError):
            validate_extension("../../../../etc/passwd", audio_cfg)

    def test_empty_upload_rejected(self, audio_cfg):
        with pytest.raises(AudioValidationError) as excinfo:
            validate_size(0, audio_cfg)
        assert excinfo.value.code == "EMPTY_FILE"

    def test_oversized_upload_rejected(self, audio_cfg):
        with pytest.raises(AudioValidationError) as excinfo:
            validate_size(audio_cfg.max_upload_bytes + 1, audio_cfg)
        assert excinfo.value.code == "FILE_TOO_LARGE"

    def test_size_boundary_is_inclusive(self, audio_cfg):
        validate_size(audio_cfg.max_upload_bytes, audio_cfg)  # must not raise


# ---------------------------------------------------------------------------
# End-to-end entry points
# ---------------------------------------------------------------------------
class TestEntryPoints:
    def test_path_and_bytes_agree(self, speech_wav, tmp_path, audio_cfg):
        """The CLI (path) and the API (bytes) must produce identical features."""
        payload, name = speech_wav
        path = tmp_path / name
        path.write_bytes(payload)

        from_path, _ = load_and_prepare(str(path), audio_cfg)
        from_bytes, _ = prepare_from_bytes(payload, audio_cfg, filename=name)

        np.testing.assert_allclose(from_path, from_bytes, atol=1e-6)

    def test_bytes_peak_normalised(self, wav_bytes_factory, audio_cfg):
        payload, name = wav_bytes_factory(duration_sec=2.0)
        window, _ = prepare_from_bytes(payload, audio_cfg, filename=name)
        assert np.isclose(np.max(np.abs(window)), audio_cfg.peak_target, atol=1e-3)

    def test_reports_native_and_analysed_rate(self, wav_bytes_factory, audio_cfg):
        payload, name = wav_bytes_factory(duration_sec=2.0, sample_rate=48000)
        _, info = prepare_from_bytes(payload, audio_cfg, filename=name)

        assert info.native_sample_rate == 48000
        assert info.sample_rate == audio_cfg.sample_rate == 16000
        assert info.channels == 1

    def test_audio_info_serialises_to_api_shape(self, speech_wav, audio_cfg):
        _, info = prepare_from_bytes(speech_wav[0], audio_cfg, filename="clip.wav")
        payload = info.to_dict()

        # Exactly the keys declared by AudioInfoResponse.
        assert set(payload) == {
            "duration_sec",
            "sample_rate",
            "channels",
            "native_sample_rate",
            "trimmed_sec",
            "peak_amplitude",
        }
        assert all(isinstance(v, (int, float)) for v in payload.values())

    def test_rejects_bytes_whose_name_disagrees(self, speech_wav, audio_cfg):
        payload, _ = speech_wav
        with pytest.raises(AudioValidationError) as excinfo:
            prepare_from_bytes(payload, audio_cfg, filename="clip.exe")
        assert excinfo.value.code == "UNSUPPORTED_FORMAT"
