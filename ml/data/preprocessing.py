"""Waveform-level audio preprocessing.

This module is the *only* place where raw audio is turned into the fixed-length,
16 kHz, mono float32 tensor that the MFCC front-end consumes. Training
(:mod:`ml.data.dataset`), the CLI and the HTTP API all call
:func:`load_and_prepare`, which is what guarantees that a prediction at inference
time is produced by exactly the same signal chain as the one the model was
trained on.

Pipeline
--------
1. **Decode** - soundfile first (wav/flac/ogg/mp3, fast), librosa as fallback.
2. **Mono** - average channels. RAVDESS is already mono, uploads may not be.
3. **Resample** - to ``AudioConfig.sample_rate`` (16 kHz by default) using
   ``soxr_hq``. 16 kHz covers the full 0-8 kHz mel band and is 3x cheaper than
   RAVDESS' native 48 kHz, which matters on a CPU-only machine.
4. **Validate** - reject empty, too short, too long or near-silent audio.
5. **Trim silence** - ``librosa.effects.trim`` removes the dead air that RAVDESS
   pads around each utterance, so that fixed-length padding is not dominated by
   silence.
6. **Fit to window** - centre/random crop to ``target_duration_sec``, else pad.
7. **Peak-normalise** - scale so the loudest sample is ``peak_target``. This
   removes microphone gain and speaking distance, and being a scalar multiply it
   preserves all *relative* dynamics inside the window, which is where the
   emotion cue lives.

Steps 5-7 are done in that order on purpose: normalising after trimming/cropping
means every window uses the same amplitude scale, and the peak is guaranteed to
come from actual speech rather than from a click in a silent lead-in.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import numpy as np

from ml.config import AudioConfig

logger = logging.getLogger(__name__)

AudioSource = Union[str, Path, bytes, bytearray, io.IOBase]
CropMode = Literal["center", "random", "start"]

#: Matches a heap address such as ``0x0000020a1b3c4d50``.
_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]+")
#: Matches an absolute filesystem path - ``C:\data\x.wav``, ``\\\\host\\share``,
#: ``/srv/audio/x.wav``. A leading drive letter, UNC prefix or slash is enough
#: to tell it apart from the ``.wav``-style suffixes that legitimately appear
#: in these messages.
_ABS_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\|/)[^\s'\"<>|,;]*")
#: ``<_io.BytesIO object at 0x...>`` and friends. Applied *after* the address has
#: been flattened, which is why the pattern expects ``0x...``.
_BUFFER_REPR_RE = re.compile(r"<(?:_io\.)?\w+ object at 0x\.\.\.>")


def redact_for_client(text: str) -> str:
    """Make a third-party library's error message safe to return to a client.

    Decoder exceptions happily echo the object they failed on. For an upload
    that is an in-memory buffer whose repr carries a heap address; for a file
    source it would be the *server's own path*. Neither belongs in an HTTP
    response, and the guarantee should not depend on which decoder happened to
    fail, so it is enforced here rather than at each call site.

    >>> redact_for_client("Error opening <_io.BytesIO object at 0x7f8a1c0d5e40>")
    'Error opening <buffer>'
    >>> redact_for_client("cannot read C:\\\\secret\\\\corpus\\\\x.wav")
    'cannot read <path>'
    """
    cleaned = _ADDRESS_RE.sub("0x...", text)
    cleaned = _BUFFER_REPR_RE.sub("<buffer>", cleaned)
    return _ABS_PATH_RE.sub("<path>", cleaned)


class AudioValidationError(ValueError):
    """Raised when an input cannot be used for emotion recognition.

    Deliberately a ``ValueError`` subclass so API layers can map it onto HTTP
    400 without catching unrelated programming errors.

    Attributes:
        code: stable machine-readable identifier (``AUDIO_TOO_SHORT``,
            ``SILENT_AUDIO``, ...). The API maps this onto an HTTP status and a
            response body. String-matching on the message would silently break
            the moment a message is reworded.
    """

    def __init__(self, message: str, *, code: str = "INVALID_AUDIO") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass(frozen=True)
class AudioInfo:
    """Metadata about a decoded file, surfaced in API responses."""

    duration_sec: float
    sample_rate: int
    channels: int
    native_sample_rate: int
    #: Seconds of leading+trailing silence removed by the trim stage.
    trimmed_sec: float = 0.0
    peak_amplitude: float = 0.0
    #: Samples of real (post-trim, post-crop) audio inside the padded window.
    #: The tail beyond this offset is zero padding, and the downstream attention
    #: pooling masks it out so the classifier never attends to silence it did not
    #: ask for. Always ``target_duration_sec * sample_rate`` when the clip was
    #: long enough to fill the window.
    n_valid_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "duration_sec": round(self.duration_sec, 3),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "native_sample_rate": self.native_sample_rate,
            "trimmed_sec": round(self.trimmed_sec, 3),
            "peak_amplitude": round(self.peak_amplitude, 4),
        }


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------
def _decode_soundfile(source: AudioSource):
    import soundfile as sf

    if isinstance(source, (bytes, bytearray)):
        data = io.BytesIO(bytes(source))
        return sf.read(data, dtype="float32", always_2d=True)
    if isinstance(source, io.IOBase):
        return sf.read(source, dtype="float32", always_2d=True)
    return sf.read(str(source), dtype="float32", always_2d=True)


def _decode_librosa(source: AudioSource):
    """Fallback decoder. Handles containers soundfile cannot open."""
    import librosa

    path = source
    if isinstance(source, (bytes, bytearray, io.IOBase)):
        data = bytes(source.read()) if isinstance(source, io.IOBase) else bytes(source)
        path = io.BytesIO(data)

    # mono=True mixes down and resamples to AudioConfig.sample_rate in one step.
    waveform, sr = librosa.load(path, sr=None, mono=False)
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    return waveform.T.astype(np.float32), int(sr)  # (frames, channels)


def decode_audio(
    source: AudioSource, cfg: AudioConfig | None = None
) -> tuple[np.ndarray, int, int]:
    """Decode any supported audio source to ``(samples, channels, native_sr)``.

    Raises:
        AudioValidationError: if no decoder could read the input.
    """
    cfg = cfg or AudioConfig()
    errors = []
    for decoder in (_decode_soundfile, _decode_librosa):
        try:
            data, sr = decoder(source)
            break
        except Exception as exc:  # noqa: BLE001 - both decoders raise library types
            errors.append(
                f"{decoder.__name__}: {type(exc).__name__}: {redact_for_client(str(exc))}"
            )
    else:
        raise AudioValidationError(
            "Could not decode the audio. Supported formats: "
            f"{', '.join(cfg.allowed_extensions)}. Decoder errors: "
            + " | ".join(errors),
            code="INVALID_AUDIO",
        )

    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if data.size == 0:
        raise AudioValidationError("The audio file contains no samples.", code="EMPTY_FILE")

    return data, int(data.shape[1]), int(sr)


def to_mono(data: np.ndarray) -> np.ndarray:
    """Average a multi-channel array down to one channel."""
    if data.ndim == 1:
        return data
    return data.mean(axis=1).astype(np.float32)


def resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Band-limited resampling via ``soxr_hq``."""
    if orig_sr == target_sr:
        return waveform.astype(np.float32, copy=False)
    import librosa

    return librosa.resample(
        y=waveform.astype(np.float32, copy=False),
        orig_sr=orig_sr,
        target_sr=target_sr,
        res_type="soxr_hq",
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Upload-level validation (no decoding required)
# ---------------------------------------------------------------------------
def validate_extension(filename: str, cfg: AudioConfig | None = None) -> str:
    """Check the suffix against the allow-list. Returns the lowercased suffix.

    This is a *cheap* first line of defence only. A caller can always rename
    ``payload.exe`` to ``payload.wav``; the decode step is what actually decides
    whether the bytes are audio.
    """
    cfg = cfg or AudioConfig()
    suffix = Path(str(filename or "")).suffix.lower()
    if suffix not in cfg.allowed_extensions:
        raise AudioValidationError(
            f"Unsupported file type '{suffix or 'unknown'}'. "
            f"Allowed: {', '.join(cfg.allowed_extensions)}.",
            code="UNSUPPORTED_FORMAT",
        )
    return suffix


def validate_size(n_bytes: int, cfg: AudioConfig | None = None) -> None:
    cfg = cfg or AudioConfig()
    if n_bytes <= 0:
        raise AudioValidationError("The uploaded file is empty.", code="EMPTY_FILE")
    if n_bytes > cfg.max_upload_bytes:
        limit_mb = cfg.max_upload_bytes / (1024 * 1024)
        raise AudioValidationError(
            f"File is too large ({n_bytes / (1024 * 1024):.1f} MB). "
            f"The limit is {limit_mb:.0f} MB.",
            code="FILE_TOO_LARGE",
        )


# ---------------------------------------------------------------------------
# Core transform
# ---------------------------------------------------------------------------
def preprocess_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    cfg: AudioConfig | None = None,
    *,
    crop: CropMode = "center",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, AudioInfo]:
    """Apply steps 4-7 of the pipeline to an already-decoded, mono waveform.

    Args:
        waveform: mono float32 samples at ``sample_rate``.
        crop: ``"random"`` during training (data augmentation by position),
            ``"center"`` at inference so predictions are deterministic.

    Returns:
        ``(window, info)`` where ``window`` has exactly
        ``cfg.target_duration_sec * cfg.sample_rate`` samples.
    """
    cfg = cfg or AudioConfig()
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)

    if waveform.size == 0:
        raise AudioValidationError("The audio file contains no samples.", code="EMPTY_FILE")

    duration = waveform.size / float(sample_rate)
    if duration < cfg.min_duration_sec:
        raise AudioValidationError(
            f"Audio is too short ({duration:.2f}s); "
            f"at least {cfg.min_duration_sec:.2f}s is required.",
            code="AUDIO_TOO_SHORT",
        )
    if duration > cfg.max_duration_sec:
        raise AudioValidationError(
            f"Audio is too long ({duration:.1f}s); "
            f"the limit is {cfg.max_duration_sec:.0f}s.",
            code="AUDIO_TOO_LONG",
        )

    # --- 5. silence trim -------------------------------------------------
    trimmed_sec = 0.0
    if cfg.trim_silence:
        import librosa

        waveform, _span = librosa.effects.trim(y=waveform, top_db=cfg.trim_top_db)
        trimmed_sec = (waveform.size / float(sample_rate)) - duration
        if waveform.size < int(cfg.min_duration_sec * sample_rate):
            raise AudioValidationError(
                "The file is silent - no speech was detected after "
                "silence trimming.",
                code="SILENT_AUDIO",
            )

    # --- 6. fit to fixed window -----------------------------------------
    target_len = int(round(cfg.target_duration_sec * sample_rate))
    valid_samples = waveform.size
    if waveform.size > target_len:
        if crop == "random":
            rng = rng or np.random.default_rng()
            start_idx = int(rng.integers(0, waveform.size - target_len + 1))
        elif crop == "start":
            start_idx = 0
        else:
            start_idx = (waveform.size - target_len) // 2
        waveform = waveform[start_idx : start_idx + target_len]
        valid_samples = target_len
    elif waveform.size < target_len:
        waveform = np.pad(waveform, (0, target_len - waveform.size), mode="constant")

    # --- energy gate ------------------------------------------------------
    rms = float(np.sqrt(np.mean(np.square(waveform))))
    if rms < cfg.min_rms:
        raise AudioValidationError(
            "The file is silent - the average signal energy is too low to "
            "analyse. Check the recording volume.",
            code="SILENT_AUDIO",
        )

    # --- 7. peak normalisation ------------------------------------------
    peak = float(np.max(np.abs(waveform)))
    if cfg.peak_normalize and peak > 0:
        waveform = waveform * (cfg.peak_target / peak)
    np.clip(waveform, -1.0, 1.0, out=waveform)

    info = AudioInfo(
        duration_sec=duration,
        sample_rate=sample_rate,
        channels=1,
        native_sample_rate=sample_rate,
        trimmed_sec=abs(trimmed_sec),
        peak_amplitude=peak,
        n_valid_samples=valid_samples,
    )
    return waveform.astype(np.float32, copy=False), info


def load_and_prepare(
    source: AudioSource,
    cfg: AudioConfig | None = None,
    *,
    crop: CropMode = "center",
    rng: np.random.Generator | None = None,
    filename: str | None = None,
) -> tuple[np.ndarray, AudioInfo]:
    """Decode -> mono -> resample -> validate -> trim -> window -> normalise.

    The one function the dataset loader, the CLI and the API all share.

    Args:
        source: path, raw bytes, or a binary file object.
        filename: optional name used only for the extension check.

    Returns:
        ``(window, info)``; ``window`` is mono float32 at ``cfg.sample_rate``
        with exactly ``target_duration_sec * sample_rate`` samples.
    """
    cfg = cfg or AudioConfig()

    if filename:
        validate_extension(filename, cfg)

    if isinstance(source, (bytes, bytearray)):
        validate_size(len(source), cfg)

    data, channels, native_sr = decode_audio(source, cfg)
    native_duration = data.shape[0] / float(native_sr)
    native_channels = channels

    mono = to_mono(data) if cfg.mono else data.reshape(-1)
    if not cfg.mono and mono.ndim > 1:  # pragma: no cover - mono=True by default
        raise AudioValidationError("Multi-channel processing is not enabled.")

    waveform = resample(mono, native_sr, cfg.sample_rate)
    waveform, info = preprocess_waveform(
        waveform, cfg.sample_rate, cfg, crop=crop, rng=rng
    )

    info = AudioInfo(
        duration_sec=info.duration_sec,
        sample_rate=info.sample_rate,
        channels=native_channels,
        native_sample_rate=native_sr,
        trimmed_sec=info.trimmed_sec,
        peak_amplitude=info.peak_amplitude,
        n_valid_samples=info.n_valid_samples,
    )
    logger.debug(
        "prepared audio: native=%dHz/%dch/%.2fs -> %.2fs @ %dHz",
        native_sr,
        native_channels,
        native_duration,
        cfg.target_duration_sec,
        cfg.sample_rate,
    )
    return waveform, info


def prepare_from_bytes(
    payload: bytes,
    cfg: AudioConfig | None = None,
    *,
    filename: str | None = None,
    crop: CropMode = "center",
) -> tuple[np.ndarray, AudioInfo]:
    """Convenience wrapper for the API path (already-read upload bytes)."""
    cfg = cfg or AudioConfig()
    validate_size(len(payload), cfg)
    if filename:
        validate_extension(filename, cfg)
    return load_and_prepare(payload, cfg, crop=crop, filename=filename)


__all__ = [
    "AudioInfo",
    "AudioValidationError",
    "decode_audio",
    "load_and_prepare",
    "prepare_from_bytes",
    "preprocess_waveform",
    "resample",
    "to_mono",
    "validate_extension",
    "validate_size",
]
