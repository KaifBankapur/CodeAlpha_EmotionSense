"""MFCC feature extraction - the acoustic front-end of the whole system.

What MFCC is and why speech needs it
------------------------------------
An emotion recogniser looks at *how* someone says something, and that information
is carried by three things:

1. **The spectral envelope** - the overall shape of how energy is distributed
   across frequency. Anger tends to be low-frequency and heavy; sadness tends to
   be high-frequency and light. A raw FFT gives the envelope only implicitly,
   buried under the harmonic structure (pitch) and spread linearly across Hz.
2. **The frequency axis being perceptually spaced.** Human hearing is roughly
   logarithmic, so two frequencies at 200 Hz and 400 Hz matter much less to a
   listener than 2 kHz and 4 kHz. A linear spectrogram treats them as equally
   spaced.
3. **Energy being compactly encoded.** A spectrum has ~257 bins for a 512-point
   FFT, but the envelope is smooth and low-dimensional.

MFCCs fix all three:

* **Mel filterbank** - triangular filters spaced on the mel scale
  ``m = 2595 * log10(1 + f/700)`` resample the spectrum into perceptually
  meaningful bands and integrate out the fine structure, keeping the envelope.
* **log compression** - ``log(max(x, eps))`` converts multiplicative energy into
  additive dB-like units, matching the roughly logarithmic response of human
  hearing and preventing loud bands from dominating.
* **DCT** - the Orthogonal Discrete Cosine Transform decorrelates the log-mel
  coefficients and concentrates the *envelope* into the first ~20 coefficients,
  giving a compact, low-dimensional representation. This is the "C" in MFCC.

Why 40 coefficients and not the 13 typical of ASR
-------------------------------------------------
Speech-recognition front-ends want phoneme identity, which lives in the first
13 cepstra. Emotion instead depends on voice quality, bandwidth, spectral
tilt and phonation - energy distribution across the *whole* cepstral sequence -
so the higher coefficients are informative here rather than noise.

Why delta features
------------------
Emotion is largely expressed as *movement* over time: rising pitch and
increasing energy for surprise, a falling trajectory for sadness. Derivatives of
the cepstra (``delta``, ``delta2``) expose that dynamics directly and typically
buy several accuracy points on paralinguistic tasks. They are stacked
frame-wise below, giving a ``(n_features, n_frames)`` matrix.
"""

from __future__ import annotations

import logging

import numpy as np

from ml.config import FeatureConfig

logger = logging.getLogger(__name__)

_EPS = 1e-10


class MFCCExtractor:
    """Stateless, serialisable MFCC front-end.

    Stateless matters: there is no fitting step, so a saved extractor config is
    always sufficient to reproduce the features exactly. This is what allows the
    inference service to reconstruct the front-end from a JSON file.

    Example:
        >>> extractor = MFCCExtractor(FeatureConfig())
        >>> feats = extractor.transform(window_16k, 16000)
        >>> feats.shape
        (120, 301)
    """

    def __init__(self, cfg: FeatureConfig | None = None) -> None:
        self.cfg = cfg or FeatureConfig()

    # -- properties -------------------------------------------------------
    @property
    def n_features(self) -> int:
        return self.cfg.n_output_features()

    def output_shape(self, n_samples: int, sample_rate: int) -> tuple:
        """Predicted ``(n_features, n_frames)`` - used by tests and the cache."""
        n_frames = 1 + int(n_samples // self.cfg.hop_length)
        return (self.n_features, n_frames)

    # -- core -------------------------------------------------------------
    def transform(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Compute the stacked MFCC / delta / delta-delta matrix.

        Args:
            waveform: mono float32 samples, peak-normalised to ``[-1, 1]``.
            sample_rate: sample rate of ``waveform`` (16 kHz by default).

        Returns:
            ``(n_features, n_frames)`` float32 array, CMVN applied if enabled.
            Frame 0 is time, features are in rows.
        """
        import librosa

        cfg = self.cfg
        y = np.asarray(waveform, dtype=np.float32).reshape(-1)

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sample_rate,
            n_mfcc=cfg.n_mfcc,
            n_mels=cfg.n_mels,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
            window=cfg.window,
        ).astype(np.float32)

        planes = [mfcc]
        if cfg.use_delta:
            planes.append(librosa.feature.delta(mfcc, order=1, axis=-1).astype(np.float32))
        if cfg.use_delta2:
            planes.append(librosa.feature.delta(mfcc, order=2, axis=-1).astype(np.float32))

        features = np.concatenate(planes, axis=0)

        if cfg.cmvn:
            features = apply_cmvn(features)

        return np.ascontiguousarray(features, dtype=np.float32)

    __call__ = transform

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self.cfg)

    @classmethod
    def from_dict(cls, payload: dict) -> MFCCExtractor:
        from dataclasses import fields

        cfg = FeatureConfig()
        known = {f.name for f in fields(FeatureConfig)}
        for key, value in (payload or {}).items():
            if key in known:
                setattr(cfg, key, value)
        return cls(cfg)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"MFCCExtractor(n_mfcc={self.cfg.n_mfcc}, n_mels={self.cfg.n_mels}, "
            f"n_fft={self.cfg.n_fft}, hop={self.cfg.hop_length}, "
            f"fmax={self.cfg.fmax}, {self.cfg.describe()}, cmvn={self.cfg.cmvn})"
        )


def apply_cmvn(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Cepstral mean-and-variance normalisation over the time axis.

    For each feature plane, subtract the utterance mean and divide by the
    utterance standard deviation. This is a cheap, training-free form of speaker
    and channel normalisation: it removes the constant offset and scale that
    come from *who is speaking and what microphone was used*, leaving the
    temporal shape that carries the emotion.

    The per-utterance standard deviation is a real quantity here - unlike
    speaker-independent ASR, in emotion recognition utterance-level loudness
    variation is itself a cue - so it is retained rather than replaced by a
    global constant.
    """
    features = np.asarray(features, dtype=np.float32)
    mean = features.mean(axis=1, keepdims=True)
    std = features.std(axis=1, keepdims=True) + eps
    return ((features - mean) / std).astype(np.float32)


def log_mel_spectrogram(
    waveform: np.ndarray, sample_rate: int, cfg: FeatureConfig | None = None
) -> np.ndarray:
    """Log-mel spectrogram - the intermediate representation MFCCs are built from.

    Not fed to the model (the DCT already discards the redundant phase and the
    envelope redundancy), but exposed for the visualisations in the README and
    for anyone who wants to reason about what the network actually sees.
    """
    import librosa

    cfg = cfg or FeatureConfig()
    return librosa.feature.melspectrogram(
        y=np.asarray(waveform, dtype=np.float32).reshape(-1),
        sr=sample_rate,
        n_mels=cfg.n_mels,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        window=cfg.window,
    ).astype(np.float32)


__all__ = ["MFCCExtractor", "apply_cmvn", "log_mel_spectrogram"]
