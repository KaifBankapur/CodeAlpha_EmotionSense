"""The inference engine: one trained model, reusable by the CLI and the API.

Design contract
---------------
* **Load once, reuse.** Constructing an :class:`EmotionPredictor` reads the
  checkpoint and its config from disk; every subsequent prediction is a forward
  pass. The API loads a single instance at startup.
* **The front-end comes from the artifact.** Audio and MFCC settings are read
  out of the checkpoint, never out of the current defaults, so editing
  ``ml/config.py`` cannot silently change how an existing model is served.
* **Same pipeline as training.** Prediction calls the identical
  ``load_and_prepare`` -> ``MFCCExtractor`` path used to build the training
  cache. There is no second, "simplified" inference path.

Usage:
    predictor = EmotionPredictor.from_run_dir(Path("artifacts/models/expA"))
    result = predictor.predict("clip.wav")
    print(result.emotion, result.confidence)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
import torch

from ml.config import ExperimentConfig, get_active_run_dir, resolve_num_threads
from ml.data.preprocessing import (
    AudioValidationError,
    load_and_prepare,
    validate_extension,
    validate_size,
)
from ml.features.mfcc import MFCCExtractor
from ml.models.emotion_cnn_lstm import EmotionCNNBiLSTM

logger = logging.getLogger(__name__)

AudioInput = Union[str, Path, bytes, bytearray]

ERROR_CODE_PREFIX = "AUDIO_"


@dataclass
class PredictionResult:
    """A single prediction, in a shape both the CLI and the API can serialise."""

    emotion: str
    confidence: float
    probabilities: dict[str, float]
    emotion_index: int
    ranked: list[tuple[str, float]]
    audio: dict[str, object]
    processing_ms: float
    model: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "emotion": self.emotion,
            "confidence": round(float(self.confidence), 6),
            "emotion_index": self.emotion_index,
            "probabilities": {
                label: round(float(value), 6) for label, value in self.probabilities.items()
            },
            "ranked_emotions": [
                {"emotion": label, "probability": round(float(value), 6)}
                for label, value in self.ranked
            ],
            "audio": self.audio,
            "processing_ms": round(float(self.processing_ms), 2),
            "model": self.model,
        }


class EmotionPredictor:
    """Loads a trained model and turns audio into an emotion prediction.

    Thread safety: a single lock serialises the forward pass. PyTorch's forward
    is re-entrant, but concurrent requests would otherwise each try to use the
    whole intra-op thread pool and thrash on a CPU-only host.
    """

    def __init__(
        self,
        model: EmotionCNNBiLSTM,
        experiment: ExperimentConfig,
        *,
        run_dir: Path | None = None,
        metadata: dict[str, object] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.experiment = experiment
        self.run_dir = Path(run_dir) if run_dir else None
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels: list[str] = list(experiment.labels)
        self.audio_cfg = experiment.audio
        self.feature_cfg = experiment.features
        self.extractor = MFCCExtractor(self.feature_cfg)
        self.metadata: dict[str, object] = dict(metadata or {})

        self.model.to(self.device)
        self.model.eval()
        self._lock = threading.Lock()

        expected = self.extractor.n_features
        if self.model.n_input_features != expected:
            raise ValueError(
                f"Model expects {self.model.n_input_features} input features but "
                f"the saved feature config produces {expected}. The checkpoint "
                "and its config disagree - refusing to serve."
            )

    # -- construction -----------------------------------------------------
    @classmethod
    def from_run_dir(
        cls,
        run_dir: str | Path,
        *,
        device: torch.device | None = None,
    ) -> EmotionPredictor:
        """Load a specific run directory."""
        from ml.training.trainer import load_checkpoint

        run_dir = Path(run_dir)
        checkpoint = run_dir / "best_model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"No 'best_model.pt' in run '{run_dir.name}'. "
                "Train a model first: python scripts/train.py"
            )

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, experiment, payload = load_checkpoint(checkpoint, device)

        metadata = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "dataset": experiment.dataset,
            "checkpoint_epoch": payload.get("epoch"),
            "selection_metric": payload.get("monitor"),
            "selection_value": payload.get("monitor_value"),
            "n_parameters": model.num_parameters(),
            "n_input_features": payload.get("n_input_features"),
            "feature_description": experiment.features.describe(),
            "window_sec": experiment.audio.target_duration_sec,
            "sample_rate": experiment.audio.sample_rate,
            "notes": experiment.notes,
            "training_metrics": payload.get("metrics", {}),
        }
        logger.info(
            "loaded model '%s' (%s, %s params, %d classes) from %s",
            run_dir.name,
            payload.get("monitor"),
            f"{model.num_parameters():,}",
            len(experiment.labels),
            checkpoint,
        )
        return cls(model, experiment, run_dir=run_dir, metadata=metadata, device=device)

    @classmethod
    def load_active(cls, *, device: torch.device | None = None) -> EmotionPredictor:
        """Load whichever run the pointer file designates."""
        return cls.from_run_dir(get_active_run_dir(), device=device)

    # -- internals --------------------------------------------------------
    def _features_to_tensor(
        self, source: AudioInput, *, filename: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        window, info = load_and_prepare(source, self.audio_cfg, crop="center", filename=filename)
        matrix = self.extractor.transform(window, self.audio_cfg.sample_rate)

        if matrix.shape[0] != self.model.n_input_features:
            raise RuntimeError(
                f"Feature mismatch: extractor produced {matrix.shape[0]} planes "
                f"but the model expects {self.model.n_input_features}."
            )

        valid = min(matrix.shape[1], 1 + int(info.n_valid_samples // self.feature_cfg.hop_length))
        mask = np.zeros(matrix.shape[1], dtype=np.float32)
        mask[: max(1, valid)] = 1.0

        features = torch.from_numpy(matrix).unsqueeze(0).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        return features, mask_tensor, info.to_dict()

    # -- public API -------------------------------------------------------
    def predict(self, source: AudioInput, *, filename: str | None = None) -> PredictionResult:
        """Predict the emotion in an audio file or byte string.

        Args:
            source: filesystem path, raw bytes, or a readable binary object.
            filename: optional name, used for the extension check only.

        Raises:
            AudioValidationError: for empty, oversized, undecodable, too short,
                too long or silent audio.
        """
        started = time.perf_counter()

        if isinstance(source, (bytes, bytearray)):
            validate_size(len(source), self.audio_cfg)
            if filename:
                validate_extension(filename, self.audio_cfg)

        features, mask, audio_info = self._features_to_tensor(source, filename=filename)

        with self._lock, torch.inference_mode():
            logits = self.model(features, frame_mask=mask)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)

        scores = probabilities.detach().cpu().numpy().astype(np.float64)
        # Renormalise against float round-off so the returned distribution sums
        # to exactly 1.0 - the frontend draws bars from these numbers.
        total = float(scores.sum())
        if total > 0:
            scores = scores / total

        best_index = int(np.argmax(scores))
        ranked_indices = np.argsort(scores)[::-1]
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return PredictionResult(
            emotion=self.labels[best_index],
            confidence=float(scores[best_index]),
            probabilities={label: float(scores[i]) for i, label in enumerate(self.labels)},
            emotion_index=best_index,
            ranked=[(self.labels[i], float(scores[i])) for i in ranked_indices.tolist()],
            audio=audio_info,
            processing_ms=elapsed_ms,
            model=self.served_model_ref(),
        )

    def served_model_ref(self) -> dict[str, object]:
        """Compact identity of the served model, for per-prediction responses.

        See ``backend.app.schemas.prediction.ServedModelRef`` for why this is a
        subset of :meth:`model_info` rather than the whole thing.
        """
        info = self.model_info()
        selection = info.get("selection") or {}
        return {
            "run_name": info["run_name"],
            "parameters": info["parameters"],
            "selection_metric": selection.get("metric", "unknown"),
            "selection_value": selection.get("value"),
            "trained_epoch": info.get("trained_epoch"),
            "device": info["device"],
        }

    def predict_from_bytes(self, payload: bytes, filename: str | None = None) -> PredictionResult:
        """Convenience wrapper used by the HTTP upload path."""
        return self.predict(payload, filename=filename)

    def _synthetic_upload(self) -> bytes:
        """A tiny in-memory WAV for warming the front-end, never the model.

        Deliberately encoded at 22.05 kHz rather than the model's 16 kHz: if the
        rate already matched, ``resample`` would short-circuit and the band-limited
        resampler would stay cold until a real upload arrived.
        """
        import io

        import soundfile as sf

        native_sr = 22050
        seconds = self.audio_cfg.target_duration_sec * 0.6
        t = np.arange(int(native_sr * seconds)) / native_sr
        # A decaying tone: enough energy to clear the RMS gate and the silence
        # trimmer, and cheap to synthesise.
        tone = (0.35 * np.sin(2 * np.pi * 220.0 * t) * np.exp(-2.0 * t)).astype(np.float32)
        buffer = io.BytesIO()
        sf.write(buffer, tone, native_sr, format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    def warmup(self, repeats: int = 1) -> float:
        """Run one complete prediction on synthetic audio so the first real
        request is not slow.

        Warming only the model forward pass is not enough. Measured on this
        machine, a freshly started server answered its first ``POST /predict`` in
        ~21.6 s and every subsequent one in ~15 ms: PyTorch does lazy kernel
        selection and one-off allocation, and librosa/soundfile/soxr likewise
        defer work until an actual waveform is decoded and resampled. Warming
        zeros skips all of that, so the cost simply moved to the first real user.

        This runs the identical path - decode, resample, trim, window,
        normalise, MFCC, forward - on a synthesised clip, which is what
        ``processing_ms`` will actually measure when a user clicks the button.
        """
        started = time.perf_counter()

        try:
            payload = self._synthetic_upload()
        except Exception:  # warmup must never block startup
            logger.warning("warmup: could not synthesise a test clip", exc_info=True)
            payload = None

        with self._lock, torch.inference_mode():
            for _ in range(max(1, repeats)):
                if payload is None:
                    # Fall back to warming the forward pass alone.
                    n_frames = 1 + int(
                        self.audio_cfg.target_duration_sec
                        * self.audio_cfg.sample_rate
                        // self.feature_cfg.hop_length
                    )
                    features = torch.zeros(
                        1, 1, self.model.n_input_features, n_frames, device=self.device
                    )
                    mask = torch.ones(1, n_frames, device=self.device)
                    self.model(features, frame_mask=mask)
                else:
                    features, mask, _info = self._features_to_tensor(payload, filename="warmup.wav")
                    self.model(features, frame_mask=mask)

        elapsed = (time.perf_counter() - started) * 1000.0
        logger.info(
            "warmup complete in %.1f ms (%d pass(es), front-end %s)",
            elapsed,
            max(1, repeats),
            "warmed" if payload is not None else "unavailable",
        )
        return elapsed

    def model_info(self) -> dict[str, object]:
        """Public description of the loaded model, surfaced by ``GET /model``."""
        return {
            "labels": list(self.labels),
            "num_classes": len(self.labels),
            "architecture": "CNN(2-D) -> BiLSTM -> masked attention pooling",
            "parameters": self.model.num_parameters(),
            "feature_extraction": {
                "type": "MFCC + delta + delta-delta",
                "n_mfcc": self.feature_cfg.n_mfcc,
                "n_mels": self.feature_cfg.n_mels,
                "n_fft": self.feature_cfg.n_fft,
                "hop_length": self.feature_cfg.hop_length,
                "window": self.feature_cfg.win_length,
                "sample_rate": self.audio_cfg.sample_rate,
                "cmvn": self.feature_cfg.cmvn,
                "features_per_frame": self.model.n_input_features,
            },
            "preprocessing": {
                "window_sec": self.audio_cfg.target_duration_sec,
                "resample_to_hz": self.audio_cfg.sample_rate,
                "mono": self.audio_cfg.mono,
                "trim_silence": self.audio_cfg.trim_silence,
                "trim_top_db": self.audio_cfg.trim_top_db,
                "peak_normalize": self.audio_cfg.peak_normalize,
                "accepted_formats": list(self.audio_cfg.allowed_extensions),
                "max_upload_mb": round(self.audio_cfg.max_upload_bytes / 1048576, 1),
                "min_duration_sec": self.audio_cfg.min_duration_sec,
                "max_duration_sec": self.audio_cfg.max_duration_sec,
            },
            "dataset": self.metadata.get("dataset"),
            "run_name": self.metadata.get("run_name"),
            "trained_epoch": self.metadata.get("checkpoint_epoch"),
            "selection": {
                "metric": self.metadata.get("selection_metric"),
                "value": self.metadata.get("selection_value"),
            },
            "device": str(self.device),
            **{
                k: v
                for k, v in self.metadata.items()
                if k
                in {
                    "training_metrics",
                    "feature_description",
                    "window_sec",
                    "sample_rate",
                    "notes",
                }
            },
        }


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------
_PREDICTOR: EmotionPredictor | None = None
_PREDICTOR_LOCK = threading.Lock()


def get_predictor(reload: bool = False) -> EmotionPredictor:
    """Return the shared predictor, loading the active model on first use.

    Raises:
        FileNotFoundError: if no trained model exists. Callers (e.g. the API's
            startup hook) should surface this as a clear configuration error
            rather than letting every request fail.
    """
    global _PREDICTOR
    with _PREDICTOR_LOCK:
        if _PREDICTOR is None or reload:
            threads = resolve_num_threads()
            torch.set_num_threads(threads)
            _PREDICTOR = EmotionPredictor.load_active()
            _PREDICTOR.warmup()
        return _PREDICTOR


def set_predictor(predictor: EmotionPredictor | None) -> None:
    """Install a predictor explicitly (used by tests)."""
    global _PREDICTOR
    with _PREDICTOR_LOCK:
        _PREDICTOR = predictor


__all__ = [
    "AudioValidationError",
    "EmotionPredictor",
    "PredictionResult",
    "get_predictor",
    "set_predictor",
]
