"""
The API's inference service.

Responsibilities kept out of the route handlers on purpose:

* load the model **once** at startup and hold it (see
  :meth:`InferenceService.load_model`);
* read uploads **incrementally with a hard byte ceiling**, so a client cannot
  exhaust server memory by claiming a huge ``Content-Length``;
* translate ML-layer failures (:class:`AudioValidationError`) into typed API
  errors with the right HTTP status.

A deliberate security decision: **uploads are never written to disk.** The bytes
are decoded straight from memory (``soundfile`` accepts a ``BytesIO``), which
removes the whole class of temp-file problems at once - no path traversal, no
leftover files, no cleanup on crash, and nothing for another request to read.
With a 20 MB ceiling the memory cost is bounded.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

from fastapi import UploadFile

from backend.app.api.errors import (
    AudioTooLongError,
    AudioTooShortError,
    EmptyFileError,
    FileTooLargeError,
    InvalidAudioError,
    ModelUnavailableError,
    SilentAudioError,
    UnsupportedFormatError,
)
from backend.app.core.config import Settings
from ml.config import MODELS_DIR
from ml.data.preprocessing import AudioValidationError
from ml.inference.predictor import EmotionPredictor, PredictionResult

logger = logging.getLogger(__name__)

#: Read granularity for uploads. 64 KB keeps syscall overhead low without
#: holding the whole body in one buffer.
READ_CHUNK = 64 * 1024

#: Maps a machine-readable AudioValidationError code onto an API error type.
#:
#: Each code keeps its own error class so the response carries the *specific*
#: reason. Collapsing them all into ``InvalidAudioError`` would work, but the
#: client can then only say "unreadable audio" for a clip that is perfectly
#: readable and merely silent - a materially different problem for the user,
#: with a different fix (check the microphone vs. re-export the file).
_ERROR_MAP = {
    "INVALID_AUDIO": InvalidAudioError,
    "EMPTY_FILE": EmptyFileError,
    "FILE_TOO_LARGE": FileTooLargeError,
    "UNSUPPORTED_FORMAT": UnsupportedFormatError,
    "AUDIO_TOO_SHORT": AudioTooShortError,
    "AUDIO_TOO_LONG": AudioTooLongError,
    "SILENT_AUDIO": SilentAudioError,
}


def safe_filename(name: str | None) -> str:
    """Reduce a client-supplied filename to a bare basename.

    We never use the client's name to build a path, but stripping directory
    components means even a future change that logs or echoes it cannot leak a
    traversal path such as ``../../etc/passwd``.
    """
    if not name:
        return "upload"
    cleaned = str(name).replace("\\", "/").split("/")[-1].strip()
    return cleaned or "upload"


class InferenceService:
    """Owns the single model instance used by all requests."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._predictor: EmotionPredictor | None = None
        self._load_error: str | None = None
        #: Bounds how many blocking forward passes run at once. Inference is
        #: CPU-bound, so unbounded concurrency makes every request slower rather
        #: than faster.
        self._semaphore = threading.Semaphore(settings.max_concurrent_predictions)
        self._lock = threading.Lock()

    # -- lifecycle --------------------------------------------------------
    def load_model(self) -> EmotionPredictor | None:
        """Load the configured (or active) model. Returns ``None`` on failure."""
        with self._lock:
            if self._predictor is not None:
                return self._predictor
            try:
                run = self.settings.model_run
                if run:
                    candidate = Path(run)
                    if not candidate.is_dir():
                        candidate = MODELS_DIR / run
                    logger.info("loading model from explicit run: %s", candidate)
                    predictor = EmotionPredictor.from_run_dir(candidate)
                else:
                    logger.info("loading active model")
                    predictor = EmotionPredictor.load_active()

                predictor.warmup()
                self._predictor = predictor
                self._load_error = None
                logger.info(
                    "model ready: %s (%d classes, %s)",
                    predictor.metadata.get("run_name"),
                    len(predictor.labels),
                    predictor.device,
                )
                return predictor
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                logger.error("failed to load model: %s", self._load_error)
                return None

    @property
    def is_ready(self) -> bool:
        return self._predictor is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def get_predictor(self) -> EmotionPredictor:
        if self._predictor is None:
            self.load_model()
        if self._predictor is None:
            raise ModelUnavailableError(
                "No trained model is loaded. Train one with "
                "`python scripts/train.py` and restart the service.",
                detail=self._load_error,
            )
        return self._predictor

    def reload(self) -> EmotionPredictor | None:
        """Drop the cached model and load it again."""
        with self._lock:
            self._predictor = None
            self._load_error = None
        return self.load_model()

    # -- upload handling --------------------------------------------------
    async def read_upload(self, upload: UploadFile) -> bytes:
        """Read an upload into memory, refusing to exceed the configured limit.

        The limit is enforced *while reading* rather than by trusting
        ``Content-Length``, because that header is client-supplied and a
        malicious client can simply understate it.
        """
        limit = self.settings.max_upload_bytes
        chunks: list[bytes] = []
        total = 0

        while True:
            chunk = await upload.read(READ_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                await upload.close()
                raise FileTooLargeError(
                    f"File exceeds the {limit / (1024 * 1024):.0f} MB limit.",
                    detail=f"Stopped reading after {total:,} bytes.",
                )
            chunks.append(chunk)

        if total == 0:
            raise EmptyFileError("The uploaded file contains no data.")

        return b"".join(chunks)

    # -- prediction -------------------------------------------------------
    def predict_bytes(self, payload: bytes, filename: str | None = None) -> PredictionResult:
        """Blocking prediction. Call from a worker thread, not the event loop."""
        predictor = self.get_predictor()
        display_name = safe_filename(filename)

        with self._semaphore:
            try:
                return predictor.predict(payload, filename=display_name)
            except AudioValidationError as exc:
                error_type = _ERROR_MAP.get(exc.code, InvalidAudioError)
                raise error_type(exc.message) from exc

    async def predict_upload(self, upload: UploadFile) -> PredictionResult:
        """Read (async) then predict (in a worker thread)."""
        payload = await self.read_upload(upload)
        return await asyncio.to_thread(self.predict_bytes, payload, upload.filename)

    # -- reporting --------------------------------------------------------
    def model_info(self) -> dict:
        return self.get_predictor().model_info()


__all__ = ["READ_CHUNK", "InferenceService", "safe_filename"]
