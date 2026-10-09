"""Pydantic response models.

These are the API's public contract. The emotion labels themselves are **not**
hard-coded here - the API validates against whatever the loaded model declares,
so the schema cannot disagree with the model it serves.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AudioInfoResponse(BaseModel):
    """Facts about the audio that was actually analysed."""

    duration_sec: float = Field(..., description="Duration of the source file.")
    sample_rate: int = Field(..., description="Rate the model analysed (16 kHz).")
    channels: int = Field(..., description="Channel count of the source file.")
    native_sample_rate: int = Field(
        ..., description="Sample rate of the source file before resampling."
    )
    trimmed_sec: float = Field(..., description="Seconds of leading/trailing silence removed.")
    peak_amplitude: float = Field(..., description="Peak amplitude before normalisation (0-1).")


class RankedEmotion(BaseModel):
    emotion: str
    probability: float


class ServedModelRef(BaseModel):
    """Just enough to identify which model answered.

    Deliberately not the whole of ``GET /model``. A prediction response repeats
    on every request, and full metadata - architecture, feature settings,
    preprocessing limits - is constant for the lifetime of the process. Sending
    ~40 lines of identical JSON with every prediction would be bandwidth spent on
    nothing; clients that want the detail call ``GET /model`` once.

    Included here: the run name, the selection metric that produced it, and the
    validation score it achieved. Those are the three facts that make a
    prediction auditable - without them a response could have come from any
    checkpoint and there would be no way to tell.
    """

    run_name: str
    parameters: int
    selection_metric: str
    selection_value: float | None = None
    trained_epoch: int | None = None
    device: str


class PredictionResponse(BaseModel):
    """Successful prediction."""

    emotion: str = Field(..., description="Predicted emotion label.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Probability assigned to the top emotion."
    )
    emotion_index: int
    #: Full distribution over every class the model can output. Keys always
    #: match the model's label list exactly.
    probabilities: dict[str, float]
    ranked_emotions: list[RankedEmotion]
    audio: AudioInfoResponse
    processing_ms: float = Field(..., description="Server-side processing time.")
    model: ServedModelRef = Field(..., description="Which model produced this result.")
    request_id: str | None = None


class EmotionInfo(BaseModel):
    """One supported emotion, with what the model does with it."""

    index: int
    label: str
    ravdess_code: str | None = Field(None, description="RAVDESS filename code, when applicable.")


class ModelInfoResponse(BaseModel):
    """Description of the loaded model, served by ``GET /model``."""

    labels: list[str]
    num_classes: int
    architecture: str
    parameters: int
    feature_extraction: dict[str, object]
    preprocessing: dict[str, object]
    dataset: str | None = None
    run_name: str | None = None
    trained_epoch: int | None = None
    selection: dict[str, object] = Field(default_factory=dict)
    device: str | None = None


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' or 'degraded'.")
    model_loaded: bool
    model_run: str | None = None
    num_classes: int | None = None
    version: str
    detail: str | None = None


class ErrorDetail(BaseModel):
    """Uniform error body for every failure mode."""

    code: str = Field(
        ...,
        description=(
            "Stable machine-readable code: NO_FILE, UNSUPPORTED_FORMAT, "
            "FILE_TOO_LARGE, EMPTY_FILE, INVALID_AUDIO, AUDIO_TOO_SHORT, "
            "AUDIO_TOO_LONG, SILENT_AUDIO, MODEL_UNAVAILABLE, "
            "INTERNAL_ERROR."
        ),
    )
    message: str = Field(..., description="Human-readable explanation.")
    detail: str | None = Field(
        None, description="Extra context. Never contains server filesystem paths."
    )


class ErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: str | None = None


__all__ = [
    "AudioInfoResponse",
    "EmotionInfo",
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
    "ModelInfoResponse",
    "PredictionResponse",
    "RankedEmotion",
]
