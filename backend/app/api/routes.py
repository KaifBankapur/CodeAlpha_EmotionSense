"""HTTP route handlers.

Endpoint contract
-----------------
``GET  /``          service banner and a link map
``GET  /health``    liveness + whether a model is loaded (never raises)
``GET  /emotions``  the label set, read from the loaded model
``GET  /model``     architecture, features and preprocessing actually in use
``POST /predict``   multipart upload of one audio file -> emotion prediction

Every response shape is declared by :mod:`backend.app.schemas`, so the generated
OpenAPI document is an accurate contract rather than a guess.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, File, Request, UploadFile, status
from fastapi.responses import JSONResponse

from backend.app.api.errors import ApiError, NoFileError
from backend.app.core.config import Settings, get_settings
from backend.app.schemas.prediction import (
    EmotionInfo,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    RankedEmotion,
)
from backend.app.services.inference import InferenceService
from ml.config import RAVDESS_EMOTION_CODES

logger = logging.getLogger(__name__)

router = APIRouter()

#: Reverse lookup so /emotions can show the RAVDESS filename code next to a label.
_CODE_BY_LABEL = {label: code for code, label in RAVDESS_EMOTION_CODES.items()}

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "Empty, unreadable, silent or out-of-range audio"},
    413: {"model": ErrorResponse, "description": "Upload exceeds the size limit"},
    415: {"model": ErrorResponse, "description": "Unsupported audio format"},
    429: {"model": ErrorResponse, "description": "Too many concurrent predictions"},
    503: {"model": ErrorResponse, "description": "No trained model loaded"},
}


def _service(request: Request) -> InferenceService:
    return request.app.state.inference


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
@router.get("/", summary="Service information")
async def root(settings: Settings = get_settings()) -> dict[str, Any]:
    """Human-facing banner so hitting the bare host is not a dead end."""
    return {
        "service": settings.app_name,
        "version": settings.version,
        "description": settings.description,
        "endpoints": {
            "health": "GET /health",
            "emotions": "GET /emotions",
            "model": "GET /model",
            "predict": "POST /predict (multipart/form-data, field: file)",
            "docs": "GET /docs",
        },
    }


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and model readiness",
)
async def health(request: Request, settings: Settings = get_settings()) -> HealthResponse:
    """Never raises, even when the model failed to load.

    A health endpoint that 500s cannot tell a load balancer *why* the service is
    unhealthy, so failures are reported as data (``status='degraded'``).
    """
    service = _service(request)
    ready = service.is_ready
    predictor = service._predictor

    return HealthResponse(
        status="ok" if ready else "degraded",
        model_loaded=ready,
        model_run=(predictor.metadata.get("run_name") if predictor else None),
        num_classes=len(predictor.labels) if predictor else None,
        version=settings.version,
        detail=None if ready else service.load_error,
    )


@router.get(
    "/emotions",
    response_model=list[EmotionInfo],
    summary="Emotion classes the model can predict",
)
async def emotions(request: Request) -> list[EmotionInfo]:
    """Labels come from the loaded model, never from a hard-coded list here."""
    predictor = _service(request).get_predictor()
    return [
        EmotionInfo(
            index=index,
            label=label,
            ravdess_code=_CODE_BY_LABEL.get(label),
        )
        for index, label in enumerate(predictor.labels)
    ]


@router.get(
    "/model",
    response_model=ModelInfoResponse,
    summary="Loaded model, features and preprocessing",
)
async def model_info(request: Request) -> ModelInfoResponse:
    return ModelInfoResponse(**_service(request).model_info())


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict the emotion in an audio file",
    responses=_ERROR_RESPONSES,
)
async def predict(
    request: Request,
    file: UploadFile = File(
        ...,
        description="Audio file: wav, flac, ogg, mp3 or m4a.",
    ),
) -> PredictionResponse:
    """Accepts one audio upload and returns the model's prediction.

    The response contains the winning emotion, its confidence, the full
    probability distribution over every class, facts about the audio that was
    analysed, and metadata identifying the model that produced the answer.
    """
    service = _service(request)

    if file is None or not file.filename:
        raise NoFileError(
            "No audio file was uploaded. Send multipart/form-data with a 'file' field."
        )

    try:
        result = await service.predict_upload(file)
    except ApiError:
        raise
    except Exception as exc:
        # Never leak internals (paths, stack traces) to the client; the request
        # id ties the client-visible failure to this server-side log line.
        logger.exception("unhandled error while predicting request_id=%s", _request_id(request))
        raise ApiError(
            "The prediction failed because of an internal error.",
            detail=type(exc).__name__,
        ) from exc
    finally:
        await file.close()

    payload = result.to_dict()
    return PredictionResponse(
        emotion=payload["emotion"],
        confidence=payload["confidence"],
        emotion_index=payload["emotion_index"],
        probabilities=payload["probabilities"],
        ranked_emotions=[RankedEmotion(**item) for item in payload["ranked_emotions"]],
        audio=payload["audio"],
        processing_ms=payload["processing_ms"],
        model=payload["model"],
        request_id=_request_id(request),
    )


@router.post(
    "/model/reload",
    summary="Reload the model from disk",
    include_in_schema=False,
)
async def reload_model(request: Request) -> JSONResponse:
    """Operational helper: pick up a newly trained checkpoint without a restart."""
    service = _service(request)
    predictor = service.reload()
    if predictor is None:
        raise ApiError("Reload failed: " + str(service.load_error))
    return JSONResponse({"status": "reloaded", "run_name": predictor.metadata.get("run_name")})


__all__ = ["router"]
