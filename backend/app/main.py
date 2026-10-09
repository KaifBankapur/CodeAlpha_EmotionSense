"""FastAPI application factory and process wiring.

Responsibilities
----------------
* build the app, mount the router, configure CORS;
* load the model **once** during startup (and only once);
* translate every error into the same JSON envelope, with a request id that
  links a client-visible failure to a server-side log line.

Run it::

    uvicorn backend.app.main:app --reload --port 8000
    python -m backend.app.main            # equivalent, uses Settings.host/port
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app import __doc__ as _package_doc  # noqa: F401  (path bootstrap)
from backend.app.api.errors import ApiError
from backend.app.api.routes import router
from backend.app.core.config import Settings, get_settings
from backend.app.schemas.prediction import ErrorDetail, ErrorResponse
from backend.app.services.inference import InferenceService
from ml.utils.logging_utils import setup_logging

logger = logging.getLogger("backend")

REQUEST_ID_HEADER = "X-Request-ID"

# Starlette renamed this constant and deprecated the old name. Reading it off
# ``status`` keeps working on either version without emitting a warning, and a
# warning promoted to an error by ``filterwarnings = ["error"]`` is a test
# failure, so the fallback is load-bearing rather than decorative.
_HTTP_422 = (
    getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", None) or status.HTTP_422_UNPROCESSABLE_ENTITY
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown.

    The model is loaded here rather than lazily per request: the alternative
    would let the first request after a restart pay seconds of disk I/O and
    model construction, and would allow a half-loaded state to serve traffic.
    """
    settings: Settings = app.state.settings
    setup_logging(settings.log_level)

    logger.info("starting %s v%s", settings.app_name, settings.version)
    logger.info("model run override: %s", settings.model_run or "(active pointer)")

    service: InferenceService = app.state.inference
    predictor = service.load_model()

    if predictor is None and settings.require_model_on_startup:
        # Fail loudly at boot. Serving 503 on every request hides a
        # misconfiguration behind an apparently healthy process.
        raise RuntimeError(
            "No trained model could be loaded. Train one first:\n"
            "    python scripts/download_dataset.py\n"
            "    python scripts/prepare_data.py\n"
            "    python scripts/train.py --set-active\n"
            f"Underlying error: {service.load_error}"
        )
    if predictor is None:
        logger.warning(
            "starting without a model; /predict will return 503 until one is "
            "trained. Set SER_REQUIRE_MODEL_ON_STARTUP=true to fail instead."
        )

    yield

    logger.info("shutting down")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Exposed separately so tests can build variants."""
    settings = settings or get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=settings.description,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.state.settings = settings
    # Instantiated eagerly so routes never see a missing attribute, even if the
    # lifespan hook is skipped by an unusual test harness.
    app.state.inference = InferenceService(settings)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            # An explicit origin list, never "*". Credentials plus a wildcard
            # origin is rejected by browsers anyway, and pairing it with
            # allow_credentials is a classic security mistake.
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "Accept", REQUEST_ID_HEADER],
            expose_headers=[REQUEST_ID_HEADER],
            max_age=600,
        )

    app.middleware("http")(request_id_middleware)
    app.include_router(router)

    _register_exception_handlers(app)
    return app


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
async def request_id_middleware(request: Request, call_next):
    """Attach a request id used in logs and echoed back to the client.

    Trusting a client-supplied ``X-Request-ID`` is safe here because it is only
    ever logged, never used to build a path or query.
    """
    request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled exception request_id=%s path=%s", request_id, request.url.path)
        raise

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    response.headers[REQUEST_ID_HEADER] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    if elapsed_ms > 500:
        logger.info("slow request %s %s -> %.0f ms", request.method, request.url.path, elapsed_ms)
    return response


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
def _error_body(request: Request, code: str, message: str, detail: str | None) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorDetail(code=code, message=message, detail=detail),
        request_id=getattr(request.state, "request_id", None),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=payload.model_dump(exclude_none=True),
    )


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        """Our own typed errors: the message is safe to return verbatim."""
        if exc.status_code >= 500:
            logger.error(
                "api error request_id=%s code=%s detail=%s",
                getattr(request.state, "request_id", None),
                exc.code,
                exc.detail,
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=ErrorDetail(code=exc.code, message=exc.message, detail=exc.detail),
                request_id=getattr(request.state, "request_id", None),
            ).model_dump(exclude_none=True),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Map FastAPI's 422 onto our envelope, without echoing the input back.

        Validation error bodies can contain the offending value; a submitted
        file payload must never be reflected in a response.
        """
        problems = [
            f"{'.'.join(str(p) for p in err.get('loc', [])[1:])}: {err.get('msg')}"
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=_HTTP_422,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="VALIDATION_ERROR",
                    message="The request could not be processed.",
                    detail="; ".join(problems) or None,
                ),
                request_id=getattr(request.state, "request_id", None),
            ).model_dump(exclude_none=True),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        """Last resort. The client gets a generic message; the log gets the trace."""
        logger.exception(
            "unhandled error request_id=%s path=%s",
            getattr(request.state, "request_id", None),
            request.url.path,
        )
        return _error_body(
            request,
            "INTERNAL_ERROR",
            "An unexpected internal error occurred.",
            type(exc).__name__,
        )


app = create_app()


def main() -> None:
    """Entry point for ``python -m backend.app.main``."""
    import uvicorn

    settings = get_settings()
    setup_logging(settings.log_level)
    uvicorn.run(
        "backend.app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
