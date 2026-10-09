"""Typed API errors.

The API distinguishes *the client sent something unacceptable* (4xx, actionable
message) from *something on the server broke* (5xx, generic message). Internal
exception text and filesystem paths never reach the client; they are logged with
the request id instead, so an operator can still find them.
"""

from __future__ import annotations

from typing import Optional


class ApiError(Exception):
    """Base class for errors that map onto a specific HTTP status."""

    status_code = 500
    code = "INTERNAL_ERROR"
    default_message = "An unexpected error occurred."

    def __init__(self, message: Optional[str] = None, *, detail: Optional[str] = None) -> None:
        self.message = message or self.default_message
        self.detail = detail
        super().__init__(self.message)


class NoFileError(ApiError):
    status_code = 400
    code = "NO_FILE"
    default_message = "No audio file was included in the request."


class UnsupportedFormatError(ApiError):
    status_code = 415
    code = "UNSUPPORTED_FORMAT"
    default_message = "That audio format is not supported."


class FileTooLargeError(ApiError):
    status_code = 413
    code = "FILE_TOO_LARGE"
    default_message = "The uploaded file is too large."


class EmptyFileError(ApiError):
    status_code = 400
    code = "EMPTY_FILE"
    default_message = "The uploaded file is empty."


class InvalidAudioError(ApiError):
    status_code = 400
    code = "INVALID_AUDIO"
    default_message = "The uploaded file could not be read as audio."


class AudioTooShortError(ApiError):
    status_code = 400
    code = "AUDIO_TOO_SHORT"
    default_message = "The audio clip is too short to analyse."


class AudioTooLongError(ApiError):
    status_code = 400
    code = "AUDIO_TOO_LONG"
    default_message = "The audio clip is too long to analyse."


class SilentAudioError(ApiError):
    status_code = 400
    code = "SILENT_AUDIO"
    default_message = "No speech was detected in the audio."


class ModelUnavailableError(ApiError):
    status_code = 503
    code = "MODEL_UNAVAILABLE"
    default_message = "No trained model is currently loaded."


class TooManyRequestsError(ApiError):
    status_code = 429
    code = "TOO_MANY_REQUESTS"
    default_message = "The server is busy. Please retry shortly."


__all__ = [
    "ApiError",
    "AudioTooLongError",
    "AudioTooShortError",
    "EmptyFileError",
    "FileTooLargeError",
    "InvalidAudioError",
    "ModelUnavailableError",
    "NoFileError",
    "SilentAudioError",
    "TooManyRequestsError",
    "UnsupportedFormatError",
]
