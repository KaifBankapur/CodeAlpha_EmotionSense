"""Security-focused tests.

The threat model is narrow and stated up front, because "security tests" without
one just assert that nothing crashed:

* an unauthenticated public upload endpoint;
* an untrusted, client-supplied filename;
* an untrusted request body of unbounded declared size;
* a server-side error surface that must not leak filesystem paths.

Everything else (TLS, auth, rate limiting beyond the concurrency cap) is out of
scope for this project and is not claimed.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.services.inference import safe_filename


def make_client(**overrides) -> TestClient:
    return TestClient(create_app(Settings(require_model_on_startup=False, **overrides)))


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A fully started client with the real model loaded (skipped if untrained).

    Module-scoped: loading the checkpoint once keeps the suite fast, so
    function-scoped fixtures (``tmp_path`` and friends) cannot be mixed in.
    """
    from ml.config import get_active_run_dir

    try:
        get_active_run_dir()
    except FileNotFoundError:
        pytest.skip("No trained model. Run `python scripts/train.py --set-active`.")

    with make_client() as test_client:
        yield test_client


@pytest.fixture
def bare_client() -> TestClient:
    """A client whose model load is forced to fail, for the 503 path."""
    return make_client(model_run="__no_such_run__")


# ---------------------------------------------------------------------------
# Filenames are untrusted input
# ---------------------------------------------------------------------------
class TestFilenameHardening:
    @pytest.mark.parametrize(
        "supplied,expected",
        [
            ("clip.wav", "clip.wav"),
            ("../../../../etc/passwd", "passwd"),
            ("..\\..\\windows\\system32\\cmd.exe", "cmd.exe"),
            ("/absolute/path/to/file.wav", "file.wav"),
            ("C:\\Users\\victim\\secret.wav", "secret.wav"),
            ("....//....//etc/shadow.wav", "shadow.wav"),
            ("", "upload"),
            (None, "upload"),
            ("   ", "upload"),
            ("/", "upload"),
            ("C:\\", "upload"),
            ("a/b/", "upload"),
        ],
    )
    def test_directory_components_are_stripped(self, supplied, expected):
        """The name is reduced to a bare basename before anything else sees it."""
        assert safe_filename(supplied) == expected

    def test_no_separator_survives(self):
        for name in ("a/b/c.wav", "a\\b\\c.wav", "//x//y.wav"):
            assert "/" not in safe_filename(name)
            assert "\\" not in safe_filename(name)

    def test_traversal_names_do_not_escape_the_upload_limit(self):
        """A traversal name is judged on its extension, like any other."""
        from ml.data.preprocessing import AudioValidationError, validate_extension

        assert validate_extension("../../../../etc/passwd.wav") == ".wav"
        with pytest.raises(AudioValidationError):
            validate_extension("../../../../etc/passwd")

    def test_traversal_filename_does_not_break_prediction(self, client, speech_wav):
        """A hostile name is stored as a basename and never used as a path."""
        payload, _ = speech_wav
        response = client.post(
            "/predict",
            files={"file": ("../../../../../../etc/passwd.wav", payload, "audio/wav")},
        )
        assert response.status_code == 200

    def test_null_bytes_in_a_filename_are_harmless(self, client, speech_wav):
        payload, _ = speech_wav
        response = client.post("/predict", files={"file": ("evil\x00.wav", payload, "audio/wav")})
        # Either the name is rejected or it is neutralised; it must never be
        # 500 and must never appear as a path.
        assert response.status_code in {200, 400, 415}
        if response.status_code != 200:
            assert "error" in response.json()


# ---------------------------------------------------------------------------
# Upload size is enforced while reading
# ---------------------------------------------------------------------------
class TestUploadSizeEnforcement:
    def test_limit_applies_without_a_content_length_header(self, speech_wav):
        """A lying or absent Content-Length must not bypass the cap.

        The service reads the body in chunks and stops at the ceiling, so a
        client that declares nothing still cannot exhaust memory.
        """
        payload, name = speech_wav
        padded = payload + b"\0" * (512 * 1024)

        with make_client(max_upload_bytes=8192) as client:
            response = client.post("/predict", files={"file": (name, padded)})

        assert response.status_code == 413
        assert response.json()["error"]["code"] == "FILE_TOO_LARGE"

    def test_a_lying_content_length_does_not_help(self, speech_wav):
        payload, name = speech_wav
        padded = payload + b"\0" * (256 * 1024)

        with make_client(max_upload_bytes=8192) as client:
            response = client.post(
                "/predict",
                files={"file": (name, padded)},
                headers={"Content-Length": "10"},
            )

        assert response.status_code == 413

    def test_under_the_limit_is_accepted(self, speech_wav):
        payload, name = speech_wav
        with make_client(max_upload_bytes=4 * 1024 * 1024) as client:
            response = client.post("/predict", files={"file": (name, payload)})
        assert response.status_code == 200

    def test_reading_stops_early_rather_than_buffering_everything(self):
        """The cap must abort mid-read, not after the whole body is in memory."""
        import inspect

        from backend.app.services.inference import READ_CHUNK, InferenceService

        assert READ_CHUNK > 0
        source = inspect.getsource(InferenceService.read_upload)
        # The limit is compared inside the read loop, not after it.
        assert "while True:" in source
        assert source.index("total > limit") < source.index('return b"".join(chunks)')


# ---------------------------------------------------------------------------
# Nothing is written to disk
# ---------------------------------------------------------------------------
class TestNoDiskWrites:
    def test_prediction_leaves_no_files_behind(self, client, speech_wav):
        """Uploads are decoded from memory, so there is nothing to clean up.

        This is the design that removes path traversal, stale temp files and
        crash-cleanup concerns in one move.
        """
        import os
        import tempfile

        before = set(os.listdir(tempfile.gettempdir()))
        payload, name = speech_wav

        for _ in range(3):
            assert client.post("/predict", files={"file": (name, payload)}).status_code == 200

        after = set(os.listdir(tempfile.gettempdir()))
        assert after - before <= {"__pycache__"}

    def test_no_temporary_file_is_created_in_the_project(self, client, speech_wav):
        payload, name = speech_wav
        client.post("/predict", files={"file": (name, payload)})

        leftovers = [
            p
            for p in Path(".").rglob("*")
            if p.is_file() and p.suffix in {".tmp", ".part", ".upload"} and ".venv" not in str(p)
        ]
        assert leftovers == []

    def test_failures_also_leave_nothing_behind(self, client):
        for bad in (b"junk" * 40, b"", b"RIFF" + b"\x00" * 200):
            client.post("/predict", files={"file": ("x.wav", bad)})

        leftovers = [p for p in Path(".").rglob("*.tmp") if ".venv" not in str(p)]
        assert leftovers == []


# ---------------------------------------------------------------------------
# Error surfaces do not leak internals
# ---------------------------------------------------------------------------
class TestNoInformationLeakage:
    def _body_text(self, response) -> str:
        return response.text.lower()

    @pytest.mark.parametrize(
        "filename,content",
        [
            ("clip.exe", b"MZ\x90\x00binary"),
            ("clip.wav", b"garbage" * 50),
            ("clip.zip", b"PK\x03\x04"),
            ("clip.wav", b""),
        ],
    )
    def test_error_bodies_contain_no_filesystem_paths(self, client, filename, content):
        response = client.post("/predict", files={"file": (filename, content)})
        text = self._body_text(response)

        assert response.status_code >= 400
        for leak in (
            "emotion recognition from speech",
            "artifacts/models",
            "data/raw",
            ".venv",
            "site-packages",
            "d:\\python312",
            "traceback",
            "site-packages\\",
        ):
            assert leak not in text, f"response leaked {leak!r}"

    def test_error_bodies_do_not_echo_the_uploaded_content(self, client):
        marker = "SECRETVALUE12345"
        response = client.post("/predict", files={"file": ("clip.wav", (marker * 40).encode())})
        assert marker.lower() not in self._body_text(response)

    def test_missing_model_error_does_not_expose_a_path(self, bare_client, speech_wav):
        payload, name = speech_wav
        with bare_client as client:
            response = client.post("/predict", files={"file": (name, payload)})

        assert response.status_code == 503
        text = self._body_text(response)
        assert "artifacts/models" not in text
        assert "traceback" not in text
        # The user still gets something actionable.
        assert "train" in text

    def test_settings_contain_no_secret_shaped_fields(self):
        """Nothing here should be a credential waiting to be logged."""
        for name in Settings.model_fields:
            lowered = name.lower()
            assert not any(
                token in lowered
                for token in ("secret", "password", "token", "api_key", "credential")
            ), f"Settings.{name} looks like a secret"

    def test_model_endpoint_exposes_no_local_paths(self, client):
        payload = client.get("/model").json()
        text = str(payload).lower()
        assert "artifacts" not in text
        assert ".venv" not in text
        assert "data/raw" not in text


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
class TestCORSHardening:
    def test_default_origins_are_the_dev_ports_only(self):
        settings = Settings()
        assert set(settings.cors_origins) == {
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        }

    def test_no_origin_is_allowed_by_default_when_the_list_is_empty(self):
        with make_client(cors_origins=[]) as client:
            response = client.get("/health", headers={"Origin": "https://any.test"})
            assert "access-control-allow-origin" not in response.headers

    def test_a_wildcard_origin_is_refused_at_startup(self):
        """Loudly, rather than shipping a blanket cross-origin grant.

        The API needs no cookies or credentials, so ``*`` would appear to work
        while silently opening the endpoint to every origin. Rejecting it in the
        settings makes that a startup error instead.
        """
        with pytest.raises(ValueError, match="must list explicit origins"):
            Settings(cors_origins=["*"])

        with pytest.raises(ValueError, match="must list explicit origins"):
            Settings(cors_origins="http://localhost:5173,*")

    def test_a_valid_origin_list_still_passes(self):
        assert Settings(cors_origins=["http://localhost:5173"]).cors_origins == [
            "http://localhost:5173"
        ]

    def test_only_the_declared_methods_are_allowed(self, client):
        response = client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "DELETE",
            },
        )
        assert "delete" not in response.headers.get("access-control-allow-methods", "").lower()


# ---------------------------------------------------------------------------
# Resource limits
# ---------------------------------------------------------------------------
class TestResourceLimits:
    def test_concurrency_cap_is_configurable_and_bounded(self):
        assert Settings(max_concurrent_predictions=1).max_concurrent_predictions == 1
        with pytest.raises(Exception):
            Settings(max_concurrent_predictions=0)
        with pytest.raises(Exception):
            Settings(max_concurrent_predictions=999)

    def test_a_zip_bomb_looking_payload_is_not_accepted_as_audio(self, client):
        """A large highly-compressible blob is not decodable audio."""
        with io.BytesIO() as buffer:
            sf.write(buffer, np.zeros(32000, dtype=np.float32), 16000, format="WAV")
            payload = buffer.getvalue()

        # Prepend a zip-style local header; the decoders must still refuse it.
        hostile = b"PK\x03\x04" + b"\x00" * 400 + payload
        response = client.post("/predict", files={"file": ("clip.wav", hostile)})
        assert response.status_code >= 400

    def test_service_refuses_to_buffer_an_impossible_upload(self):
        """Sanity check on the configured floor for the upload limit."""
        with pytest.raises(Exception):
            Settings(max_upload_bytes=1)

    def test_a_very_long_gentle_wav_is_rejected_not_queued(self, client, wav_bytes_factory):
        payload, name = wav_bytes_factory(duration_sec=2.0)
        padded = payload + b"\0" * (30 * 1024 * 1024)  # 30 MB, over the 20 MB cap

        response = client.post("/predict", files={"file": (name, padded)})
        assert response.status_code == 413
