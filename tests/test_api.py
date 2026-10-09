"""HTTP contract tests for the FastAPI backend.

Everything here goes through the real app: real routing, real Pydantic
validation, real error handlers. Nothing is monkeypatched except the model
*availability*, which is deliberately tested in both states.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


def make_client(**overrides) -> TestClient:
    settings = Settings(require_model_on_startup=False, **overrides)
    return TestClient(create_app(settings))


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A client with the model loaded (skipped when nothing is trained)."""
    from ml.config import get_active_run_dir

    try:
        get_active_run_dir()
    except FileNotFoundError:
        pytest.skip("No trained model. Run `python scripts/train.py --set-active`.")

    with make_client() as test_client:
        yield test_client


@pytest.fixture
def bare_client() -> TestClient:
    """A client whose model load is guaranteed to fail, for the 503 path."""
    return make_client(model_run="__no_such_run__")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
class TestMetadataEndpoints:
    def test_root_advertises_the_endpoints(self, client):
        response = client.get("/")
        assert response.status_code == 200
        payload = response.json()
        assert payload["version"]
        assert set(payload["endpoints"]) == {"health", "emotions", "model", "predict", "docs"}

    def test_health_reports_a_loaded_model(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()

        assert payload["status"] == "ok"
        assert payload["model_loaded"] is True
        assert payload["num_classes"] == 8
        assert payload["model_run"]
        assert payload["detail"] is None

    def test_health_never_raises_when_the_model_is_missing(self, bare_client):
        """A health check that 500s cannot explain *why* it is unhealthy."""
        with bare_client as test_client:
            response = test_client.get("/health")

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "degraded"
        assert payload["model_loaded"] is False
        assert payload["num_classes"] is None
        assert payload["detail"]

    def test_emotions_come_from_the_model(self, client):
        payload = client.get("/emotions").json()
        assert len(payload) == 8

        labels = [item["label"] for item in payload]
        assert labels == [
            "neutral",
            "calm",
            "happy",
            "sad",
            "angry",
            "fearful",
            "disgust",
            "surprised",
        ]
        assert [item["index"] for item in payload] == list(range(8))
        # RAVDESS codes come from the shared mapping, not a duplicated literal.
        assert payload[0]["ravdess_code"] == "01"
        assert payload[7]["ravdess_code"] == "08"

    def test_model_endpoint_describes_the_served_artifact(self, client):
        payload = client.get("/model").json()

        assert payload["num_classes"] == 8
        assert payload["architecture"]
        assert payload["parameters"] > 0
        assert payload["feature_extraction"]["n_mfcc"] == 40
        assert payload["feature_extraction"]["cmvn"] is True
        assert payload["preprocessing"]["resample_to_hz"] == 16000
        assert payload["preprocessing"]["window_sec"] == 3.0
        assert payload["preprocessing"]["max_upload_mb"] > 0
        assert payload["device"] in {"cpu", "cuda"}
        assert payload["selection"]["metric"] == "val_macro_f1"

    def test_openapi_schema_documents_every_endpoint(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        assert set(paths) >= {"/", "/health", "/emotions", "/model", "/predict"}

        predict = paths["/predict"]["post"]
        assert "requestBody" in predict
        assert {str(code) for code in predict["responses"]} >= {"200", "400", "413", "415", "503"}

    def test_docs_are_reachable(self, client):
        assert client.get("/docs").status_code == 200


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
class TestPredictHappyPath:
    def test_predicts_from_an_upload(self, client, speech_wav):
        payload, name = speech_wav
        response = client.post("/predict", files={"file": (name, payload, "audio/wav")})

        assert response.status_code == 200
        body = response.json()

        labels = client.get("/emotions").json()
        expected_labels = [item["label"] for item in labels]

        assert body["emotion"] in expected_labels
        assert 0.0 <= body["confidence"] <= 1.0
        # The index must agree with the served label order, or a frontend that
        # colours bars by index would mislabel the winner.
        assert body["emotion_index"] == expected_labels.index(body["emotion"])
        assert list(body["probabilities"]) == expected_labels

    def test_response_matches_the_declared_schema(self, client, speech_wav):
        payload, name = speech_wav
        body = client.post("/predict", files={"file": (name, payload)}).json()

        assert set(body) == {
            "emotion",
            "confidence",
            "emotion_index",
            "probabilities",
            "ranked_emotions",
            "audio",
            "processing_ms",
            "model",
            "request_id",
        }
        assert len(body["probabilities"]) == 8
        assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)

        ranked = body["ranked_emotions"]
        assert len(ranked) == 8
        assert [item["probability"] for item in ranked] == sorted(
            [item["probability"] for item in ranked], reverse=True
        )
        assert ranked[0]["emotion"] == body["emotion"]

        assert set(body["audio"]) == {
            "duration_sec",
            "sample_rate",
            "channels",
            "native_sample_rate",
            "trimmed_sec",
            "peak_amplitude",
        }
        assert body["audio"]["sample_rate"] == 16000
        assert body["processing_ms"] > 0
        assert body["request_id"]

    def test_echoes_the_model_that_answered(self, client, speech_wav):
        payload, name = speech_wav
        model = client.get("/model").json()
        body = client.post("/predict", files={"file": (name, payload)}).json()

        assert body["model"]["run_name"] == model["run_name"]
        assert body["model"]["parameters"] == model["parameters"]

    def test_stereo_and_mono_agree_on_the_contract(self, client, wav_bytes_factory):
        mono = wav_bytes_factory(duration_sec=2.0, channels=1)
        stereo = wav_bytes_factory(duration_sec=2.0, channels=2)

        a = client.post("/predict", files={"file": (mono[1], mono[0])}).json()
        b = client.post("/predict", files={"file": (stereo[1], stereo[0])}).json()

        assert a["audio"]["channels"] == 1
        assert b["audio"]["channels"] == 2

    def test_repeated_uploads_are_stable(self, client, speech_wav):
        payload, name = speech_wav
        first = client.post("/predict", files={"file": (name, payload)}).json()
        second = client.post("/predict", files={"file": (name, payload)}).json()

        assert first["emotion"] == second["emotion"]
        assert first["probabilities"] == pytest.approx(second["probabilities"], abs=1e-6)

    def test_request_id_header_is_echoed(self, client, speech_wav):
        payload, name = speech_wav
        response = client.post(
            "/predict",
            files={"file": (name, payload)},
            headers={"X-Request-ID": "test-request-123"},
        )
        assert response.headers["X-Request-ID"] == "test-request-123"
        assert response.json()["request_id"] == "test-request-123"

    def test_a_request_id_is_generated_when_absent(self, client, speech_wav):
        payload, name = speech_wav
        response = client.post("/predict", files={"file": (name, payload)})
        assert response.headers.get("X-Request-ID")

    def test_processing_time_header_is_present(self, client):
        response = client.get("/health")
        assert float(response.headers["X-Process-Time-Ms"]) >= 0


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------
class TestPredictErrors:
    def _error(self, response):
        body = response.json()
        assert set(body) <= {"error", "request_id"}
        assert set(body["error"]) <= {"code", "message", "detail"}
        return body["error"]

    def test_missing_file_field_is_rejected(self, client):
        response = client.post("/predict", data={"not_a_file": "x"})
        assert response.status_code == 422
        assert self._error(response)["code"] == "VALIDATION_ERROR"

    def test_empty_filename_is_rejected(self, client):
        """An empty ``filename`` is a missing part, not an unusable one.

        Starlette's multipart parser drops a part with no filename before the
        route ever sees it, so this arrives as a schema violation (422) rather
        than our own ``NO_FILE`` 400. Both are correct client errors; the test
        pins the actual behaviour so a future parser change is noticed.
        """
        response = client.post("/predict", files={"file": ("", b"data", "audio/wav")})
        assert response.status_code == 422
        assert self._error(response)["code"] == "VALIDATION_ERROR"

    def test_a_missing_file_field_is_still_a_validation_error(self, client):
        response = client.post("/predict", files={"other": ("x.wav", b"data")})
        assert response.status_code == 422
        assert self._error(response)["code"] == "VALIDATION_ERROR"

    def test_empty_body_is_rejected(self, client):
        response = client.post("/predict", files={"file": ("clip.wav", b"", "audio/wav")})
        assert response.status_code == 400
        assert self._error(response)["code"] == "EMPTY_FILE"

    def test_unsupported_extension_is_415(self, client):
        response = client.post(
            "/predict", files={"file": ("payload.exe", b"MZ\x90\x00", "application/octet-stream")}
        )
        assert response.status_code == 415
        assert self._error(response)["code"] == "UNSUPPORTED_FORMAT"

    def test_unsupported_extension_lists_the_alternatives(self, client):
        response = client.post(
            "/predict", files={"file": ("a.zip", b"PK\x03\x04", "application/zip")}
        )
        message = self._error(response)["message"]
        for extension in (".wav", ".flac", ".ogg", ".mp3"):
            assert extension in message

    def test_garbage_with_an_audio_extension_is_400(self, client):
        response = client.post(
            "/predict", files={"file": ("clip.wav", b"not audio at all" * 64, "audio/wav")}
        )
        assert response.status_code == 400
        assert self._error(response)["code"] == "INVALID_AUDIO"

    def test_truncated_wav_is_400(self, client, speech_wav):
        """A header-only prefix decodes to almost nothing, so it is "too short".

        Which specific code comes back depends on how many bytes survive, and
        every one of these is an accurate description of the problem - a
        truncated WAV is not a decoding failure, it is an audio clip with no
        usable content.
        """
        payload, name = speech_wav
        response = client.post("/predict", files={"file": (name, payload[:128])})
        assert response.status_code == 400
        assert self._error(response)["code"] in {
            "INVALID_AUDIO",
            "EMPTY_FILE",
            "AUDIO_TOO_SHORT",
        }

    def test_a_header_with_no_data_chunk_is_rejected(self, client):
        """The classic 'truncated in transit' file: valid header, zero samples."""
        header = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        response = client.post("/predict", files={"file": ("stub.wav", header)})
        assert response.status_code == 400
        assert self._error(response)["code"] in {"EMPTY_FILE", "AUDIO_TOO_SHORT", "INVALID_AUDIO"}

    def test_silence_is_400_with_a_specific_code(self, client):
        import numpy as np
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, np.zeros(32000, dtype=np.float32), 16000, format="WAV", subtype="PCM_16")

        response = client.post("/predict", files={"file": ("silence.wav", buffer.getvalue())})
        assert response.status_code == 400
        assert self._error(response)["code"] == "SILENT_AUDIO"

    def test_audio_shorter_than_the_minimum_is_400(self, client, wav_bytes_factory):
        """A distinct code, so the client can say "too short" rather than "broken"."""
        payload, name = wav_bytes_factory(duration_sec=0.1)
        response = client.post("/predict", files={"file": (name, payload)})
        assert response.status_code == 400
        assert self._error(response)["code"] == "AUDIO_TOO_SHORT"

    def test_oversized_upload_is_413(self, wav_bytes_factory):
        payload, name = wav_bytes_factory(duration_sec=2.0)
        padded = payload + b"\0" * (1024 * 1024)

        with make_client(max_upload_bytes=4096) as test_client:
            response = test_client.post("/predict", files={"file": (name, padded)})

        assert response.status_code == 413
        assert response.json()["error"]["code"] == "FILE_TOO_LARGE"

    def test_predict_without_a_model_is_503(self, bare_client, speech_wav):
        payload, name = speech_wav
        with bare_client as test_client:
            response = test_client.post("/predict", files={"file": (name, payload)})

        assert response.status_code == 503
        body = response.json()["error"]
        assert body["code"] == "MODEL_UNAVAILABLE"
        assert "train" in body["message"].lower()

    def test_emotions_without_a_model_is_503(self, bare_client):
        with bare_client as test_client:
            response = test_client.get("/emotions")
        assert response.status_code == 503

    def test_model_without_a_model_is_503(self, bare_client):
        with bare_client as test_client:
            response = test_client.get("/model")
        assert response.status_code == 503

    def test_unknown_route_is_404(self, client):
        assert client.get("/does-not-exist").status_code == 404

    def test_wrong_method_is_405(self, client):
        assert client.get("/predict").status_code == 405


# ---------------------------------------------------------------------------
# Cross-cutting behaviour
# ---------------------------------------------------------------------------
class TestCrossCutting:
    def test_every_error_body_is_uniform(self, client, speech_wav):
        """Whatever fails, the client can parse it the same way."""
        payload, name = speech_wav
        responses = [
            client.post("/predict", data={"x": "y"}),  # 422
            client.post("/predict", files={"file": ("a.zip", b"x")}),  # 415
            client.post("/predict", files={"file": ("a.wav", b"junk" * 40)}),  # 400
        ]
        for response in responses:
            body = response.json()
            assert "error" in body
            assert isinstance(body["error"]["code"], str)
            assert isinstance(body["error"]["message"], str)

    def test_cors_preflight_from_the_dev_origin_is_allowed(self, client):
        response = client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert response.status_code in {200, 204}
        assert response.headers["access-control-allow-origin"] == "http://localhost:5173"

    def test_cors_from_an_unknown_origin_is_not_allowed(self, client):
        response = client.get("/health", headers={"Origin": "https://evil.example.com"})
        assert "access-control-allow-origin" not in response.headers

    def test_cors_is_never_a_wildcard_with_credentials(self, client):
        response = client.get("/health", headers={"Origin": "http://localhost:5173"})
        origin = response.headers.get("access-control-allow-origin")
        assert origin != "*"

    def test_the_service_survives_a_burst_of_bad_requests(self, client, speech_wav):
        """One bad upload must not poison the process."""
        for _ in range(5):
            client.post("/predict", files={"file": ("bad.wav", b"junk" * 50)})

        payload, name = speech_wav
        assert client.post("/predict", files={"file": (name, payload)}).status_code == 200

    def test_startup_fails_loudly_when_a_model_is_required(self):
        """A misconfigured deployment should die at boot, not 500 per request."""
        from ml.config import get_active_run_dir

        try:
            get_active_run_dir()
        except FileNotFoundError:
            pytest.skip("No model pointer; the failure case is not distinct here.")

        settings = Settings(require_model_on_startup=True, model_run="__no_such_run__")
        app = create_app(settings)
        with pytest.raises(RuntimeError, match="No trained model"):
            with TestClient(app):
                pass


class TestSettings:
    def test_cors_origins_accept_a_comma_separated_string(self):
        settings = Settings(cors_origins="http://a.test, http://b.test")
        assert settings.cors_origins == ["http://a.test", "http://b.test"]

    def test_upload_limit_is_validated(self):
        with pytest.raises(Exception):
            Settings(max_upload_bytes=10)  # below the floor
        with pytest.raises(Exception):
            Settings(max_upload_bytes=999 * 1024 * 1024)  # above the ceiling

    def test_invalid_log_level_is_rejected(self):
        with pytest.raises(Exception):
            Settings(log_level="CHATTY")

    def test_upload_limit_flows_into_the_audio_config(self):
        settings = Settings(max_upload_bytes=5 * 1024 * 1024)
        assert settings.audio_config().max_upload_bytes == 5 * 1024 * 1024
