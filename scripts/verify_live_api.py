"""End-to-end verification against a *running* backend.

``tests/test_api.py`` exercises the ASGI app in-process. This script drives a
real uvicorn over HTTP instead, which is the only way to catch problems that
live in the server layer rather than the app: the ASGI transport, multipart
parsing, the CORS headers, and the actual bytes on the wire.

It needs a live server and a real model, so it reports clearly and exits
non-zero when either is missing rather than failing obscurely.

    python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
    python scripts/verify_live_api.py
    python scripts/verify_live_api.py --base-url http://127.0.0.1:9000
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import wave
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.config import MODELS_DIR, get_active_run_dir, resolve_ravdess_root
from ml.data.ravdess import assign_splits, build_manifest

PASS, FAIL = "PASS", "FAIL"


def wav_bytes(samples: np.ndarray, sample_rate: int = 16_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return buffer.getvalue()


def tone_wav(seconds: float) -> bytes:
    n = max(1, int(seconds * 16_000))
    t = np.linspace(0, n / 16_000, n, endpoint=False)
    return wav_bytes(0.4 * np.sin(2 * np.pi * 220 * t) * 32_767)


def silence_wav() -> bytes:
    return wav_bytes(np.zeros(16_000))


class Checks:
    def __init__(self) -> None:
        self.results: list[tuple[str, str, str]] = []

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        self.results.append((PASS if passed else FAIL, name, detail))
        return bool(passed)

    @property
    def failed(self) -> list[tuple[str, str, str]]:
        return [row for row in self.results if row[0] == FAIL]

    def report(self) -> int:
        width = max(len(name) for _, name, _ in self.results)
        for status, name, detail in self.results:
            line = f"  [{status}] {name:<{width}}"
            if detail:
                line += f"  {detail}"
            print(line)
        print(f"\n{len(self.results) - len(self.failed)}/{len(self.results)} live checks passed")
        return 1 if self.failed else 0


def find_paths(payload: object, needles: list[str], where: str = "response") -> list[str]:
    """Every place a needle appears in a nested JSON payload."""
    hits: list[str] = []
    if isinstance(payload, str):
        hits += [f"{where}: {n}" for n in needles if n.lower() in payload.lower()]
    elif isinstance(payload, dict):
        for key, value in payload.items():
            hits += find_paths(value, needles, f"{where}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            hits += find_paths(value, needles, f"{where}[{index}]")
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--samples", type=int, default=3, help="real test-split files to classify")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    checks = Checks()

    # ---- service metadata --------------------------------------------------
    try:
        health = httpx.get(f"{base}/health", timeout=15)
    except httpx.HTTPError as exc:
        print(f"No server at {base} ({exc}).")
        print("Start one with:")
        print("  python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000")
        return 2

    checks.check("GET /health returns 200", health.status_code == 200, str(health.status_code))
    body = health.json()
    checks.check("/health reports a loaded model", body.get("model_loaded") is True)
    checks.check("/health reports 8 classes", body.get("num_classes") == 8)

    emotions = httpx.get(f"{base}/emotions", timeout=15)
    checks.check("GET /emotions returns 200", emotions.status_code == 200)
    listed = emotions.json()
    labels = [entry["label"] for entry in listed]
    checks.check("/emotions lists 8 labels", len(labels) == 8, ", ".join(labels[:3]) + ", ...")
    checks.check(
        "/emotions indices are 0..7 in order",
        [entry["index"] for entry in listed] == list(range(8)),
    )

    model = httpx.get(f"{base}/model", timeout=15)
    checks.check("GET /model returns 200", model.status_code == 200)
    served = model.json()
    checks.check(
        "/model reports the architecture",
        bool(served.get("architecture")),
        served.get("architecture", ""),
    )
    checks.check(
        "/model reports a parameter count",
        served.get("parameters", 0) > 0,
        str(served.get("parameters")),
    )
    checks.check(
        "/model names the selection metric",
        served.get("selection", {}).get("metric") == "val_macro_f1",
        f"{served.get('selection', {}).get('metric')}={served.get('selection', {}).get('value')}",
    )

    # ---- the served model is the one that was selected ---------------------
    try:
        active = get_active_run_dir().name
        checks.check("/model serves the active run", served.get("run_name") == active, active)
    except Exception:
        checks.check("/model names a run", bool(served.get("run_name")), served.get("run_name", ""))

    # ---- real inference on held-out audio ----------------------------------
    try:
        root = resolve_ravdess_root()
        test_samples = assign_splits(build_manifest(root))["test"]
    except Exception as exc:
        print(f"\nSkipping the inference checks - no corpus available ({exc}).")
        return checks.report()

    correct = 0
    for sample in test_samples[: args.samples]:
        path = root / sample.path
        started = time.perf_counter()
        response = httpx.post(
            f"{base}/predict",
            files={"file": (path.name, path.read_bytes(), "audio/x-wav")},
            timeout=180,
        )
        wall_ms = (time.perf_counter() - started) * 1000
        if not checks.check(
            f"POST /predict {path.name}", response.status_code == 200, str(response.status_code)
        ):
            print(f"        body: {response.text[:220]}")
            continue

        payload = response.json()
        top = payload.get("emotion")
        distribution = payload.get("probabilities", {})
        checks.check(
            f"  truth={sample.label} -> {top} ({payload.get('confidence')})",
            top in labels,
            f"server {payload.get('processing_ms')} ms, round trip {wall_ms:.0f} ms",
        )
        checks.check("  probabilities cover every class", set(distribution) == set(labels))
        total = sum(distribution.values())
        checks.check("  probabilities sum to 1", abs(total - 1.0) < 1e-3, f"{total:.6f}")
        checks.check(
            "  argmax agrees with the top label",
            max(distribution, key=distribution.get) == top,
        )
        checks.check(
            "  ranking is sorted descending",
            [row["probability"] for row in payload.get("ranked_emotions", [])]
            == sorted(
                (row["probability"] for row in payload.get("ranked_emotions", [])), reverse=True
            ),
        )
        checks.check(
            "  audio metadata is present",
            payload.get("audio", {}).get("duration_sec", 0) > 0,
            f"{payload.get('audio', {}).get('duration_sec')}s, trimmed "
            f"{payload.get('audio', {}).get('trimmed_sec')}s",
        )
        checks.check(
            "  response names the serving run",
            payload.get("model", {}).get("run_name") == served.get("run_name"),
        )
        checks.check("  a request id is echoed back", bool(payload.get("request_id")))
        correct += top == sample.label

    if args.samples > 1:
        print(
            f"        ({correct}/{min(args.samples, len(test_samples))} of the spot-checked "
            f"files classified correctly - a 3-sample probe, not a metric)"
        )

    # ---- validation and security paths -------------------------------------
    cases = [
        ("empty file", {"files": {"file": ("x.wav", b"", "audio/x-wav")}}, 400, "EMPTY_FILE"),
        (
            "non-audio bytes",
            {"files": {"file": ("x.wav", b"hello world" * 50, "audio/x-wav")}},
            400,
            "INVALID_AUDIO",
        ),
        (
            "truncated wav",
            {"files": {"file": ("x.wav", tone_wav(1.0)[:200], "audio/x-wav")}},
            400,
            None,
        ),
        (
            "digital silence",
            {"files": {"file": ("x.wav", silence_wav(), "audio/x-wav")}},
            400,
            "SILENT_AUDIO",
        ),
        (
            "below minimum length",
            {"files": {"file": ("x.wav", tone_wav(0.05), "audio/x-wav")}},
            400,
            "AUDIO_TOO_SHORT",
        ),
        (
            "disallowed extension",
            {"files": {"file": ("x.exe", b"MZ" + bytes(400), "application/octet-stream")}},
            415,
            "UNSUPPORTED_FORMAT",
        ),
        (
            "over the size cap",
            {"files": {"file": ("x.wav", b"RIFF" + bytes(21 * 1024 * 1024), "audio/x-wav")}},
            413,
            "FILE_TOO_LARGE",
        ),
        ("no file part", {"data": {}}, 422, "VALIDATION_ERROR"),
    ]
    for name, kwargs, expected, expected_code in cases:
        response = httpx.post(f"{base}/predict", timeout=300, **kwargs)
        ok = checks.check(
            f"rejects {name}",
            response.status_code == expected,
            f"{response.status_code} (want {expected})",
        )
        if not ok:
            continue
        detail = response.json().get("error", {})
        checks.check(f"  {name} -> stable code", bool(detail.get("code")), str(detail.get("code")))
        if expected_code:
            checks.check(
                f"  {name} -> {expected_code}",
                detail.get("code") == expected_code,
                str(detail.get("code")),
            )
        checks.check(f"  {name} -> actionable message", len(detail.get("message", "")) > 20)
        checks.check(f"  {name} -> no heap address", "0x" not in json.dumps(response.json()))

    # ---- routing and headers -----------------------------------------------
    checks.check("GET /predict is 405", httpx.get(f"{base}/predict", timeout=15).status_code == 405)
    checks.check("GET /nope is 404", httpx.get(f"{base}/nope", timeout=15).status_code == 404)
    checks.check(
        "OpenAPI schema is served", httpx.get(f"{base}/openapi.json", timeout=15).status_code == 200
    )

    allowed = httpx.options(
        f"{base}/predict",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
        timeout=15,
    )
    checks.check(
        "CORS preflight from the dev origin",
        allowed.status_code == 200
        and "http://localhost:5173" in allowed.headers.get("access-control-allow-origin", ""),
        allowed.headers.get("access-control-allow-origin", "(none)"),
    )
    checks.check(
        "an unknown origin is not reflected",
        httpx.options(
            f"{base}/predict",
            headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
            timeout=15,
        ).headers.get("access-control-allow-origin", "")
        != "http://evil.example",
    )

    # ---- nothing internal leaks --------------------------------------------
    needles = [str(MODELS_DIR), str(root), str(MODELS_DIR.parent), "artifacts", "Traceback"]
    leaks: list[str] = []
    for route in ("/health", "/model", "/emotions"):
        leaks += find_paths(httpx.get(f"{base}{route}", timeout=15).json(), needles, route)
    leaks += find_paths(
        httpx.post(
            f"{base}/predict",
            files={
                "file": (
                    test_samples[0].filename,
                    (root / test_samples[0].path).read_bytes(),
                    "audio/x-wav",
                )
            },
            timeout=180,
        ).json(),
        needles,
        "/predict",
    )
    checks.check("no filesystem paths in any response", not leaks, "; ".join(leaks[:3]))

    return checks.report()


if __name__ == "__main__":
    raise SystemExit(main())
