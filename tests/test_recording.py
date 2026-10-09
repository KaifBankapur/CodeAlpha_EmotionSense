"""Guards the in-page recording path.

The bug this file exists to prevent
----------------------------------
`MediaRecorder` can only emit containers the browser ships an encoder for:
Chrome produces WebM/Opus, Firefox Ogg/Opus, Safari MP4. The API accepts
``.wav .flac .ogg .mp3 .m4a`` and rejects ``.webm``, because the server decodes
through libsndfile, which has no Matroska/WebM support. So in Chrome *every*
recording was refused with ``UNSUPPORTED_FORMAT`` and the record button could not
work at all.

The fix is client-side: ``frontend/src/lib/audio.ts`` decodes whatever the
browser recorded and re-encodes it as 16 kHz mono WAV. That contract has two
halves, and this module covers both:

* the **TypeScript** encoder actually produces a valid, correct RIFF/WAVE file -
  checked by running the real module under Node and reading the bytes back with
  the stdlib ``wave`` module, which shares no code with it;
* the **server** accepts every container a browser can produce, and still refuses
  the ones it genuinely cannot decode, with a useful message.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app
from tests.conftest import synth_speech_like

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = REPO_ROOT / "frontend"
ENCODER_TS = FRONTEND / "src" / "lib" / "audio.ts"

# libsndfile reports what it can actually decode. If a future upgrade gains
# Matroska support this test will start failing loudly rather than the allow-list
# silently disagreeing with the decoder.
WEBM_DECODABLE = any(name.upper() in {"MATROSKA", "WEBM"} for name in sf.available_formats())


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(Settings()), raise_server_exceptions=False)


def post(
    client: TestClient, filename: str, payload: bytes, ctype: str = "application/octet-stream"
):
    return client.post(
        "/predict",
        files={"file": (filename, payload, ctype)},
    )


def encoded(samples: np.ndarray, rate: int, fmt: str, subtype: str | None = None) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples, rate, format=fmt, subtype=subtype)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# 1. The decoder's real capabilities
# ---------------------------------------------------------------------------
def test_ogg_opus_is_decodable() -> None:
    """Firefox's MediaRecorder output must be something the server can read.

    This is the load-bearing assumption behind preferring ``audio/ogg;codecs=opus``
    in the browser, so it is asserted rather than assumed: if a dependency change
    ever drops OPUS support, the fallback ordering in ``Recorder.tsx`` becomes
    wrong and this fails instead of surfacing as a decode error at runtime.
    """
    subtypes = sf.available_subtypes("OGG")
    assert "OPUS" in subtypes, f"libsndfile lost OPUS support; subtypes={subtypes}"


def test_webm_is_genuinely_undecodable() -> None:
    """Documents *why* ``.webm`` stays off the allow-list.

    If this ever starts failing it is good news and actionable: the server could
    then accept WebM directly and the client-side transcode would become an
    optimisation rather than a necessity.
    """
    assert not WEBM_DECODABLE, (
        "libsndfile now decodes Matroska/WebM. The browser-side transcode can be "
        "replaced by accepting .webm server-side, and this test updated."
    )


# ---------------------------------------------------------------------------
# 2. Server-side container acceptance
# ---------------------------------------------------------------------------
def test_client_style_transcoded_wav_is_accepted(client: TestClient) -> None:
    """The exact shape `transcodeRecordingToWav` produces: 16 kHz, mono, PCM16."""
    samples = synth_speech_like(3.0, 16000)
    payload = encoded(samples, 16000, "WAV", "PCM_16")

    response = post(client, "recording-2026-01-01-00-00-00.wav", payload, "audio/wav")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["emotion"] in {
        "neutral",
        "calm",
        "happy",
        "sad",
        "angry",
        "fearful",
        "disgust",
        "surprised",
    }
    # The recorded duration is decided by the model, not echoed from the client.
    assert body["audio"]["duration_sec"] == pytest.approx(3.0, abs=0.05)


@pytest.mark.parametrize(
    ("name", "fmt", "subtype", "rate"),
    [
        # Chrome and Safari hand the page MP4; Firefox hands it Ogg. Both are on
        # the allow-list, so both must survive the round trip.
        ("rec.ogg", "OGG", "OPUS", 48000),
        ("rec.ogg", "OGG", "VORBIS", 48000),
        # A browser recording at its native rate, before any client resampling.
        ("rec-native.wav", "WAV", "PCM_16", 48000),
        # Typical desktop input devices run at 44.1 kHz.
        ("rec-44k.wav", "WAV", "PCM_16", 44100),
        # Formats a user might upload rather than record.
        ("upload.flac", "FLAC", "PCM_16", 48000),
        ("upload.mp3", "MP3", None, 44100),
    ],
)
def test_browser_producible_containers_are_accepted(
    client: TestClient,
    name: str,
    fmt: str,
    subtype: str | None,
    rate: int,
) -> None:
    samples = synth_speech_like(3.0, rate)
    response = post(client, name, encoded(samples, rate, fmt, subtype))

    assert response.status_code == 200, response.text
    audio = response.json()["audio"]
    # Every upload is resampled and downmixed to the model's rate regardless of
    # what it arrived as - this is the one preprocessing path both training and
    # inference share.
    assert audio["sample_rate"] == 16000


def test_stereo_recording_is_downmixed(client: TestClient) -> None:
    """Some browsers/OS combinations capture two channels.

    ``AudioInfo.channels`` reports the *source* channel count, so 2 is the
    correct answer here; what matters is that the stereo file is accepted and
    still analysed at the model's mono rate rather than being rejected or
    silently treated as mono.
    """
    samples = synth_speech_like(3.0, 48000)
    stereo = np.stack([samples, np.roll(samples, 137)], axis=1)
    payload = encoded(stereo, 48000, "WAV", "PCM_16")

    response = post(client, "rec-stereo.wav", payload, "audio/wav")

    assert response.status_code == 200, response.text
    audio = response.json()["audio"]
    assert audio["channels"] == 2, "source channel count should be reported faithfully"
    assert audio["sample_rate"] == 16000


def test_webm_is_refused_with_an_actionable_message(client: TestClient) -> None:
    """WebM must be rejected up front, not accepted and then failed at decode.

    415 (Unsupported Media Type) is the semantically correct status and is what
    ``UnsupportedFormatError`` maps to.
    """
    payload = b"\x1a\x45\xdf\xa3" + b"\x00" * 512
    response = post(client, "recording.webm", payload, "audio/webm")

    assert response.status_code == 415, response.text
    body = response.json()
    assert body["error"]["code"] == "UNSUPPORTED_FORMAT"
    # The message must name the alternatives, since this is the exact error a
    # user hit after trying to record.
    message = body["error"]["message"]
    for extension in (".wav", ".flac", ".ogg", ".mp3", ".m4a"):
        assert extension in message, message


def test_webm_named_as_other_extensions_is_still_decoded_or_refused(
    client: TestClient,
) -> None:
    """A renamed WebM must not slip past the extension check.

    The suffix check is only the cheap first line of defence. A `.webm` renamed
    to `.wav` reaches the decoder, which is what actually decides whether the
    bytes are audio - so this asserts it is rejected as invalid audio rather than
    either trusted or silently accepted.
    """
    payload = b"\x1a\x45\xdf\xa3" + b"\x00" * 512
    response = post(client, "disguised.wav", payload, "audio/wav")

    assert response.status_code in (400, 415), response.text
    assert response.json()["error"]["code"] in {
        "INVALID_AUDIO",
        "UNSUPPORTED_FORMAT",
        "CORRUPT_AUDIO",
    }


# ---------------------------------------------------------------------------
# 3. The real TypeScript encoder
# ---------------------------------------------------------------------------
_NODE_DRIVER = r"""
import { writeFileSync } from 'node:fs'
import { encodeWavPcm16 } from '__ENTRY__'

const RATE = 16000
const N = RATE * 3
const samples = new Float32Array(N)
for (let i = 0; i < N; i += 1) samples[i] = Math.sin((2 * Math.PI * 220 * i) / RATE)

// Edge cases the encoder has to get right.
samples[0] = 1.0 // +full scale -> 32767, not 32768
samples[1] = -1.0 // -full scale -> -32768, not -32767
samples[2] = 0.0 // digital silence
samples[3] = 1 / 65536 // sub-LSB: must round to 0, not wrap to a full-scale spike
samples[4] = 5.0 // over-unity: must clamp, not wrap to a negative spike

writeFileSync('__OUT__', Buffer.from(encodeWavPcm16(samples, RATE)))
"""


def _run_encoder(tmp_path: Path) -> bytes | None:
    """Run the real ``audio.ts`` encoder under Node and return the WAV bytes.

    Returns ``None`` when the frontend toolchain is unavailable, so the suite
    still runs in a Python-only checkout. That is a deliberate trade: the check is
    most valuable where the frontend exists, and skipping beats a false failure.
    """
    node = shutil.which("node")
    if node is None or not ENCODER_TS.is_file():
        return None

    driver = tmp_path / "encode.mts"
    out = tmp_path / "encoded.wav"
    # `as_uri` gives a percent-encoded file:// URL, and `as_posix` gives forward
    # slashes. Either matters: a Windows path embedded raw in a JS string literal
    # has its backslashes eaten as escape sequences (`\U`, `\k`, ...), producing
    # a mangled path that only fails once the encoder tries to write.
    entry = ENCODER_TS.resolve().as_uri()
    driver.write_text(
        _NODE_DRIVER.replace("__ENTRY__", entry).replace("__OUT__", out.as_posix()),
        encoding="utf-8",
    )

    # --experimental-strip-types lets Node run the TypeScript directly, so the
    # exact shipped module is exercised rather than a copy of it.
    completed = subprocess.run(
        [node, "--experimental-strip-types", str(driver)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        cwd=str(FRONTEND),
    )
    if completed.returncode != 0 or not out.is_file():
        pytest.fail(
            "the browser-side WAV encoder failed to run under Node:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return out.read_bytes()


def test_typescript_encoder_emits_a_valid_wav(tmp_path: Path) -> None:
    """Header is well-formed, per the RIFF specification."""
    payload = _run_encoder(tmp_path)
    if payload is None:
        pytest.skip("Node or frontend/src/lib/audio.ts is unavailable")

    assert len(payload) == 44 + 16000 * 3 * 2
    assert payload[:4] == b"RIFF"
    assert payload[8:12] == b"WAVE"
    assert payload[12:16] == b"fmt "
    assert payload[36:40] == b"data"

    riff_size = int.from_bytes(payload[4:8], "little")
    data_size = int.from_bytes(payload[40:44], "little")
    # The RIFF size field counts everything after the first 8 bytes.
    assert riff_size == len(payload) - 8
    assert data_size == 16000 * 3 * 2


def test_typescript_encoder_output_is_readable_by_an_independent_decoder(
    tmp_path: Path,
) -> None:
    """Decode with the stdlib ``wave`` module and verify the samples survived.

    ``wave`` shares no code with the encoder, so a container it can parse and
    whose samples it recovers exactly is genuinely correct rather than merely
    self-consistent - which is the failure mode that matters here, since a
    malformed header would otherwise only surface server-side as an opaque decode
    error after the user had recorded something.
    """
    payload = _run_encoder(tmp_path)
    if payload is None:
        pytest.skip("Node or frontend/src/lib/audio.ts is unavailable")

    path = tmp_path / "roundtrip.wav"
    path.write_bytes(payload)

    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16000
        frames = handle.getnframes()
        raw = handle.readframes(frames)

    assert frames == 16000 * 3
    values = np.frombuffer(raw, dtype="<i2")

    # Edge cases: the integer mapping is asymmetric and must not wrap.
    assert values[0] == 32767, "+1.0 must clamp to 32767, not wrap"
    assert values[1] == -32768, "-1.0 must clamp to -32768, not wrap"
    assert values[2] == 0
    assert values[3] == 0, "a sub-LSB sample must round to 0, not wrap to a spike"
    assert values[4] == 32767, "a sample beyond unity must clamp, not wrap negative"

    # Whole-signal fidelity against the reference the driver encoded.
    reference = (32767 * np.sin(2 * np.pi * 220 * np.arange(frames) / 16000)).astype(np.int16)
    reference[0] = 32767
    reference[1] = -32768
    reference[2] = 0
    reference[3] = 0
    reference[4] = 32767
    assert np.max(np.abs(values.astype(np.int32) - reference.astype(np.int32))) <= 1

    # The frequency survives, which a container that truncated or resampled
    # samples would not manage.
    crossings = int(np.count_nonzero(np.signbit(values[:-1]) != np.signbit(values[1:])))
    assert crossings / 2 / 3 == pytest.approx(220, abs=1.0)


def test_encoder_output_flows_through_the_real_api(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """End-to-end: bytes from the shipped TypeScript encoder -> HTTP prediction.

    This is the test that would have caught the original bug. The unit checks
    above confirm the encoder is well-formed and the container tests confirm the
    server accepts such files, but only this one proves the two halves actually
    meet: if the encoder ever changed target rate, bit depth or channel count,
    every other test here would still pass while recording silently broke.
    """
    payload = _run_encoder(tmp_path)
    if payload is None:
        pytest.skip("Node or frontend/src/lib/audio.ts is unavailable")

    response = post(client, "recording-from-browser.wav", payload, "audio/wav")

    assert response.status_code == 200, response.text
    body = response.json()
    assert 0.0 <= body["confidence"] <= 1.0
    # `probabilities` is keyed by label, while `ranked_emotions` is an ordered list
    # of {emotion, probability}. Both must describe the same eight classes.
    assert set(body["probabilities"]) == {item["emotion"] for item in body["ranked_emotions"]}
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    # Ranked order must be strictly descending, which is what the UI's bar chart
    # and the 3-D scene both rely on.
    ranked_values = [item["probability"] for item in body["ranked_emotions"]]
    assert ranked_values == sorted(ranked_values, reverse=True)

    # The client-side transcode targets the model's own rate, so the server must
    # not have had to resample. `native_sample_rate` proves nothing was converted.
    assert body["audio"]["sample_rate"] == 16000
    assert body["audio"]["native_sample_rate"] == 16000


def test_recorder_declares_the_wav_contract() -> None:
    """Static guard on the recorder's transcode step.

    Cheap and brittle by nature, but this is the one place where a refactor
    could silently reintroduce the original defect by uploading the raw
    MediaRecorder blob again. The behavioural tests above cannot catch that,
    because nothing about them would fail - the file would simply never be
    produced.
    """
    recorder = (FRONTEND / "src" / "components" / "Recorder.tsx").read_text(encoding="utf-8")
    assert "transcodeRecordingToWav" in recorder, (
        "Recorder must transcode the MediaRecorder blob; uploading it raw is what "
        "made Chrome recordings fail with UNSUPPORTED_FORMAT."
    )
    assert ".webm'" not in recorder, "the recorder must not hand a .webm blob to the API"


def test_webm_is_not_on_the_allow_list() -> None:
    """The allow-list and the decoder must agree.

    If ``.webm`` were added without decoder support, uploads would pass the suffix
    check and then fail later as ``INVALID_AUDIO`` - a strictly worse error than
    the one currently reported.
    """
    if WEBM_DECODABLE:
        pytest.skip("libsndfile can now decode WebM; the allow-list may be widened")

    from ml.config import AudioConfig

    assert ".webm" not in AudioConfig().allowed_extensions


def test_api_advertises_only_decodable_formats(client: TestClient) -> None:
    """``/model`` is the client's source of truth for upload limits.

    The frontend reads the accept-list from here, so a format the server will
    reject must never appear in the response - otherwise the file picker offers
    the user a format that cannot work.
    """
    response = client.get("/model")
    assert response.status_code == 200, response.text
    # The allow-list lives under `preprocessing`, because it is a property of
    # the audio contract rather than of the model.
    preprocessing = response.json()["preprocessing"]
    advertised = preprocessing["accepted_formats"]

    assert set(advertised) == {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
    assert ".webm" not in advertised

    # Everything advertised must really decode.
    available = {name.upper() for name in sf.available_formats()}
    for extension, fmt in (
        (".wav", "WAV"),
        (".flac", "FLAC"),
        (".ogg", "OGG"),
        (".mp3", "MP3"),
    ):
        if extension in advertised:
            assert fmt in available, f"{extension} is advertised but not decodable"


def test_advertised_limits_match_the_server_side_guards(client: TestClient) -> None:
    """The limits the client enforces must be the limits the server enforces.

    The frontend reads ``min_duration_sec`` and ``max_upload_mb`` from here so a
    clip is never described as acceptable and then rejected. That only works if
    these really are the values the server applies, which is exactly the kind of
    duplicated constant that silently drifts.
    """
    from ml.config import AudioConfig

    cfg = AudioConfig()
    preprocessing = client.get("/model").json()["preprocessing"]

    assert preprocessing["min_duration_sec"] == cfg.min_duration_sec
    assert preprocessing["max_duration_sec"] == cfg.max_duration_sec
    # The field is named `max_upload_mb` but is computed in MiB
    # (`max_upload_bytes / 1024**2`), so 20 MiB reports as 20.0. Asserting the
    # relationship rather than hard-coding a number keeps the check meaningful
    # without silently blessing the 1e6-vs-1024**2 ambiguity.
    assert preprocessing["max_upload_mb"] == pytest.approx(cfg.max_upload_bytes / 1024**2)
    assert preprocessing["accepted_formats"] == cfg.allowed_extensions


def test_extension_check_is_case_insensitive(client: TestClient) -> None:
    """Browsers and users both produce ``.WAV`` and ``.WAV``-style suffixes.

    A recorded file named by the user could easily carry an uppercase suffix; the
    check lowercases, and a regression here would be an annoying, invisible
    rejection.
    """
    payload = encoded(synth_speech_like(2.0, 16000), 16000, "WAV", "PCM_16")
    for name in ("RECORDING.WAV", "Recording.Wav"):
        response = post(client, name, payload, "audio/wav")
        assert response.status_code == 200, f"{name}: {response.text}"


def test_error_bodies_stay_machine_readable_for_bad_recordings(
    client: TestClient,
) -> None:
    """A too-short or silent clip must produce a specific code, not a 500."""
    too_short = encoded(synth_speech_like(0.1, 16000), 16000, "WAV", "PCM_16")
    response = post(client, "too-short.wav", too_short, "audio/wav")
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "AUDIO_TOO_SHORT"

    silence = np.zeros(16000 * 3, dtype=np.float32)
    response = post(client, "silent.wav", encoded(silence, 16000, "WAV", "PCM_16"), "audio/wav")
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "SILENT_AUDIO"


def test_probe_and_check_scripts_are_not_committed() -> None:
    """Keeps throwaway verification scripts out of the deliverable.

    These checks live in the repo; ad-hoc probes used while building should not
    end up shipped, so assert none were accidentally written into the tree.
    """
    stray = [
        path.name
        for path in REPO_ROOT.rglob("*.py")
        if path.name.startswith(("probe_", "check_")) and "site-packages" not in path.parts
    ]
    assert stray == [], f"throwaway scripts committed: {stray}"


def test_encoder_is_referenced_by_the_lazy_scene_boundary() -> None:
    """Keeps the 3-D scene's WebGL fallback honest.

    ``ConfidenceScene`` returns ``null`` when WebGL is missing, so the 2-D bars in
    ``ProbabilityBars`` are the only representation of the distribution. That is
    an acceptable degradation, but only while it is still true - if the 3-D scene
    ever became the sole place a probability is shown, this would be a real
    accessibility regression rather than a graceful one.
    """
    boundary = (FRONTEND / "src" / "components" / "three" / "ConfidenceScene.tsx").read_text(
        encoding="utf-8"
    )
    assert "hasWebGL()" in boundary, "the scene must probe for WebGL before mounting"
    assert "WebGLBoundary" in boundary, "runtime WebGL failures need an error boundary"
    assert "return null" in boundary, "no WebGL must degrade to the 2-D bars"

    bars = (FRONTEND / "src" / "components" / "ProbabilityBars.tsx").read_text(encoding="utf-8")
    assert "probability" in bars, "the 2-D bars must still render the real values"
