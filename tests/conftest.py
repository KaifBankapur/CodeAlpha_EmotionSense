"""Shared pytest fixtures.

Two principles:

1. **Tests that need real artifacts skip instead of failing.** Whether a trained
   checkpoint or the RAVDESS corpus is present depends on how far the pipeline
   has been run. A missing artifact is a skipped test with an actionable
   message, not a spurious failure.
2. **Audio used by tests is synthesised**, so the suite never depends on
   committed binary data and can assert on exact expected properties.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

# `ml` and `backend` are importable because pyproject.toml sets
# `pythonpath = ["."]` for pytest, so no sys.path manipulation is needed here.
from ml.config import (
    MODELS_DIR,
    AudioConfig,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    get_active_run_dir,
    resolve_ravdess_root,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def audio_cfg() -> AudioConfig:
    return AudioConfig()


@pytest.fixture(scope="session")
def feature_cfg() -> FeatureConfig:
    return FeatureConfig()


@pytest.fixture(scope="session")
def model_cfg() -> ModelConfig:
    return ModelConfig()


@pytest.fixture(scope="session")
def experiment_cfg() -> ExperimentConfig:
    return ExperimentConfig()


# ---------------------------------------------------------------------------
# Synthetic audio
# ---------------------------------------------------------------------------
def synth_speech_like(
    duration_sec: float,
    sample_rate: int = 16000,
    *,
    f0: float = 120.0,
    n_harmonics: int = 24,
    seed: int = 0,
) -> np.ndarray:
    """Build a voiced, formant-shaped signal with speech-like dynamics.

    Not real speech - it is a harmonic stack under a moving fundamental, passed
    through a fixed formant envelope and given syllable-rate amplitude
    modulation. That gives the MFCC front-end and the network realistic
    structure (harmonics, spectral tilt, temporal variation) without shipping
    binary fixtures.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_sec * sample_rate)
    t = np.arange(n, dtype=np.float32) / sample_rate

    # Pitch contour: f0 drifts slowly, which is what makes a voice sound voiced
    # rather than like a static buzzer.
    f0_t = f0 * (1.0 + 0.10 * np.sin(2 * np.pi * 1.7 * t))
    phase = (2 * np.pi * np.cumsum(f0_t) / sample_rate).astype(np.float32)

    # Harmonic stack with 1/k roll-off: the spectral tilt of a real voice.
    signal = np.zeros(n, dtype=np.float32)
    for k in range(1, n_harmonics + 1):
        if f0 * k >= sample_rate / 2:
            break
        signal += (1.0 / k) * np.sin(k * phase, dtype=np.float32)

    # Two fixed resonances approximate the F1/F2 emphasis of voiced speech.
    bins = np.fft.rfftfreq(n, 1.0 / sample_rate)
    for centre, width, gain in ((700.0, 180.0, 1.0), (2100.0, 300.0, 0.55)):
        envelope = gain * np.exp(-(((bins - centre) / width) ** 2))
        signal = np.fft.irfft(np.abs(np.fft.rfft(signal)) * envelope, n=n).astype(np.float32)

    # Syllable-rate amplitude modulation plus a slow utterance contour.
    syllable = 0.55 + 0.45 * np.sin(2 * np.pi * 3.1 * t - np.pi / 2)
    contour = np.clip(np.sin(np.pi * np.clip(t / max(duration_sec, 1e-6), 0, 1)), 0.1, 1.0)
    signal = signal * syllable * contour

    noise = rng.normal(0.0, 0.004, size=n).astype(np.float32)
    out = signal + noise
    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * 0.8).astype(np.float32)


@pytest.fixture(scope="session")
def speech_waveform() -> np.ndarray:
    """2.0 s of synthetic speech at 16 kHz - short enough to be padded."""
    return synth_speech_like(2.0)


@pytest.fixture(scope="session")
def long_waveform() -> np.ndarray:
    """5.0 s, so it must be centre-cropped to the 3 s window."""
    return synth_speech_like(5.0, seed=1)


@pytest.fixture(scope="session")
def wav_bytes_factory() -> Callable[..., tuple[bytes, str]]:
    """Factory returning ``(payload, filename)`` for an in-memory WAV file."""
    import soundfile as sf

    def make(
        duration_sec: float = 2.0,
        sample_rate: int = 16000,
        name: str = "sample.wav",
        channels: int = 1,
        seed: int = 0,
        waveform: np.ndarray | None = None,
    ) -> tuple[bytes, str]:
        data = (
            waveform
            if waveform is not None
            else synth_speech_like(duration_sec, sample_rate, seed=seed)
        )
        if channels == 2:
            data = np.stack([data, np.roll(data, 7)], axis=1)
        buffer = io.BytesIO()
        sf.write(buffer, data, sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue(), name

    return make


@pytest.fixture(scope="session")
def speech_wav(wav_bytes_factory) -> tuple[bytes, str]:
    """A valid 2 s mono WAV, the canonical happy-path upload."""
    return wav_bytes_factory(duration_sec=2.0)


# ---------------------------------------------------------------------------
# Real artifacts (skipped when absent)
# ---------------------------------------------------------------------------
def _require_ravdess() -> Path:
    try:
        root = resolve_ravdess_root()
    except FileNotFoundError:
        pytest.skip("RAVDESS is not present. Run `python scripts/download_dataset.py`.")
    if not any(root.glob("Actor_*/*.wav")):
        pytest.skip("RAVDESS directory exists but contains no audio.")
    return root


@pytest.fixture(scope="session")
def ravdess_root() -> Path:
    return _require_ravdess()


@pytest.fixture(scope="session")
def ravdess_sample_paths(ravdess_root: Path) -> list[Path]:
    """A handful of real files, one per emotion where possible."""
    paths = sorted(ravdess_root.rglob("*.wav"))
    return paths[:12]


@pytest.fixture(scope="session")
def active_run_dir() -> Path:
    """The run directory the API would serve. Skips when nothing is trained."""
    try:
        run_dir = get_active_run_dir()
    except FileNotFoundError:
        pytest.skip(
            "No trained model. Run `python scripts/prepare_data.py` then "
            "`python scripts/train.py --set-active`."
        )
    if not (run_dir / "best_model.pt").is_file():
        pytest.skip(f"Active run '{run_dir.name}' has no best_model.pt.")
    return run_dir


@pytest.fixture(scope="session")
def predictor(active_run_dir: Path):
    """A real trained predictor, loaded once for the whole session."""
    import torch

    from ml.inference.predictor import EmotionPredictor

    torch.set_num_threads(2)
    try:
        return EmotionPredictor.from_run_dir(active_run_dir)
    except Exception as exc:
        pytest.skip(f"Could not load model from '{active_run_dir}': {exc}")


@pytest.fixture(scope="session")
def manifest_path() -> Path:
    path = MODELS_DIR.parent.parent / "data" / "processed" / "manifest.json"
    if not path.is_file():
        # prepare_data.py may write it elsewhere; fall back to a glob.
        candidates = sorted((PROJECT_ROOT / "data" / "processed").glob("**/manifest*.json"))
        if not candidates:
            pytest.skip("No dataset manifest. Run `python scripts/prepare_data.py`.")
        path = candidates[0]
    return path


@pytest.fixture(scope="session")
def split_manifest_path() -> Path:
    candidates = sorted((PROJECT_ROOT / "data" / "processed").glob("**/*split*.json"))
    if not candidates:
        pytest.skip("No split manifest. Run `python scripts/prepare_data.py`.")
    return candidates[0]
