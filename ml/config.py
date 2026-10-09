"""
Central, serialisable configuration for the whole project.

Everything that influences a prediction lives here and is round-tripped to JSON
next to the model weights. That has one concrete benefit: the inference service
reads its preprocessing and feature settings *from the artifact it loads*, not
from whatever the current source tree happens to say. Upgrading the code later
therefore cannot change how an already-trained model is served.

Design rules for this module
----------------------------
1. No torch / librosa imports - it must be importable from the CLI, the API and
   the tests without paying for heavy dependencies.
2. Dataclasses only, so serialisation is a one-liner (``dataclasses.asdict``).
3. ``from_dict`` ignores unknown keys, so adding a field later does not break
   older artifacts.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
FIGURES_DIR: Path = ARTIFACTS_DIR / "figures"
MODELS_DIR: Path = ARTIFACTS_DIR / "models"

#: Points at the run directory the API should serve.
ACTIVE_MODEL_POINTER: Path = ARTIFACTS_DIR / "active_model.json"


@dataclasses.dataclass(frozen=True)
class RunFilenames:
    """The three files that make up a training run directory.

    Defined here rather than in ``ml.training.trainer`` because
    :func:`get_active_run_dir` needs to recognise a *finished* run, and
    ``ml.config`` cannot import from a module that imports ``ml.config``.
    """

    checkpoint: str = "best_model.pt"
    history: str = "history.csv"
    summary: str = "training_summary.json"


RUN_FILENAMES = RunFilenames()


# ---------------------------------------------------------------------------
# Dataset: RAVDESS
# ---------------------------------------------------------------------------
# Source of truth (Zenodo record 1188976, "Filename identifiers"):
#
#   Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad,
#            05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
#
# NOTE: a very common internet "RAVDESS mapping" (01=neutral, 03=disgust,
# 05=happy, 07=surprise ...) is WRONG and silently relabels the dataset. The
# mapping below is the one published with the dataset itself. It is asserted
# against the filename parser in tests/test_data.py.
RAVDESS_EMOTION_CODES: dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

#: Canonical class order. Index in this list == index of the model's output logit.
EMOTION_LABELS: list[str] = list(RAVDESS_EMOTION_CODES.values())
NUM_CLASSES: int = len(EMOTION_LABELS)

RAVDESS_FILENAME_EXAMPLE: str = "03-01-05-01-02-01-12.wav"
#: Modalality code for audio-only files (03).
RAVDESS_AUDIO_MODALITY: str = "03"

RAVDESS_DOWNLOAD_URL: str = (
    "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
)
RAVDESS_ARCHIVE_NAME: str = "Audio_Speech_Actors_01-24.zip"
RAVDESS_ARCHIVE_MD5: str = "bc696df654c87fed845eb13823edef8a"
RAVDESS_LICENSE: str = "CC BY-NC-SA 4.0 (non-commercial)"

RAVDESS_N_ACTORS: int = 24


def resolve_ravdess_root(raw_dir: Path = RAW_DIR) -> Path:
    """Locate the extracted ``Audio_Speech_Actors_01-24`` directory.

    The archive layout has changed between Zenodo revisions, so rather than
    hard-coding one exact path we try the canonical location first and then fall
    back to a bounded search for a directory containing ``Actor_*`` folders.

    Raises:
        FileNotFoundError: if no candidate directory looks like RAVDESS audio.
    """
    candidates = [
        raw_dir / "RAVDESS" / "Audio_Speech_Actors_01-24",
        raw_dir / "Audio_Speech_Actors_01-24",
        raw_dir,
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("Actor_*")):
            return candidate

    # Bounded fallback search (avoid walking the whole project tree).
    if raw_dir.is_dir():
        for path in sorted(raw_dir.rglob("Actor_*")):
            if path.is_dir():
                return path.parent

    raise FileNotFoundError(
        f"Could not locate the extracted RAVDESS speech audio under '{raw_dir}'. "
        "Run `python scripts/download_dataset.py` first."
    )


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
@dataclass
class SplitConfig:
    """Speaker-disjoint train/validation/test split.

    RAVDESS contains 24 actors x 60 utterances. Splitting at the *file* level
    would put the same voice in train and test, letting the network memorise
    speaker identity and inflating accuracy by several points. We therefore
    partition **actors**, never files.
    """

    #: Held out for the final, one-shot evaluation.
    test_actors: list[int] = field(default_factory=lambda: [3, 8, 14, 21])
    #: Used for early stopping / model selection / hyper-parameter choice.
    val_actors: list[int] = field(default_factory=lambda: [6, 11, 19, 22])
    #: Everything else (16 actors -> 960 utterances).
    seed: int = 42

    def train_actors(self) -> list[int]:
        held_out = set(self.test_actors) | set(self.val_actors)
        return [a for a in range(1, RAVDESS_N_ACTORS + 1) if a not in held_out]


# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------
@dataclass
class AudioConfig:
    """Waveform-level preprocessing.

    Every value here is applied identically at training and inference time by
    :mod:`ml.data.preprocessing`.
    """

    #: Analysis rate fed to the MFCC front-end. RAVDESS ships 48 kHz; 16 kHz is
    #: the standard rate for MFCC pipelines (covers the whole 0-8 kHz mel range)
    #: and is 3x cheaper, which matters on a CPU-only box.
    sample_rate: int = 16000
    #: Mixed down to a single channel. RAVDESS is already mono but user uploads
    #: may not be.
    mono: bool = True
    #: Energy threshold (dB below the peak) for leading/trailing silence removal.
    trim_top_db: float = 30.0
    #: Enable silence trimming.
    trim_silence: bool = True
    #: Peak-normalise to this amplitude so that recording gain and microphone
    #: distance cannot shift the model's input distribution.
    peak_normalize: bool = True
    peak_target: float = 0.95
    #: Fixed-length analysis window. Longer clips are centre-cropped at
    #: inference and randomly cropped during training; shorter clips are padded.
    target_duration_sec: float = 3.0
    #: Absolute guard rails - a 3 s window cannot contain a 4 hour recording.
    min_duration_sec: float = 0.35
    max_duration_sec: float = 30.0
    #: Reject clips that are (almost) entirely silence - they carry no emotion
    #: and would otherwise produce confident nonsense predictions.
    min_rms: float = 1e-4

    # ---- upload validation (shared by CLI + API) ----
    allowed_extensions: list[str] = field(
        default_factory=lambda: [".wav", ".flac", ".ogg", ".mp3", ".m4a"]
    )
    max_upload_bytes: int = 20 * 1024 * 1024  # 20 MB


# ---------------------------------------------------------------------------
# MFCC feature extraction
# ---------------------------------------------------------------------------
@dataclass
class FeatureConfig:
    """MFCC front-end.

    Defaults follow the standard librosa configuration for 16 kHz speech, with
    40 coefficients (rather than the 13 used for ASR) because the extra
    coefficients carry paralinguistic information - bandwidth, voice quality,
    spectral envelope detail - that correlates with emotional state.
    """

    n_mfcc: int = 40
    n_mels: int = 128
    n_fft: int = 512  # 32 ms window @ 16 kHz
    hop_length: int = 160  # 10 ms stride @ 16 kHz -> 100 frames/second
    win_length: int = 400  # 25 ms window @ 16 kHz
    fmin: float = 0.0
    fmax: float = 8000.0
    window: str = "hann"
    #: First-order temporal derivatives of the cepstra. Emotion is largely
    #: expressed as *movement* (pitch/energy dynamics), which lives in delta.
    use_delta: bool = True
    #: Second-order derivatives.
    use_delta2: bool = True
    #: Cepstral mean-and-variance normalisation, computed over the utterance.
    #: Acts as a cheap, training-free speaker/channel normalisation step.
    cmvn: bool = True

    def n_output_features(self) -> int:
        """Number of feature planes produced per frame.

        MFCC, delta and delta-delta are each a full ``n_mfcc``-wide block, so the
        stacked representation is ``n_mfcc * (1 + use_delta + use_delta2)``.
        """
        stacks = 1 + int(self.use_delta) + int(self.use_delta2)
        return self.n_mfcc * stacks

    def describe(self) -> str:
        planes = ["mfcc"]
        if self.use_delta:
            planes.append("delta")
        if self.use_delta2:
            planes.append("delta2")
        suffix = "+".join(planes)
        return f"{self.n_mfcc} x ({suffix}) = {self.n_output_features()} features/frame"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """CNN(2-D, time-frequency) -> BiLSTM -> attention pooling classifier."""

    #: Channels after each 2-D convolution block. The first block also halves
    #: the frequency axis; see EmotionCNNBiLSTM.
    channels: list[int] = field(default_factory=lambda: [32, 64, 128])
    #: Stride of the first convolution.
    #:
    #: At a 10 ms hop a 3 s window yields 301 frames, and neighbouring MFCC
    #: frames are heavily correlated. Running 32 convolution channels over the
    #: full 120x301 grid costs ~1.3 GFLOP per sample and dominated training time
    #: (~89 s/epoch measured on this CPU). A stride-2 stem reaches the same
    #: receptive-field coverage for roughly 1/16th of the early-layer cost, and
    #: the downsampling is *learned* rather than applied to the features, so no
    #: information is discarded. Measured effect: ~3.0 s -> ~0.3 s per step.
    stem_stride: list[int] = field(default_factory=lambda: [2, 2])
    #: Dropout after each conv block.
    conv_dropout: float = 0.15
    #: Dropout before the classifier head.
    head_dropout: float = 0.30
    #: LSTM hidden size (per direction).
    lstm_hidden: int = 128
    lstm_layers: int = 1
    bidirectional: bool = True
    #: Dropout *between* stacked LSTM layers. PyTorch only applies it when
    #: ``num_layers > 1``; with a single layer this is ignored.
    lstm_dropout: float = 0.1
    #: Enable the additive attention-pooling layer over LSTM timesteps. With
    #: fixed-length padding, mean/last pooling is biased by the padding.
    use_attention_pooling: bool = True
    #: BatchNorm adds a small dependency on batch statistics; harmless here.
    use_batch_norm: bool = True
    num_classes: int = NUM_CLASSES

    def lstm_output_dim(self) -> int:
        return self.lstm_hidden * (2 if self.bidirectional else 1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Optimisation hyper-parameters."""

    batch_size: int = 32
    max_epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    label_smoothing: float = 0.05
    #: Monitored metric for checkpointing. macro-F1 rather than loss/accuracy,
    #: because RAVDESS is imbalanced (see README -> Dataset).
    monitor: str = "val_macro_f1"
    monitor_mode: str = "max"
    early_stopping_patience: int = 12
    #: Decay LR by half when the monitor plateaus for this many epochs.
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    lr_scheduler_min_lr: float = 1e-5
    #: Number of CPU threads for torch intra-op parallelism.
    num_threads: int = 0  # 0 -> resolved from the machine at runtime
    seed: int = 42
    #: Deterministic cuDNN / data-order behaviour.
    deterministic: bool = True
    #: EMA-free but cheap: warmup-free cosine schedule is replaced by ReduceLROnPlateau
    #: (see :class:`TrainConfig.lr_scheduler_*`). Kept explicit for readability.


# ---------------------------------------------------------------------------
# Aggregate artifact config
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    """Everything needed to reproduce a prediction, saved beside the weights."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    #: Class order. Never infer this from the filesystem at load time.
    labels: list[str] = field(default_factory=lambda: list(EMOTION_LABELS))
    dataset: str = "RAVDESS (Audio_Speech_Actors_01-24)"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def sub_config(self, name: str) -> dict[str, Any]:
        """One nested config as a plain dict.

        Reports want the audio settings and the feature settings side by side;
        without this, every caller has to know that only the top-level config
        owns ``to_dict``.
        """
        config = getattr(self, name, None)
        if not dataclasses.is_dataclass(config):
            raise AttributeError(f"ExperimentConfig has no config field '{name}'")
        return dataclasses.asdict(config)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExperimentConfig:
        def build(kls: type, data: Any):
            if not isinstance(data, Mapping):
                return kls()
            known = {f.name for f in dataclasses.fields(kls)}
            return kls(**{k: v for k, v in data.items() if k in known})

        return cls(
            audio=build(AudioConfig, payload.get("audio")),
            features=build(FeatureConfig, payload.get("features")),
            model=build(ModelConfig, payload.get("model")),
            train=build(TrainConfig, payload.get("train")),
            split=build(SplitConfig, payload.get("split")),
            labels=list(payload.get("labels") or EMOTION_LABELS),
            dataset=str(payload.get("dataset", "RAVDESS (Audio_Speech_Actors_01-24)")),
            notes=str(payload.get("notes", "")),
        )

    # ---- persistence helpers ----
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Active model pointer
# ---------------------------------------------------------------------------
def set_active_model(run_name: str) -> None:
    """Record which trained run the inference service should load."""
    ACTIVE_MODEL_POINTER.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_MODEL_POINTER.write_text(
        json.dumps({"run": run_name, "run_dir": str(MODELS_DIR / run_name)}, indent=2),
        encoding="utf-8",
    )


def _run_is_complete(run_dir: Path) -> bool:
    """True when a run wrote a summary saying it finished.

    ``best_model.pt`` appears on the first epoch that improves, so its presence
    proves only that *something* was learned. ``training_summary.json`` with
    ``"complete": true`` is written once the loop exits, which is the only
    evidence that the checkpoint is final rather than mid-run.
    """
    summary = run_dir / RUN_FILENAMES.summary
    if not summary.is_file():
        return False
    try:
        return bool(json.loads(summary.read_text(encoding="utf-8")).get("complete"))
    except (json.JSONDecodeError, OSError, AttributeError):
        return False


def get_active_run_dir(explicit: Path | None = None) -> Path:
    """Resolve the run directory holding the model the API should serve.

    Resolution order:
      1. an explicit path (CLI override) - used verbatim,
      2. ``artifacts/active_model.json``,
      3. the most recently finished run directory containing ``best_model.pt``.

    Discovery (step 3) only considers **completed** runs. The asymmetry with
    step 1 is deliberate: naming a run explicitly is a human decision, while
    discovery is automatic and must never hand a service a checkpoint from a
    training job that is still running. Without this filter, starting a new
    training run silently changes what a running API serves, and it serves
    whatever that job has written so far.

    Raises:
        FileNotFoundError: if no trained model is available anywhere.
    """
    if explicit is not None:
        run_dir = Path(explicit)
        if not (run_dir / "best_model.pt").is_file():
            raise FileNotFoundError(f"No best_model.pt in run '{run_dir.name}'.")
        return run_dir

    if ACTIVE_MODEL_POINTER.is_file():
        try:
            payload = json.loads(ACTIVE_MODEL_POINTER.read_text(encoding="utf-8"))
            run_dir = Path(payload["run_dir"])
            if (run_dir / "best_model.pt").is_file():
                return run_dir
        except (json.JSONDecodeError, KeyError, OSError, TypeError):
            pass  # fall through to discovery

    candidates = [
        checkpoint.parent
        for checkpoint in MODELS_DIR.glob("*/best_model.pt")
        if checkpoint.is_file() and _run_is_complete(checkpoint.parent)
    ]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(
        "No completed trained model found. Train one and select it:\n"
        "    python scripts/prepare_data.py\n"
        "    python scripts/train.py --set-active"
    )


def resolve_num_threads(requested: int = 0) -> int:
    """Pick a sane torch thread count.

    On CPU, letting torch spawn one thread per core on a 12-core laptop competes
    with the OS and can be slower than using a handful. We cap it.
    """
    import os

    if requested and requested > 0:
        return requested
    cpu_count = os.cpu_count() or 4
    return max(1, min(cpu_count, 8))


__all__ = [
    "ARTIFACTS_DIR",
    "DATA_DIR",
    "EMOTION_LABELS",
    "FIGURES_DIR",
    "MODELS_DIR",
    "NUM_CLASSES",
    "PROJECT_ROOT",
    "RAVDESS_ARCHIVE_MD5",
    "RAVDESS_ARCHIVE_NAME",
    "RAVDESS_DOWNLOAD_URL",
    "RAVDESS_EMOTION_CODES",
    "RAVDESS_LICENSE",
    "RAW_DIR",
    "AudioConfig",
    "ExperimentConfig",
    "FeatureConfig",
    "ModelConfig",
    "SplitConfig",
    "TrainConfig",
    "get_active_run_dir",
    "resolve_num_threads",
    "resolve_ravdess_root",
    "set_active_model",
]
