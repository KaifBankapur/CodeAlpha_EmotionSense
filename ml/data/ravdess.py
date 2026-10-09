"""RAVDESS corpus parsing, manifest building and speaker-disjoint splitting.

Filename grammar
----------------
Every RAVDESS file is a 7-part numeric identifier::

    03-01-05-01-02-01-12.wav
    MM-VC-EI-SS-RV-RA-AA

    MM  modality        01 = audio-visual, 02 = video-only, 03 = audio-only
    VC  vocal channel   01 = speech,        02 = song
    EI  emotion         01 neutral, 02 calm, 03 happy, 04 sad,
                        05 angry, 06 fearful, 07 disgust, 08 surprised
    SS  intensity       01 normal, 02 strong  (neutral has no "strong")
    RV  statement       01 "Kids are talking by the door"
                        02 "Dogs are sitting by the door"
    RA  repetition      01 first, 02 second
    AA  actor           01-24  (odd male, even female, per the dataset docs)

Note the field order: the emotion code is the **third** field, not the second.
``03-01-05-...`` is audio-only, *speech*, anger.

We train on ``MM == 03`` (audio-only) and ``EI``/``RV == 01`` (speech), which is
exactly the content of ``Audio_Speech_Actors_01-24.zip``: 1440 files,
60 per actor, 24 actors.

The emotion code -> label table is **not** guessed here; it is imported from
:data:`ml.config.RAVDESS_EMOTION_CODES`, which is transcribed from the Zenodo
record for the dataset. Getting this wrong is the single most common silent
error in RAVDESS projects - the network still trains to a high accuracy and every
prediction is simply attached to the wrong word.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from ml.config import (
    EMOTION_LABELS,
    RAVDESS_AUDIO_MODALITY,
    RAVDESS_EMOTION_CODES,
    SplitConfig,
    resolve_ravdess_root,
)

logger = logging.getLogger(__name__)

_FILENAME_RE = re.compile(
    r"^(?P<modality>\d{2})-(?P<vocal_channel>\d{2})-(?P<emotion>\d{2})"
    r"-(?P<intensity>\d{2})-(?P<statement>\d{2})-(?P<repetition>\d{2})"
    r"-(?P<actor>\d{2})\.wav$",
    re.IGNORECASE,
)

VOCAL_CHANNEL_SPEECH = "01"
SPLIT_NAMES = ("train", "val", "test")

#: RAVDESS has exactly 24 professional actors (Actor_01 .. Actor_24).
RAVDESS_N_ACTORS = 24


@dataclass(frozen=True)
class RavdessSample:
    """One utterance plus everything derivable from its filename."""

    path: str          # path relative to the dataset root (portable)
    filename: str
    label: str
    label_index: int
    emotion_code: str
    intensity: int
    statement: int
    repetition: int
    actor: int
    speaker_id: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RavdessSample:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


class RavdessParseError(ValueError):
    """Raised when a ``.wav`` filename does not follow the RAVDESS convention."""


def parse_filename(path: Path, root: Path | None = None) -> RavdessSample:
    """Turn ``Audio_Speech_Actors_01-24/Actor_07/03-01-05-01-01-01-07.wav``
    into a :class:`RavdessSample`.

    Raises:
        RavdessParseError: on a malformed name or an unknown emotion code.
    """
    match = _FILENAME_RE.match(path.name)
    if not match:
        raise RavdessParseError(
            f"'{path.name}' does not match the RAVDESS naming convention "
            "MM-VC-EI-SS-RV-RA-AA.wav."
        )

    fields = {k: v for k, v in match.groupdict().items()}
    emotion_code = fields["emotion"]
    if emotion_code not in RAVDESS_EMOTION_CODES:
        raise RavdessParseError(
            f"Unknown emotion code '{emotion_code}' in '{path.name}'."
        )

    label = RAVDESS_EMOTION_CODES[emotion_code]
    actor = int(fields["actor"])
    if not 1 <= actor <= RAVDESS_N_ACTORS:
        # RAVDESS only ever has actors 01-24. Accepting 99 would silently
        # manufacture a speaker id that splits cannot be reasoned about.
        raise RavdessParseError(
            f"Actor {actor:02d} in '{path.name}' is outside the RAVDESS range "
            f"01-{RAVDESS_N_ACTORS:02d}."
        )

    try:
        relative = str(path.relative_to(root)) if root else str(path)
    except ValueError:
        relative = str(path)

    return RavdessSample(
        path=relative.replace("\\", "/"),
        filename=path.name,
        label=label,
        label_index=EMOTION_LABELS.index(label),
        emotion_code=emotion_code,
        intensity=int(fields["intensity"]),
        statement=int(fields["statement"]),
        repetition=int(fields["repetition"]),
        actor=actor,
        speaker_id=f"Actor_{actor:02d}",
    )


def iter_wav_files(root: Path) -> Iterable[Path]:
    """Yield every ``.wav`` under ``root`` in deterministic order."""
    yield from sorted(root.rglob("*.wav"))


def build_manifest(
    root: Path | None = None,
    *,
    audio_only: bool = True,
    speech_only: bool = True,
    on_malformed: str = "raise",
) -> list[RavdessSample]:
    """Scan the corpus and return one :class:`RavdessSample` per usable file.

    Args:
        root: dataset root; auto-resolved from ``data/raw`` when omitted.
        audio_only: keep only ``MM == 03`` (audio-only files).
        speech_only: keep only vocal channel ``01`` (speech, not song).
        on_malformed: ``"raise"``, ``"warn"`` (skip) or ``"ignore"``.
    """
    root = Path(root) if root is not None else resolve_ravdess_root()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root '{root}' does not exist.")

    samples: list[RavdessSample] = []
    skipped: list[str] = []

    for path in iter_wav_files(root):
        try:
            sample = parse_filename(path, root)
        except RavdessParseError as exc:
            if on_malformed == "raise":
                raise
            logger.warning("skipping %s (%s)", path.name, exc)
            skipped.append(path.name)
            continue

        if audio_only and sample.filename[:2] != RAVDESS_AUDIO_MODALITY:
            continue
        if speech_only and int(sample.filename[3:5]) != int(VOCAL_CHANNEL_SPEECH):
            continue
        samples.append(sample)

    if skipped:
        logger.warning("skipped %d malformed filenames", len(skipped))

    samples.sort(key=lambda s: s.path)
    logger.info("manifest: %d samples from %s", len(samples), root)
    return samples


def summarise(samples: Sequence[RavdessSample]) -> dict[str, object]:
    """Dataset statistics - counts per emotion, per actor, per intensity.

    Deliberately computed from the files that are actually present rather than
    hard-coded from the dataset paper, so the README numbers cannot drift.
    """
    by_emotion = Counter(s.label for s in samples)
    by_actor = Counter(s.actor for s in samples)
    by_intensity = Counter(s.intensity for s in samples)

    return {
        "n_samples": len(samples),
        "n_speakers": len(by_actor),
        "n_classes": len(by_emotion),
        "samples_per_emotion": {label: by_emotion.get(label, 0) for label in EMOTION_LABELS},
        "samples_per_actor": {f"Actor_{a:02d}": by_actor.get(a, 0) for a in sorted(by_actor)},
        "samples_per_intensity": {
            "normal": by_intensity.get(1, 0),
            "strong": by_intensity.get(2, 0),
        },
        "class_imbalance_ratio": (
            round(max(by_emotion.values()) / min(by_emotion.values()), 3)
            if by_emotion and min(by_emotion.values()) > 0
            else None
        ),
    }


def assign_splits(
    samples: Sequence[RavdessSample], split_cfg: SplitConfig | None = None
) -> dict[str, list[RavdessSample]]:
    """Partition by **actor** so no voice appears in two splits.

    Why this matters: RAVDESS gives every actor the same 60 utterances, so a
    random file-level split puts near-identical recordings of the *same* speaker
    on both sides of the boundary. A model can then score well by recognising
    *who* is talking rather than *how*, which inflates test accuracy and makes
    the number meaningless for real users whose voice the model has never heard.

    Args:
        samples: manifest entries.
        split_cfg: which actors go where.

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}``.

    Raises:
        ValueError: if the actor sets overlap or do not cover all actors.
    """
    split_cfg = split_cfg or SplitConfig()
    val = set(split_cfg.val_actors)
    test = set(split_cfg.test_actors)
    train = set(split_cfg.train_actors())

    if (val & test) or (val & train) or (test & train):
        raise ValueError(
            f"Split actors overlap: train={sorted(train)}, "
            f"val={sorted(val)}, test={sorted(test)}"
        )
    missing = set(range(1, 25)) - (train | val | test)
    if missing:
        raise ValueError(f"Splits do not cover actor(s): {sorted(missing)}")

    buckets: dict[str, list[RavdessSample]] = {name: [] for name in SPLIT_NAMES}
    for sample in samples:
        if sample.actor in test:
            buckets["test"].append(sample)
        elif sample.actor in val:
            buckets["val"].append(sample)
        else:
            buckets["train"].append(sample)

    for name in SPLIT_NAMES:
        buckets[name].sort(key=lambda s: s.path)
    return buckets


def save_manifest(samples: Sequence[RavdessSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([s.to_dict() for s in samples], indent=2), encoding="utf-8"
    )


def load_manifest(path: Path) -> list[RavdessSample]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [RavdessSample.from_dict(item) for item in payload]


def save_split_manifest(
    splits: Mapping[str, Sequence[RavdessSample]],
    split_cfg: SplitConfig,
    path: Path,
) -> None:
    """Persist exactly which actor went where, so runs stay auditable."""
    payload = {
        "methodology": (
            "speaker-disjoint: the 24 RAVDESS actors are partitioned into "
            "train/val/test and no actor appears in more than one split. "
            "Selection used the validation macro-F1; the test split was "
            "evaluated once, after model selection was frozen."
        ),
        "train_actors": split_cfg.train_actors(),
        "val_actors": split_cfg.val_actors,
        "test_actors": split_cfg.test_actors,
        "counts": {name: len(items) for name, items in splits.items()},
        "speakers_per_split": {
            name: sorted({s.actor for s in items}) for name, items in splits.items()
        },
        "emotions_per_split": {
            name: dict(Counter(s.label for s in items)) for name, items in splits.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "SPLIT_NAMES",
    "RavdessParseError",
    "RavdessSample",
    "assign_splits",
    "build_manifest",
    "load_manifest",
    "parse_filename",
    "save_manifest",
    "save_split_manifest",
    "summarise",
]
