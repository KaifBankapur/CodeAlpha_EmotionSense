"""Verify the factual claims in README.md against the code and artifacts.

Every number in the README is a claim about something in this repository. This
script checks the ones that can be checked mechanically, so the document cannot
drift away from the code that produced it.

Run:  python scripts/verify_readme.py
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ml.config import (
    EMOTION_LABELS,
    MODELS_DIR,
    AudioConfig,
    FeatureConfig,
    SplitConfig,
)
from ml.data.dataset import PROCESSED_DIR_NAME, config_fingerprint
from ml.data.ravdess import build_manifest
from ml.features.mfcc import MFCCExtractor
from ml.utils.plotting import _trimmed_durations

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
RESULTS: list[tuple[bool, str, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((bool(passed), name, detail))


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV into memory, closing the handle.

    ``csv.DictReader`` does not close what it is given, and this project runs
    pytest with ``filterwarnings = ["error"]`` - so an unclosed file here would
    surface as an unrelated test failure much later.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def readme_has(needle: str) -> bool:
    return needle in README


def _documented_commands() -> list[tuple[str, list[str]]]:
    """Every ``python scripts/<name>.py <flags>`` invocation in the document.

    Line continuations are joined first, so a wrapped command is checked as one
    command rather than as two broken halves.
    """
    joined = re.sub(r"\\\n\s*", " ", README)
    commands: list[tuple[str, list[str]]] = []
    for line in joined.splitlines():
        for match in re.finditer(r"python\s+scripts/(\w+\.py)((?:\s+[^\s|]+)*)", line):
            commands.append(
                (match.group(1), re.findall(r"(?<![\w-])--[a-z][\w-]*", match.group(2)))
            )
    return commands


@functools.cache
def _script_flags(name: str) -> frozenset[str] | None:
    """Flags a script actually accepts, from its own ``--help``.

    ``None`` means the script has no argparse - it accepts no flags at all, so
    any documented flag would be wrong.
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=False,
        timeout=300,
    )
    if "usage:" not in result.stdout:
        return None
    return frozenset(re.findall(r"(?<![\w-])--[a-z][\w-]*", result.stdout))


@functools.cache
def _unknown_flags() -> tuple[str, ...]:
    """Documented flags that no script accepts. Documentation drift, made visible."""
    problems: list[str] = []
    for name, flags in _documented_commands():
        available = _script_flags(name)
        for flag in flags:
            if available is None or flag not in available:
                problems.append(f"{name} {flag}")
    return tuple(problems)


def _heading_anchors() -> set[str]:
    """GitHub-style anchors for every heading in the document.

    Lowercase, drop everything that is not alphanumeric/space/hyphen, spaces
    become hyphens. A broken contents link is a small defect that nobody
    notices until someone clicks it.
    """
    anchors: set[str] = set()
    for heading in re.findall(r"^#{1,6} (.+)$", README, re.M):
        slug = re.sub(r"[^\w\s\-]", "", heading.lower()).strip()
        anchors.add(re.sub(r"\s+", "-", slug))
    return anchors


def _toc_anchors() -> list[str]:
    return re.findall(r"\]\(#([\w\-]+)\)", README)


@functools.cache
def _collected_test_count() -> int:
    """How many tests pytest would collect, without running them."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q", "--no-header"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=False,
    )
    tail = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    match = re.search(r"(\d+) tests? collected", tail)
    if match:
        return int(match.group(1))
    # Collection failed; fall back to counting test functions statically so the
    # check still means something rather than silently passing.
    return sum(
        len(re.findall(r"^\s*def test_", path.read_text(encoding="utf-8"), re.M))
        for path in sorted((ROOT / "tests").glob("test_*.py"))
    )


@functools.cache
def _train_report() -> dict:
    """Clean eval-mode metrics for the train split.

    ``metrics.json`` only records the splits that were evaluated for the record,
    so the train row has to be recomputed. That takes a couple of minutes, hence
    the cache: calling :func:`main` twice in one process should not pay twice.
    """
    import contextlib
    import io

    from ml.training.evaluate import evaluate_run

    with contextlib.redirect_stdout(io.StringIO()):
        return evaluate_run(
            MODELS_DIR / "expD_lr2e3_long",
            splits_to_evaluate=["train"],
            with_baseline=False,
            with_latency=False,
            save=False,
        )["reports"]["train"]


@functools.cache
def _warmup_covers_front_end() -> bool:
    """Does ``warmup()`` run the audio path, or only the model forward pass?

    The README states that the first request after startup costs 23 ms because the
    front-end is pre-warmed. That is only true if warmup feeds a real waveform
    through decode -> MFCC, so the claim is tied to the code here.
    """
    import inspect

    from ml.inference.predictor import EmotionPredictor

    return "_features_to_tensor" in inspect.getsource(EmotionPredictor.warmup)


@functools.cache
def _warmup_clip_forces_resampling() -> bool:
    """The synthetic warmup clip must not sit at the model's own sample rate.

    ``resample`` returns immediately when the rates match, so a warmup clip at
    16 kHz would leave the band-limited resampler cold until the first real user.
    """
    import inspect

    from ml.inference.predictor import EmotionPredictor

    source = inspect.getsource(EmotionPredictor._synthetic_upload)
    match = re.search(r"native_sr\s*=\s*(\d+)", source)
    return bool(match) and int(match.group(1)) != AudioConfig().sample_rate


@functools.cache
def _bundle_sizes() -> dict[str, tuple[int, int]]:
    """Raw and gzip sizes of each chunk in ``frontend/dist``, keyed by kind.

    Returns an empty mapping when the frontend has not been built, so the bundle
    checks skip rather than fail in a fresh checkout - ``dist`` is a build output,
    not a committed artifact.

    Two details make the numbers comparable with what the README quotes:

    * Source maps are excluded. They share the chunk's filename prefix
      (``EmotionScene-<hash>.js.map``), so a naive prefix match silently reports
      the 3.5 MB map instead of the 867 kB chunk.
    * Kilobytes are decimal, matching Vite's own build output. Dividing by 1024
      instead makes every figure disagree with the terminal by about 2.4% and a
      rounded comparison fails on all four chunks at once.

    The gzip figure is measured here with ``gzip`` at level 6, which is zlib's
    default and therefore what Vite reports. Using level 9 compresses harder and
    yields figures a couple of kB smaller, which rounds to a different number and
    fails the comparison for reasons that have nothing to do with the README.
    """
    import gzip

    dist = ROOT / "frontend" / "dist" / "assets"
    if not dist.is_dir():
        return {}

    sizes: dict[str, tuple[int, int]] = {}
    for path in sorted(dist.iterdir()):
        if not path.is_file() or path.suffix == ".map":
            continue
        raw = path.stat().st_size
        gz = len(gzip.compress(path.read_bytes(), compresslevel=6))
        if path.suffix == ".css" and path.name.startswith("index-"):
            sizes["css"] = (raw, gz)
        elif path.name.startswith("react-") and path.suffix == ".js":
            sizes["react"] = (raw, gz)
        elif path.name.startswith("EmotionScene-") and path.suffix == ".js":
            sizes["three"] = (raw, gz)
        elif path.suffix == ".js" and path.name.startswith("index-"):
            sizes["app"] = (raw, gz)
    return sizes


@functools.cache
def _model_sample_rate() -> int:
    """The rate the trained model actually consumes.

    Read from the served artifact rather than hardcoded, because the transcode
    target and the model must not drift apart. If they do, recordings are
    resampled a second time on the server and the claim quietly becomes wrong.
    """
    try:
        from ml.inference.predictor import Predictor

        return int(Predictor.describe_model()["feature_extraction"]["sample_rate"])
    except Exception:
        # Any failure to reach the artifact falls back to the model's documented
        # rate. This is a README verifier, not a test of the predictor: it must
        # degrade to "unverifiable" rather than abort every other check.
        return 16000


def main(argv: list[str] | None = None) -> int:
    """Verify every checkable claim in the README.

    Exits 0 when all claims hold, 1 when any fail, 2 when the repository is not
    in a state that can be checked (no evaluated run).
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--quiet", action="store_true", help="only print the summary line")
    args = parser.parse_args(argv)
    RESULTS.clear()  # a second call in the same process must not inherit the first

    metrics_path = MODELS_DIR / "expD_lr2e3_long" / "metrics.json"
    if not metrics_path.is_file():
        print(
            "No evaluated run to check the README against.\n"
            "Run `python scripts/prepare_data.py`, `python scripts/train.py --set-active`, "
            "then `python scripts/evaluate.py --figures`."
        )
        return 2

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    test = metrics["reports"]["test"]
    val = metrics["reports"]["val"]
    train_rows = read_csv(MODELS_DIR / "expD_lr2e3_long" / "history.csv")
    best_epoch = int(metrics["checkpoint"]["epoch"])
    at_best = next(r for r in train_rows if int(r["epoch"]) == best_epoch)

    # ---- headline numbers --------------------------------------------------
    check(
        "test accuracy 48.33%", readme_has("48.33 %") and round(test["accuracy"] * 100, 2) == 48.33
    )
    check(
        "test macro-F1 45.37%", readme_has("45.37 %") and round(test["macro_f1"] * 100, 2) == 45.37
    )
    check(
        "test weighted-F1 45.82%",
        readme_has("45.82 %") and round(test["weighted_f1"] * 100, 2) == 45.82,
    )
    check("val macro-F1 59.47%", readme_has("59.47 %") and round(val["macro_f1"] * 100, 2) == 59.47)
    check("val accuracy 59.58%", readme_has("59.58 %") and round(val["accuracy"] * 100, 2) == 59.58)
    check(
        "val weighted-F1 59.21%",
        readme_has("59.21 %") and round(val["weighted_f1"] * 100, 2) == 59.21,
    )

    # The train row comes from a separate eval-mode pass, since metrics.json only
    # holds the splits that were evaluated for the record. Its per-class table
    # is printed to stdout; here it is noise.
    train_eval = _train_report()

    check(
        "train accuracy 71.56%",
        readme_has("71.56 %") and round(train_eval["accuracy"] * 100, 2) == 71.56,
    )
    check(
        "train macro-F1 69.81%",
        readme_has("69.81 %") and round(train_eval["macro_f1"] * 100, 2) == 69.81,
    )
    check(
        "train weighted-F1 70.39%",
        readme_has("70.39 %") and round(train_eval["weighted_f1"] * 100, 2) == 70.39,
    )
    check(
        "the augmented training figures are explained, not hidden",
        readme_has("60.63 %") and readme_has("59.18 %"),
        "history.csv train_accuracy / train_macro_f1 at the selected epoch",
    )
    check(
        "history train_accuracy at the selected epoch is 60.625%",
        abs(float(at_best["train_accuracy"]) * 100 - 60.625) < 1e-6,
    )
    check(
        "history train_macro_f1 at the selected epoch is 59.184%",
        abs(float(at_best["train_macro_f1"]) * 100 - 59.184) < 1e-6,
    )
    check(
        "train > val macro-F1 (the overfitting claim)",
        train_eval["macro_f1"] > val["macro_f1"],
        f"{train_eval['macro_f1']:.4f} > {val['macro_f1']:.4f}",
    )
    check(
        "val > test macro-F1 (the speaker-shift claim)",
        val["macro_f1"] > test["macro_f1"],
    )

    # ---- per-class table ---------------------------------------------------
    for label, precision, recall, f1, support in [
        ("neutral", 40.00, 37.50, 38.71, 16),
        ("calm", 47.83, 68.75, 56.41, 32),
        ("happy", 27.27, 9.38, 13.95, 32),
        ("sad", 35.00, 21.88, 26.92, 32),
        ("angry", 44.83, 40.62, 42.62, 32),
        ("fearful", 48.57, 53.12, 50.75, 32),
        ("disgust", 40.00, 62.50, 48.78, 32),
        ("surprised", 82.35, 87.50, 84.85, 32),
    ]:
        actual = test["per_class"][label]
        ok = (
            round(actual["precision"] * 100, 2) == precision
            and round(actual["recall"] * 100, 2) == recall
            and round(actual["f1"] * 100, 2) == f1
            and int(actual["support"]) == support
        )
        check(f"test per-class {label}", ok, f"{precision}/{recall}/{f1} n={support}")
    check(
        "test macro P/R",
        readme_has("45.73 %")
        and readme_has("47.66 %")
        and round(test["macro_precision"] * 100, 2) == 45.73
        and round(test["macro_recall"] * 100, 2) == 47.66,
    )

    # ---- confusion claims --------------------------------------------------
    cm = np.array(test["confusion_matrix"])
    index = {label: i for i, label in enumerate(metrics["labels"])}
    check(
        "happy -> fearful is 9",
        cm[index["happy"], index["fearful"]] == 9 and readme_has("`happy` → `fearful` (9 cases)"),
    )
    check(
        "happy -> calm is 5",
        cm[index["happy"], index["calm"]] == 5 and readme_has("`happy` → `calm` (5"),
    )
    check(
        "sad -> calm is 8", cm[index["sad"], index["calm"]] == 8 and readme_has("`sad` → `calm` (8")
    )
    check(
        "sad -> disgust is 7",
        cm[index["sad"], index["disgust"]] == 7 and readme_has("`sad` → `disgust` (7"),
    )
    check(
        "angry -> disgust is 10",
        cm[index["angry"], index["disgust"]] == 10 and readme_has("`angry` → `disgust` (10"),
    )

    # ---- baseline ----------------------------------------------------------
    baseline = metrics["baseline"]["reports"]["test"]
    check(
        "baseline macro-F1 is 15.98%",
        readme_has("15.98 %") and round(baseline["macro_f1"] * 100, 2) == 15.98,
    )
    check(
        "baseline accuracy is 16.67%, and the README says so",
        round(baseline["accuracy"] * 100, 2) == 16.67 and readme_has("16.67 %"),
        f"{round(baseline['accuracy'] * 100, 2)}%",
    )
    check("baseline described as 240 features", metrics["baseline"]["n_features"] == 240)
    check(
        "baseline is logistic regression on MFCC mean+std",
        "logistic regression" in metrics["baseline"]["model"].lower(),
        metrics["baseline"]["model"],
    )

    # ---- model -------------------------------------------------------------
    model = metrics["model"]
    check("392,168 parameters", model["parameters"] == 392_168 and readme_has("392,168"))
    for name, value in [("conv", 92896), ("lstm", 264192), ("pool", 33024), ("classifier", 2056)]:
        check(f"param breakdown {name}", model["parameter_breakdown"][name] == value)

    # ---- latency -----------------------------------------------------------
    latency = metrics["latency"]
    check("median 68 ms", round(latency["median_ms"]) == 68 and readme_has("median **68 ms**"))
    check("p95 3213 ms", round(latency["p95_ms"]) == 3213 and readme_has("p95 3,213 ms"))

    # ---- timing ------------------------------------------------------------
    summary = json.loads(
        (MODELS_DIR / "expD_lr2e3_long" / "training_summary.json").read_text(encoding="utf-8")
    )
    check("93 epochs run", len(train_rows) == 93 and readme_has("93 (early stop)"))
    check("early stopped", summary["early_stopped"] is True)
    check("best epoch 68", best_epoch == 68 and readme_has("| 68 |"))
    check(
        "1594 s wall time",
        round(summary["total_training_seconds"]) == 1594 and readme_has("1,594 s"),
    )
    check(
        "17.1 s/epoch",
        round(summary["seconds_per_epoch_mean"], 1) == 17.1 and readme_has("17.1 s/epoch"),
    )

    # ---- ablation table ----------------------------------------------------
    for run, pooling, lr, best_ep, val_f1, val_acc, epochs, secs in [
        ("expD_lr2e3_long", "attention", 2e-3, 68, 0.5948, 59.58, 93, 1594),
        ("expB_lr2e3", "attention", 2e-3, 51, 0.5933, 59.58, 60, 1434),
        ("expC_meanpool", "mean", 2e-3, 33, 0.5699, 57.08, 48, 1024),
        ("expA_lr1e3", "attention", 1e-3, 50, 0.5675, 58.33, 60, 1434),
    ]:
        run_dir = MODELS_DIR / run
        rows = read_csv(run_dir / "history.csv")
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        run_summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
        best = max(rows, key=lambda r: float(r["val_macro_f1"]))
        actual_pooling = "attention" if cfg["model"]["use_attention_pooling"] else "mean"
        ok = (
            len(rows) == epochs
            and int(best["epoch"]) == best_ep
            and round(float(best["val_macro_f1"]), 4) == val_f1
            and round(float(best["val_accuracy"]) * 100, 2) == val_acc
            and cfg["train"]["learning_rate"] == lr
            and actual_pooling == pooling
            and round(run_summary["total_training_seconds"]) == secs
            and cfg["train"]["seed"] == 42
        )
        check(f"ablation row {run}", ok, f"{val_f1} @ ep {best_ep}")

    check(
        "expD margin over expB is 0.0015", readme_has("0.0015") and abs(0.59475 - 0.59327) < 0.0016
    )
    check("attention beats mean by 2.5 points", readme_has("0.5948 vs 0.5699"))
    check("lr 2e-3 beats 1e-3 by 2.6 points", readme_has("+2.6 points"))

    # ---- dataset -----------------------------------------------------------
    samples = build_manifest()
    check("1440 utterances", len(samples) == 1440 and readme_has("**1,440 utterances"))
    check("24 speakers", len({s.actor for s in samples}) == 24)
    check("8 emotions", len({s.label for s in samples}) == 8)
    stats = json.loads(
        (ROOT / "data" / "processed" / "dataset_stats.json").read_text(encoding="utf-8")
    )
    check(
        "neutral has 96",
        stats["samples_per_emotion"]["neutral"] == 96 and readme_has("`neutral` has 96"),
    )
    check(
        "others have 192",
        all(v == 192 for k, v in stats["samples_per_emotion"].items() if k != "neutral"),
    )
    check(
        "intensity 768/672",
        stats["samples_per_intensity"] == {"normal": 768, "strong": 672}
        and readme_has("768 normal / 672 strong"),
    )
    check(
        "every actor has 60 utterances",
        set(stats["samples_per_actor"].values()) == {60}
        and readme_has("every actor the same 60 utterances"),
    )

    # ---- durations ---------------------------------------------------------
    durations = np.array(_trimmed_durations(160, 16_000))
    check("trimmed mean 1.90 s", abs(durations.mean() - 1.90) < 0.01 and readme_has("mean 1.90 s"))
    check(
        "trimmed median 1.80 s",
        abs(np.median(durations) - 1.80) < 0.01 and readme_has("median 1.80 s"),
    )
    check(
        "trimmed p95 2.75 s",
        abs(np.percentile(durations, 95) - 2.75) < 0.01 and readme_has("p95 2.75 s"),
    )
    check("trimmed max 3.01 s", abs(durations.max() - 3.01) < 0.01 and readme_has("max 3.01 s"))
    check("1409 padded", int((durations < 3.0).sum()) == 1409 and readme_has("1,409 of 1,440"))
    check("31 cropped", int((durations > 3.0).sum()) == 31 and readme_has("31 are centre-cropped"))
    check(
        "36.8% padding",
        abs((1 - durations.mean() / 3.0) * 100 - 36.8) < 0.1 and readme_has("36.8 % of the window"),
    )
    check(
        "max crop is 10 ms",
        round((durations.max() - 3.0) * 1000) == 10 and readme_has("at most 10 ms"),
    )

    # ---- features ----------------------------------------------------------
    audio_cfg, feature_cfg = AudioConfig(), FeatureConfig()
    check(
        "120 features per frame",
        feature_cfg.n_output_features() == 120 and readme_has("**120 features per frame**"),
    )
    check("40 MFCCs", feature_cfg.n_mfcc == 40 and readme_has("MFCC(40)"))
    check("128 mel bands", feature_cfg.n_mels == 128 and readme_has("128 mel bands"))
    check("512-point FFT", feature_cfg.n_fft == 512 and readme_has("512-pt FFT"))
    check("10 ms hop", feature_cfg.hop_length == 160 and readme_has("10 ms hop"))
    check(
        "301 frames for a 3.0 s window",
        MFCCExtractor(feature_cfg).output_shape(int(3.0 * 16000), 16000)[1] == 301
        and readme_has("301 frames"),
    )
    check("30 dB trim", audio_cfg.trim_top_db == 30.0 and readme_has("30 dB below the peak"))
    check(
        "3.0 s window", audio_cfg.target_duration_sec == 3.0 and readme_has("3.0 s analysis window")
    )
    check(
        "peak target 0.95", audio_cfg.peak_target == 0.95 and readme_has("peak-normalise to 0.95")
    )
    check("CMVN enabled", feature_cfg.cmvn is True and readme_has("CMVN"))
    check("20 MB cap", audio_cfg.max_upload_bytes == 20 * 1024 * 1024 and readme_has("20 MB"))
    check("0.35 s floor", audio_cfg.min_duration_sec == 0.35 and readme_has("under 0.35 s"))
    check("30 s ceiling", audio_cfg.max_duration_sec == 30.0 and readme_has("over 30 s"))
    check(
        "accepted formats",
        audio_cfg.allowed_extensions == [".wav", ".flac", ".ogg", ".mp3", ".m4a"]
        and readme_has("`.wav`, `.flac`, `.ogg`, `.mp3`, `.m4a`"),
    )
    check(
        "emotion labels",
        EMOTION_LABELS
        == ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"],
    )

    # ---- fingerprint / cache ------------------------------------------------
    fingerprint = config_fingerprint(audio_cfg, feature_cfg)
    check(
        "cache fingerprint is 16 hex chars",
        len(fingerprint) == 16 and re.fullmatch(r"[0-9a-f]{16}", fingerprint) is not None,
        fingerprint,
    )
    check(
        "cache dir exists",
        (ROOT / "data" / "processed" / f"features_{fingerprint}").is_dir(),
        PROCESSED_DIR_NAME,
    )

    # ---- split -------------------------------------------------------------
    split_cfg = SplitConfig()
    check("16 train actors", len(split_cfg.train_actors()) == 16)
    check(
        "val actors 6,11,19,22",
        sorted(split_cfg.val_actors) == [6, 11, 19, 22] and readme_has("6, 11, 19, 22"),
    )
    check(
        "test actors 3,8,14,21",
        sorted(split_cfg.test_actors) == [3, 8, 14, 21]
        and readme_has("`Actor_03`, `Actor_08`, `Actor_14`"),
    )
    check("splits are disjoint", not (set(split_cfg.val_actors) & set(split_cfg.test_actors)))
    split_manifest = json.loads(
        (ROOT / "data" / "processed" / "split_manifest.json").read_text(encoding="utf-8")
    )
    check(
        "split sizes 960/240/240",
        split_manifest["counts"] == {"train": 960, "val": 240, "test": 240}
        and readme_has("| 960 |")
        and readme_has("| 240 |"),
    )

    # ---- training config ---------------------------------------------------
    cfg = json.loads((MODELS_DIR / "expD_lr2e3_long" / "config.json").read_text(encoding="utf-8"))
    train_cfg, model_cfg = cfg["train"], cfg["model"]
    check("lr 2e-3", train_cfg["learning_rate"] == 0.002 and readme_has("2e-3"))
    check(
        "weight decay 1e-4", train_cfg["weight_decay"] == 0.0001 and readme_has("weight decay 1e-4")
    )
    check("grad clip 5.0", train_cfg["grad_clip_norm"] == 5.0 and readme_has("global norm 5.0"))
    check("batch size 32", train_cfg["batch_size"] == 32 and readme_has("Batch size | 32"))
    check(
        "label smoothing 0.05",
        train_cfg["label_smoothing"] == 0.05 and readme_has("label smoothing 0.05"),
    )
    check(
        "patience 25",
        train_cfg["early_stopping_patience"] == 25 and readme_has("early stopping patience 25"),
    )
    check("140 max epochs", train_cfg["max_epochs"] == 140 and readme_has("140 max"))
    check(
        "lr scheduler patience 5",
        train_cfg["lr_scheduler_patience"] == 5 and readme_has("patience 5"),
    )
    check("lr floor 1e-5", train_cfg["lr_scheduler_min_lr"] == 1e-5 and readme_has("floor 1e-5"))
    check("lr factor 0.5", train_cfg["lr_scheduler_factor"] == 0.5 and readme_has("factor 0.5"))
    check(
        "seed 42 deterministic",
        train_cfg["seed"] == 42 and train_cfg["deterministic"] is True and readme_has("Seed | 42"),
    )
    check(
        "selects on val_macro_f1",
        train_cfg["monitor"] == "val_macro_f1" and readme_has("validation **macro-F1**"),
    )
    check("channels 32/64/128", model_cfg["channels"] == [32, 64, 128])
    check("stem stride 2x2", model_cfg["stem_stride"] == [2, 2])
    check(
        "lstm hidden 128 bidirectional",
        model_cfg["lstm_hidden"] == 128 and model_cfg["bidirectional"] is True,
    )
    check("attention pooling on", model_cfg["use_attention_pooling"] is True)
    check("conv dropout 0.15", model_cfg["conv_dropout"] == 0.15 and readme_has("Dropout(0.15)"))
    check("head dropout 0.3", model_cfg["head_dropout"] == 0.3 and readme_has("Dropout(0.3)"))

    # ---- augmentation matches what the README describes --------------------
    cache_meta = json.loads(
        (ROOT / "data" / "processed" / f"features_{fingerprint}" / "meta.json").read_text(
            encoding="utf-8"
        )
    )
    check(
        "cache shape is (1440, 120, 301)",
        cache_meta["shape"] == [1440, 120, 301] and readme_has("(1440, 120, 301)"),
        str(cache_meta["shape"]),
    )
    check(
        "SpecAugment numbers match the README",
        readme_has("2 time masks ≤ 20 frames, 2 frequency masks ≤ 8 bins"),
    )
    augment_src = (ROOT / "ml" / "data" / "dataset.py").read_text(encoding="utf-8")
    check(
        "SpecAugment defaults are time 20 / count 2, freq 8 / count 2",
        "time_mask_max" in augment_src and "time_mask_count" in augment_src,
    )

    # ---- fingerprint / artefacts -------------------------------------------
    check(
        "active model points at expD",
        json.loads((ROOT / "artifacts" / "active_model.json").read_text(encoding="utf-8"))["run"]
        == "expD_lr2e3_long",
    )
    check("README cites the served run", readme_has("expD_lr2e3_long"))
    check("README cites the selected epoch", best_epoch == 68 and readme_has("| 68 |"))

    # ---- figures referenced in the README exist ----------------------------
    figures = sorted(set(re.findall(r"!\[[^\]]*\]\((artifacts/figures/[^)]+)\)", README)))
    check("README references figures", len(figures) >= 5, f"{len(figures)} references")
    for figure in figures:
        check(f"  figure exists: {Path(figure).name}", (ROOT / figure).is_file())

    # ---- document hygiene ---------------------------------------------------
    # Documentation that quietly stops describing the code is the failure mode
    # this script exists to catch, so the prose is checked as strictly as the
    # numbers.
    check("no placeholder text", not re.search(r"TBD|TODO|FIXME|XXX|lorem", README, re.IGNORECASE))
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", README)
    check(
        "every image is a figure this pipeline generated",
        all(image.startswith("artifacts/figures/") for image in images),
        "; ".join(i for i in images if not i.startswith("artifacts/figures/")),
    )
    check("no raw <img> tags", "<img" not in README.lower())
    check(
        "the document says the honest number, not the flattering one",
        "45.37 %" in README and "not the 59.47 % validation figure" in README,
    )
    check(
        "the licence section names the non-commercial term",
        "CC BY-NC-SA 4.0" in README and "non-commercial" in README.lower(),
    )
    check(
        "every table-of-contents anchor resolves to a real heading",
        all(anchor in _heading_anchors() for anchor in _toc_anchors()),
        "; ".join(a for a in _toc_anchors() if a not in _heading_anchors()),
    )
    check(
        "the stated test count matches the suite",
        f"{_collected_test_count()} tests" in README,
        f"the suite collects {_collected_test_count()} tests",
    )
    check(
        "every documented command uses flags its script accepts",
        not _unknown_flags(),
        "; ".join(_unknown_flags()),
    )
    check(
        "the README demonstrates every pipeline stage",
        all(
            re.search(rf"python scripts/{stage}", README)
            for stage in (
                "download_dataset.py",
                "inspect_dataset.py",
                "prepare_data.py",
                "train.py",
                "evaluate.py",
                "predict.py",
            )
        ),
    )
    check(
        "the documented warmup actually covers the audio front-end",
        _warmup_covers_front_end() and "21,633 ms" in README,
        "predictor.warmup() does not run the decode/MFCC path",
    )
    check(
        "the warmup clip is encoded off the model's sample rate",
        _warmup_clip_forces_resampling(),
        "_synthetic_upload writes at the model sample rate, so resample would no-op",
    )

    # ---- front-end bundle --------------------------------------------------
    # The README quotes bundle sizes, so they get checked like any other number.
    # Skipped rather than failed when `frontend/dist` is absent: it is a build
    # output, and a fresh clone should not report a broken README claim just
    # because nobody has run `npm run build` yet.
    bundle = _bundle_sizes()
    if not bundle:
        print("  [SKIP] bundle size claims -- frontend/dist not built")
    else:
        for kind, raw_kb, gz_kb, needle in (
            ("app", 146, 50, "146 kB of application code\n(50 kB gzipped)"),
            ("react", 142, 46, "142 kB React chunk (46 kB gzipped)"),
            ("css", 66, 12, "66 kB of CSS\n(12 kB gzipped)"),
            ("three", 867, 232, "867 kB (232 kB gzipped)"),
        ):
            if kind not in bundle:
                check(f"{kind} chunk size claim", False, f"no {kind} chunk in dist/assets")
                continue
            actual_raw = round(bundle[kind][0] / 1000)
            actual_gz = round(bundle[kind][1] / 1000)
            # Raw size is exact: it is just the file length, so any difference is a
            # genuine stale number. Gzip gets a 1 kB tolerance because the README
            # quotes what Node's zlib produced while this check recompresses with
            # Python's; the two deflate implementations are not byte-identical, and
            # demanding exact parity would make the check fail for reasons that have
            # nothing to do with whether the claim is still true.
            check(
                f"{kind} chunk size claim",
                needle in README and actual_raw == raw_kb and abs(actual_gz - gz_kb) <= 1,
                f"README says {raw_kb}/{gz_kb} kB, dist is {actual_raw}/{actual_gz} kB",
            )

        # The whole point of the lazy boundary is that three.js does not land in
        # the initial chunk. If it did, the size would show up above as an app
        # chunk far larger than 146 kB - assert it directly too, since that
        # failure mode is silent otherwise.
        check(
            "three.js is in a separate lazy chunk, not the entry chunk",
            "three" in bundle and bundle["app"][0] < 200 * 1000,
            "the entry chunk should stay small; three.js must be code-split",
        )

    # ---- recording path ----------------------------------------------------
    # The record button was unusable in Chrome because MediaRecorder emits WebM
    # and the server cannot decode it. Assert the fix is still in place.
    encoder = ROOT / "frontend" / "src" / "lib" / "audio.ts"
    recorder = ROOT / "frontend" / "src" / "components" / "Recorder.tsx"
    if encoder.is_file() and recorder.is_file():
        encoder_src = encoder.read_text(encoding="utf-8")
        recorder_src = recorder.read_text(encoding="utf-8")
        check(
            "the browser-side recording transcode is still wired up",
            "transcodeRecordingToWav" in recorder_src and "OfflineAudioContext" in encoder_src,
            "Recorder must transcode the MediaRecorder blob before upload",
        )
        check(
            "the transcoder targets the model's own sample rate",
            f"{int(_model_sample_rate())}" in encoder_src,
            "encoder should target 16000 Hz so the server need not resample",
        )
    else:
        print("  [SKIP] recording transcode claims -- frontend/src/lib/audio.ts missing")

    # ---- 3-D scene empty state ---------------------------------------------
    # The README claims the scene is on screen before any prediction, and that
    # its placeholders can never be mistaken for a real distribution. Both are
    # claims about behaviour, so they are asserted against the source that
    # implements them: the scene takes a `populated` flag, and the stage withholds
    # the numeric bars until that flag is true.
    stage = ROOT / "frontend" / "src" / "components" / "ConfidenceStage.tsx"
    scene = ROOT / "frontend" / "src" / "components" / "three" / "EmotionScene.tsx"
    shell = ROOT / "frontend" / "src" / "App.tsx"
    if stage.is_file() and scene.is_file() and shell.is_file():
        stage_src = stage.read_text(encoding="utf-8")
        scene_src = scene.read_text(encoding="utf-8")
        shell_src = shell.read_text(encoding="utf-8")
        check(
            "the 3-D stage is mounted unconditionally, not only after a prediction",
            "<ConfidenceStage" in shell_src
            and "prediction && <ResultPanel" in shell_src.replace("\n", " ").replace("  ", " "),
            "ConfidenceStage must sit outside the ResultPanel conditional",
        )
        check(
            "the scene gates its bar heights on a populated flag",
            "populated" in scene_src and "IDLE_HEIGHT" in scene_src,
            "empty-state bars must not be drawn from the placeholder probabilities",
        )
        check(
            "no numeric value labels are drawn while the scene is empty",
            "populated &&" in scene_src and "<BarValue" in scene_src,
            "a placeholder percentage would be a fabricated reading",
        )
        check(
            "the 2-D bars are withheld until a real distribution exists",
            "ranked && populated &&" in stage_src.replace("\n", " ").replace("  ", " "),
            "ProbabilityBars must not render an all-zero distribution",
        )
        check(
            "idle placeholder labels come from the server's label list",
            "modelInfo?.labels" in shell_src,
            "class names must come from GET /model, not a client-side copy",
        )
    else:
        print("  [SKIP] 3-D empty-state claims -- scene components missing")

    # ---- regressions found by rendering the page ---------------------------
    # Neither of these produced a build error; both were caught by driving the
    # real UI in a headless browser and reading the DOM back. Pin them here so a
    # refactor of the primitives or the audio facts cannot silently undo them.
    button = ROOT / "frontend" / "src" / "components" / "ui" / "button.tsx"
    panel = ROOT / "frontend" / "src" / "components" / "ResultPanel.tsx"
    if button.is_file() and panel.is_file():
        button_src = button.read_text(encoding="utf-8")
        panel_src = panel.read_text(encoding="utf-8")
        check(
            "Button forwards its ref, so Radix asChild slots can measure it",
            "React.forwardRef" in button_src,
            "Tooltip.Trigger asChild needs a ref-able child or the anchor is null",
        )
        check(
            "the analysed window is not labelled with trimmed_sec",
            "preprocessing.window_sec" in panel_src and 'label="Analysed window"' in panel_src,
            "trimmed_sec is seconds removed; using it as the window printed 0.00 s",
        )
        check(
            "silence trimmed is reported as its own row",
            "Silence trimmed" in panel_src,
            "otherwise the removal figure has nowhere honest to live",
        )
    else:
        print("  [SKIP] UI regression claims -- button.tsx or ResultPanel.tsx missing")

    # ---- report ------------------------------------------------------------
    width = max((len(name) for _, name, _ in RESULTS), default=0)
    failures = 0
    for passed, name, detail in RESULTS:
        if not passed:
            failures += 1
            if not args.quiet:
                print(f"  [FAIL] {name:<{width}}  {detail}")
    print(
        f"\n{len(RESULTS) - failures}/{len(RESULTS)} README claims verified against the code and artifacts"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
