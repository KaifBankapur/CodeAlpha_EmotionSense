"""
Predict the emotion in one or more audio files from the command line.

The same :class:`~ml.inference.predictor.EmotionPredictor` serves this script,
the HTTP API and the tests, so a command-line result and an API result for the
same file are identical by construction.

Usage
-----
    python scripts/predict.py clip.wav
    python scripts/predict.py clip.wav --json
    python scripts/predict.py data/raw/RAVDESS/Actor_03 --recursive --limit 10
    python scripts/predict.py clip.wav --run expA_lr1e3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.config import MODELS_DIR
from ml.data.preprocessing import AudioValidationError
from ml.inference.predictor import EmotionPredictor
from ml.utils.logging_utils import setup_logging

BAR_WIDTH = 28


def render_result(result, *, show_bars: bool = True) -> str:
    """Human-readable prediction block."""
    lines: list[str] = []
    lines.append("")
    lines.append(f"  DETECTED EMOTION : {result.emotion.upper()}")
    lines.append(f"  CONFIDENCE       : {result.confidence * 100:.2f}%")
    lines.append("")

    if show_bars:
        lines.append("  Emotion distribution")
        lines.append("  " + "-" * (BAR_WIDTH + 24))
        for label, probability in result.ranked:
            filled = round(probability * BAR_WIDTH)
            bar = "#" * filled + "." * (BAR_WIDTH - filled)
            marker = " <=" if label == result.emotion else ""
            lines.append(f"  {label:<11s} |{bar}| {probability * 100:5.2f}%{marker}")
        lines.append("  " + "-" * (BAR_WIDTH + 24))

    audio = result.audio
    lines.append("")
    lines.append(
        f"  Audio     : {audio['duration_sec']:.2f}s, {audio['sample_rate']} Hz, "
        f"{audio['channels']} ch (source {audio['native_sample_rate']} Hz), "
        f"peak {audio['peak_amplitude']:.3f}"
    )
    lines.append(f"  Processing: {result.processing_ms:.1f} ms")
    lines.append(
        f"  Model     : {result.model.get('run_name')} "
        f"({result.model.get('parameters'):,} params, {result.model.get('device')})"
    )
    lines.append("")
    return "\n".join(lines)


def collect_inputs(paths: Sequence[str], recursive: bool) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            found = [
                p
                for p in path.glob(pattern)
                if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
            ]
            files.extend(sorted(found))
        elif path.is_file():
            files.append(path)
        else:
            print(f"warning: '{raw}' does not exist, skipping", file=sys.stderr)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inputs", nargs="+", help="audio file(s) or directory(ies)")
    parser.add_argument("--run", default=None, help="run name or path (default: active)")
    parser.add_argument("--recursive", action="store_true", help="recurse into directories")
    parser.add_argument("--limit", type=int, default=None, help="max files to process")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    setup_logging()

    run_dir: Path | None = None
    if args.run:
        candidate = Path(args.run)
        if not candidate.is_dir():
            candidate = MODELS_DIR / args.run
        run_dir = candidate

    try:
        predictor = (
            EmotionPredictor.from_run_dir(run_dir) if run_dir else EmotionPredictor.load_active()
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    files = collect_inputs(args.inputs, args.recursive)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("error: no audio files to process", file=sys.stderr)
        return 2

    results = []
    failures = []
    for path in files:
        try:
            results.append((path, predictor.predict(path)))
        except AudioValidationError as exc:
            failures.append((path, str(exc)))
        except Exception as exc:
            failures.append((path, f"{type(exc).__name__}: {exc}"))

    if args.json:
        payload = {
            "results": [{"file": str(path), **result.to_dict()} for path, result in results],
            "failures": [{"file": str(path), "error": err} for path, err in failures],
            "model": predictor.model_info(),
        }
        print(json.dumps(payload, indent=2))
    else:
        for path, result in results:
            print(f"\n{path.name}")
            print(render_result(result))

        if len(results) > 1:
            print("  " + "=" * 52)
            print(f"  BATCH SUMMARY ({len(results)} files)")
            print("  " + "=" * 52)
            from collections import Counter

            counts = Counter(result.emotion for _, result in results)
            for emotion, count in counts.most_common():
                print(f"  {emotion:<12s} {count:>4d}")

    if failures:
        print(f"\n  {len(failures)} file(s) could not be processed:", file=sys.stderr)
        for path, error in failures[:10]:
            print(f"    {path.name}: {error}", file=sys.stderr)

    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
