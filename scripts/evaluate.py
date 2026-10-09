"""
Evaluate a trained run and write a full metric record.

Usage
-----
    python scripts/evaluate.py                        # active run, val + test
    python scripts/evaluate.py --run expA_lr1e3
    python scripts/evaluate.py --splits train,val,test
    python scripts/evaluate.py --no-baseline --no-latency

Outputs (in ``artifacts/models/<run>/``):
    metrics.json   per-split metrics, error analysis, baseline, latency

With ``--figures``:
    artifacts/figures/*.png   confusion matrix, per-class bars, training
                              curves, corpus overview, one MFCC example
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.config import get_active_run_dir
from ml.training.evaluate import evaluate_run
from ml.utils.logging_utils import setup_logging

VALID_SPLITS = ("train", "val", "test")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", default=None, help="run name or path (default: active)")
    parser.add_argument(
        "--splits",
        default="val,test",
        help=f"comma-separated subset of {','.join(VALID_SPLITS)}",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--no-baseline", action="store_true", help="skip the logistic-regression reference"
    )
    parser.add_argument("--no-latency", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--figures",
        action="store_true",
        help="also render the documentation figures into artifacts/figures/",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="also write the full record to this path"
    )
    args = parser.parse_args()

    setup_logging()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    unknown = [s for s in splits if s not in VALID_SPLITS]
    if unknown:
        parser.error(f"unknown split(s) {unknown}; choose from {VALID_SPLITS}")

    run_dir = Path(args.run) if args.run else get_active_run_dir()
    if not run_dir.is_absolute() and args.run and not (Path.cwd() / run_dir).exists():
        from ml.config import MODELS_DIR

        candidate = MODELS_DIR / run_dir
        if candidate.is_dir():
            run_dir = candidate

    print(f"evaluating: {run_dir}")
    print(f"splits    : {', '.join(splits)}")

    result = evaluate_run(
        run_dir,
        splits_to_evaluate=splits,
        device_str=args.device,
        with_baseline=not args.no_baseline,
        with_latency=not args.no_latency,
        save=not args.no_save,
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")

    # ---- side-by-side summary ------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'model':<28s} {'split':<6s} {'accuracy':>9s} {'macro-F1':>9s}")
    print("-" * 72)

    for split, report in result["reports"].items():  # type: ignore[union-attr]
        print(
            f"{result['run_name']:<28s} {split:<6s} "
            f"{report['accuracy'] * 100:8.2f}% {report['macro_f1'] * 100:8.2f}%"
        )
        baseline_split = (result.get("baseline") or {}).get("reports", {}).get(split)
        if baseline_split:
            print(
                f"{'  baseline (MFCC mean+std LR)':<28s} {split:<6s} "
                f"{baseline_split['accuracy'] * 100:8.2f}% "
                f"{baseline_split['macro_f1'] * 100:8.2f}%"
            )

    latency = result.get("latency")
    if latency:
        print(
            f"\ninference latency (single file, CPU): median "
            f"{latency['median_ms']:.1f} ms   p95 {latency['p95_ms']:.1f} ms"
        )
    print(f"\nfull record: {run_dir / 'metrics.json'}")

    if args.figures:
        from ml.utils.plotting import FIGURES_DIR, generate_run_figures

        # Performance figures follow the last split requested, which is the one
        # the README quotes.
        written = generate_run_figures(run_dir, result, split=splits[-1])
        print(f"\n{len(written)} figure(s) in {FIGURES_DIR}:")
        for path in written:
            print(f"  {path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
