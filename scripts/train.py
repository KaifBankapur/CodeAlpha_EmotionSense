"""
Train the speech-emotion classifier.

This is the single training entry point. It writes a self-contained run
directory that the inference service can load without knowing anything about
this script:

    artifacts/models/<run_name>/
        best_model.pt        weights + architecture + audio/feature config
        config.json          the full ExperimentConfig, human readable
        history.csv          per-epoch metrics
        training_summary.json timing, parameter count, best epoch
        split_manifest.json  exactly which actor went where

Usage
-----
    python scripts/train.py
    python scripts/train.py --name cnn_blstm_ablation --no-attention
    python scripts/train.py --epochs 40 --lr 5e-4 --n-mfcc 20
    python scripts/train.py --test-actors 1 2 3 4 --val-actors 5 6 7 8

Model selection uses validation macro-F1 only. The test split is not evaluated
here; run scripts/evaluate.py after selection is frozen.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.config import (
    MODELS_DIR,
    AudioConfig,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    SplitConfig,
    TrainConfig,
    set_active_model,
)
from ml.data.dataset import AugmentConfig
from ml.data.loading import PROCESSED_DIR, load_prepared
from ml.training.trainer import train
from ml.utils.logging_utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ---- identity -------------------------------------------------------
    parser.add_argument("--name", default=None, help="run name (default: auto)")
    parser.add_argument("--notes", default="", help="free-text note stored in the run config")

    # ---- audio / features ----------------------------------------------
    parser.add_argument("--target-duration", type=float, default=None)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--n-mfcc", type=int, default=None)
    parser.add_argument("--n-mels", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=None)
    parser.add_argument("--no-delta", action="store_true")
    parser.add_argument("--no-delta2", action="store_true")
    parser.add_argument("--no-cmvn", action="store_true")

    # ---- model ----------------------------------------------------------
    parser.add_argument("--channels", type=int, nargs="+", default=None)
    parser.add_argument("--lstm-hidden", type=int, default=None)
    parser.add_argument("--lstm-layers", type=int, default=None)
    parser.add_argument("--no-bidirectional", action="store_true")
    parser.add_argument(
        "--no-attention", action="store_true", help="use masked mean pooling instead of attention"
    )
    parser.add_argument("--conv-dropout", type=float, default=None)
    parser.add_argument("--head-dropout", type=float, default=None)

    # ---- optimisation ---------------------------------------------------
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--num-threads", type=int, default=None)

    # ---- data -----------------------------------------------------------
    parser.add_argument("--test-actors", type=int, nargs="+", default=None)
    parser.add_argument("--val-actors", type=int, nargs="+", default=None)
    parser.add_argument(
        "--no-augment", action="store_true", help="disable SpecAugment time/frequency masking"
    )

    # ---- post-training --------------------------------------------------
    parser.add_argument(
        "--set-active", action="store_true", help="make this run the one the API serves"
    )
    return parser


def apply_overrides(args: argparse.Namespace) -> ExperimentConfig:
    """Build the experiment config from defaults + CLI overrides."""
    audio = AudioConfig()
    features = FeatureConfig()
    model = ModelConfig()
    train_cfg = TrainConfig()
    split = SplitConfig()

    if args.target_duration is not None:
        audio.target_duration_sec = args.target_duration
    if args.sample_rate is not None:
        audio.sample_rate = args.sample_rate

    if args.n_mfcc is not None:
        features.n_mfcc = args.n_mfcc
    if args.n_mels is not None:
        features.n_mels = args.n_mels
    if args.hop_length is not None:
        features.hop_length = args.hop_length
    if args.no_delta:
        features.use_delta = False
    if args.no_delta2:
        features.use_delta2 = False
    if args.no_cmvn:
        features.cmvn = False

    if args.channels is not None:
        model.channels = list(args.channels)
    if args.lstm_hidden is not None:
        model.lstm_hidden = args.lstm_hidden
    if args.lstm_layers is not None:
        model.lstm_layers = args.lstm_layers
    if args.no_bidirectional:
        model.bidirectional = False
    if args.no_attention:
        model.use_attention_pooling = False
    if args.conv_dropout is not None:
        model.conv_dropout = args.conv_dropout
    if args.head_dropout is not None:
        model.head_dropout = args.head_dropout

    if args.epochs is not None:
        train_cfg.max_epochs = args.epochs
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.lr is not None:
        train_cfg.learning_rate = args.lr
    if args.weight_decay is not None:
        train_cfg.weight_decay = args.weight_decay
    if args.label_smoothing is not None:
        train_cfg.label_smoothing = args.label_smoothing
    if args.grad_clip is not None:
        train_cfg.grad_clip_norm = args.grad_clip
    if args.early_stopping_patience is not None:
        train_cfg.early_stopping_patience = args.early_stopping_patience
    if args.seed is not None:
        train_cfg.seed = args.seed
    if args.num_threads is not None:
        train_cfg.num_threads = args.num_threads

    if args.test_actors:
        split.test_actors = list(args.test_actors)
    if args.val_actors:
        split.val_actors = list(args.val_actors)

    return ExperimentConfig(
        audio=audio,
        features=features,
        model=model,
        train=train_cfg,
        split=split,
        notes=args.notes,
    )


def auto_run_name(experiment: ExperimentConfig) -> str:
    feature = experiment.features
    planes = "mfcc"
    if feature.use_delta:
        planes += "_d"
    if feature.use_delta2:
        planes += "_d2"
    planes += "_cmvn" if feature.cmvn else "_nocmvn"
    pooling = "attn" if experiment.model.use_attention_pooling else "mean"
    return (
        f"cnnblstm_{planes}_{feature.n_mfcc}mfcc_{pooling}_"
        f"{experiment.audio.target_duration_sec:g}s_"
        # Local time, so run names sort the way a human reading them expects.
        f"{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}"
    )


def main() -> int:
    args = build_parser().parse_args()
    setup_logging()

    experiment = apply_overrides(args)
    run_name = args.name or auto_run_name(experiment)
    run_dir = MODELS_DIR / run_name

    print("=" * 68)
    print(f"RUN: {run_name}")
    print("=" * 68)
    print(
        f"  window       : {experiment.audio.target_duration_sec}s @ "
        f"{experiment.audio.sample_rate} Hz"
    )
    print(f"  features     : {experiment.features.describe()}, cmvn={experiment.features.cmvn}")
    print(f"  conv channels: {experiment.model.channels}")
    print(
        f"  lstm         : hidden={experiment.model.lstm_hidden}, "
        f"layers={experiment.model.lstm_layers}, "
        f"bidirectional={experiment.model.bidirectional}"
    )
    print(
        f"  pooling      : "
        f"{'attention' if experiment.model.use_attention_pooling else 'masked-mean'}"
    )
    print(
        f"  optimiser    : AdamW lr={experiment.train.learning_rate} "
        f"wd={experiment.train.weight_decay}"
    )
    print(
        f"  monitor      : {experiment.train.monitor} "
        f"(patience {experiment.train.early_stopping_patience})"
    )
    print()

    prepared = load_prepared(
        audio_cfg=experiment.audio,
        feature_cfg=experiment.features,
        split_cfg=experiment.split,
        batch_size=experiment.train.batch_size,
        seed=experiment.train.seed,
        augment=not args.no_augment,
        augment_cfg=AugmentConfig(enabled=not args.no_augment),
    )

    summary = train(
        prepared.cache,
        prepared.datasets,
        prepared.loaders,
        experiment,
        run_dir,
        augment_cfg=prepared.augment_cfg,
        device_str=args.device,
    )

    experiment.save(run_dir / "config.json")

    split_manifest = PROCESSED_DIR / "split_manifest.json"
    if split_manifest.is_file():
        shutil.copy2(split_manifest, run_dir / "split_manifest.json")

    if args.set_active:
        set_active_model(run_name)
        print(f"\nactive model set to: {run_name}")

    print("\n" + "=" * 68)
    print("TRAINING COMPLETE")
    print("=" * 68)
    print(f"  run directory      : {run_dir}")
    print(f"  best epoch         : {summary['best_epoch']} / {summary['epochs_run']} run")
    print(f"  best val macro-F1  : {summary['best_val_macro_f1'] * 100:.2f}%")
    print(f"  early stopped      : {summary['early_stopped']}")
    print(
        f"  wall clock         : {summary['total_training_seconds']:.1f}s "
        f"({summary['seconds_per_epoch_mean']:.2f}s/epoch)"
    )
    print(f"  parameters         : {summary['n_parameters']:,}")
    print(f"  device             : {summary['device']}")
    print("\nNext:  python scripts/evaluate.py --run " + str(run_name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
