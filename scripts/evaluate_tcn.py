"""
scripts/evaluate_tcn.py — HelioForge AI Model Testing & Evaluation Script
========================================================================

Evaluates a trained Baseline TCN checkpoint (e.g., best_macro_f1.pt)
on the held-out TEST dataset split (test_feat32_w512.pt).

Usage (EC2 — auto-finds the latest run and best_macro_f1.pt checkpoint):
    python scripts/evaluate_tcn.py

Specify a custom checkpoint:
    python scripts/evaluate_tcn.py \
        --checkpoint /opt/helioforge-ai/experiments/baseline_tcn/runs/<run_name>/checkpoints/best_macro_f1.pt

Options:
    --checkpoint  Path to checkpoint .pt file [default: latest run's best_macro_f1.pt]
    --data-dir    Windows dataset directory   [default: PATH_CFG.windows.root]
    --test-file   Test split tensor filename  [default: test_feat32_w512.pt]
    --scaler-file Scaler JSON filename        [default: scaler_f32_w512.json]
    --no-cuda     Disable GPU evaluation
"""

import sys
import json
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import PATH_CFG
from src.HPINA.models.baseline_tcn import (
    HelioForgeTCN,
    evaluate,
    confusion_matrix_str,
    format_metrics_table,
    CLASS_NAMES,
)
from scripts.train import load_split, _fatal, DataError, ScalerError


def find_latest_checkpoint(runs_dir: Path) -> Path:
    """Find the best_macro_f1.pt checkpoint from the most recent run directory."""
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in: {runs_dir}")

    # Sort runs by directory creation/modification time
    run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

    for latest_run in run_dirs:
        ckpt_path = latest_run / "checkpoints" / "best_macro_f1.pt"
        if ckpt_path.exists():
            return ckpt_path

    raise FileNotFoundError(f"No 'best_macro_f1.pt' found in recent runs under: {runs_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HelioForge TCN Test Evaluation Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint .pt file. If omitted, auto-finds latest run's best_macro_f1.pt.",
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(PATH_CFG.windows.root),
        help="Directory containing test tensor and scaler JSON.",
    )
    parser.add_argument("--test-file",   type=str, default="test_feat32_w512.pt")
    parser.add_argument("--scaler-file", type=str, default="scaler_f32_w512.json")
    parser.add_argument("--no-cuda",     action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device   = torch.device("cuda" if use_cuda else "cpu")

    runs_dir = PATH_CFG.experiments.baseline_tcn.runs

    # Resolve checkpoint path
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        print("Searching for latest best_macro_f1.pt checkpoint...")
        ckpt_path = find_latest_checkpoint(runs_dir)

    if not ckpt_path.exists():
        print(f"[FATAL] Checkpoint file not found: {ckpt_path}")
        sys.exit(1)

    print("=" * 70)
    print("  HELIO-FORGE AI  |  Model Evaluation on Held-Out Test Split")
    print("=" * 70)
    print(f"  Checkpoint  : {ckpt_path}")
    print(f"  Device      : {device}")
    print(f"  Data dir    : {args.data_dir}")
    print(f"  Test file   : {args.test_file}")
    print(f"  Scaler file : {args.scaler_file}")
    print("=" * 70)

    # ── Load Checkpoint ────────────────────────────────────────────────────────
    try:
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    except Exception as exc:
        print(f"[FATAL] Failed to load checkpoint {ckpt_path}: {exc}")
        sys.exit(1)

    saved_args = ckpt.get("args", {})
    epoch      = ckpt.get("epoch", "unknown")
    val_loss   = ckpt.get("val_loss", None)
    val_f1     = ckpt.get("val_metrics", {}).get("macro_f1", None)

    print(f"  Checkpoint Epoch  : {epoch}")
    if val_loss is not None:
        print(f"  Checkpoint Val Loss: {val_loss:.4f}")
    if val_f1 is not None:
        print(f"  Checkpoint Val F1  : {val_f1:.4f}")

    # Extract model hyperparams from checkpoint
    in_channels = saved_args.get("in_channels", 32)
    n_classes   = saved_args.get("n_classes", 5)
    dropout     = saved_args.get("dropout", 0.2)
    norm_type   = saved_args.get("norm_type", "batch")
    head_dims   = saved_args.get("head_dims", [256, 128])

    # ── Load Model ────────────────────────────────────────────────────────────
    model = HelioForgeTCN(
        in_channels=in_channels,
        n_classes=n_classes,
        dropout=dropout,
        norm_type=norm_type,
        head_dims=head_dims,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("  ✓ Model architecture and weights restored successfully.")

    # ── Load Scaler & Test Dataset ──────────────────────────────────────────────
    data_dir    = Path(args.data_dir)
    scaler_path = data_dir / args.scaler_file
    if not scaler_path.exists():
        print(f"[FATAL] Scaler file not found: {scaler_path}")
        sys.exit(1)

    with open(scaler_path) as f:
        scaler = json.load(f)

    test_ds = load_split(data_dir, args.test_file, scaler)
    print(f"  Test windows: {len(test_ds):,}")

    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Dummy criterion for loss computation
    test_labels = test_ds.tensors[1]
    criterion   = torch.nn.CrossEntropyLoss()

    # ── Evaluate on Test Set ──────────────────────────────────────────────────
    print("\nRunning evaluation on held-out test split...")
    test_loss, test_metrics = evaluate(
        model, test_loader, criterion, device,
        n_classes=n_classes, class_names=CLASS_NAMES[:n_classes],
    )

    # Collect all predictions for confusion matrix
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            logits = model(X_b.to(device))
            preds  = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(y_b.tolist())

    # ── Print Final Results ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  HELD-OUT TEST SET EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_metrics['accuracy']:.4f}  ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  Test Macro F1 : {test_metrics['macro_f1']:.4f}")
    print(f"  Macro Precision: {test_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall   : {test_metrics['macro_recall']:.4f}")
    print("=" * 70)

    print("\n  Per-Class Performance Table:")
    print(format_metrics_table(test_metrics, class_names=CLASS_NAMES[:n_classes]))

    print("\n  Test Split Confusion Matrix:")
    print(confusion_matrix_str(
        all_preds, all_targets,
        n_classes=n_classes,
        class_names=CLASS_NAMES[:n_classes],
    ))
    print("=" * 70)


if __name__ == "__main__":
    main()
