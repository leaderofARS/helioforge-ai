"""
scripts/train.py — HelioForge AI Baseline TCN Training Script
=============================================================

Usage (EC2):
    python scripts/train.py [OPTIONS]

Options:
    --data-dir      Path to windows data directory          [default: /opt/helioforge-ai/data/windows]
    --train-file    Train tensor filename                   [default: train_feat32_w512.pt]
    --val-file      Val tensor filename                     [default: val_feat32_w512.pt]
    --output-dir    Checkpoint and log output directory     [default: experiments/baseline_tcn/runs]
    --run-name      Experiment run name                     [default: auto-generated with timestamp]
    --n-epochs      Number of training epochs               [default: 80]
    --batch-size    Training batch size                     [default: 32]
    --lr            Initial learning rate for AdamW         [default: 1e-3]
    --weight-decay  AdamW weight decay                      [default: 1e-4]
    --dropout       Dropout probability                     [default: 0.2]
    --norm-type     Normalization type: batch|layer|none    [default: batch]
    --patience      Early stopping patience (epochs)        [default: 15]
    --grad-clip     Gradient clipping max-norm              [default: 1.0]
    --label-smooth  Label smoothing epsilon                 [default: 0.0]
    --n-classes     Number of output classes                [default: 5]
    --seed          Random seed                             [default: 42]
    --no-cuda       Disable GPU even if available

Example:
    python scripts/train.py \\
        --data-dir /opt/helioforge-ai/data/windows \\
        --output-dir experiments/baseline_tcn/runs \\
        --run-name baseline_batchnorm_lr1e3 \\
        --n-epochs 80 \\
        --batch-size 32
"""

import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to sys.path so we can import src.HPINA
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.HPINA.models.baseline_tcn import (
    HelioForgeTCN,
    build_weighted_criterion,
    evaluate,
    confusion_matrix_str,
    format_metrics_table,
    CLASS_NAMES,
)


# ── Logging setup ──────────────────────────────────────────────────────────────

def setup_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("helioforge.train")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HelioForge Baseline TCN Training")

    # Data
    parser.add_argument("--data-dir",    type=str, default="/opt/helioforge-ai/data/windows")
    parser.add_argument("--train-file",  type=str, default="train_feat32_w512.pt")
    parser.add_argument("--val-file",    type=str, default="val_feat32_w512.pt")
    parser.add_argument("--label-offset",type=int, default=0,
                        help="If labels are 1-indexed, set to -1 to shift to 0-indexed.")

    # Output
    parser.add_argument("--output-dir",  type=str, default="experiments/baseline_tcn/runs")
    parser.add_argument("--run-name",    type=str, default=None,
                        help="Experiment run name. Auto-generated if not set.")

    # Training hyperparams
    parser.add_argument("--n-epochs",    type=int,   default=80)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--weight-decay",type=float, default=1e-4)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--norm-type",   type=str,   default="batch",
                        choices=["batch", "layer", "none"])
    parser.add_argument("--patience",    type=int,   default=15,
                        help="Early stopping patience in epochs.")
    parser.add_argument("--grad-clip",   type=float, default=1.0)
    parser.add_argument("--label-smooth",type=float, default=0.0)

    # Model
    parser.add_argument("--n-classes",   type=int,   default=5)
    parser.add_argument("--in-channels", type=int,   default=32)

    # Misc
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--no-cuda",     action="store_true")
    parser.add_argument("--num-workers", type=int,   default=4)

    return parser.parse_args()


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_split(data_dir: Path, filename: str, label_offset: int = 0) -> TensorDataset:
    """
    Load a (X, y) tensor pair from a single .pt file.

    Expects the file to contain either:
      - A dict with keys "X" and "y" (preferred), or
      - A tuple (X, y)

    X shape: (N, F, L) — e.g. (1874, 32, 512)
    y shape: (N,)      — integer class labels

    Parameters
    ----------
    data_dir : Path
        Directory containing the .pt tensor file.
    filename : str
        Filename of the .pt file.
    label_offset : int
        Offset to add to labels (e.g. -1 to shift 1-indexed → 0-indexed).

    Returns
    -------
    TensorDataset
        PyTorch dataset of (X, y) pairs.
    """
    path = data_dir / filename
    obj  = torch.load(str(path), map_location="cpu", weights_only=False)

    if isinstance(obj, dict):
        X = obj["X"].float()
        y = obj["y"].long()
    elif isinstance(obj, (tuple, list)) and len(obj) == 2:
        X = obj[0].float()
        y = obj[1].long()
    else:
        raise ValueError(
            f"Unexpected tensor format in {path}. "
            f"Expected dict with 'X'/'y' keys or a (X, y) tuple."
        )

    if label_offset != 0:
        y = y + label_offset

    return TensorDataset(X, y)


# ── Checkpoint utilities ───────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: Path) -> None:
    torch.save(state, str(path))


# ── Main training loop ─────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Reproducibility ──────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ── Device ───────────────────────────────────────────────────────────────
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device   = torch.device("cuda" if use_cuda else "cpu")

    # ── Run directory ─────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = args.run_name or f"tcn_{args.norm_type}_lr{args.lr}_{timestamp}"
    run_dir   = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir  = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # ── Logging ───────────────────────────────────────────────────────────────
    log = setup_logging(run_dir / "train.log")
    log.info("=" * 70)
    log.info("  HELIO-FORGE AI  |  Baseline TCN Training")
    log.info("=" * 70)
    log.info(f"  Run          : {run_name}")
    log.info(f"  Device       : {device}")
    log.info(f"  Data dir     : {args.data_dir}")
    log.info(f"  Epochs       : {args.n_epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    log.info(f"  Norm         : {args.norm_type}  |  Dropout: {args.dropout}")
    log.info(f"  Patience     : {args.patience}  |  Grad clip: {args.grad_clip}")
    log.info("=" * 70)

    # Save config
    config = vars(args)
    config["run_name"] = run_name
    config["device"]   = str(device)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Load data ─────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    log.info(f"Loading train split: {args.train_file}")
    train_ds = load_split(data_dir, args.train_file, label_offset=args.label_offset)
    log.info(f"  Train: {len(train_ds)} windows")

    log.info(f"Loading val split  : {args.val_file}")
    val_ds = load_split(data_dir, args.val_file, label_offset=args.label_offset)
    log.info(f"  Val  : {len(val_ds)} windows")

    # Extract all training labels for class weight computation
    train_labels = train_ds.tensors[1]
    log.info(f"  Label distribution: { {i: (train_labels == i).sum().item() for i in range(args.n_classes)} }")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    pin_memory = use_cuda
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = HelioForgeTCN(
        in_channels=args.in_channels,
        n_classes=args.n_classes,
        dropout=args.dropout,
        norm_type=args.norm_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters : {n_params:,}")

    # ── Loss, optimizer, scheduler ────────────────────────────────────────────
    criterion = build_weighted_criterion(
        train_labels=train_labels,
        n_classes=args.n_classes,
        device=device,
        label_smoothing=args.label_smooth,
    )
    log.info(f"Class weights    : {criterion.weight.tolist()}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.patience // 2, factor=0.5, verbose=False
    )

    # ── Training state ────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    best_macro_f1  = 0.0
    patience_count = 0
    history        = []

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, args.n_epochs + 1):
        epoch_start = time.time()

        # ── Train phase ───────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Val phase ─────────────────────────────────────────────────────────
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device,
            n_classes=args.n_classes, class_names=CLASS_NAMES[:args.n_classes]
        )
        scheduler.step(val_loss)

        elapsed = time.time() - epoch_start
        lr_now  = optimizer.param_groups[0]["lr"]

        log.info(
            f"Epoch {epoch:3d}/{args.n_epochs}  "
            f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
            f"macro_f1: {val_metrics['macro_f1']:.4f}  "
            f"acc: {val_metrics['accuracy']:.4f}  "
            f"lr: {lr_now:.2e}  [{elapsed:.1f}s]"
        )

        # ── Record history ────────────────────────────────────────────────────
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr_now}
        row.update(val_metrics)
        history.append(row)

        # ── Checkpoint best val loss ──────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            save_checkpoint(
                {
                    "epoch"     : epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss"  : val_loss,
                    "val_metrics": val_metrics,
                    "args"      : vars(args),
                },
                ckpt_dir / "best_val_loss.pt"
            )
            log.info(f"  ✓ Checkpoint saved  (best val_loss = {best_val_loss:.4f})")

        # ── Checkpoint best macro F1 ──────────────────────────────────────────
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            save_checkpoint(
                {
                    "epoch"     : epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss"  : val_loss,
                    "val_metrics": val_metrics,
                    "args"      : vars(args),
                },
                ckpt_dir / "best_macro_f1.pt"
            )
            log.info(f"  ✓ Checkpoint saved  (best macro_f1 = {best_macro_f1:.4f})")

        else:
            patience_count += 1

        # ── Early stopping ────────────────────────────────────────────────────
        if patience_count >= args.patience:
            log.info(f"Early stopping triggered after {args.patience} epochs without improvement.")
            break

    # ── Save final model ──────────────────────────────────────────────────────
    save_checkpoint(
        {
            "epoch"      : epoch,
            "model_state": model.state_dict(),
            "val_loss"   : val_loss,
            "val_metrics": val_metrics,
            "args"       : vars(args),
        },
        ckpt_dir / "final.pt"
    )

    # ── Save training history ─────────────────────────────────────────────────
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Final evaluation summary ──────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("  TRAINING COMPLETE")
    log.info(f"  Best val_loss  : {best_val_loss:.4f}")
    log.info(f"  Best macro_F1  : {best_macro_f1:.4f}")
    log.info("")
    log.info("  Final Epoch Val Metrics:")
    log.info(format_metrics_table(val_metrics, class_names=CLASS_NAMES[:args.n_classes]))
    log.info("")
    log.info("  Confusion Matrix (val, final epoch):")
    # Re-collect predictions for confusion matrix
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = model(X_batch.to(device)).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(y_batch.tolist())
    log.info(confusion_matrix_str(all_preds, all_labels, n_classes=args.n_classes,
                                   class_names=CLASS_NAMES[:args.n_classes]))
    log.info("=" * 70)
    log.info(f"  Checkpoints saved to : {ckpt_dir}")
    log.info(f"  Training log saved to: {run_dir / 'train.log'}")
    log.info(f"  History saved to     : {run_dir / 'history.json'}")


if __name__ == "__main__":
    main()
