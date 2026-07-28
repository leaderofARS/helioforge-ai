"""
scripts/train.py — HelioForge AI Baseline TCN Training Script
=============================================================

All default paths are resolved from configs/data_paths.yaml via PATH_CFG —
the single source of truth for this project. On EC2:

    dataset root    : /opt/helioforge-ai
    windows         : /opt/helioforge-ai/data/windows/
    train tensor    : /opt/helioforge-ai/data/windows/train_feat32_w512.pt
    val tensor      : /opt/helioforge-ai/data/windows/val_feat32_w512.pt
    scaler          : /opt/helioforge-ai/data/windows/scaler_f32_w512.json
    runs (output)   : /opt/helioforge-ai/experiments/baseline_tcn/runs/
    checkpoints     : /opt/helioforge-ai/experiments/baseline_tcn/checkpoints/
    training log    : /opt/helioforge-ai/logs/training.log

Data format:
    The .pt files contain ONLY {"sequences": Tensor(N, F, L)}.
    There are NO stored labels. Labels are derived on load by reversing the
    MinMax normalisation on channel 0 (soft_mean) using the persisted scaler
    JSON, then applying physical SoLEXS COUNTS/sec thresholds:
        0 — Quiet     (<100 counts/s)
        1 — B-class   (100–500 counts/s)
        2 — C-class   (500–2,000 counts/s)
        3 — M-class   (2,000–8,000 counts/s)
        4 — X-class   (>=8,000 counts/s)

Usage (EC2 — zero required args, all paths from config):
    python scripts/train.py

Override any default:
    python scripts/train.py \\
        --run-name baseline_v1 \\
        --n-epochs 80 \\
        --batch-size 32 \\
        --norm-type batch

Options:
    --data-dir      Windows directory        [default: PATH_CFG.windows.root]
    --train-file    Train tensor filename    [default: train_feat32_w512.pt]
    --val-file      Val tensor filename      [default: val_feat32_w512.pt]
    --scaler-file   Scaler JSON filename     [default: scaler_f32_w512.json]
    --output-dir    Run output base dir      [default: PATH_CFG.experiments.baseline_tcn.runs]
    --log-dir       Global log directory     [default: PATH_CFG.logs.root]
    --run-name      Experiment name          [default: auto timestamp]
    --n-epochs      Training epochs          [default: 80]
    --batch-size    Batch size               [default: 32]
    --lr            AdamW learning rate      [default: 1e-3]
    --weight-decay  AdamW weight decay       [default: 1e-4]
    --dropout       Encoder dropout          [default: 0.2]
    --norm-type     batch | layer | none     [default: batch]
    --patience      Early stopping patience  [default: 15]
    --grad-clip     Gradient clip max-norm   [default: 1.0]
    --label-smooth  Label smoothing epsilon  [default: 0.0]
    --n-classes     Number of output classes [default: 5]
    --in-channels   Input feature channels   [default: 32]
    --seed          Random seed              [default: 42]
    --no-cuda       Disable GPU
    --num-workers   DataLoader workers       [default: 4]
"""

import sys
import json
import argparse
import logging
import time
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Project root on sys.path ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Config — single source of truth for all EC2 paths ─────────────────────────
from src.utils.config import PATH_CFG

from src.HPINA.models.baseline_tcn import (
    HelioForgeTCN,
    build_weighted_criterion,
    evaluate,
    confusion_matrix_str,
    format_metrics_table,
    CLASS_NAMES,
)


# ── Physical SoLEXS flare-class thresholds (COUNTS/sec) ───────────────────────
# Must match verify_pre_training.py and datasets.py exactly.
THRESHOLDS = [100, 500, 2_000, 8_000]


# ── Label derivation from raw tensor ──────────────────────────────────────────

def derive_labels(
    sequences: torch.Tensor,
    scaler: dict,
    soft_channel: int = 0,
) -> torch.Tensor:
    """
    Derive integer flare-class labels (0–4) from the normalised window tensor.

    The .pt files on EC2 store ONLY {"sequences": Tensor(N, F, L)}.
    Labels are computed here by:
      1. Reversing MinMax normalisation on channel 0 (soft_mean) via scaler bounds.
      2. Taking per-window peak physical COUNTS/sec across L=512 timesteps.
      3. Binning with physical SoLEXS thresholds [100, 500, 2000, 8000].

    Parameters
    ----------
    sequences    : Tensor shape (N, F, L)
    scaler       : dict loaded from scaler_f32_w512.json
    soft_channel : index of soft_mean feature in the F dimension (default 0)

    Returns
    -------
    torch.Tensor  long tensor of shape (N,), values 0–4
    """
    soft_min = scaler["min"][soft_channel]
    soft_max = scaler["max"][soft_channel]

    # Peak normalised value per window  →  shape (N,)
    soft_norm_peak = sequences[:, soft_channel, :].max(dim=1).values.numpy()

    # Reverse MinMax normalisation → physical COUNTS/sec
    raw_peak = soft_norm_peak * (soft_max - soft_min) + soft_min

    labels = np.zeros(len(raw_peak), dtype=np.int64)
    labels[(raw_peak >= THRESHOLDS[0]) & (raw_peak < THRESHOLDS[1])] = 1
    labels[(raw_peak >= THRESHOLDS[1]) & (raw_peak < THRESHOLDS[2])] = 2
    labels[(raw_peak >= THRESHOLDS[2]) & (raw_peak < THRESHOLDS[3])] = 3
    labels[raw_peak >= THRESHOLDS[3]] = 4

    return torch.from_numpy(labels)


# ── Split loader ───────────────────────────────────────────────────────────────

def load_split(data_dir: Path, filename: str, scaler: dict) -> TensorDataset:
    """
    Load a window split .pt file and derive its labels.

    Expected EC2 location: /opt/helioforge-ai/data/windows/<filename>
    File format          : {"sequences": Tensor(N, F, L)}

    Parameters
    ----------
    data_dir : Path   directory containing the .pt file
    filename : str    e.g. "train_feat32_w512.pt"
    scaler   : dict   loaded from scaler_f32_w512.json

    Returns
    -------
    TensorDataset(X float32 (N,F,L),  y long (N,))
    """
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"\n  Tensor file not found: {path}"
            f"\n  Expected in          : {data_dir}"
            f"\n  Check --data-dir points to the windows directory on EC2."
        )

    obj = torch.load(str(path), map_location="cpu", weights_only=False)

    if isinstance(obj, dict):
        X = obj.get("sequences", obj.get("X"))
        if X is None:
            raise ValueError(
                f"Cannot find 'sequences' key in {path}. "
                f"Keys present: {list(obj.keys())}"
            )
        X = X.float()
    elif isinstance(obj, torch.Tensor):
        X = obj.float()
    else:
        raise ValueError(f"Unexpected object type {type(obj)} in {path}.")

    y = derive_labels(X, scaler)
    return TensorDataset(X, y)


# ── Logging ────────────────────────────────────────────────────────────────────

def setup_logging(run_log_file: Path, global_log_file: Path) -> logging.Logger:
    logger = logging.getLogger("helioforge.train")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Per-run log   (run_dir/train.log)
    # Global log    (/opt/helioforge-ai/logs/training.log)
    for log_path in (run_log_file, global_log_file):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HelioForge Baseline TCN Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data — all paths default from PATH_CFG ─────────────────────────────────
    parser.add_argument(
        "--data-dir", type=str,
        default=str(PATH_CFG.windows.root),          # /opt/helioforge-ai/data/windows
        help="Directory containing .pt tensors and scaler JSON.",
    )
    parser.add_argument("--train-file",  type=str, default="train_feat32_w512.pt")
    parser.add_argument("--val-file",    type=str, default="val_feat32_w512.pt")
    parser.add_argument("--scaler-file", type=str, default="scaler_f32_w512.json")

    # ── Output — from PATH_CFG ─────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PATH_CFG.experiments.baseline_tcn.runs),  # /opt/helioforge-ai/experiments/baseline_tcn/runs
        help="Base directory for per-run output (checkpoints, logs, history.json).",
    )
    parser.add_argument(
        "--log-dir", type=str,
        default=str(PATH_CFG.logs.root),             # /opt/helioforge-ai/logs
        help="Directory for the global training.log.",
    )
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name. Auto-generated from timestamp if omitted.")

    # ── Training hyperparams ───────────────────────────────────────────────────
    parser.add_argument("--n-epochs",     type=int,   default=80)
    parser.add_argument("--batch-size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout",      type=float, default=0.2)
    parser.add_argument("--norm-type",    type=str,   default="batch",
                        choices=["batch", "layer", "none"])
    parser.add_argument("--patience",     type=int,   default=15)
    parser.add_argument("--grad-clip",    type=float, default=1.0)
    parser.add_argument("--label-smooth", type=float, default=0.0)

    # ── Model ──────────────────────────────────────────────────────────────────
    parser.add_argument("--n-classes",   type=int, default=5)
    parser.add_argument("--in-channels", type=int, default=32)

    # ── Misc ───────────────────────────────────────────────────────────────────
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--no-cuda",     action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)

    return parser.parse_args()


# ── Checkpoint helper ──────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device   = torch.device("cuda" if use_cuda else "cpu")

    # ── Run directory ──────────────────────────────────────────────────────────
    # EC2: /opt/helioforge-ai/experiments/baseline_tcn/runs/<run_name>/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = args.run_name or f"tcn_{args.norm_type}_lr{args.lr}_{timestamp}"
    run_dir   = Path(args.output_dir) / run_name
    ckpt_dir  = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    run_log    = run_dir / "train.log"
    global_log = Path(args.log_dir) / "training.log"
    log = setup_logging(run_log, global_log)

    log.info("=" * 70)
    log.info("  HELIO-FORGE AI  |  Baseline TCN Training")
    log.info("=" * 70)
    log.info(f"  Run name     : {run_name}")
    log.info(f"  Device       : {device}")
    log.info(f"  Data dir     : {args.data_dir}")
    log.info(f"  Train file   : {args.train_file}")
    log.info(f"  Val file     : {args.val_file}")
    log.info(f"  Scaler file  : {args.scaler_file}")
    log.info(f"  Output dir   : {run_dir}")
    log.info(f"  Global log   : {global_log}")
    log.info(f"  Epochs/Batch : {args.n_epochs} / {args.batch_size}")
    log.info(f"  LR / WD      : {args.lr} / {args.weight_decay}")
    log.info(f"  Norm / Drop  : {args.norm_type} / {args.dropout}")
    log.info(f"  Patience     : {args.patience}  |  Grad clip: {args.grad_clip}")
    log.info("=" * 70)

    # Save full config to run directory
    config = vars(args)
    config.update({"run_name": run_name, "device": str(device)})
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Load scaler (required for label derivation) ────────────────────────────
    data_dir    = Path(args.data_dir)
    scaler_path = data_dir / args.scaler_file
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"\n  Scaler not found : {scaler_path}"
            f"\n  Expected in      : {data_dir}"
            f"\n  Run build_windows.py first to generate scaler_f32_w512.json."
        )
    with open(scaler_path) as f:
        scaler = json.load(f)
    log.info(f"Scaler loaded  : {scaler_path}  ({scaler['n_features']} features, w={scaler['window_size']})")

    # ── Load splits ────────────────────────────────────────────────────────────
    log.info(f"Loading train  : {data_dir / args.train_file}")
    train_ds = load_split(data_dir, args.train_file, scaler)
    log.info(f"  Train windows: {len(train_ds)}")

    log.info(f"Loading val    : {data_dir / args.val_file}")
    val_ds = load_split(data_dir, args.val_file, scaler)
    log.info(f"  Val windows  : {len(val_ds)}")

    train_labels = train_ds.tensors[1]
    dist = {CLASS_NAMES[i]: (train_labels == i).sum().item() for i in range(args.n_classes)}
    log.info(f"  Label dist   : {dist}")

    # ── DataLoaders ────────────────────────────────────────────────────────────
    pin_memory   = use_cuda
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = HelioForgeTCN(
        in_channels=args.in_channels,
        n_classes=args.n_classes,
        dropout=args.dropout,
        norm_type=args.norm_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Parameters     : {n_params:,}")

    # ── Loss, optimiser, scheduler ─────────────────────────────────────────────
    criterion = build_weighted_criterion(
        train_labels=train_labels,
        n_classes=args.n_classes,
        device=device,
        label_smoothing=args.label_smooth,
    )
    log.info(f"Class weights  : {[round(w, 4) for w in criterion.weight.tolist()]}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.patience // 2, factor=0.5, verbose=False,
    )

    # ── Training state ─────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    best_macro_f1  = 0.0
    patience_count = 0
    history        = []

    # ── Epoch loop ─────────────────────────────────────────────────────────────
    for epoch in range(1, args.n_epochs + 1):
        t0 = time.time()

        # Train phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Val phase
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device,
            n_classes=args.n_classes, class_names=CLASS_NAMES[:args.n_classes],
        )
        scheduler.step(val_loss)

        lr_now  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        log.info(
            f"Epoch {epoch:3d}/{args.n_epochs}  "
            f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
            f"macro_f1: {val_metrics['macro_f1']:.4f}  "
            f"acc: {val_metrics['accuracy']:.4f}  "
            f"lr: {lr_now:.2e}  [{elapsed:.1f}s]"
        )

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "lr": lr_now, **val_metrics,
        })

        # Checkpoint: best val loss
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            save_checkpoint(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(),
                 "val_loss": val_loss, "val_metrics": val_metrics, "args": vars(args)},
                ckpt_dir / "best_val_loss.pt",
            )
            log.info(f"  ✓ best_val_loss.pt  (val_loss={best_val_loss:.4f})")

        # Checkpoint: best macro F1
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            save_checkpoint(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(),
                 "val_loss": val_loss, "val_metrics": val_metrics, "args": vars(args)},
                ckpt_dir / "best_macro_f1.pt",
            )
            log.info(f"  ✓ best_macro_f1.pt  (macro_f1={best_macro_f1:.4f})")
        else:
            patience_count += 1

        if patience_count >= args.patience:
            log.info(f"Early stopping after {args.patience} epochs without improvement.")
            break

    # ── Final checkpoint & history ─────────────────────────────────────────────
    save_checkpoint(
        {"epoch": epoch, "model_state": model.state_dict(),
         "val_loss": val_loss, "val_metrics": val_metrics, "args": vars(args)},
        ckpt_dir / "final.pt",
    )
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Final summary ──────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("  TRAINING COMPLETE")
    log.info(f"  Best val_loss : {best_val_loss:.4f}")
    log.info(f"  Best macro_F1 : {best_macro_f1:.4f}")
    log.info("")
    log.info("  Final Val Metrics:")
    log.info(format_metrics_table(val_metrics, class_names=CLASS_NAMES[:args.n_classes]))
    log.info("")

    # Re-collect predictions for final confusion matrix
    all_preds, all_labels_list = [], []
    model.eval()
    with torch.no_grad():
        for X_b, y_b in val_loader:
            all_preds.extend(model(X_b.to(device)).argmax(dim=1).cpu().tolist())
            all_labels_list.extend(y_b.tolist())
    log.info("  Confusion Matrix (val, final epoch):")
    log.info(confusion_matrix_str(
        all_preds, all_labels_list,
        n_classes=args.n_classes,
        class_names=CLASS_NAMES[:args.n_classes],
    ))
    log.info("=" * 70)
    log.info(f"  Checkpoints → {ckpt_dir}")
    log.info(f"  Run log     → {run_log}")
    log.info(f"  Global log  → {global_log}")
    log.info(f"  History     → {run_dir / 'history.json'}")


if __name__ == "__main__":
    main()
