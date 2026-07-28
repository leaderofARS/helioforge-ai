"""
scripts/train.py — HelioForge AI Baseline TCN Training Script
=============================================================

EC2 RESOURCE CONSTRAINTS:
    RAM  : 8 GB total (OS + Python ~1.5 GB overhead → ~6.5 GB usable)
    CPU  : 2 cores

Memory budget (batch_size=16, 8.4M-param model):
    Model weights          :  ~33 MB
    Gradients              :  ~33 MB
    AdamW moments (×2)     :  ~66 MB
    Activations (8 blocks) : ~132 MB   (16 × 512 × 512 × 4B × 8 blocks)
    Dataset in RAM         : ~150 MB   (train + val tensors)
    Python / OS overhead   : ~800 MB
    ─────────────────────────────────
    Estimated total        : ~1.2 GB   (safe within 8 GB)

Gradient accumulation:
    batch_size=16, accum_steps=2 → effective batch = 32
    Keeps memory low while matching the original training dynamics.

DataLoader workers:
    num_workers=0  — each worker is a full Python process (~300–500 MB each).
    On a 2-core CPU, workers add IPC overhead. Single-threaded loading is faster.

All default paths resolve from configs/data_paths.yaml via PATH_CFG.

Usage (EC2 — zero required args):
    python scripts/train.py

Override any default:
    python scripts/train.py \\
        --run-name baseline_v1 \\
        --batch-size 16 \\
        --accum-steps 2 \\
        --norm-type batch
"""

import sys
import json
import argparse
import logging
import math
import time
import traceback
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Config ─────────────────────────────────────────────────────────────────────
try:
    from src.utils.config import PATH_CFG
except Exception as exc:
    print("\n" + "!" * 70)
    print("  [FATAL] CONFIG LOAD FAILURE")
    print("!" * 70)
    print(f"  Cannot import PATH_CFG from src.utils.config")
    print(f"  Reason  : {exc}")
    print(f"  Check   : configs/data_paths.yaml exists and is valid YAML")
    print(f"  CWD     : {Path.cwd()}")
    print("!" * 70 + "\n")
    sys.exit(1)

try:
    from src.HPINA.models.baseline_tcn import (
        HelioForgeTCN,
        build_weighted_criterion,
        evaluate,
        confusion_matrix_str,
        format_metrics_table,
        CLASS_NAMES,
    )
except Exception as exc:
    print("\n" + "!" * 70)
    print("  [FATAL] MODEL IMPORT FAILURE")
    print("!" * 70)
    print(f"  Cannot import from src.HPINA.models.baseline_tcn")
    print(f"  Reason  : {exc}")
    print(f"  Check   : src/HPINA/models/baseline_tcn/__init__.py exports all symbols")
    print("!" * 70 + "\n")
    sys.exit(1)


# =============================================================================
# Custom exception hierarchy
# =============================================================================

class HelioForgeError(RuntimeError):
    """Base exception for all HelioForge training errors."""

class DataError(HelioForgeError):
    """Missing, corrupt, or mis-shaped data files."""

class ScalerError(HelioForgeError):
    """Scaler JSON missing, malformed, or incompatible."""

class LabelError(HelioForgeError):
    """Label derivation fails or produces a degenerate distribution."""

class ModelError(HelioForgeError):
    """Model fails to initialise or produces invalid outputs."""

class TrainingError(HelioForgeError):
    """Training loop encounters an unrecoverable error (NaN loss, crash)."""

class CheckpointError(HelioForgeError):
    """Checkpoint cannot be written (disk full, permission denied)."""


# =============================================================================
# Constants
# =============================================================================

# Physical SoLEXS flare-class thresholds (COUNTS/sec).
# Must match verify_pre_training.py and datasets.py exactly.
THRESHOLDS = [100, 500, 2_000, 8_000]

_SEP       = "─" * 70
_FATAL_SEP = "!" * 70


# =============================================================================
# Memory utility
# =============================================================================

def _mem_mb() -> float:
    """
    Return current process RSS memory in MB.
    Uses /proc/self/status (Linux EC2) with psutil as fallback.
    Returns -1.0 if neither is available.
    """
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1024  # kB → MB
    except OSError:
        pass
    try:
        import psutil, os
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        pass
    return -1.0


def _total_ram_mb() -> float:
    """Return total system RAM in MB from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return -1.0


# =============================================================================
# Error reporting helpers
# =============================================================================

def _fatal(log: logging.Logger, error_type: str, msg: str, hint: str = "") -> None:
    """
    Print a clearly visible [FATAL] banner to log + console.

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      [FATAL] <ERROR_TYPE>
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      <message lines>
      Hint : <hint>
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    log.error(_FATAL_SEP)
    log.error(f"  [FATAL] {error_type}")
    log.error(_FATAL_SEP)
    for line in msg.strip().splitlines():
        log.error(f"  {line}")
    if hint:
        log.error(f"  Hint    : {hint}")
    log.error(_FATAL_SEP)


def _warn(log: logging.Logger, warning_type: str, msg: str) -> None:
    """Print a clearly visible [WARNING] banner."""
    log.warning(_SEP)
    log.warning(f"  [WARNING] {warning_type}")
    log.warning(_SEP)
    for line in msg.strip().splitlines():
        log.warning(f"  {line}")
    log.warning(_SEP)


# =============================================================================
# Label derivation
# =============================================================================

def derive_labels(
    sequences: torch.Tensor,
    scaler: dict,
    soft_channel: int = 0,
) -> torch.Tensor:
    """
    Derive integer flare-class labels (0–4) from the normalised window tensor.

    The .pt files store ONLY {"sequences": Tensor(N, F, L)} — no labels.
    Labels are computed by:
      1. Reversing MinMax normalisation on channel 0 (soft_mean) via scaler.
      2. Per-window peak physical COUNTS/sec across L timesteps.
      3. Binning with SoLEXS thresholds [100, 500, 2000, 8000].

    Raises
    ------
    ScalerError   — scaler keys missing or zero-range bounds
    DataError     — tensor shape wrong or channel out of range
    LabelError    — all windows collapse to one class
    """
    # Validate scaler
    for key in ("min", "max"):
        if key not in scaler:
            raise ScalerError(
                f"Scaler JSON missing required key: '{key}'.\n"
                f"  Keys present : {list(scaler.keys())}\n"
                f"  Action       : Re-run build_windows.py to regenerate scaler_f32_w512.json."
            )

    if soft_channel >= len(scaler["min"]):
        raise ScalerError(
            f"soft_channel={soft_channel} is out of range for scaler.\n"
            f"  Scaler channels : {len(scaler['min'])}  (indices 0–{len(scaler['min'])-1})\n"
            f"  Action          : Verify the .pt file and scaler were built together."
        )

    soft_min = scaler["min"][soft_channel]
    soft_max = scaler["max"][soft_channel]

    if soft_min == soft_max:
        raise ScalerError(
            f"Scaler channel {soft_channel} (soft_mean) has zero range: min=max={soft_min}.\n"
            f"  The feature had no variation in the training data.\n"
            f"  Action : Inspect the raw SoLEXS data or re-run build_windows.py."
        )

    # Validate tensor
    if sequences.ndim != 3:
        raise DataError(
            f"Expected 3D tensor (N, F, L) but got shape {tuple(sequences.shape)}.\n"
            f"  Action : Ensure the .pt file was generated by build_windows.py."
        )

    if soft_channel >= sequences.shape[1]:
        raise DataError(
            f"soft_channel={soft_channel} exceeds tensor F dimension.\n"
            f"  Tensor shape : {tuple(sequences.shape)}  (N, F={sequences.shape[1]}, L)\n"
            f"  Action       : Check --in-channels matches the dataset feature count."
        )

    # Derive labels
    soft_norm_peak = sequences[:, soft_channel, :].max(dim=1).values.numpy()
    raw_peak       = soft_norm_peak * (soft_max - soft_min) + soft_min

    labels = np.zeros(len(raw_peak), dtype=np.int64)
    labels[(raw_peak >= THRESHOLDS[0]) & (raw_peak < THRESHOLDS[1])] = 1
    labels[(raw_peak >= THRESHOLDS[1]) & (raw_peak < THRESHOLDS[2])] = 2
    labels[(raw_peak >= THRESHOLDS[2]) & (raw_peak < THRESHOLDS[3])] = 3
    labels[raw_peak >= THRESHOLDS[3]] = 4

    unique_classes = np.unique(labels)
    if len(unique_classes) == 1:
        raise LabelError(
            f"Label derivation collapsed all {len(labels)} windows into class {unique_classes[0]}.\n"
            f"  raw_peak range : [{raw_peak.min():.2f}, {raw_peak.max():.2f}] COUNTS/sec\n"
            f"  soft_min/max   : {soft_min:.4f} / {soft_max:.4f}\n"
            f"  Thresholds     : {THRESHOLDS}\n"
            f"  Possible causes:\n"
            f"    - Wrong scaler bounds (re-run build_windows.py)\n"
            f"    - Dataset has only one flare class (inspect raw data)\n"
            f"    - Wrong soft_channel index (expected 0 = soft_mean)"
        )

    return torch.from_numpy(labels)


# =============================================================================
# Split loader
# =============================================================================

def load_split(data_dir: Path, filename: str, scaler: dict) -> TensorDataset:
    """
    Load a window split .pt file and derive its labels.

    File format: {"sequences": Tensor(N, F, L)}
    Labels are NOT stored — derived from channel 0 peak flux via derive_labels().

    Raises
    ------
    DataError    — file missing, corrupt, wrong shape, NaN/Inf
    ScalerError  — propagated from derive_labels()
    LabelError   — propagated from derive_labels()
    """
    path = data_dir / filename

    if not path.exists():
        present = ([f.name for f in data_dir.glob("*.pt")]
                   if data_dir.exists() else "DIR NOT FOUND")
        raise DataError(
            f"Tensor file not found.\n"
            f"  Expected path : {path}\n"
            f"  Data dir      : {data_dir}\n"
            f"  .pt files present : {present}\n"
            f"  Action        : Confirm --data-dir points to the EC2 windows directory.\n"
            f"                  Run build_windows.py if tensors have not been generated yet."
        )

    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise DataError(
            f"Failed to load tensor file.\n"
            f"  Path   : {path}\n"
            f"  Reason : {exc}\n"
            f"  Action : File may be corrupted. Re-run build_windows.py to regenerate."
        ) from exc

    if isinstance(obj, dict):
        X = obj.get("sequences", obj.get("X"))
        if X is None:
            raise DataError(
                f"Dict in {filename} has no 'sequences' key.\n"
                f"  Keys found : {list(obj.keys())}\n"
                f"  Expected   : 'sequences'  (written by build_windows.py)\n"
                f"  Action     : Regenerate tensors with the current build_windows.py."
            )
        X = X.float()
    elif isinstance(obj, torch.Tensor):
        X = obj.float()
    else:
        raise DataError(
            f"Unexpected object type in {filename}.\n"
            f"  Got    : {type(obj).__name__}\n"
            f"  Expect : dict with 'sequences' key  OR  a raw torch.Tensor\n"
            f"  Action : Regenerate tensors with build_windows.py."
        )

    if X.ndim != 3:
        raise DataError(
            f"Tensor in {filename} is not 3-dimensional.\n"
            f"  Shape  : {tuple(X.shape)}  (expected: (N, F, L))\n"
            f"  Action : Regenerate tensors with build_windows.py."
        )

    N, F, L = X.shape
    if N == 0:
        raise DataError(
            f"Tensor in {filename} has zero windows (N=0).\n"
            f"  Shape  : {tuple(X.shape)}\n"
            f"  Action : Stride may be too large. Check build_windows.py settings."
        )

    n_nan = torch.isnan(X).sum().item()
    n_inf = torch.isinf(X).sum().item()
    if n_nan > 0 or n_inf > 0:
        raise DataError(
            f"Tensor {filename} contains invalid values.\n"
            f"  NaN count : {n_nan:,}  ({100*n_nan/X.numel():.2f}% of {X.numel():,} elements)\n"
            f"  Inf count : {n_inf:,}  ({100*n_inf/X.numel():.2f}% of {X.numel():,} elements)\n"
            f"  Shape     : {tuple(X.shape)}\n"
            f"  Action    : Re-run build_windows.py (nan_to_num is applied during generation)."
        )

    y = derive_labels(X, scaler)
    return TensorDataset(X, y)


# =============================================================================
# Logging
# =============================================================================

def setup_logging(run_log_file: Path, global_log_file: Path) -> logging.Logger:
    logger = logging.getLogger("helioforge.train")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    for log_path in (run_log_file, global_log_file):
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_path, mode="a")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except OSError as exc:
            logger.warning(f"Could not create log file {log_path}: {exc}")

    return logger


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HelioForge Baseline TCN Training  [EC2: 8 GB RAM, 2 CPU cores]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data — all paths default from PATH_CFG = configs/data_paths.yaml
    parser.add_argument("--data-dir",    type=str, default=str(PATH_CFG.windows.root),
                        help="Directory with .pt tensors and scaler JSON.")
    parser.add_argument("--train-file",  type=str, default="train_feat32_w512.pt")
    parser.add_argument("--val-file",    type=str, default="val_feat32_w512.pt")
    parser.add_argument("--scaler-file", type=str, default="scaler_f32_w512.json")

    # Output — from PATH_CFG
    parser.add_argument("--output-dir",  type=str,
                        default=str(PATH_CFG.experiments.baseline_tcn.runs),
                        help="Base dir for per-run output.")
    parser.add_argument("--log-dir",     type=str, default=str(PATH_CFG.logs.root),
                        help="Global training.log directory.")
    parser.add_argument("--run-name",    type=str, default=None,
                        help="Run name. Auto-generated from timestamp if omitted.")

    # Training — EC2-safe defaults
    parser.add_argument("--n-epochs",     type=int,   default=80)
    parser.add_argument("--batch-size",   type=int,   default=16,
                        help="Mini-batch size per gradient step. "
                             "EC2 default=16 (low RAM). Use --accum-steps to scale effective batch.")
    parser.add_argument("--accum-steps",  type=int,   default=2,
                        help="Gradient accumulation steps. "
                             "Effective batch = batch_size × accum_steps. "
                             "Default: 2 → effective batch = 32, matching full-size training.")
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout",      type=float, default=0.2)
    parser.add_argument("--norm-type",    type=str,   default="batch",
                        choices=["batch", "layer", "none"])
    parser.add_argument("--patience",     type=int,   default=15)
    parser.add_argument("--grad-clip",    type=float, default=1.0)
    parser.add_argument("--label-smooth", type=float, default=0.0)

    # Model
    parser.add_argument("--n-classes",   type=int, default=5)
    parser.add_argument("--in-channels", type=int, default=32)

    # EC2 resource controls
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker processes. "
                             "EC2 default=0: no subprocesses. "
                             "Each worker uses ~300–500 MB RAM on a 2-core machine.")
    parser.add_argument("--num-threads", type=int, default=2,
                        help="PyTorch intra-op threads (torch.set_num_threads). "
                             "Set to number of physical cores (EC2 default=2).")

    # Misc
    parser.add_argument("--seed",    type=int,  default=42)
    parser.add_argument("--no-cuda", action="store_true",
                        help="Disable GPU. Auto-disabled if no CUDA device found.")

    return parser.parse_args()


# =============================================================================
# Checkpoint helper
# =============================================================================

def save_checkpoint(state: dict, path: Path, log: logging.Logger) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, str(path))
    except OSError as exc:
        raise CheckpointError(
            f"Failed to save checkpoint.\n"
            f"  Target path : {path}\n"
            f"  Reason      : {exc}\n"
            f"  Action      : Check disk space (df -h /opt) and directory permissions (ls -la)."
        ) from exc


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    # ── EC2 resource constraints ───────────────────────────────────────────────
    torch.set_num_threads(args.num_threads)        # match physical cores
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device   = torch.device("cuda" if use_cuda else "cpu")

    # Effective batch size after gradient accumulation
    effective_batch = args.batch_size * args.accum_steps

    # ── Run directory ─────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = args.run_name or f"tcn_{args.norm_type}_lr{args.lr}_{timestamp}"
    run_dir   = Path(args.output_dir) / run_name
    ckpt_dir  = run_dir / "checkpoints"

    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(exist_ok=True)
    except OSError as exc:
        print(f"\n{_FATAL_SEP}")
        print(f"  [FATAL] CANNOT CREATE RUN DIRECTORY")
        print(f"{_FATAL_SEP}")
        print(f"  Target  : {run_dir}")
        print(f"  Reason  : {exc}")
        print(f"  Action  : Check --output-dir is writable.  (chmod 755 /opt/helioforge-ai/experiments)")
        print(f"{_FATAL_SEP}\n")
        sys.exit(1)

    run_log    = run_dir / "train.log"
    global_log = Path(args.log_dir) / "training.log"
    log        = setup_logging(run_log, global_log)

    # ── System resource report ─────────────────────────────────────────────────
    total_ram  = _total_ram_mb()
    start_mem  = _mem_mb()
    ram_str    = f"{total_ram/1024:.1f} GB" if total_ram > 0 else "unknown"
    mem_str    = f"{start_mem:.0f} MB" if start_mem > 0 else "unknown"

    log.info("=" * 70)
    log.info("  HELIO-FORGE AI  |  Baseline TCN Training")
    log.info("=" * 70)
    log.info(f"  Run name       : {run_name}")
    log.info(f"  Device         : {device}  {'(GPU ✓)' if use_cuda else '(CPU — no GPU detected)'}")
    log.info(f"  PyTorch threads: {args.num_threads}  (torch.set_num_threads)")
    log.info(f"  System RAM     : {ram_str}  |  Process at start: {mem_str}")
    log.info("─" * 70)
    log.info(f"  Data dir       : {args.data_dir}")
    log.info(f"  Train file     : {args.train_file}")
    log.info(f"  Val file       : {args.val_file}")
    log.info(f"  Scaler file    : {args.scaler_file}")
    log.info("─" * 70)
    log.info(f"  Output dir     : {run_dir}")
    log.info(f"  Global log     : {global_log}")
    log.info("─" * 70)
    log.info(f"  Batch size     : {args.batch_size}  (micro-batch per step)")
    log.info(f"  Accum steps    : {args.accum_steps}  →  effective batch = {effective_batch}")
    log.info(f"  DataLoader wkr : {args.num_workers}  (0 = no subprocesses, safest on 8 GB RAM)")
    log.info(f"  Epochs         : {args.n_epochs}")
    log.info(f"  LR / WD        : {args.lr} / {args.weight_decay}")
    log.info(f"  Norm / Dropout : {args.norm_type} / {args.dropout}")
    log.info(f"  Patience       : {args.patience}  |  Grad clip: {args.grad_clip}")
    log.info("=" * 70)

    # Warn if total RAM is very tight
    if 0 < total_ram < 6 * 1024:
        _warn(log, "LOW SYSTEM RAM DETECTED",
              f"System RAM: {ram_str}.\n"
              f"  Recommended minimum for this model : 6 GB\n"
              f"  Current settings  : batch_size={args.batch_size}, num_workers={args.num_workers}\n"
              f"  If OOM occurs     : reduce --batch-size to 8 and keep --accum-steps=4")

    # Save config snapshot
    config_snap = vars(args).copy()
    config_snap.update({
        "run_name": run_name,
        "device": str(device),
        "effective_batch_size": effective_batch,
        "num_threads": args.num_threads,
    })
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_snap, f, indent=2)

    # ── Load scaler ───────────────────────────────────────────────────────────
    data_dir    = Path(args.data_dir)
    scaler_path = data_dir / args.scaler_file

    if not data_dir.exists():
        _fatal(log, "DATA DIRECTORY NOT FOUND",
               f"data-dir does not exist on this machine.\n"
               f"  Provided : {data_dir}\n"
               f"  Config   : PATH_CFG.windows.root = {PATH_CFG.windows.root}",
               hint="Confirm this is the correct EC2 instance and the dataset has been transferred.")
        raise DataError(f"Data directory not found: {data_dir}")

    if not scaler_path.exists():
        present = [f.name for f in data_dir.iterdir()] if data_dir.exists() else []
        _fatal(log, "SCALER FILE NOT FOUND",
               f"Cannot derive labels without the scaler JSON.\n"
               f"  Expected  : {scaler_path}\n"
               f"  Dir holds : {present}",
               hint="Run  python scripts/build_windows.py --mode features  to regenerate scaler_f32_w512.json.")
        raise ScalerError(f"Scaler not found: {scaler_path}")

    try:
        with open(scaler_path) as f:
            scaler = json.load(f)
    except json.JSONDecodeError as exc:
        _fatal(log, "SCALER JSON PARSE ERROR",
               f"Scaler file exists but cannot be parsed.\n"
               f"  Path   : {scaler_path}\n"
               f"  Reason : {exc}",
               hint="File may be truncated or corrupted. Delete it and re-run build_windows.py.")
        raise ScalerError(f"Malformed scaler JSON: {scaler_path}") from exc

    for required_key in ("min", "max", "n_features", "window_size"):
        if required_key not in scaler:
            _fatal(log, "SCALER SCHEMA MISMATCH",
                   f"Scaler JSON is missing required field: '{required_key}'\n"
                   f"  Path       : {scaler_path}\n"
                   f"  Keys found : {list(scaler.keys())}",
                   hint="Regenerate the scaler by re-running build_windows.py.")
            raise ScalerError(f"Missing key '{required_key}' in scaler: {scaler_path}")

    if scaler["n_features"] != args.in_channels:
        _warn(log, "SCALER / MODEL CHANNEL MISMATCH",
              f"Scaler was built for F={scaler['n_features']} features "
              f"but --in-channels={args.in_channels}.\n"
              f"  Labels may be incorrect if channel 0 is not soft_mean.\n"
              f"  Scaler path : {scaler_path}")

    log.info(f"Scaler loaded  : {scaler_path}  (F={scaler['n_features']}, w={scaler['window_size']})")

    # ── Load datasets ─────────────────────────────────────────────────────────
    try:
        log.info(f"Loading train  : {data_dir / args.train_file}")
        train_ds = load_split(data_dir, args.train_file, scaler)
        t_mem = _mem_mb()
        log.info(f"  Train windows: {len(train_ds):,}  |  RAM after load: {t_mem:.0f} MB")
    except (DataError, ScalerError, LabelError) as exc:
        _fatal(log, "TRAIN DATA LOAD FAILURE", str(exc))
        raise

    try:
        log.info(f"Loading val    : {data_dir / args.val_file}")
        val_ds = load_split(data_dir, args.val_file, scaler)
        v_mem = _mem_mb()
        log.info(f"  Val windows  : {len(val_ds):,}    |  RAM after load: {v_mem:.0f} MB")
    except (DataError, ScalerError, LabelError) as exc:
        _fatal(log, "VAL DATA LOAD FAILURE", str(exc))
        raise

    train_labels = train_ds.tensors[1]
    dist = {CLASS_NAMES[i]: (train_labels == i).sum().item() for i in range(args.n_classes)}
    log.info(f"  Label dist   : {dist}")

    absent = [CLASS_NAMES[i] for i in range(args.n_classes) if dist[CLASS_NAMES[i]] == 0]
    if absent:
        _warn(log, "ABSENT CLASSES IN TRAINING SET",
              f"Classes with zero training windows: {absent}\n"
              f"  Label dist : {dist}\n"
              f"  The model cannot learn these classes without examples.\n"
              f"  They will still receive maximum loss weight.")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    # pin_memory=False — no GPU, and pinning wastes RAM on CPU-only machines.
    # num_workers=0    — no subprocess workers; each worker costs ~300–500 MB RAM.
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    try:
        model = HelioForgeTCN(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            dropout=args.dropout,
            norm_type=args.norm_type,
        ).to(device)
    except Exception as exc:
        _fatal(log, "MODEL CONSTRUCTION FAILURE",
               f"HelioForgeTCN failed to initialise.\n"
               f"  in_channels : {args.in_channels}\n"
               f"  n_classes   : {args.n_classes}\n"
               f"  norm_type   : {args.norm_type}\n"
               f"  dropout     : {args.dropout}\n"
               f"  Reason      : {exc}",
               hint="Check that norm_type is one of 'batch', 'layer', 'none'.")
        raise ModelError("HelioForgeTCN construction failed.") from exc

    n_params   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_mb   = n_params * 4 / 1024 / 1024          # float32 = 4 bytes
    model_mem  = _mem_mb()
    log.info(f"Parameters     : {n_params:,}  (~{model_mb:.1f} MB weights)")
    log.info(f"RAM after model: {model_mem:.0f} MB")

    # Estimate training memory cost and warn if tight
    activation_mb = args.batch_size * 512 * 512 * 4 / 1024 / 1024 * 8  # 8 blocks
    total_est_mb  = model_mb * 3 + activation_mb + (v_mem if v_mem > 0 else 0)
    log.info(
        f"Memory estimate: weights={model_mb:.0f}MB  "
        f"activations~{activation_mb:.0f}MB  "
        f"total~{total_est_mb:.0f}MB"
    )
    if total_ram > 0 and total_est_mb > total_ram * 0.75:
        _warn(log, "HIGH MEMORY USAGE ESTIMATED",
              f"Estimated training memory ({total_est_mb:.0f} MB) "
              f"exceeds 75% of system RAM ({total_ram:.0f} MB).\n"
              f"  Action : Reduce --batch-size to 8 and set --accum-steps 4 "
              f"(effective batch stays 32).")

    # ── Loss, optimiser, scheduler ────────────────────────────────────────────
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

    # ── Training state ────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    best_macro_f1  = 0.0
    patience_count = 0
    history        = []

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, args.n_epochs + 1):
        t0 = time.time()

        # ── Train phase with gradient accumulation ─────────────────────────
        model.train()
        train_loss   = 0.0
        accum_loss   = 0.0
        optimizer.zero_grad()

        try:
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                # Scale loss by accum_steps so accumulated gradient matches
                # a single large-batch gradient exactly.
                loss   = criterion(logits, y_batch) / args.accum_steps
                loss.backward()

                accum_loss += loss.item()

                # Step every accum_steps batches, or at end of epoch
                is_accum_step = (batch_idx + 1) % args.accum_steps == 0
                is_last_batch = (batch_idx + 1) == len(train_loader)

                if is_accum_step or is_last_batch:
                    # Mid-step NaN/Inf detection
                    if not math.isfinite(accum_loss):
                        raise TrainingError(
                            f"Loss became non-finite.\n"
                            f"  Epoch        : {epoch}\n"
                            f"  Batch        : {batch_idx + 1} / {len(train_loader)}\n"
                            f"  Accum loss   : {accum_loss}\n"
                            f"  Current LR   : {optimizer.param_groups[0]['lr']:.2e}\n"
                            f"  Likely causes:\n"
                            f"    - LR too high → restart with --lr 3e-4\n"
                            f"    - Gradient explosion → restart with --grad-clip 0.5\n"
                            f"    - Input NaN/Inf → re-verify dataset with verify_pre_training.py"
                        )

                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                    train_loss += accum_loss
                    accum_loss  = 0.0

        except TrainingError:
            raise
        except Exception as exc:
            _fatal(log, "TRAINING LOOP CRASHED",
                   f"Unexpected error in epoch {epoch}, batch {batch_idx + 1}.\n"
                   f"  Error type : {type(exc).__name__}\n"
                   f"  Message    : {exc}\n"
                   f"  RAM usage  : {_mem_mb():.0f} MB\n\n"
                   + traceback.format_exc(),
                   hint="If OOM: reduce --batch-size 8 --accum-steps 4. "
                        "If crash: check nvidia-smi or free -h.")
            raise TrainingError(f"Training loop crashed at epoch {epoch}.") from exc

        train_loss /= math.ceil(len(train_loader) / args.accum_steps)

        # ── Val phase ──────────────────────────────────────────────────────
        try:
            val_loss, val_metrics = evaluate(
                model, val_loader, criterion, device,
                n_classes=args.n_classes, class_names=CLASS_NAMES[:args.n_classes],
            )
        except Exception as exc:
            _fatal(log, "VALIDATION LOOP CRASHED",
                   f"Evaluation failed at epoch {epoch}.\n"
                   f"  Error type : {type(exc).__name__}\n"
                   f"  Message    : {exc}\n"
                   f"  RAM usage  : {_mem_mb():.0f} MB\n\n"
                   + traceback.format_exc(),
                   hint="Check RAM usage with  free -h  on EC2.")
            raise TrainingError(f"Validation loop crashed at epoch {epoch}.") from exc

        scheduler.step(val_loss)

        lr_now   = optimizer.param_groups[0]["lr"]
        elapsed  = time.time() - t0
        epoch_mb = _mem_mb()
        mem_info = f"  mem={epoch_mb:.0f}MB" if epoch_mb > 0 else ""

        log.info(
            f"Epoch {epoch:3d}/{args.n_epochs}  "
            f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
            f"macro_f1: {val_metrics['macro_f1']:.4f}  "
            f"acc: {val_metrics['accuracy']:.4f}  "
            f"lr: {lr_now:.2e}  [{elapsed:.1f}s{mem_info}]"
        )

        history.append({
            "epoch"     : epoch,
            "train_loss": train_loss,
            "val_loss"  : val_loss,
            "lr"        : lr_now,
            "ram_mb"    : epoch_mb,
            **val_metrics,
        })

        # ── Checkpoint: best val loss ──────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            try:
                save_checkpoint(
                    {"epoch": epoch, "model_state": model.state_dict(),
                     "optimizer_state": optimizer.state_dict(),
                     "val_loss": val_loss, "val_metrics": val_metrics, "args": vars(args)},
                    ckpt_dir / "best_val_loss.pt", log,
                )
                log.info(f"  ✓ best_val_loss.pt  (val_loss={best_val_loss:.4f})")
            except CheckpointError as exc:
                _warn(log, "CHECKPOINT SAVE FAILED", str(exc))

        # ── Checkpoint: best macro F1 ──────────────────────────────────────
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            try:
                save_checkpoint(
                    {"epoch": epoch, "model_state": model.state_dict(),
                     "optimizer_state": optimizer.state_dict(),
                     "val_loss": val_loss, "val_metrics": val_metrics, "args": vars(args)},
                    ckpt_dir / "best_macro_f1.pt", log,
                )
                log.info(f"  ✓ best_macro_f1.pt  (macro_f1={best_macro_f1:.4f})")
            except CheckpointError as exc:
                _warn(log, "CHECKPOINT SAVE FAILED", str(exc))
        else:
            patience_count += 1

        if patience_count >= args.patience:
            log.info(
                f"Early stopping triggered. "
                f"No macro_f1 improvement for {args.patience} consecutive epochs."
            )
            break

    # ── Final checkpoint & history ─────────────────────────────────────────────
    try:
        save_checkpoint(
            {"epoch": epoch, "model_state": model.state_dict(),
             "val_loss": val_loss, "val_metrics": val_metrics, "args": vars(args)},
            ckpt_dir / "final.pt", log,
        )
    except CheckpointError as exc:
        _warn(log, "FINAL CHECKPOINT SAVE FAILED", str(exc))

    try:
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
    except OSError as exc:
        _warn(log, "HISTORY JSON SAVE FAILED",
              f"Could not write history.json.\n"
              f"  Path   : {run_dir / 'history.json'}\n"
              f"  Reason : {exc}")

    # ── Final summary ──────────────────────────────────────────────────────────
    final_mem = _mem_mb()
    log.info("")
    log.info("=" * 70)
    log.info("  TRAINING COMPLETE")
    log.info(f"  Best val_loss  : {best_val_loss:.4f}")
    log.info(f"  Best macro_F1  : {best_macro_f1:.4f}")
    log.info(f"  Final RAM      : {final_mem:.0f} MB")
    log.info("")
    log.info("  Final Val Metrics:")
    log.info(format_metrics_table(val_metrics, class_names=CLASS_NAMES[:args.n_classes]))
    log.info("")

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


# =============================================================================
# Entry point — top-level exception handler
# =============================================================================

if __name__ == "__main__":
    try:
        main()
    except (DataError, ScalerError, LabelError, ModelError, TrainingError, CheckpointError) as exc:
        print(f"\n{_FATAL_SEP}")
        print(f"  [FATAL] {type(exc).__name__}")
        print(f"{_FATAL_SEP}")
        for line in str(exc).strip().splitlines():
            print(f"  {line}")
        print(f"{_FATAL_SEP}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n{_SEP}")
        print("  [INTERRUPTED]  Training stopped by user (Ctrl+C).")
        print(f"  Partial checkpoints may be in the run directory.")
        print(f"{_SEP}\n")
        sys.exit(130)
    except Exception as exc:
        print(f"\n{_FATAL_SEP}")
        print("  [FATAL] UNEXPECTED ERROR — Full traceback:")
        print(f"{_FATAL_SEP}")
        traceback.print_exc()
        print(f"{_FATAL_SEP}\n")
        sys.exit(1)
