"""
scripts/train.py — HelioForge AI Baseline TCN Training Script
=============================================================

EC2 RESOURCE CONSTRAINTS:
    RAM  : 8 GB total (OS + Python ~1.5 GB overhead → ~6.5 GB usable)
    CPU  : 2 cores  |  GPU: None

Memory budget (batch_size=16, 8.4M-param model):
    Model weights + grads + AdamW moments :  ~132 MB
    Activations (8 blocks, batch=16)      :  ~132 MB
    Dataset in RAM (train + val)          :  ~150 MB
    Python / OS overhead                  :  ~800 MB
    ─────────────────────────────────────────────────
    Estimated total                       :  ~1.2 GB  (safe within 8 GB)

Progress display:
    ┌ Epoch bar  — overall training progress with ETA + RAM + best F1
    └ Batch bar  — per-epoch batch progress with running loss + RAM

All paths resolve from configs/data_paths.yaml via PATH_CFG.

Usage:
    python scripts/train.py
    python scripts/train.py --run-name baseline_v1 --batch-size 16 --accum-steps 2
"""

import sys
import json
import argparse
import logging
import math
import time
import traceback
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── tqdm — required for progress bars ─────────────────────────────────────────
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

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
    """Training loop encounters an unrecoverable error."""

class CheckpointError(HelioForgeError):
    """Checkpoint cannot be written."""


# =============================================================================
# Constants
# =============================================================================

THRESHOLDS = [100, 500, 2_000, 8_000]   # SoLEXS COUNTS/sec class boundaries
_SEP       = "─" * 70
_FATAL_SEP = "!" * 70


# =============================================================================
# Memory utility
# =============================================================================

def _mem_mb() -> float:
    """Current process RSS in MB (reads /proc/self/status on Linux EC2)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    try:
        import psutil, os
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        pass
    return -1.0


def _total_ram_mb() -> float:
    """Total system RAM in MB from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return -1.0


def _mem_str() -> str:
    """Compact RAM string for progress bar: '1142MB' or '1.1GB'."""
    mb = _mem_mb()
    if mb < 0:
        return "?MB"
    if mb >= 1024:
        return f"{mb/1024:.1f}GB"
    return f"{mb:.0f}MB"


def _eta_str(seconds: float) -> str:
    """Format seconds as 'Xh Ym Zs' or 'Ym Zs' or 'Zs'."""
    if seconds < 0:
        return "--"
    td  = timedelta(seconds=int(seconds))
    h   = td.seconds // 3600
    m   = (td.seconds % 3600) // 60
    s   = td.seconds % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# =============================================================================
# tqdm-aware logging handler
# =============================================================================

class _TqdmHandler(logging.StreamHandler):
    """
    Logging handler that uses tqdm.write() so log messages don't corrupt bars.
    Falls back to stderr.write() if tqdm is not available.
    """
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if TQDM_AVAILABLE:
                tqdm.write(msg, file=sys.stdout)
            else:
                sys.stdout.write(msg + "\n")
                sys.stdout.flush()
        except Exception:
            self.handleError(record)


# =============================================================================
# Error reporting helpers
# =============================================================================

def _fatal(log: logging.Logger, error_type: str, msg: str, hint: str = "") -> None:
    """Emit a clearly visible [FATAL] banner to log + console."""
    log.error(_FATAL_SEP)
    log.error(f"  [FATAL] {error_type}")
    log.error(_FATAL_SEP)
    for line in msg.strip().splitlines():
        log.error(f"  {line}")
    if hint:
        log.error(f"  Hint    : {hint}")
    log.error(_FATAL_SEP)


def _warn(log: logging.Logger, warning_type: str, msg: str) -> None:
    """Emit a clearly visible [WARNING] banner."""
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

    The .pt files store ONLY {"sequences": Tensor(N, F, L)}.
    Labels computed by reversing MinMax on channel 0 → peak COUNTS/sec → bin.

    Raises: ScalerError, DataError, LabelError
    """
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
            f"  Action : Re-run build_windows.py — raw data may have no variation."
        )

    if sequences.ndim != 3:
        raise DataError(
            f"Expected 3D tensor (N, F, L) but got shape {tuple(sequences.shape)}.\n"
            f"  Action : Ensure .pt file was generated by build_windows.py."
        )

    if soft_channel >= sequences.shape[1]:
        raise DataError(
            f"soft_channel={soft_channel} exceeds tensor F={sequences.shape[1]}.\n"
            f"  Action : Check --in-channels matches the dataset feature count."
        )

    soft_norm_peak = sequences[:, soft_channel, :].max(dim=1).values.numpy()
    raw_peak       = soft_norm_peak * (soft_max - soft_min) + soft_min

    labels = np.zeros(len(raw_peak), dtype=np.int64)
    labels[(raw_peak >= THRESHOLDS[0]) & (raw_peak < THRESHOLDS[1])] = 1
    labels[(raw_peak >= THRESHOLDS[1]) & (raw_peak < THRESHOLDS[2])] = 2
    labels[(raw_peak >= THRESHOLDS[2]) & (raw_peak < THRESHOLDS[3])] = 3
    labels[raw_peak >= THRESHOLDS[3]] = 4

    if len(np.unique(labels)) == 1:
        raise LabelError(
            f"All {len(labels)} windows collapsed into class {labels[0]}.\n"
            f"  raw_peak range : [{raw_peak.min():.2f}, {raw_peak.max():.2f}] COUNTS/sec\n"
            f"  soft_min/max   : {soft_min:.4f} / {soft_max:.4f}\n"
            f"  Thresholds     : {THRESHOLDS}\n"
            f"  Action         : Re-run build_windows.py — scaler bounds may be wrong."
        )

    return torch.from_numpy(labels)


# =============================================================================
# Split loader
# =============================================================================

def load_split(data_dir: Path, filename: str, scaler: dict) -> TensorDataset:
    """Load a .pt window file and derive labels. Format: {"sequences": Tensor(N,F,L)}."""
    path = data_dir / filename

    if not path.exists():
        present = [f.name for f in data_dir.glob("*.pt")] if data_dir.exists() else "DIR NOT FOUND"
        raise DataError(
            f"Tensor file not found.\n"
            f"  Expected path  : {path}\n"
            f"  .pt files here : {present}\n"
            f"  Action         : Confirm --data-dir and run build_windows.py if needed."
        )

    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise DataError(
            f"Failed to load tensor file.\n"
            f"  Path   : {path}\n"
            f"  Reason : {exc}\n"
            f"  Action : File may be corrupted. Re-run build_windows.py."
        ) from exc

    if isinstance(obj, dict):
        X = obj.get("sequences", obj.get("X"))
        if X is None:
            raise DataError(
                f"Dict in {filename} has no 'sequences' key.\n"
                f"  Keys found : {list(obj.keys())}\n"
                f"  Action     : Regenerate tensors with build_windows.py."
            )
        X = X.float()
    elif isinstance(obj, torch.Tensor):
        X = obj.float()
    else:
        raise DataError(
            f"Unexpected object type in {filename}.\n"
            f"  Got    : {type(obj).__name__}\n"
            f"  Action : Regenerate tensors with build_windows.py."
        )

    if X.ndim != 3 or X.shape[0] == 0:
        raise DataError(
            f"Tensor shape invalid in {filename}.\n"
            f"  Shape  : {tuple(X.shape)}  (expected: (N>0, F, L))\n"
            f"  Action : Regenerate tensors with build_windows.py."
        )

    n_nan = torch.isnan(X).sum().item()
    n_inf = torch.isinf(X).sum().item()
    if n_nan > 0 or n_inf > 0:
        raise DataError(
            f"Tensor {filename} contains invalid values.\n"
            f"  NaN : {n_nan:,}  |  Inf : {n_inf:,}  "
            f"(out of {X.numel():,} elements)\n"
            f"  Action : Re-run build_windows.py — nan_to_num is applied during generation."
        )

    return TensorDataset(X, derive_labels(X, scaler))


# =============================================================================
# Logging setup
# =============================================================================

def setup_logging(run_log_file: Path, global_log_file: Path) -> logging.Logger:
    logger = logging.getLogger("helioforge.train")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")

    # Console handler — tqdm-aware so bars aren't corrupted
    ch = _TqdmHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handlers (run-specific + global)
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
        description="HelioForge Baseline TCN  [EC2: 8 GB RAM, 2 CPU cores]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--data-dir",    type=str, default=str(PATH_CFG.windows.root))
    parser.add_argument("--train-file",  type=str, default="train_feat32_w512.pt")
    parser.add_argument("--val-file",    type=str, default="val_feat32_w512.pt")
    parser.add_argument("--scaler-file", type=str, default="scaler_f32_w512.json")
    parser.add_argument("--output-dir",  type=str, default=str(PATH_CFG.experiments.baseline_tcn.runs))
    parser.add_argument("--log-dir",     type=str, default=str(PATH_CFG.logs.root))
    parser.add_argument("--run-name",    type=str, default=None)

    # Training — EC2-safe defaults
    parser.add_argument("--n-epochs",     type=int,   default=80)
    parser.add_argument("--batch-size",   type=int,   default=16,
                        help="Micro-batch size. EC2 default=16 (low RAM).")
    parser.add_argument("--accum-steps",  type=int,   default=2,
                        help="Gradient accumulation. Effective batch = batch × accum.")
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout",      type=float, default=0.2)
    parser.add_argument("--norm-type",    type=str,   default="batch",
                        choices=["batch", "layer", "none"])
    parser.add_argument("--patience",     type=int,   default=15)
    parser.add_argument("--grad-clip",    type=float, default=1.0)
    parser.add_argument("--label-smooth", type=float, default=0.0)
    parser.add_argument("--n-classes",    type=int,   default=5)
    parser.add_argument("--in-channels",  type=int,   default=32)

    # EC2 resource controls
    parser.add_argument("--num-workers",  type=int, default=0,
                        help="DataLoader workers. EC2 default=0 (no subprocess RAM cost).")
    parser.add_argument("--num-threads",  type=int, default=2,
                        help="PyTorch intra-op threads. Match physical core count.")

    parser.add_argument("--seed",         type=int,  default=42)
    parser.add_argument("--no-cuda",      action="store_true")
    parser.add_argument("--no-progress",  action="store_true",
                        help="Disable tqdm progress bars (plain log output only).")

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
            f"  Action      : Check disk space (df -h /opt) and permissions (ls -la)."
        ) from exc


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    use_cuda       = torch.cuda.is_available() and not args.no_cuda
    device         = torch.device("cuda" if use_cuda else "cpu")
    use_bars       = TQDM_AVAILABLE and not args.no_progress
    effective_batch = args.batch_size * args.accum_steps

    # ── Run directory ──────────────────────────────────────────────────────────
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
        print(f"  Action  : chmod 755 /opt/helioforge-ai/experiments")
        print(f"{_FATAL_SEP}\n")
        sys.exit(1)

    run_log    = run_dir / "train.log"
    global_log = Path(args.log_dir) / "training.log"
    log        = setup_logging(run_log, global_log)

    # ── System info ────────────────────────────────────────────────────────────
    total_ram = _total_ram_mb()
    ram_str   = f"{total_ram/1024:.1f} GB" if total_ram > 0 else "unknown"

    log.info("=" * 70)
    log.info("  HELIO-FORGE AI  |  Baseline TCN Training")
    log.info("=" * 70)
    log.info(f"  Run name       : {run_name}")
    log.info(f"  Device         : {device}  {'(GPU ✓)' if use_cuda else '(CPU)'}")
    log.info(f"  PyTorch threads: {args.num_threads}")
    log.info(f"  System RAM     : {ram_str}  |  Process start: {_mem_str()}")
    log.info("─" * 70)
    log.info(f"  Data dir       : {args.data_dir}")
    log.info(f"  Train / Val    : {args.train_file}  /  {args.val_file}")
    log.info(f"  Scaler         : {args.scaler_file}")
    log.info("─" * 70)
    log.info(f"  Batch (micro)  : {args.batch_size}  |  Accum: {args.accum_steps}  →  effective: {effective_batch}")
    log.info(f"  DataLoader wkr : {args.num_workers}  (0 = no subprocesses, safe on 8 GB)")
    log.info(f"  Epochs         : {args.n_epochs}  |  LR: {args.lr}  |  WD: {args.weight_decay}")
    log.info(f"  Norm / Dropout : {args.norm_type} / {args.dropout}")
    log.info(f"  Patience       : {args.patience}  |  Grad clip: {args.grad_clip}")
    log.info(f"  Progress bars  : {'tqdm ✓' if use_bars else 'disabled (--no-progress)'}")
    log.info("=" * 70)

    if 0 < total_ram < 6 * 1024:
        _warn(log, "LOW SYSTEM RAM",
              f"RAM: {ram_str}. Recommended ≥ 6 GB.\n"
              f"  If OOM: --batch-size 8 --accum-steps 4")

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump({**vars(args), "run_name": run_name,
                   "effective_batch": effective_batch, "device": str(device)}, f, indent=2)

    # ── Scaler ─────────────────────────────────────────────────────────────────
    data_dir    = Path(args.data_dir)
    scaler_path = data_dir / args.scaler_file

    if not data_dir.exists():
        _fatal(log, "DATA DIRECTORY NOT FOUND",
               f"Provided  : {data_dir}\n"
               f"Config    : PATH_CFG.windows.root = {PATH_CFG.windows.root}",
               hint="Confirm this is the correct EC2 instance and the dataset has been transferred.")
        raise DataError(f"Data directory not found: {data_dir}")

    if not scaler_path.exists():
        present = [f.name for f in data_dir.iterdir()] if data_dir.exists() else []
        _fatal(log, "SCALER FILE NOT FOUND",
               f"Expected  : {scaler_path}\n"
               f"Dir holds : {present}",
               hint="Run  python scripts/build_windows.py --mode features")
        raise ScalerError(f"Scaler not found: {scaler_path}")

    try:
        with open(scaler_path) as f:
            scaler = json.load(f)
    except json.JSONDecodeError as exc:
        _fatal(log, "SCALER JSON PARSE ERROR",
               f"Path   : {scaler_path}\n"
               f"Reason : {exc}",
               hint="Delete and re-run build_windows.py.")
        raise ScalerError(f"Malformed scaler: {scaler_path}") from exc

    for key in ("min", "max", "n_features", "window_size"):
        if key not in scaler:
            _fatal(log, "SCALER SCHEMA MISMATCH",
                   f"Missing field: '{key}'\n"
                   f"Path       : {scaler_path}\n"
                   f"Keys found : {list(scaler.keys())}",
                   hint="Re-run build_windows.py.")
            raise ScalerError(f"Missing key '{key}' in scaler")

    if scaler["n_features"] != args.in_channels:
        _warn(log, "SCALER / MODEL CHANNEL MISMATCH",
              f"Scaler F={scaler['n_features']} ≠ --in-channels={args.in_channels}.\n"
              f"  Labels may be wrong if channel 0 ≠ soft_mean.")

    log.info(f"Scaler loaded  : {scaler_path}  (F={scaler['n_features']}, w={scaler['window_size']})")

    # ── Load datasets ──────────────────────────────────────────────────────────
    try:
        log.info(f"Loading train  : {data_dir / args.train_file}")
        train_ds = load_split(data_dir, args.train_file, scaler)
        log.info(f"  Train windows: {len(train_ds):,}  |  RAM: {_mem_str()}")
    except (DataError, ScalerError, LabelError) as exc:
        _fatal(log, "TRAIN DATA LOAD FAILURE", str(exc))
        raise

    try:
        log.info(f"Loading val    : {data_dir / args.val_file}")
        val_ds = load_split(data_dir, args.val_file, scaler)
        log.info(f"  Val windows  : {len(val_ds):,}  |  RAM: {_mem_str()}")
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
              f"  Model cannot learn these classes without examples.")

    # ── DataLoaders ────────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False,
        drop_last=False, persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
        persistent_workers=False,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    try:
        model = HelioForgeTCN(
            in_channels=args.in_channels, n_classes=args.n_classes,
            dropout=args.dropout, norm_type=args.norm_type,
        ).to(device)
    except Exception as exc:
        _fatal(log, "MODEL CONSTRUCTION FAILURE",
               f"HelioForgeTCN failed to initialise.\n"
               f"  in_channels={args.in_channels}  n_classes={args.n_classes}\n"
               f"  norm_type={args.norm_type}  dropout={args.dropout}\n"
               f"  Reason : {exc}",
               hint="norm_type must be one of 'batch', 'layer', 'none'.")
        raise ModelError("HelioForgeTCN construction failed.") from exc

    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_mb  = n_params * 4 / 1024 / 1024
    log.info(f"Parameters     : {n_params:,}  (~{model_mb:.1f} MB)  |  RAM: {_mem_str()}")

    # ── Loss / optimiser / scheduler ───────────────────────────────────────────
    criterion = build_weighted_criterion(
        train_labels=train_labels, n_classes=args.n_classes,
        device=device, label_smoothing=args.label_smooth,
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
    train_start    = time.time()

    # ── Progress bar format ────────────────────────────────────────────────────
    #
    #  Epoch bar (outer):
    #  Training  [██████████████░░░░░░] 15/80 epochs │ val_loss=0.8821 │ F1=0.4312 │ RAM=1.1GB │ ETA: 1h 23m 44s
    #
    #  Batch bar (inner, erased after each epoch):
    #  Epoch  15  [███████████████████░] 85% │ loss=0.9104 │ RAM=1.1GB │ 12s/batch
    #
    EPOCH_BAR_FMT = (
        "{desc}  {bar}  {n_fmt}/{total_fmt} epochs"
        "  │  {postfix}  │  ETA: {remaining}"
    )
    BATCH_BAR_FMT = (
        "  {desc}  {bar}  {percentage:3.0f}%"
        "  │  {postfix}  │  {elapsed}<{remaining}"
    )

    epoch_bar = (
        tqdm(
            range(1, args.n_epochs + 1),
            desc="Training ",
            unit="epoch",
            dynamic_ncols=True,
            bar_format=EPOCH_BAR_FMT,
            colour="cyan",
        )
        if use_bars else range(1, args.n_epochs + 1)
    )

    # ── Epoch loop ─────────────────────────────────────────────────────────────
    for epoch in epoch_bar:
        t0 = time.time()

        # ── Train phase ────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        accum_loss = 0.0
        optimizer.zero_grad()
        running_loss = 0.0   # smoothed for batch bar postfix

        batch_bar = (
            tqdm(
                train_loader,
                desc=f"Epoch {epoch:3d}/{args.n_epochs}",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
                bar_format=BATCH_BAR_FMT,
                colour="green",
            )
            if use_bars else train_loader
        )

        try:
            for batch_idx, (X_batch, y_batch) in enumerate(batch_bar):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                loss   = criterion(logits, y_batch) / args.accum_steps
                loss.backward()

                accum_loss   += loss.item()
                running_loss  = accum_loss * args.accum_steps   # unscaled display

                is_accum_step = (batch_idx + 1) % args.accum_steps == 0
                is_last_batch = (batch_idx + 1) == len(train_loader)

                if is_accum_step or is_last_batch:
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
                            f"    - Input NaN/Inf → re-verify with verify_pre_training.py"
                        )
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()
                    train_loss += accum_loss
                    accum_loss  = 0.0

                # Update batch bar postfix every 5 batches to reduce overhead
                if use_bars and batch_idx % 5 == 0:
                    batch_bar.set_postfix(ordered_dict={
                        "loss": f"{running_loss:.4f}",
                        "RAM" : _mem_str(),
                    })

        except TrainingError:
            if use_bars:
                batch_bar.close()
            raise
        except Exception as exc:
            if use_bars:
                batch_bar.close()
            _fatal(log, "TRAINING LOOP CRASHED",
                   f"Epoch {epoch}  Batch {batch_idx + 1}/{len(train_loader)}\n"
                   f"  Error : {type(exc).__name__}: {exc}\n"
                   f"  RAM   : {_mem_str()}\n\n"
                   + traceback.format_exc(),
                   hint="OOM → --batch-size 8 --accum-steps 4.  Crash → free -h on EC2.")
            raise TrainingError(f"Training loop crashed at epoch {epoch}.") from exc

        if use_bars:
            batch_bar.close()

        n_steps     = math.ceil(len(train_loader) / args.accum_steps)
        train_loss /= max(n_steps, 1)

        # ── Val phase ──────────────────────────────────────────────────────────
        try:
            val_loss, val_metrics = evaluate(
                model, val_loader, criterion, device,
                n_classes=args.n_classes, class_names=CLASS_NAMES[:args.n_classes],
            )
        except Exception as exc:
            _fatal(log, "VALIDATION LOOP CRASHED",
                   f"Epoch {epoch}\n"
                   f"  Error : {type(exc).__name__}: {exc}\n"
                   f"  RAM   : {_mem_str()}\n\n"
                   + traceback.format_exc(),
                   hint="Check RAM with  free -h  on EC2.")
            raise TrainingError(f"Validation crashed at epoch {epoch}.") from exc

        scheduler.step(val_loss)

        lr_now  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # ETA for full training (based on elapsed per epoch so far)
        elapsed_total = time.time() - train_start
        epochs_done   = epoch - (1 if args.run_name else 0)
        secs_per_epoch = elapsed_total / max(epoch, 1)
        remaining_secs = secs_per_epoch * (args.n_epochs - epoch)

        # ── Update epoch bar postfix ───────────────────────────────────────────
        if use_bars:
            epoch_bar.set_postfix(ordered_dict={
                "loss"    : f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "F1"      : f"{val_metrics['macro_f1']:.4f}",
                "LR"      : f"{lr_now:.1e}",
                "RAM"     : _mem_str(),
            })

        # ── Log line (visible in train.log + global log) ───────────────────────
        log.info(
            f"Epoch {epoch:3d}/{args.n_epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"F1={val_metrics['macro_f1']:.4f}  "
            f"acc={val_metrics['accuracy']:.4f}  "
            f"lr={lr_now:.2e}  "
            f"[{elapsed:.1f}s  RAM={_mem_str()}  "
            f"ETA={_eta_str(remaining_secs)}]"
        )

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "lr": lr_now,
            "elapsed_s": elapsed, "ram_mb": _mem_mb(),
            **val_metrics,
        })

        # ── Checkpoints ────────────────────────────────────────────────────────
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

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            try:
                save_checkpoint(
                    {"epoch": epoch, "model_state": model.state_dict(),
                     "optimizer_state": optimizer.state_dict(),
                     "val_loss": val_loss, "val_metrics": val_metrics, "args": vars(args)},
                    ckpt_dir / "best_macro_f1.pt", log,
                )
                log.info(f"  ✓ best_macro_f1.pt  (F1={best_macro_f1:.4f})")
            except CheckpointError as exc:
                _warn(log, "CHECKPOINT SAVE FAILED", str(exc))
        else:
            patience_count += 1

        if patience_count >= args.patience:
            log.info(
                f"Early stopping: no macro_f1 improvement for {args.patience} epochs."
            )
            break

    if use_bars:
        epoch_bar.close()

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
              f"Path : {run_dir / 'history.json'}\nReason : {exc}")

    # ── Final summary ──────────────────────────────────────────────────────────
    total_secs = time.time() - train_start
    log.info("")
    log.info("=" * 70)
    log.info("  TRAINING COMPLETE")
    log.info(f"  Epochs trained : {epoch}")
    log.info(f"  Total time     : {_eta_str(total_secs)}")
    log.info(f"  Best val_loss  : {best_val_loss:.4f}")
    log.info(f"  Best macro_F1  : {best_macro_f1:.4f}")
    log.info(f"  Final RAM      : {_mem_str()}")
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
# Entry point
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
        print("  [FATAL] UNEXPECTED ERROR")
        print(f"{_FATAL_SEP}")
        traceback.print_exc()
        print(f"{_FATAL_SEP}\n")
        sys.exit(1)
