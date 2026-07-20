"""
scripts/demo.py
───────────────
Visualise the 3D tensor structure of train.pt.

Run on EC2:
    python scripts/demo.py

Or with a custom path:
    python scripts/demo.py --path /opt/helioforge-ai/data/windows/train.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import PATH_CFG


def hr(char: str = "─", width: int = 60) -> str:
    return char * width


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise HPINA tensor structure")
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Path to .pt file (default: PATH_CFG.windows.train)",
    )
    parser.add_argument(
        "--feature-names",
        type=Path,
        default=None,
        help="Optional: path to selected_feature_names.csv to show real feature names",
    )
    args = parser.parse_args()

    # ── Load tensor ───────────────────────────────────────────────────────────
    pt_path = args.path or PATH_CFG.windows.train
    print(hr("="))
    print("  HELIO-FORGE AI  |  TENSOR VISUALISER")
    print(hr("="))
    print(f"\n  Loading: {pt_path}\n")

    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    tensor = data["sequences"] if isinstance(data, dict) else data

    N, F, L = tensor.shape

    # ── Optional feature names ────────────────────────────────────────────────
    feature_names = None
    csv_path = args.feature_names
    if csv_path and csv_path.exists():
        import csv
        with open(csv_path) as fh:
            reader = csv.reader(fh)
            next(reader, None)  # skip header
            feature_names = [row[0] for row in reader if row]

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Tensor shape
    # ══════════════════════════════════════════════════════════════════════════
    print(hr())
    print("  SECTION 1 — Shape")
    print(hr())
    print(f"\n  tensor.shape = {tuple(tensor.shape)}\n")
    print(f"  Axis 0  →  N = {N:,}   windows      (independent training examples)")
    print(f"  Axis 1  →  F = {F}     features     (channels, physics measurements)")
    print(f"  Axis 2  →  L = {L}     timesteps    (~{L/60:.1f} minutes at 1 Hz)")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Global statistics
    # ══════════════════════════════════════════════════════════════════════════
    print(hr())
    print("  SECTION 2 — Global statistics")
    print(hr())
    arr = tensor.numpy()
    print(f"\n  dtype   : {tensor.dtype}")
    print(f"  min     : {arr.min():.6f}")
    print(f"  max     : {arr.max():.6f}")
    print(f"  mean    : {arr.mean():.6f}")
    print(f"  std     : {arr.std():.6f}")
    print(f"  NaNs    : {np.isnan(arr).sum()}")
    print(f"  Infs    : {np.isinf(arr).sum()}")
    total_values = N * F * L
    print(f"\n  Total stored values  : {total_values:,}   ({N} × {F} × {L})")
    size_mb = arr.nbytes / (1024 ** 2)
    print(f"  Memory footprint     : {size_mb:.1f} MB")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Index meaning explained
    # ══════════════════════════════════════════════════════════════════════════
    print(hr())
    print("  SECTION 3 — How to index this tensor")
    print(hr())
    print()
    print("  tensor[window_idx, feature_idx, timestep_idx]")
    print()

    mid_f = F // 2  # safe middle feature index regardless of F size
    mid_w = N // 2  # safe middle window index

    examples = [
        (0,     0,     0,   "first window, first feature, first second"),
        (0,     0,     L-1, "first window, first feature, last second"),
        (0,     F-1,   0,   "first window, last feature, first second"),
        (N-1,   0,     0,   "last window,  first feature, first second"),
        (mid_w, mid_f, 100, f"middle window #{mid_w}, middle feature #{mid_f}, second 100"),
    ]

    for w_idx, f_idx, t_idx, description in examples:
        val = tensor[w_idx, f_idx, t_idx].item()
        feat_label = (
            feature_names[f_idx] if feature_names and f_idx < len(feature_names)
            else f"feature_{f_idx}"
        )
        print(f"  tensor[{w_idx:>5}, {f_idx:>2}, {t_idx:>3}]  =  {val:.5f}   ← {description}")
        print(f"                          ({feat_label})")
        print()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Slice one window and inspect it
    # ══════════════════════════════════════════════════════════════════════════
    print(hr())
    print("  SECTION 4 — Inspect one window (window index 0)")
    print(hr())
    print()
    window = tensor[0]   # shape: (F, L)
    print(f"  tensor[0].shape    = {tuple(window.shape)}   ← one sheet of graph paper")
    print(f"  → F={F} rows (features), L={L} columns (timesteps)\n")

    print("  Per-feature stats across the 512 timesteps of window 0:")
    print()
    header = f"  {'idx':>4}  {'feature':<35}  {'min':>8}  {'max':>8}  {'mean':>8}  {'std':>8}"
    print(header)
    print("  " + hr("-", 72))
    for i in range(F):
        row = window[i].numpy()
        name = feature_names[i] if feature_names and i < len(feature_names) else f"feat_{i:02d}"
        print(
            f"  {i:>4}  {name:<35}  "
            f"{row.min():>8.4f}  {row.max():>8.4f}  "
            f"{row.mean():>8.4f}  {row.std():>8.4f}"
        )
    print()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — Slice one feature across all windows
    # ══════════════════════════════════════════════════════════════════════════
    print(hr())
    print("  SECTION 5 — Inspect one feature across all windows (feature index 0)")
    print(hr())
    print()
    feature_all_windows = tensor[:, 0, :]   # shape: (N, L)
    print(f"  tensor[:, 0, :].shape = {tuple(feature_all_windows.shape)}")
    print(f"  → {N} windows, each containing {L} timesteps of feature_0\n")
    arr_f = feature_all_windows.numpy()
    print(f"  Global stats for feature_0 across ALL {N:,} windows:")
    print(f"    min  : {arr_f.min():.6f}")
    print(f"    max  : {arr_f.max():.6f}")
    print(f"    mean : {arr_f.mean():.6f}")
    print(f"    std  : {arr_f.std():.6f}")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — Batch simulation (what the model actually receives)
    # ══════════════════════════════════════════════════════════════════════════
    print(hr())
    print("  SECTION 6 — Simulating a training batch (batch_size=4)")
    print(hr())
    print()
    batch_size = 4
    batch = tensor[:batch_size]          # shape: (4, F, L)
    print(f"  batch = tensor[:4]")
    print(f"  batch.shape = {tuple(batch.shape)}")
    print()
    print("  This is exactly what the TCN model receives at each training step:")
    print(f"    Axis 0 → batch dimension  : {batch_size} windows processed in parallel")
    print(f"    Axis 1 → channel dimension: {F} physics features (TCN's 'in_channels')")
    print(f"    Axis 2 → time dimension   : {L} timesteps the TCN reads left-to-right")
    print()
    print("  After TCNEncoder:")
    print(f"    (4, {F}, {L})  →  (4, 64, {L})   ← 38 raw features → 64 learned features")
    print()
    print("  After Global Average Pool:")
    print(f"    (4, 64, {L})  →  (4, 64)         ← 512 timesteps collapsed to 1 vector")
    print()
    print("  After ClassifierHead:")
    print(f"    (4, 64)       →  (4, n_classes)  ← one logit per flare class")
    print()

    print(hr("="))
    print("  Done. Every window is a chunk of 8.5 minutes of solar X-ray history.")
    print("  The TCN reads all 38 features simultaneously, second by second,")
    print("  building up temporal patterns to predict solar flare class.")
    print(hr("="))


if __name__ == "__main__":
    main()
