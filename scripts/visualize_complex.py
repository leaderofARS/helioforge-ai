"""
scripts/visualize_complex.py
──────────────────────────────
Generates a comprehensive suite of 7 complex, publication-grade visualizations
for both the 32 Selected Feature Matrix and the 3D TCN Window Tensors.

Saves all rendered PNG figures directly to the assets/ directory.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA

# ── Style setup (Sleek dark theme) ──────────────────────────────────────────
plt.style.use("dark_background")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.titlesize": 14,
    "figure.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.color": "#555555",
})

ACCENT_CYAN   = "#00E5FF"
ACCENT_MAGENTA= "#FF007F"
ACCENT_YELLOW = "#FFD700"
ACCENT_GREEN  = "#00E676"
ACCENT_ORANGE = "#FF9100"
BG_DARK       = "#0D1117"
PANEL_BG      = "#161B22"


def find_data_files(repo_root: Path) -> tuple[Path, Path]:
    """Locate the 32-feature CSV and the 3D window tensor .pt file."""
    # Check features paths
    feat_candidates = [
        repo_root / "data" / "features_second" / "selected_features.csv",
        repo_root / "data" / "features" / "selected_features.csv",
    ]
    feat_csv = next((p for p in feat_candidates if p.exists() and p.stat().st_size > 100000), None)
    if feat_csv is None:
        feat_csv = next((p for p in feat_candidates if p.exists()), feat_candidates[0])

    # Check windows paths
    win_candidates = [
        repo_root / "data" / "windows_second" / "train_feat32_w512.pt",
        repo_root / "data" / "windows_third" / "train_feat32_w512.pt",
        repo_root / "data" / "windows" / "train_feat32_w512.pt",
        repo_root / "data" / "windows" / "train.pt",
    ]
    win_pt = next((p for p in win_candidates if p.exists()), win_candidates[0])

    return feat_csv, win_pt


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    assets_dir = repo_root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  HELIO-FORGE AI  |  COMPLEX VISUALIZATION GENERATOR")
    print("=" * 60)

    feat_csv, win_pt = find_data_files(repo_root)
    print(f"  Feature Matrix CSV : {feat_csv}")
    print(f"  Window Tensor PT   : {win_pt}")
    print(f"  Output Directory   : {assets_dir}")
    print("=" * 60)

    # ── Load Feature Matrix ──────────────────────────────────────────────────
    print("\n[1/7] Loading Feature Matrix …")
    df = pd.read_csv(feat_csv)
    time_col = "TIME" if "TIME" in df.columns else None
    feat_cols = [c for c in df.columns if c not in ("TIME", "observation_id")]
    print(f"      Rows T={len(df)}, Features F={len(feat_cols)}")

    # ── Load 3D Tensor ───────────────────────────────────────────────────────
    print("\n[2/7] Loading 3D Window Tensor …")
    tensor_data = torch.load(win_pt, map_location="cpu", weights_only=True)
    if isinstance(tensor_data, dict):
        tensor = tensor_data.get("sequences", next(iter(tensor_data.values())))
    else:
        tensor = tensor_data
    tensor_np = tensor.numpy()
    print(f"      Tensor shape: {tensor.shape} (N={tensor.shape[0]}, F={tensor.shape[1]}, L={tensor.shape[2]})")

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 1: 32 x 32 Correlation Matrix Heatmap
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3/7] Generating Figure 1: 32×32 Feature Correlation Heatmap …")
    corr = df[feat_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10), facecolor=BG_DARK)
    ax.set_facecolor(PANEL_BG)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"},
        ax=ax,
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_title("HelioForge — 32 Physics Features Correlation Matrix", pad=15, color="white", fontsize=14, fontweight="bold")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    fig1_path = assets_dir / "01_selected_features_correlation_heatmap.png"
    plt.savefig(fig1_path, facecolor=BG_DARK)
    plt.close()
    print(f"      ✓ Saved → {fig1_path.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 2: Grid of KDE Feature Distributions
    # ─────────────────────────────────────────────────────────────────────────
    print("[4/7] Generating Figure 2: 32 Feature Distributions (KDE Density Grid) …")
    n_feats = len(feat_cols)
    cols = 4
    rows = int(np.ceil(n_feats / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(16, 3 * rows), facecolor=BG_DARK)
    axes = axes.flatten()

    for idx, col_name in enumerate(feat_cols):
        ax = axes[idx]
        ax.set_facecolor(PANEL_BG)
        vals = df[col_name].dropna().to_numpy()
        sns.kdeplot(vals, ax=ax, color=ACCENT_CYAN, fill=True, alpha=0.3, linewidth=1.5)
        ax.set_title(col_name, fontsize=9, color="white", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Density", fontsize=7, color="#888888")
        ax.tick_params(axis="both", colors="#AAAAAA")

    # Hide unused axes
    for idx in range(n_feats, len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle("HelioForge — Distribution & Density Profiles of 32 Physics Features", color="white", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    fig2_path = assets_dir / "02_feature_distributions_kde.png"
    plt.savefig(fig2_path, facecolor=BG_DARK)
    plt.close()
    print(f"      ✓ Saved → {fig2_path.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 3: Continuous Multi-Channel Time-Series Plot
    # ─────────────────────────────────────────────────────────────────────────
    print("[5/7] Generating Figure 3: Continuous Time-Series Physics Streams …")
    sample_len = min(2000, len(df))
    sub_df = df.iloc[:sample_len]
    x_axis = sub_df["TIME"].to_numpy() - sub_df["TIME"].iloc[0] if time_col else np.arange(sample_len)

    key_streams = [
        ("soft_mean", ACCENT_CYAN, "SoLEXS Soft X-Ray Mean (COUNTS/s)"),
        ("hard_max", ACCENT_MAGENTA, "HEL1OS Hard X-Ray Peak Energy (keV)"),
        ("soft_rise_rate", ACCENT_YELLOW, "Soft X-Ray Rise Rate"),
        ("soft_hard_ratio", ACCENT_GREEN, "Soft-to-Hard X-Ray Ratio"),
        ("cross_correlation", ACCENT_ORANGE, "Cross-Channel Correlation"),
    ]
    valid_streams = [s for s in key_streams if s[0] in df.columns]

    fig, axes = plt.subplots(len(valid_streams), 1, figsize=(14, 2.5 * len(valid_streams)), sharex=True, facecolor=BG_DARK)
    if len(valid_streams) == 1:
        axes = [axes]

    for ax, (col, color, label) in zip(axes, valid_streams):
        ax.set_facecolor(PANEL_BG)
        ax.plot(x_axis, sub_df[col].to_numpy(), color=color, linewidth=1.2, label=label)
        ax.set_ylabel(col, fontsize=9, color="white", fontweight="bold")
        ax.legend(loc="upper right", frameon=True, facecolor=BG_DARK, edgecolor="#444444", fontsize=8)

    axes[-1].set_xlabel("Time (seconds)", fontsize=10, color="white")
    fig.suptitle(f"HelioForge — Continuous Multichannel Physics Time-Series (First {sample_len} seconds)", color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig3_path = assets_dir / "03_continuous_feature_timeseries.png"
    plt.savefig(fig3_path, facecolor=BG_DARK)
    plt.close()
    print(f"      ✓ Saved → {fig3_path.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 4: 2D Heatmap of Single 3D Window Sheet (32 Features x 512 Steps)
    # ─────────────────────────────────────────────────────────────────────────
    print("[6/7] Generating Figure 4: Single Window Tensor Heatmap (32×512) …")
    window_idx = 0
    w_data = tensor_np[window_idx]  # shape (F=32, L=512)

    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG_DARK)
    ax.set_facecolor(PANEL_BG)
    im = ax.imshow(w_data, aspect="auto", cmap="viridis", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("MinMax Normalised Feature Value [0, 1]", color="white")

    ax.set_yticks(np.arange(min(len(feat_cols), w_data.shape[0])))
    ax.set_yticklabels(feat_cols[:w_data.shape[0]], fontsize=7)
    ax.set_xlabel("Timestep within Window (0 → 511 seconds)", fontsize=10, color="white")
    ax.set_title(f"HelioForge — 3D Tensor Window #{window_idx} Sheet (F={w_data.shape[0]} Features × L={w_data.shape[1]} Timesteps)", color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig4_path = assets_dir / "04_3d_tensor_single_window_heatmap.png"
    plt.savefig(fig4_path, facecolor=BG_DARK)
    plt.close()
    print(f"      ✓ Saved → {fig4_path.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 5: Stacked Line Trajectories of Single 3D Window
    # ─────────────────────────────────────────────────────────────────────────
    print("[7/7] Generating Figure 5: Feature Channel Trajectories over 512 Timesteps …")
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG_DARK)
    ax.set_facecolor(PANEL_BG)
    t_steps = np.arange(w_data.shape[1])

    # Plot top 8 features with highest variance in window 0
    variances = np.var(w_data, axis=1)
    top_indices = np.argsort(variances)[::-1][:8]
    colors_list = [ACCENT_CYAN, ACCENT_MAGENTA, ACCENT_YELLOW, ACCENT_GREEN, ACCENT_ORANGE, "#B388FF", "#FF80AB", "#84FFFF"]

    for idx, c_idx in enumerate(top_indices):
        f_name = feat_cols[c_idx] if c_idx < len(feat_cols) else f"Feat_{c_idx}"
        ax.plot(t_steps, w_data[c_idx], label=f"{f_name}", color=colors_list[idx % len(colors_list)], linewidth=1.5, alpha=0.85)

    ax.set_title(f"HelioForge — Top Feature Trajectories in Window #{window_idx} (~4.5 Hours of Time)", color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel("Timestep (seconds)", color="white")
    ax.set_ylabel("Normalised Value", color="white")
    ax.legend(loc="upper right", facecolor=BG_DARK, edgecolor="#444444", fontsize=8)
    plt.tight_layout()
    fig5_path = assets_dir / "05_3d_tensor_channel_trajectories.png"
    plt.savefig(fig5_path, facecolor=BG_DARK)
    plt.close()
    print(f"      ✓ Saved → {fig5_path.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 6: PCA 2D Scatter Plot of All 3D Window Embeddings
    # ─────────────────────────────────────────────────────────────────────────
    print("[8/7] Generating Figure 6: PCA 2D Projection of All Window Embeddings …")
    # Flatten each window: (N, 32, 512) -> (N, 32*512)
    N, F_dim, L_dim = tensor_np.shape
    flat_windows = tensor_np.reshape(N, F_dim * L_dim)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(flat_windows)
    var_exp = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG_DARK)
    ax.set_facecolor(PANEL_BG)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=np.arange(N), cmap="magma", alpha=0.6, s=15)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Window Sequence Order (Time Progress)", color="white")

    ax.set_title(f"HelioForge — PCA 2D Embedding Projection of N={N} Window Tensors", color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"PC 1 ({var_exp[0]:.1f}% Variance)", color="white")
    ax.set_ylabel(f"PC 2 ({var_exp[1]:.1f}% Variance)", color="white")
    plt.tight_layout()
    fig6_path = assets_dir / "06_3d_windows_pca_projection.png"
    plt.savefig(fig6_path, facecolor=BG_DARK)
    plt.close()
    print(f"      ✓ Saved → {fig6_path.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 7: Feature Variance Rankings & Cumulative PCA Variance
    # ─────────────────────────────────────────────────────────────────────────
    print("[9/7] Generating Figure 7: Feature Variance Rankings & PCA Explained Variance …")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG_DARK)
    ax1.set_facecolor(PANEL_BG)
    ax2.set_facecolor(PANEL_BG)

    # Subplot 1: Feature Variances
    var_series = df[feat_cols].var().sort_values(ascending=False)
    ax1.barh(var_series.index[:15][::-1], var_series.values[:15][::-1], color=ACCENT_CYAN, alpha=0.85)
    ax1.set_title("Top 15 Feature Variances", color="white", fontweight="bold")
    ax1.set_xlabel("Variance", color="white")

    # Subplot 2: Cumulative PCA Variance across tabular features
    pca_full = PCA().fit(df[feat_cols].dropna())
    cum_var = np.cumsum(pca_full.explained_variance_ratio_) * 100
    ax2.plot(np.arange(1, len(cum_var) + 1), cum_var, marker="o", color=ACCENT_MAGENTA, linewidth=2)
    ax2.axhline(90, color=ACCENT_YELLOW, linestyle="--", label="90% Variance Threshold")
    ax2.axhline(95, color=ACCENT_GREEN, linestyle="--", label="95% Variance Threshold")
    ax2.set_title("Cumulative PCA Explained Variance Ratio", color="white", fontweight="bold")
    ax2.set_xlabel("Number of Principal Components", color="white")
    ax2.set_ylabel("Cumulative Variance (%)", color="white")
    ax2.legend(loc="lower right", facecolor=BG_DARK, edgecolor="#444444")

    fig.suptitle("HelioForge — Feature Variance & Dimensionality Diagnostics", color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig7_path = assets_dir / "07_feature_variances_and_pca_variance.png"
    plt.savefig(fig7_path, facecolor=BG_DARK)
    plt.close()
    print(f"      ✓ Saved → {fig7_path.name}")

    print("\n" + "=" * 60)
    print("  ALL 7 COMPLEX VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print(f"  Saved to: {assets_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
