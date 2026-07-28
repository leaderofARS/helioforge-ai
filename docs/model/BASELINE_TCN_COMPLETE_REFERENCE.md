# HelioForge AI — Baseline TCN: Complete Technical Reference

> **Document scope:** Everything implemented in `src/HPINA/models/baseline_tcn/` and `scripts/train.py`. This document explains the *what*, *why*, and *how* for every file — architecture, mathematics, loss functions, training loop, metrics, and deployment.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Dataset Versioning & Split Discipline](#2-dataset-versioning--split-discipline)
3. [Architecture Overview](#3-architecture-overview)
4. [File-by-File Breakdown](#4-file-by-file-breakdown)
   - [causal_conv.py](#41-causal_convpy)
   - [residual_block.py](#42-residual_blockpy)
   - [tcn_encoder.py](#43-tcn_encoderpy)
   - [classifier.py](#44-classifierpy)
   - [model.py](#45-modelpy)
   - [losses.py](#46-lossespy)
   - [metrics.py](#47-metricspy)
   - [__init__.py](#48-__init__py)
5. [Training Pipeline & Telemetry](#5-training-pipeline--telemetry)
6. [Reproducibility & Determinism](#6-reproducibility--determinism)
7. [Automatic Mixed Precision (AMP)](#7-automatic-mixed-precision-amp)
8. [The Maths](#8-the-maths)
9. [Hyperparameter & Ablation Reference](#9-hyperparameter--ablation-reference)
10. [Checkpointing Strategy](#10-checkpointing-strategy)
11. [Interpreting Results](#11-interpreting-results)
12. [Quick Command Reference](#12-quick-command-reference)

---

## 1. The Problem

We are classifying solar flare intensity from multivariate time-series sensor data captured by the **HEL1OS** and **SoLEXS** instruments aboard ISRO's Aditya-L1 spacecraft.

### Classes

| ID | Name  | Physical Threshold (SoLEXS COUNTS/sec) | Description                                 |
|----|-------|----------------------------------------|---------------------------------------------|
| 0  | Quiet | `< 100`                                | No significant activity                     |
| 1  | B     | `100 – 500`                            | Minor flare (low energy)                    |
| 2  | C     | `500 – 2,000`                          | Moderate flare                              |
| 3  | M     | `2,000 – 8,000`                        | Strong flare — operational concern          |
| 4  | X     | `≥ 8,000`                              | Extreme flare — satellite damage risk       |

The key challenge: **extreme class imbalance**. X-class events are catastrophically rare. A naive model that always predicts "Quiet" scores >50% accuracy but is operationally worthless.

---

## 2. Dataset Versioning & Split Discipline

### Versioning History

- **`windows_fourth/` (Initial prototype):** Early split iteration (Train: 2009, Val: 238, Test: 406).
- **`windows_fifth/` / `/opt/helioforge-ai/data/windows/` (Canonical Production Dataset):** The versioned research dataset used for all baseline training and evaluation.

```
/opt/helioforge-ai/data/windows/
├── train_feat32_w512.pt    →  shape (1840, 32, 512)
├── val_feat32_w512.pt      →  shape (406,  32, 512)
├── test_feat32_w512.pt     →  shape (406,  32, 512)
└── scaler_f32_w512.json    →  persisted MinMax bounds (F=32, w=512)
```

| Split | Window Count | Class Distribution (Quiet / B / C / M / X) |
|-------|--------------|--------------------------------------------|
| **Train** | 1,840 | 644 / 689 / 352 / 139 / 16 |
| **Val** | 406 | ~18 X-class windows |
| **Test** | 406 | ~18 X-class windows |

| Axis | Meaning |
|------|---------|
| `N` — first | Number of sliding windows (independent training examples) |
| `F=32` — second | 32 physics-informed features (HEL1OS + SoLEXS channels) |
| `L=512` — third | 512 timesteps per window (~8.5 minutes at 1 Hz resolution) |

All values are MinMax-normalised to `[0, 1]` per channel using bounds computed on the **train split only**. The same bounds are applied to val and test.

### Splitting Discipline
- Stratification is enforced at the **observation level** (by full solar event, not by window) to prevent temporal data leakage.
- Overlapping sliding windows from the same observation never appear in two different splits.

---

## 3. Architecture Overview

```
Input Tensor                         (Batch, 32, 512)
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  TCN Encoder   [tcn_encoder.py]                       │
│                                                       │
│  ResidualBlock 1 — d=1,   32 → 128  (Batch,128,512)  │
│  ResidualBlock 2 — d=2,  128 → 256  (Batch,256,512)  │
│  ResidualBlock 3 — d=4,  256 → 256  (Batch,256,512)  │
│  ResidualBlock 4 — d=8,  256 → 512  (Batch,512,512)  │
│  ResidualBlock 5 — d=16, 512 → 512  (Batch,512,512)  │
│  ResidualBlock 6 — d=32, 512 → 512  (Batch,512,512)  │
│  ResidualBlock 7 — d=64, 512 → 512  (Batch,512,512)  │
│  ResidualBlock 8 — d=128,512 → 512  (Batch,512,512)  │
│                                                       │
│  Receptive Field : 511 timesteps (99.8% of window)   │
│  Parameters      : 8,573,573 parameters               │
└──────────────────────────────────────────────────────┘
        │
        ▼
  AdaptiveAvgPool1d(1)             (Batch, 512, 1)
        │
        ▼
  Flatten                          (Batch, 512)
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  Classifier Head  [classifier.py] (Configurable)      │
│                                                       │
│  Linear(512 → 256) → ReLU → Dropout(0.3)             │
│  Linear(256 → 128) → ReLU → Dropout(0.3)             │
│  Linear(128 → 5)                                      │
└──────────────────────────────────────────────────────┘
        │
        ▼
Output Logits                    (Batch, 5)
```

> **Parameter count calculation:** Dynamically evaluated as `sum(p.numel() for p in model.parameters() if p.requires_grad)`. The default `HelioForgeTCN` has **8,573,573** trainable parameters (~8.57M).

---

## 4. File-by-File Breakdown

### 4.1 `causal_conv.py`

**What it does:** A 1D convolution that is *strictly causal* — at any timestep `t`, the output only depends on inputs at `t` and earlier. Never the future.

**Implementation:**
```
Pad left by P = (kernel_size - 1) × dilation zeros
Then run standard conv1d with padding=0
Output[t] = w0*Input[t-2d] + w1*Input[t-d] + w2*Input[t]
```

---

### 4.2 `residual_block.py`

**What it does:** Stacks two dilated causal convolutions with normalization, activation, and dropout, then adds a skip connection back to the input.

**Normalization options (configurable via `norm_type`):** `"batch"`, `"layer"`, or `"none"`.

---

### 4.3 `tcn_encoder.py`

**What it does:** Stacks 8 `TemporalResidualBlock` instances in sequence with exponential dilations `[1, 2, 4, 8, 16, 32, 64, 128]`.

**Cumulative Receptive Field:**
$$\text{RF} = 1 + (k - 1) \cdot \sum_{i=0}^{7} d_i = 1 + 2 \times 255 = 511 \text{ timesteps}$$

> **Interpretation:** When the model classifies a window ending at timestep $t$, it draws evidence from **511 timesteps within the input window** (covering 99.8% of the 512-timestep window). At 1 Hz resolution, 511 timesteps corresponds to ~8.5 minutes of continuous observation history.

---

### 4.4 `classifier.py`

**What it does:** Collapses the time dimension via Global Average Pooling (`AdaptiveAvgPool1d(1)`), then passes the 512-dim summary vector through a multi-layer perceptron (MLP).

**Configurable Head Capacity (`head_dims`):**
To facilitate ablation studies, hidden layer dimensions are fully configurable:
```python
head = ClassifierHead(
    in_features=512,
    n_classes=5,
    dropout=0.3,
    head_dims=[256, 128]   # Default: 512 → 256 → 128 → 5
)
```

---

### 4.5 `model.py`

**What it does:** Top-level `HelioForgeTCN` container wiring `TCNEncoder` and `ClassifierHead` together.

```python
from src.HPINA.models.baseline_tcn import HelioForgeTCN

model = HelioForgeTCN(
    in_channels=32,
    n_classes=5,
    dropout=0.2,
    norm_type="batch",
    head_dims=[256, 128],   # Configurable classifier head
)
```

---

### 4.6 `losses.py`

**Solution: Class-Weighted Cross-Entropy Loss**

$$\mathcal{L} = -\sum_{c=0}^{4} w_c \cdot y_c \cdot \log(\hat{p}_c)$$

where weights are **computed dynamically per run from the active training split**:

$$w_c = \frac{N_{\text{total}}}{n_{\text{classes}} \times N_c}$$

> **Important:** Weights are never hardcoded. They adapt automatically to the label distribution of the dataset loaded.

---

### 4.7 `metrics.py`

### Evaluation Priority Order

On imbalanced solar flare data, metrics are evaluated in strict operational order:

1. **Macro F1 (Primary Metric):** Unweighted average of per-class F1 scores. Main target for model selection and checkpointing (`best_macro_f1.pt`).
2. **Per-class Recall:** Specifically **X-class Recall** and **M-class Recall**. Operational safety depends on catching rare high-energy flares.
3. **Confusion Matrix:** Full $5 \times 5$ inter-class confusion table to analyze misclassification patterns.
4. **Accuracy:** Overall fraction correct. Treated as a secondary baseline metric due to class imbalance bias.

---

## 5. Training Pipeline & Telemetry

`scripts/train.py` records complete telemetry per run:
- **Per-epoch time:** Seconds elapsed per epoch (`elapsed_s`).
- **Live Memory Footprint:** Process RSS RAM in MB (`ram_mb`) recorded per epoch.
- **Total Duration:** Total wall-clock execution time formatted as `Xh Ym Zs`.
- **Run Artifacts:** Full metric history saved to `history.json` and `train.log`.

---

## 6. Reproducibility & Determinism

To guarantee reproducible research across different runs and environments, `scripts/train.py` enforces deterministic seeds across all random number generators:

```python
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 7. Automatic Mixed Precision (AMP)

For GPU acceleration on CUDA-enabled instances, `scripts/train.py` supports Automatic Mixed Precision via the `--amp` flag:

```python
use_amp    = args.amp and use_cuda
amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

with torch.amp.autocast("cuda", enabled=use_amp):
    logits = model(X_batch)
    loss   = criterion(logits, y_batch) / args.accum_steps

amp_scaler.scale(loss).backward()
```

---

## 8. The Maths

### Receptive Field Formula

$$\text{RF} = 1 + (k - 1) \cdot \sum_{i=0}^{K-1} d_i = 1 + 2 \times 255 = \mathbf{511 \text{ timesteps}}$$

### Macro F1 Formula

$$\text{Macro F1} = \frac{1}{|C|} \sum_{c \in C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

---

## 9. Hyperparameter & Ablation Reference

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| `lr` | `--lr` | `1e-3` | Initial AdamW learning rate |
| `weight_decay` | `--weight-decay` | `1e-4` | AdamW weight decay |
| `batch_size` | `--batch-size` | `16` | Micro-batch size per step (optimized for 8 GB RAM) |
| `accum_steps` | `--accum-steps` | `2` | Gradient accumulation steps (effective batch = 32) |
| `norm_type` | `--norm-type` | `"batch"` | `"batch"`, `"layer"`, or `"none"` |
| `head_dims` | `--head-dims` | `256 128` | Classifier MLP hidden dimensions |
| `amp` | `--amp` | `False` | Enable Automatic Mixed Precision on GPU |
| `seed` | `--seed` | `42` | Random seed for reproducibility |

### Example Ablation Commands

```bash
# Ablation 1: Classifier head depth (512 -> 128 -> 5)
python scripts/train.py --head-dims 128 --run-name ablate_head_shallow

# Ablation 2: Layer Normalization
python scripts/train.py --norm-type layer --run-name ablate_layernorm

# Ablation 3: AMP enabled on GPU
python scripts/train.py --amp --run-name run_amp_gpu
```

---

## 10. Checkpointing Strategy

Checkpoints are saved to `experiments/baseline_tcn/runs/<run_name>/checkpoints/`:
- `best_macro_f1.pt`: Highest Macro F1 (Primary deployment target).
- `best_val_loss.pt`: Lowest validation loss.
- `final.pt`: Last epoch state.

---

## 11. Interpreting Results

### Target Performance Benchmarks

| Metric | Minimum Acceptable | Good Baseline |
|--------|-------------------|---------------|
| **Macro F1** | > 0.40 | > 0.55 |
| **X-class Recall** | > 0.30 | > 0.50 |
| **M-class Recall** | > 0.40 | > 0.60 |
| **Accuracy** | > 60% | > 75% |

---

## 12. Quick Command Reference

```bash
# EC2 Baseline Training (Zero required args)
python scripts/train.py

# Custom Run with AMP and Specific Head Architecture
python scripts/train.py --run-name custom_v1 --head-dims 256 128 --amp
```

---

*HelioForge AI — Baseline TCN Technical Reference v1.1*
*Last updated: 2026-07-28*
