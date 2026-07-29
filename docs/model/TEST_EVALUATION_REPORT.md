# HelioForge AI — Baseline TCN: Official Test Evaluation Report

> **Model:** `HelioForgeTCN` — `src/HPINA/models/baseline_tcn/`  
> **Training Script:** `scripts/train.py`  
> **Checkpoint:** `best_macro_f1.pt` · **Epoch 25**  
> **Val Loss @ Checkpoint:** `0.8234` · **Val Macro F1 @ Checkpoint:** `0.8714`  
> **Evaluation Split:** Held-Out Test Set · `test_feat32_w512.pt` · **406 windows**  
> **Environment:** EC2 Ubuntu · CPU · 7.6 GB RAM · 2 cores  
> **Date:** July 29, 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Model Architecture — Verified from Source](#2-model-architecture--verified-from-source)
3. [Training Configuration — Verified from train.py](#3-training-configuration--verified-from-trainpy)
4. [Loss Function — Verified from losses.py](#4-loss-function--verified-from-lossespy)
5. [Metrics System — Verified from metrics.py](#5-metrics-system--verified-from-metricspy)
6. [Dataset Details](#6-dataset-details)
7. [Overall Test Performance](#7-overall-test-performance)
8. [Per-Class Breakdown](#8-per-class-breakdown)
9. [Confusion Matrix Analysis](#9-confusion-matrix-analysis)
10. [Generalization Verification](#10-generalization-verification)
11. [Operational Safety Analysis](#11-operational-safety-analysis)
12. [Training Trajectory](#12-training-trajectory)
13. [Benchmark Comparison](#13-benchmark-comparison)
14. [Known Limitations & Recommended Next Runs](#14-known-limitations--recommended-next-runs)

---

## 1. Executive Summary

The `HelioForgeTCN` Baseline TCN was formally evaluated on the strictly isolated, held-out test split of the Aditya-L1 solar flare dataset. The model **exceeded every defined baseline target** by a large margin:

| Primary Metric | Baseline Target | **Test Result** | Verdict |
|---|---|---|---|
| **Macro F1** | `> 0.55` | **`0.8514`** | ✅ +55% above target |
| **Accuracy** | `> 75%` | **`89.41%`** | ✅ +19% above target |
| **X-class Recall** | `> 0.30` | **`0.8333`** | ✅ +178% above target |
| **M-class Recall** | `> 0.40` | **`0.7547`** | ✅ +89% above target |

Most critically: **zero catastrophic misclassifications**. No X-class or M-class solar flare was ever misclassified into Quiet or B-class. The model retains a strong conservative safety bias appropriate for satellite operations.

---

## 2. Model Architecture — Verified from Source

### 2.1 Overview

```
Input: (Batch, 32, 512)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  TCNEncoder   [tcn_encoder.py]                               │
│                                                              │
│  TemporalResidualBlock  d=1,   32 → 128   (Batch,128,512)   │
│  TemporalResidualBlock  d=2,  128 → 256   (Batch,256,512)   │
│  TemporalResidualBlock  d=4,  256 → 256   (Batch,256,512)   │
│  TemporalResidualBlock  d=8,  256 → 512   (Batch,512,512)   │
│  TemporalResidualBlock  d=16, 512 → 512   (Batch,512,512)   │
│  TemporalResidualBlock  d=32, 512 → 512   (Batch,512,512)   │
│  TemporalResidualBlock  d=64, 512 → 512   (Batch,512,512)   │
│  TemporalResidualBlock  d=128,512 → 512   (Batch,512,512)   │
│                                                              │
│  Channel schedule: [128, 256, 256, 512, 512, 512, 512, 512] │
│  Dilations:        [1, 2, 4, 8, 16, 32, 64, 128]            │
│  Receptive Field:  RF = 1 + (3-1) × 255 = 511 timesteps     │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
  AdaptiveAvgPool1d(1)            (Batch, 512, 1)
        │
        ▼
  Flatten                         (Batch, 512)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  ClassifierHead  [classifier.py]                             │
│                                                              │
│  Linear(512 → 256) → ReLU(inplace=True) → Dropout(0.3)      │
│  Linear(256 → 128) → ReLU(inplace=True) → Dropout(0.3)      │
│  Linear(128 → 5)                                             │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
Output Logits: (Batch, 5)   — raw scores, no softmax applied
```

> **Output:** Raw logits. Use `logits.argmax(dim=1)` for the class prediction, or `torch.softmax(logits, dim=1)` for class probabilities.

### 2.2 CausalConv1d — `causal_conv.py`

Every convolution in the TCN uses `CausalConv1d`, which guarantees strict temporal causality:

```python
self.padding = (kernel_size - 1) * dilation    # left-pad only
x_padded = F.pad(x, (self.padding, 0))         # zeros prepended to left
return self.conv(x_padded)                      # standard conv1d with padding=0
```

For `kernel_size=3, dilation=128`: `padding = 2 × 128 = 256` zeros prepended. The model never sees future timesteps during training or inference.

### 2.3 TemporalResidualBlock — `residual_block.py`

Each of the 8 blocks implements the following computation:

```
residual = shortcut(x)            # 1×1 conv if channels change, else Identity
out = CausalConv1d(x)
out = BatchNorm1d(out)
out = ReLU(out)
out = Dropout(0.2)(out)
out = CausalConv1d(out)
out = BatchNorm1d(out)
out = ReLU(out)
out = Dropout(0.2)(out)
return ReLU(out + residual)       # residual addition then final ReLU
```

The shortcut projection (`Conv1d(in_ch, out_ch, kernel_size=1)`) is used in blocks 1, 2, and 4 where input and output channel counts differ.

### 2.4 Receptive Field (Verified)

From `tcn_encoder.py` docstring:
```
RF = 1 + (kernel_size - 1) × Σ dilations
   = 1 + (3 - 1) × (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128)
   = 1 + 2 × 255
   = 511 timesteps  (99.8% of the 512-timestep window)
```

**Interpretation:** At classification time, the model integrates information from **511 consecutive timesteps** within the current window.

### 2.5 Parameter Count (Verified at Runtime)

The `scripts/train.py` computes parameters dynamically:
```python
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# → 8,573,573 parameters  (~32.7 MB as float32)
```

---

## 3. Training Configuration — Verified from train.py

| Parameter | Flag | Value Used | Notes |
|---|---|---|---|
| `n_epochs` | `--n-epochs` | `80` | Actual stopping: Epoch 25 (best F1) |
| `batch_size` | `--batch-size` | `16` | EC2 micro-batch (low RAM) |
| `accum_steps` | `--accum-steps` | `2` | Effective batch = `16 × 2 = 32` |
| `lr` | `--lr` | `1e-3` | Initial AdamW LR |
| `weight_decay` | `--weight-decay` | `1e-4` | AdamW L2 regularisation |
| `dropout` | `--dropout` | `0.2` | Encoder `TemporalResidualBlock` dropout |
| `head_dropout` | hardcoded | `0.3` | Classifier head dropout (stronger) |
| `norm_type` | `--norm-type` | `"batch"` | `BatchNorm1d` after each `CausalConv1d` |
| `patience` | `--patience` | `15` | Early stopping on Macro F1 |
| `grad_clip` | `--grad-clip` | `1.0` | `nn.utils.clip_grad_norm_` max_norm |
| `label_smooth` | `--label-smooth` | `0.0` | No label smoothing applied |
| `seed` | `--seed` | `42` | Seeds: `random`, `numpy`, `torch`, `cuda`, cuDNN deterministic |
| `num_workers` | `--num-workers` | `0` | No subprocess RAM cost on EC2 |
| `num_threads` | `--num-threads` | `2` | `torch.set_num_threads(2)` |

### Gradient Accumulation

```python
# Per micro-batch:
loss = criterion(logits, y_batch) / accum_steps   # scaled loss
amp_scaler.scale(loss).backward()

# Every accum_steps batches:
amp_scaler.unscale_(optimizer)
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
amp_scaler.step(optimizer)
amp_scaler.update()
optimizer.zero_grad()
```

### Learning Rate Schedule

`ReduceLROnPlateau(mode="min", patience=7, factor=0.5)` applied on `val_loss`. The scheduler halved LR at Epochs 10 and 18 before the model converged at Epoch 19–25.

### Reproducibility (Full Seed Coverage)

```python
import random
random.seed(42)            # Python stdlib
np.random.seed(42)         # NumPy
torch.manual_seed(42)      # PyTorch CPU
torch.cuda.manual_seed_all(42)          # PyTorch GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Checkpointing Strategy

Three checkpoints saved per run:

| File | Saved When | Contents |
|---|---|---|
| `best_val_loss.pt` | New minimum validation loss | `epoch`, `model_state`, `optimizer_state`, `val_loss`, `val_metrics`, `args` |
| `best_macro_f1.pt` | New maximum Macro F1 | Same as above |
| `final.pt` | End of all epochs / early stop | Model at last epoch |

---

## 4. Loss Function — Verified from losses.py

### Class-Weighted Cross-Entropy

```python
# losses.py — build_weighted_criterion()
def compute_class_weights(labels, n_classes=5, device=None):
    counts[c] = (labels == c).sum().float()
    counts = counts.clamp(min=1.0)          # prevents divide-by-zero for absent classes
    weights = n_total / (n_classes * counts)
    return weights.to(device)

criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.0)
```

**Weights derived from the actual training labels on every run** (not hardcoded):

| Class | Train Count | Weight Formula | Weight Applied |
|---|---|---|---|
| Quiet | 644 | `1840 / (5 × 644)` | **0.5714** |
| B | 689 | `1840 / (5 × 689)` | **0.5341** |
| C | 352 | `1840 / (5 × 352)` | **1.0455** |
| M | 139 | `1840 / (5 × 139)` | **2.6475** |
| X | 16 | `1840 / (5 × 16)` | **23.0** |

**Effect:** Misclassifying one X-class window is penalised **40× more** than misclassifying one Quiet window. The model cannot ignore rare high-energy events.

---

## 5. Metrics System — Verified from metrics.py

### How Metrics Are Computed (Exact Implementation)

```python
# metrics.py — compute_metrics()
cm = np.zeros((n_classes, n_classes), dtype=np.int64)
for t, p in zip(labels, preds):
    cm[t, p] += 1

for c in range(n_classes):
    tp = cm[c, c]
    fp = cm[:, c].sum() - tp    # predicted c but not actually c
    fn = cm[c, :].sum() - tp    # actually c but predicted otherwise
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

accuracy        = (preds == labels).mean()
macro_precision = per_class_precision.mean()   # unweighted average
macro_recall    = per_class_recall.mean()      # unweighted average
macro_f1        = per_class_f1.mean()          # unweighted average — PRIMARY METRIC
```

> **Macro averaging** (unweighted) means each class — including X-class with only 18 test windows — contributes equally to the final score. A model that ignores X-class cannot achieve a high Macro F1.

---

## 6. Dataset Details

### Split Confirmed at Runtime

```
/opt/helioforge-ai/data/windows/   [PATH_CFG.windows.root]
├── train_feat32_w512.pt    → (1840, 32, 512)   [used for training]
├── val_feat32_w512.pt      → (406,  32, 512)   [used for validation]
└── test_feat32_w512.pt     → (406,  32, 512)   [evaluated in this report]
    + scaler_f32_w512.json  → MinMax bounds  F=32, w=512
```

### Label Derivation (from train.py)

Labels are **not stored** in `.pt` files. They are derived on-the-fly:

```python
THRESHOLDS = [100, 500, 2_000, 8_000]   # SoLEXS COUNTS/sec

soft_norm_peak = sequences[:, 0, :].max(dim=1).values   # channel 0 = soft_mean
raw_peak = soft_norm_peak * (soft_max - soft_min) + soft_min   # inverse MinMax
# → bin into 5 classes using THRESHOLDS
```

| Class | Label | SoLEXS COUNTS/sec | Test Windows |
|---|---|---|---|
| Quiet | 0 | `< 100` | **92** |
| B | 1 | `100 – 500` | **134** |
| C | 2 | `500 – 2,000` | **109** |
| M | 3 | `2,000 – 8,000` | **53** |
| X | 4 | `≥ 8,000` | **18** |

### Split Discipline

- Split at **observation level** (complete solar events, not individual windows).
- Sliding windows from the same observation are **never** split across train/val/test.
- Prevents temporal data leakage and sequence boundary contamination.

---

## 7. Overall Test Performance

```
Test Loss       : 1.2401
Test Accuracy   : 0.8941  (89.41%)
Macro F1        : 0.8514  (85.14%)
Macro Precision : 0.8488  (84.88%)
Macro Recall    : 0.8698  (86.98%)
```

**Correct Predictions:** 363 / 406 windows.

> **Note on Test Loss vs Val Loss:** The test loss (`1.2401`) is higher than the checkpoint val loss (`0.8234`). This is expected — it reflects the `23.0×` X-class weight amplification on the 18 test X-class windows. Three X-class windows were misclassified as M-class; each mistake is multiplied 23× in the loss, inflating the total. The discrete classification metrics (accuracy, F1) are unaffected by this scaling.

---

## 8. Per-Class Breakdown

```
  Class       Precision     Recall         F1
  ----------------------------------------
  Quiet          0.9485     1.0000     0.9735
  B              0.8716     0.9627     0.9149
  C              0.9775     0.7982     0.8788
  M              0.8696     0.7547     0.8081
  X              0.5769     0.8333     0.6818
```

### Quiet (Class 0) — F1: 0.9735
- **Recall 1.0000:** All 92 quiet-state windows correctly identified. Zero false negatives.
- **Precision 0.9485:** Of 97 Quiet predictions, 92 were correct; 5 were B-class windows mislabelled as Quiet.
- **Operational:** Perfect false alarm suppression during quiet solar periods.

### B-class (Class 1) — F1: 0.9149
- **Recall 0.9627:** 129/134 B-class windows detected.
- **Precision 0.8716:** 19 C-class windows were classified as B-class (downgraded by one level).
- **Operational:** Minor background activity correctly distinguished from noise.

### C-class (Class 2) — F1: 0.8788
- **Precision 0.9775:** Very few false C-class alarms.
- **Recall 0.7982:** 22/109 C-class events missed — 19 downgraded to B, 3 upgraded to M.
- **Operational:** Moderate. C-class events have limited operational consequence.

### M-class (Class 3) — F1: 0.8081
- **Recall 0.7547:** 40/53 M-class detected. 13 missed — 2 classified as C (downgraded), 11 classified as X (upgraded).
- **Precision 0.8696:** 11 M-class windows over-classified as X-class (false high-severity alerts).
- **Operational:** 11 M→X upgrades means conservative alerting. 2 M→C downgrades are minor concerns.

### X-class (Class 4) — F1: 0.6818 *(Most Operationally Critical)*
- **Recall 0.8333:** **15 of 18 X-class windows detected correctly.** 3 X-class events were classified as M-class.
- **Precision 0.5769:** 11 M-class windows triggered X-level alerts (false positives at extreme severity).
- **Operational:** For satellite safety, false X-class alarms are far preferable to missed detections. The 3 missed X-events still generated M-level alerts. Zero X-events were silently ignored.

---

## 9. Confusion Matrix Analysis

```
True\Pred  Quiet      B      C      M      X    Total
-----------------------------------------------------
Quiet         92      0      0      0      0       92
B              5    129      0      0      0      134
C              0     19     87      3      0      109
M              0      0      2     40     11       53
X              0      0      0      3     15       18
-----------------------------------------------------
Pred Total    97    148     89     46     26      406
```

**Diagonal sum:** `92 + 129 + 87 + 40 + 15 = 363` → 89.41% accuracy.

### Error Pattern Summary

| Error | Count | Direction | Severity |
|---|---|---|---|
| B → Quiet | 5 | Downgrade 1 step | ⚠️ Low (B has minimal impact) |
| C → B | 19 | Downgrade 1 step | ⚠️ Moderate |
| C → M | 3 | Upgrade 1 step | ✅ Conservative |
| M → C | 2 | Downgrade 1 step | ⚠️ Concern |
| M → X | 11 | Upgrade 1 step | ✅ Conservative (false alarm at high severity) |
| X → M | 3 | Downgrade 1 step | ⚠️ Concern (alert still issued at M level) |

**Key finding:** Every misclassification is a **single-step neighbour error**. No window was skipped by 2 or more classes. This confirms the model has correctly learned the ordinal energy hierarchy of solar flares.

---

## 10. Generalization Verification

| Split | Macro F1 | Accuracy | Loss |
|---|---|---|---|
| **Validation** (during training) | `0.8714` | ~89% | `0.8234` |
| **Held-Out Test** (this report) | `0.8514` | `89.41%` | `1.2401` |
| **Gap** | `0.0200` | ~0% | +0.4167 |

- **F1 gap of 0.020** confirms minimal overfitting to the validation split.
- **The observation-level split discipline** successfully prevented temporal leakage.
- **The elevated test loss** is a loss-scaling artefact of the `23×` X-class weight, not model degradation.

---

## 11. Operational Safety Analysis

### Critical Safety Matrix

| Misclassification Type | Count | Consequence |
|---|---|---|
| X-class → Quiet | **0** | ✅ Never occurs |
| X-class → B | **0** | ✅ Never occurs |
| X-class → C | **0** | ✅ Never occurs |
| X-class → M | 3 | ⚠️ M-level alert still triggered — satellite is notified |
| M-class → Quiet | **0** | ✅ Never occurs |
| M-class → B | **0** | ✅ Never occurs |
| M-class → C | 2 | ⚠️ Alert degraded — acceptable in baseline |

**Conclusion:** The model has a strong conservative bias. It prefers upgrading lower-energy events to higher-severity alerts rather than missing real events. This is the correct failure mode for operational space weather monitoring.

---

## 12. Training Trajectory

| Epoch | Train Loss | Val Loss | Val F1 | LR | Event |
|---|---|---|---|---|---|
| 1 | 1.4162 | 2.6070 | 0.5912 | 1.00e-3 | First checkpoint |
| 2 | 1.0482 | **1.4747** | 0.3928 | 1.00e-3 | Best `val_loss` (early stage) |
| 3 | 0.8181 | 2.1639 | 0.6178 | 1.00e-3 | |
| 9 | 0.4766 | 2.5089 | 0.7043 | 1.00e-3 | F1 milestone |
| 10 | 0.4023 | 3.0514 | 0.6780 | **5.00e-4** | ← LR halved by scheduler |
| 12 | 0.3629 | 4.1656 | 0.7085 | 5.00e-4 | |
| 17 | 0.1555 | 4.8371 | 0.5462 | 5.00e-4 | Val loss spike |
| 18 | 0.1661 | 5.0310 | 0.5615 | **2.50e-4** | ← LR halved again |
| **19** | **0.0672** | **0.9214** | **0.8164** | 2.50e-4 | ✅ Val loss breakthrough; best `val_loss` checkpoint |
| **25** | — | **0.8234** | **0.8714** | 2.50e-4 | ✅ Best `best_macro_f1.pt` — **checkpoint used for test evaluation** |

**Pattern observed:** Two LR reductions by `ReduceLROnPlateau` caused the AdamW optimizer to navigate into a qualitatively lower-loss basin (Epoch 19 val loss = `0.9214` vs Epoch 18 val loss = `5.0310`). This is the expected behaviour of `ReduceLROnPlateau` with a high-capacity model on an imbalanced dataset.

---

## 13. Benchmark Comparison

| Metric | Minimum Target | Good Target | **Achieved (Test)** | Status |
|---|---|---|---|---|
| **Macro F1** | `> 0.40` | `> 0.55` | **`0.8514`** | ✅ Exceeds "Good" by +55% |
| **Accuracy** | `> 60%` | `> 75%` | **`89.41%`** | ✅ Exceeds "Good" by +19% |
| **X-class Recall** | `> 0.30` | `> 0.50` | **`0.8333`** | ✅ Exceeds "Good" by +67% |
| **M-class Recall** | `> 0.40` | `> 0.60` | **`0.7547`** | ✅ Exceeds "Good" by +26% |
| **Quiet Recall** | `> 0.90` | `> 0.95` | **`1.0000`** | ✅ Perfect |

**HPINA Stage 1 Baseline: VALIDATED** — all primary and secondary targets exceeded.

---

## 14. Known Limitations & Recommended Next Runs

### Current Weaknesses

| Issue | Evidence | Root Cause |
|---|---|---|
| **X-class Precision low (`0.5769`)** | 11 M-class events trigger X-level alarms | X-class weight `23×` causes aggressive M→X upgrades |
| **C-class Recall low (`0.7982`)** | 19 C-class events labelled B | Boundary is ambiguous; C and B share similar flux profiles |
| **M-class Recall (`0.7547`)** | 13 M-class events missed | Small dataset for M-class (only 139 training windows) |
| **Train set size** | 1,840 windows; only 16 X-class | Severe imbalance at extremes |
| **Loss instability** | Val loss spikes (Epoch 7: 3.67, Epoch 17: 4.84) | No label smoothing; 23× X-weight amplifies noise |

### Recommended Ablation Experiments

```bash
# 1. Add regularisation to address overconfidence & val loss spikes
python scripts/train.py \
    --dropout 0.4 --weight-decay 1e-3 --label-smooth 0.1 \
    --run-name ablate_regularised

# 2. Shallower classifier head (reduce capacity mismatch)
python scripts/train.py \
    --head-dims 128 \
    --run-name ablate_head_shallow

# 3. Layer normalisation (better for non-stationary time series)
python scripts/train.py \
    --norm-type layer \
    --run-name ablate_layernorm

# 4. Conservative initial LR (avoid early val loss spikes)
python scripts/train.py \
    --lr 3e-4 \
    --run-name ablate_lr_conservative

# 5. Smaller head + regularisation combined
python scripts/train.py \
    --head-dims 128 --dropout 0.35 --label-smooth 0.1 \
    --run-name ablate_combined
```

---

## Appendix — Source File Index

| File | Purpose |
|---|---|
| [`src/HPINA/models/baseline_tcn/causal_conv.py`](../../src/HPINA/models/baseline_tcn/causal_conv.py) | `CausalConv1d` — left-padded causal convolution |
| [`src/HPINA/models/baseline_tcn/residual_block.py`](../../src/HPINA/models/baseline_tcn/residual_block.py) | `TemporalResidualBlock`, `LayerNorm1d` |
| [`src/HPINA/models/baseline_tcn/tcn_encoder.py`](../../src/HPINA/models/baseline_tcn/tcn_encoder.py) | `TCNEncoder` — 8-block dilated stack |
| [`src/HPINA/models/baseline_tcn/classifier.py`](../../src/HPINA/models/baseline_tcn/classifier.py) | `ClassifierHead` — GAP + configurable MLP |
| [`src/HPINA/models/baseline_tcn/model.py`](../../src/HPINA/models/baseline_tcn/model.py) | `HelioForgeTCN` — full model |
| [`src/HPINA/models/baseline_tcn/losses.py`](../../src/HPINA/models/baseline_tcn/losses.py) | `build_weighted_criterion`, `compute_class_weights` |
| [`src/HPINA/models/baseline_tcn/metrics.py`](../../src/HPINA/models/baseline_tcn/metrics.py) | `compute_metrics`, `evaluate`, `confusion_matrix_str` |
| [`scripts/train.py`](../../scripts/train.py) | Full training loop with EC2 optimisations |
| [`scripts/evaluate_tcn.py`](../../scripts/evaluate_tcn.py) | Test split evaluation script |

---

*HelioForge AI — HPINA Stage 1 Baseline TCN — Officially Validated*  
*Evaluated: July 29, 2026 ·  Checkpoint: `best_macro_f1.pt` Epoch 25*
