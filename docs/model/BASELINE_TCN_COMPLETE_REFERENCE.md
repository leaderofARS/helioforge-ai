# HelioForge AI — Baseline TCN: Complete Technical Reference

> **Document scope:** Everything implemented in `src/HPINA/models/baseline_tcn/` and `scripts/train.py`. This document explains the *what*, *why*, and *how* for every file — architecture, mathematics, loss functions, training loop, metrics, and deployment.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Dataset Recap](#2-dataset-recap)
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
5. [Training Loop](#5-training-loop)
6. [The Maths](#6-the-maths)
7. [Hyperparameter Reference](#7-hyperparameter-reference)
8. [Checkpointing Strategy](#8-checkpointing-strategy)
9. [Interpreting Results](#9-interpreting-results)
10. [Quick Command Reference](#10-quick-command-reference)

---

## 1. The Problem

We are classifying solar flare intensity from multivariate time-series sensor data captured by the **HEL1OS** and **SoLEXS** instruments aboard ISRO's Aditya-L1 spacecraft.

### Classes

| ID | Name  | Description                                 |
|----|-------|---------------------------------------------|
| 0  | Quiet | No significant activity                     |
| 1  | B     | Minor flare (low energy)                    |
| 2  | C     | Moderate flare                              |
| 3  | M     | Strong flare — operational concern          |
| 4  | X     | Extreme flare — satellite damage risk       |

The key challenge: **extreme class imbalance**. X-class events are catastrophically rare. A naive model that always predicts "Quiet" scores >50% accuracy but is operationally worthless.

---

## 2. Dataset Recap

```
data/windows_fifth/
├── train_feat32_w512.pt    →  shape (1840, 32, 512)   ~16 X-class windows
├── val_feat32_w512.pt      →  shape (406,  32, 512)   ~18 X-class windows
└── test_feat32_w512.pt     →  shape (406,  32, 512)   ~18 X-class windows
```

| Axis | Meaning |
|------|---------|
| `N` — first  | Number of sliding windows (independent training examples) |
| `F=32` — second | 32 physics-informed features (HEL1OS + SoLEXS channels) |
| `L=512` — third | 512 timesteps per window (~4.5 hours at 32s stride) |

All values are MinMax-normalised to `[0, 1]` per channel using bounds computed on the **train split only**. The same bounds are applied to val and test.

### Splitting discipline
- Split done at the **observation level** (by full solar event, not by window) to prevent leakage.
- Overlapping windows from the same observation never appear in two different splits.

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
│  Parameters      : ~8.4 million                       │
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
│  Classifier Head  [classifier.py]                     │
│                                                       │
│  Linear(512 → 256) → ReLU → Dropout(0.3)             │
│  Linear(256 → 128) → ReLU → Dropout(0.3)             │
│  Linear(128 → 5)                                      │
└──────────────────────────────────────────────────────┘
        │
        ▼
Output Logits                    (Batch, 5)
```

> **Key insight:** The entire sequence of 512 timesteps is compressed into a single 512-dimensional vector by Global Average Pooling before classification. The TCN has already extracted all temporal patterns; GAP just collapses the time dimension.

---

## 4. File-by-File Breakdown

### 4.1 `causal_conv.py`

**What it does:** A 1D convolution that is *strictly causal* — at any timestep `t`, the output only depends on inputs at `t` and earlier. Never the future.

**Why this matters:** At deployment time, the future doesn't exist. If the model sees future data during training, it will cheat — achieving falsely high training accuracy but failing on live solar data.

**Implementation:**
```
Normal conv1d (padding='same'):
    Output[t] = w0*Input[t-1] + w1*Input[t] + w2*Input[t+1]
                                                ↑ FUTURE LEAK

Causal conv1d (left-pad only):
    Pad left by P = (kernel_size - 1) × dilation zeros
    Then run standard conv1d with padding=0
    Output[t] = w0*Input[t-2d] + w1*Input[t-d] + w2*Input[t]
                                                  ↑ only past ✓
```

**Key parameter:** `padding = (kernel_size - 1) * dilation`

For `kernel_size=3, dilation=64`: pad = 2 × 64 = 128 zeros prepended to the left.

**Shape contract:**
```
Input:  (Batch, C_in,  L)
Output: (Batch, C_out, L)   ← same length, always
```

---

### 4.2 `residual_block.py`

**What it does:** A temporal residual block — the fundamental building unit of the TCN. Stacks two dilated causal convolutions with normalization, activation, and dropout, then adds a skip connection back to the input.

**Why skip connections:** Without them, gradients shrink exponentially as they travel backwards through 8 layers. The skip path gives gradients a direct highway, so even the first layer receives a strong learning signal.

**Block structure:**
```
input ──┬─────────────────────────────────────────────► shortcut ─┐
        │                                                           │
        └► CausalConv1d → Norm → ReLU → Dropout                   │
                └► CausalConv1d → Norm → ReLU → Dropout            │
                        └───────────────────────────────────────── + ►► ReLU ►► output
```

**Shortcut projection:** If `in_channels ≠ out_channels`, a `1×1 Conv1d` projects the input to the output channel count before adding. If they match, it's an identity pass.

**Normalization options (configurable via `norm_type`):**

| `norm_type` | Layer used | Best for |
|---|---|---|
| `"batch"` | `BatchNorm1d` | Large fixed batch sizes, stable training |
| `"layer"` | `LayerNorm1d` | Small batches, non-stationary time series |
| `"none"` | `Identity` | Ablation / debugging |

`LayerNorm1d` is a custom wrapper: it transposes `(B, C, L) → (B, L, C)`, applies `nn.LayerNorm(C)`, then transposes back. This is necessary because PyTorch's LayerNorm normalises over the **last** dimension, but in 1D conv data the channel dimension comes second.

**Shape contract:**
```
Input:  (Batch, in_channels,  L)
Output: (Batch, out_channels, L)   ← sequence length preserved
```

---

### 4.3 `tcn_encoder.py`

**What it does:** Stacks 8 `TemporalResidualBlock` instances in sequence, each with a larger dilation than the last. Exposes `self.out_channels` so the downstream `ClassifierHead` knows its input size without hardcoding.

**Progressive widening schedule:**

| Block | Dilation | In Ch → Out Ch | Receptive field of this block |
|-------|----------|----------------|-------------------------------|
| 1     | 1        | 32 → 128       | 3 timesteps                   |
| 2     | 2        | 128 → 256      | 5 timesteps                   |
| 3     | 4        | 256 → 256      | 9 timesteps                   |
| 4     | 8        | 256 → 512      | 17 timesteps                  |
| 5     | 16       | 512 → 512      | 33 timesteps                  |
| 6     | 32       | 512 → 512      | 65 timesteps                  |
| 7     | 64       | 512 → 512      | 129 timesteps                 |
| 8     | 128      | 512 → 512      | 257 timesteps                 |

**Total cumulative receptive field:**
```
RF = 1 + (kernel_size - 1) × Σ dilations
   = 1 + (3 - 1) × (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128)
   = 1 + 2 × 255
   = 511 timesteps    (99.8% of our L=512 window)
```

This means: when the model predicts the flare class for a window ending at time `t`, it can draw evidence from up to 511 seconds in the past within that window.

**Shape contract:**
```
Input:  (Batch, 32,  512)
Output: (Batch, 512, 512)   ← channels expanded, sequence unchanged
```

---

### 4.4 `classifier.py`

**What it does:** Collapses the time dimension and maps the encoder's high-dimensional sequence representation to class logits.

**Step by step:**
```
(Batch, 512, 512)           ← encoder output: 512 channels × 512 timesteps
        ↓  AdaptiveAvgPool1d(1)
(Batch, 512, 1)             ← every channel's average over all 512 timesteps
        ↓  Flatten()
(Batch, 512)                ← single 512-dim summary vector per example
        ↓  Linear(512→256) → ReLU → Dropout(0.3)
(Batch, 256)
        ↓  Linear(256→128) → ReLU → Dropout(0.3)
(Batch, 128)
        ↓  Linear(128→5)
(Batch, 5)                  ← raw logits, one per flare class
```

**Global Average Pooling intuition:** Instead of taking just the last timestep (which would waste 511 steps of context) or flattening everything (which would make the Linear layers huge), GAP takes the mean across all 512 timesteps. Every timestep contributes equally to the final representation.

> **Output:** Raw *logits*, not probabilities. Apply `torch.softmax(..., dim=1)` if you need class probabilities for inference, or just `torch.argmax(..., dim=1)` for the predicted class.

---

### 4.5 `model.py`

**What it does:** The top-level container. Wires `TCNEncoder` and `ClassifierHead` together. This is the only class you need to instantiate for training or inference.

```python
from src.HPINA.models.baseline_tcn import HelioForgeTCN

model = HelioForgeTCN(
    in_channels=32,       # matches our F=32 feature count
    n_classes=5,          # Quiet, B, C, M, X
    dropout=0.2,          # encoder dropout
    norm_type="batch",    # "batch" | "layer" | "none"
)

# Inference:
x      = torch.randn(32, 32, 512)   # batch of 32 windows
logits = model(x)                    # shape: (32, 5)
preds  = logits.argmax(dim=1)        # shape: (32,)  — class index per window
```

**The classifier head always uses `dropout=0.3`** regardless of the encoder dropout setting, since the MLP head is smaller and benefits from stronger regularisation.

---

### 4.6 `losses.py`

**The problem:** With extreme class imbalance (X-class has ~16 training windows vs. hundreds of Quiet windows), standard cross-entropy treats every mistake equally. The model quickly learns to predict "Quiet" for everything and stops improving.

**Solution: Class-Weighted Cross-Entropy Loss**

$$\mathcal{L} = -\sum_{c=0}^{4} w_c \cdot y_c \cdot \log(\hat{p}_c)$$

where the weight for class $c$ is:

$$w_c = \frac{N_{\text{total}}}{n_{\text{classes}} \times N_c}$$

| Quantity | Meaning |
|---|---|
| $N_{\text{total}}$ | Total number of training windows |
| $n_{\text{classes}}$ | 5 |
| $N_c$ | Number of training windows belonging to class $c$ |

**Effect:** Rare classes get high weight → the loss penalty for misclassifying an X-class event is amplified. The model cannot ignore X-class and get away with it.

**Example weights (approximate, based on train split distribution):**

| Class | Windows | Weight (approx.) |
|-------|---------|-----------------|
| Quiet | ~900    | 0.41            |
| B     | ~450    | 0.82            |
| C     | ~380    | 0.97            |
| M     | ~80     | 4.6             |
| X     | ~16     | 23.0            |

X-class misclassification is penalised ~56× more than Quiet misclassification.

**Usage:**
```python
from src.HPINA.models.baseline_tcn import build_weighted_criterion

# Pass all training labels (1D integer tensor)
criterion = build_weighted_criterion(
    train_labels=train_ds.tensors[1],   # shape (N,)
    n_classes=5,
    device=device,
    label_smoothing=0.0,                # set to 0.1 for smoother training
)
```

Weights are computed automatically from the actual label distribution — no hardcoding.

---

### 4.7 `metrics.py`

**Why not just accuracy?** On imbalanced data, accuracy is a trap. A model predicting "Quiet" always would achieve ~50% accuracy. Macro F1 is the primary metric because it treats every class equally regardless of how many samples it has.

**Metrics computed per evaluation pass:**

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Accuracy** | `correct / total` | Overall fraction correct |
| **Macro Precision** | `mean(P_c)` | Mean precision across all 5 classes |
| **Macro Recall** | `mean(R_c)` | Mean recall across all 5 classes |
| **Macro F1** | `mean(F1_c)` | **Primary metric** — harmonic mean of P and R per class, then averaged |
| **Per-class Precision** | `TP_c / (TP_c + FP_c)` | Of all windows predicted as class `c`, how many were actually `c`? |
| **Per-class Recall** | `TP_c / (TP_c + FN_c)` | Of all actual class `c` windows, how many did we correctly catch? |
| **Per-class F1** | `2PR / (P + R)` | Harmonic mean of precision and recall for class `c` |

**Confusion matrix** is printed at the end of training:

```
True\Pred   Quiet      B      C      M      X
---------------------------------------------
Quiet         800     12      5      2      0     ← mostly correct
B              30    390     20      3      0
C              15     25    310     12      1
M               5      8     20     45      2
X               2      1      3      4      6     ← hardest class
```

The confusion matrix reveals *where* the model fails — e.g. whether it confuses M and X, or dumps everything into "Quiet".

**Key functions:**

```python
# Full eval pass — returns (mean_loss, metrics_dict)
val_loss, metrics = evaluate(model, val_loader, criterion, device)

# Just compute metrics from lists of ints
metrics = compute_metrics(all_preds, all_labels)

# Pretty-print
print(format_metrics_table(metrics))
print(confusion_matrix_str(all_preds, all_labels))
```

---

### 4.8 `__init__.py`

Exports everything public so any import from the package works cleanly:

```python
from src.HPINA.models.baseline_tcn import HelioForgeTCN           # full model
from src.HPINA.models.baseline_tcn import build_weighted_criterion  # loss
from src.HPINA.models.baseline_tcn import evaluate                  # eval loop
from src.HPINA.models.baseline_tcn import compute_metrics           # metrics
from src.HPINA.models.baseline_tcn import CLASS_NAMES               # ["Quiet","B","C","M","X"]
```

---

## 5. Training Loop

`scripts/train.py` implements the complete training pipeline.

### Data flow

```
.pt file on disk
      ↓  torch.load()
TensorDataset(X, y)
      ↓  DataLoader(shuffle=True, batch_size=32)
(X_batch, y_batch)   shape: (32, 32, 512) and (32,)
      ↓  model(X_batch)
logits               shape: (32, 5)
      ↓  criterion(logits, y_batch)
scalar loss
      ↓  loss.backward()
gradients on all ~8.4M parameters
      ↓  optimizer.step()
updated weights
```

### One epoch (train phase)

```python
model.train()
for X_batch, y_batch in train_loader:
    optimizer.zero_grad()                              # clear old gradients
    logits = model(X_batch.to(device))                 # forward pass
    loss   = criterion(logits, y_batch.to(device))     # weighted cross-entropy
    loss.backward()                                    # backpropagation
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent explosion
    optimizer.step()                                   # weight update
```

### One epoch (validation phase)

```python
model.eval()
with torch.no_grad():    # no gradients needed — saves memory and time
    val_loss, metrics = evaluate(model, val_loader, criterion, device)
```

### Optimizer: AdamW

AdamW (Adam with decoupled Weight Decay) is the standard choice for deep learning:
- **Adam** component: per-parameter adaptive learning rates based on gradient history
- **Weight decay** component: L2 regularisation applied directly to weights (not to gradients like in standard Adam), preventing overfitting

```
lr=1e-3, weight_decay=1e-4
```

### Scheduler: ReduceLROnPlateau

Monitors `val_loss`. If it doesn't improve for `patience // 2 = 7` epochs, the learning rate is halved:

```
lr  ×= 0.5  (factor)
```

This allows aggressive initial learning followed by fine-tuning as the model approaches convergence.

### Gradient clipping

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Caps the global gradient norm at 1.0. Prevents the loss from exploding to `nan` — especially important with deep dilated convolutions where gradients can accumulate multiplicatively across 8 blocks.

### Early stopping

If `val_loss` doesn't improve for `patience=15` consecutive epochs, training stops automatically. This prevents wasting compute and prevents overfitting.

```
patience_count += 1  (each epoch without improvement)
if patience_count >= 15: break
```

### Console output per epoch

```
Epoch  12/80  train_loss: 0.8821  val_loss: 0.9104  macro_f1: 0.4312  acc: 0.6820  lr: 1.00e-03  [8.3s]
  ✓ Checkpoint saved  (best val_loss = 0.9104)
```

---

## 6. The Maths

### 6.1 Cross-Entropy Loss

For a single example with ground-truth class $c$:

$$\mathcal{L}_{CE} = -\log\left(\frac{e^{z_c}}{\sum_{j=0}^{4} e^{z_j}}\right)$$

where $z_j$ are the raw logits. The denominator is the softmax normalization. This loss is maximally penalised when the model assigns very low probability to the correct class.

With class weights:

$$\mathcal{L}_{\text{weighted}} = -w_c \cdot \log\left(\text{softmax}(z)_c\right)$$

### 6.2 Receptive Field

The number of input timesteps that influence a single output timestep:

$$\text{RF} = 1 + (k - 1) \cdot \sum_{i=0}^{K-1} d_i$$

For HelioForge: $k=3$, $K=8$ blocks, $d_i = 2^i$:

$$\text{RF} = 1 + 2 \times (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128) = 1 + 2 \times 255 = \mathbf{511}$$

### 6.3 Macro F1

$$\text{Macro F1} = \frac{1}{|C|} \sum_{c \in C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

where $P_c = \frac{TP_c}{TP_c + FP_c}$ and $R_c = \frac{TP_c}{TP_c + FN_c}$.

Averaging *before* weighting by class frequency means a model that misses all X-class events is heavily penalised, even if X has only 16 training samples.

### 6.4 Causal Padding

For a kernel of size $k$ with dilation $d$:
- Left pad: $P = (k-1) \cdot d$ zeros
- Conv output length: $L + P - d \cdot (k-1) = L$ ✓ (sequence length preserved)

---

## 7. Hyperparameter Reference

| Parameter | Default | Effect |
|-----------|---------|--------|
| `lr` | `1e-3` | Initial learning rate. Too high → NaN loss. Too low → slow convergence. |
| `weight_decay` | `1e-4` | L2 regularisation strength via AdamW. Prevents overfitting. |
| `batch_size` | `32` | Windows per gradient step. Larger = more stable gradients, more memory. |
| `n_epochs` | `80` | Max training epochs before forced stop. |
| `dropout` | `0.2` | Encoder dropout. Higher → more regularisation, slower convergence. |
| `norm_type` | `"batch"` | Normalisation: `"batch"`, `"layer"`, or `"none"`. |
| `patience` | `15` | Early stopping patience. Increase if loss is noisy. |
| `grad_clip` | `1.0` | Max gradient norm. Set lower if loss explodes. |
| `label_smooth` | `0.0` | Label smoothing epsilon (e.g. `0.1` softens targets slightly). |

### Ablation suggestions

To systematically test what matters most, vary one parameter at a time:

```bash
# Ablation 1: Layer norm vs batch norm
python scripts/train.py --norm-type layer  --run-name ablate_layernorm

# Ablation 2: Higher dropout
python scripts/train.py --dropout 0.4  --run-name ablate_dropout04

# Ablation 3: Label smoothing
python scripts/train.py --label-smooth 0.1  --run-name ablate_labelsmooth

# Ablation 4: Smaller LR
python scripts/train.py --lr 3e-4  --run-name ablate_lr3e4
```

---

## 8. Checkpointing Strategy

Three checkpoints are saved per run into `experiments/baseline_tcn/runs/<run_name>/checkpoints/`:

| File | Saved when | Use for |
|---|---|---|
| `best_val_loss.pt` | Val loss reaches a new minimum | Final deployment model (most stable) |
| `best_macro_f1.pt` | Macro F1 reaches a new maximum | Best overall class-balanced performance |
| `final.pt` | End of training (regardless of performance) | Reproducing last epoch state |

Each checkpoint contains:
```python
{
  "epoch"         : int,
  "model_state"   : model.state_dict(),
  "optimizer_state": optimizer.state_dict(),
  "val_loss"      : float,
  "val_metrics"   : dict,
  "args"          : dict,   # all CLI args used to produce this run
}
```

**To resume or run inference from a checkpoint:**
```python
ckpt  = torch.load("experiments/baseline_tcn/runs/baseline_v1/checkpoints/best_macro_f1.pt")
model = HelioForgeTCN(**{k: ckpt["args"][k] for k in ["in_channels", "n_classes", "dropout", "norm_type"]})
model.load_state_dict(ckpt["model_state"])
model.eval()
```

---

## 9. Interpreting Results

### What good training looks like

```
Epoch  1/80   train_loss: 1.612  val_loss: 1.598   macro_f1: 0.200   ← random, ≈log(5)
Epoch  5/80   train_loss: 1.201  val_loss: 1.189   macro_f1: 0.310   ← learning
Epoch 10/80   train_loss: 0.887  val_loss: 0.901   macro_f1: 0.421
Epoch 20/80   train_loss: 0.623  val_loss: 0.641   macro_f1: 0.511
Epoch 30/80   train_loss: 0.521  val_loss: 0.539   macro_f1: 0.552
Epoch 40/80   train_loss: 0.498  val_loss: 0.541   macro_f1: 0.558   ← LR decay kicks in
Epoch 50/80   train_loss: 0.489  val_loss: 0.536   macro_f1: 0.562   ← convergence
```

### Red flags

| Symptom | Cause | Fix |
|---------|-------|-----|
| Loss → `nan` immediately | Gradient explosion | Lower `lr`, lower `grad_clip` |
| Both losses stuck high after 15 epochs | Too small LR or data bug | Increase `lr`, verify data loading |
| `val_loss` rising while `train_loss` falls | Overfitting | Increase `dropout`, reduce model capacity |
| Macro F1 stuck at `0.2` (random) | Class weights not applied | Check `build_weighted_criterion` is used |
| X-class recall = 0.0 throughout | Model ignores rare class | Increase weight for X-class manually |

### Target performance (baseline)

| Metric | Minimum acceptable | Good |
|--------|-------------------|------|
| Accuracy | > 60% | > 75% |
| Macro F1 | > 0.40 | > 0.55 |
| X-class Recall | > 0.30 | > 0.50 |
| M-class Recall | > 0.40 | > 0.60 |

> **Note:** X-class recall is the most important single number operationally. Missing an X-class flare has severe real-world consequences. Prioritize `best_macro_f1.pt` over `best_val_loss.pt` for deployment.

---

## 10. Quick Command Reference

### Run all module tests locally
```bash
python scripts/test_baseline_tcn.py
```

### Train on EC2 (standard run)
```bash
git pull origin main
python scripts/train.py \
    --data-dir /opt/helioforge-ai/data/windows \
    --output-dir experiments/baseline_tcn/runs \
    --run-name baseline_v1 \
    --n-epochs 80 \
    --batch-size 32
```

### Train with layer normalization (ablation)
```bash
python scripts/train.py \
    --data-dir /opt/helioforge-ai/data/windows \
    --run-name ablate_layernorm \
    --norm-type layer \
    --n-epochs 80
```

### Inspect a saved checkpoint
```python
import torch
ckpt = torch.load("experiments/baseline_tcn/runs/baseline_v1/checkpoints/best_macro_f1.pt")
print(f"Epoch      : {ckpt['epoch']}")
print(f"Val loss   : {ckpt['val_loss']:.4f}")
print(f"Macro F1   : {ckpt['val_metrics']['macro_f1']:.4f}")
print(f"X recall   : {ckpt['val_metrics']['recall_X']:.4f}")
```

### Outputs per run
```
experiments/baseline_tcn/runs/<run_name>/
├── config.json              ← all CLI args used
├── train.log                ← full epoch-by-epoch log
├── history.json             ← all metrics per epoch (for plotting)
└── checkpoints/
    ├── best_val_loss.pt     ← lowest val loss
    ├── best_macro_f1.pt     ← highest macro F1
    └── final.pt             ← end of training
```

---

*HelioForge AI — Baseline TCN Complete Reference v1.0*
*Last updated: 2026-07-28*
