# TCN Engineering Guide
## From 3D Tensors to a Trained Model — HelioForge AI

---

> **Engineering Discipline:**  
> Every file gets written. Every function gets tested. Every assumption gets verified.  
> No step is skipped. No layer is connected until the one below it passes its test.

---

## Table of Contents

1. [Mental Model Before Anything Else](#chapter-0--mental-model-before-anything-else)
2. [Understanding the 3D Tensor](#chapter-1--understanding-the-3d-tensor)
3. [What is a TCN?](#chapter-2--what-is-a-tcn)
4. [Full TCN Architecture for HelioForge](#chapter-3--full-tcn-architecture-for-helioforge)
5. [How the TCN Learns](#chapter-4--how-the-tcn-learns)
6. [Complete File Structure](#chapter-5--the-complete-file-structure)
7. [Code Skeleton](#chapter-6--code-skeleton)
8. [Key Terms Cheat Sheet](#chapter-7--key-terms-cheat-sheet)
9. [Build Order (Law, Not Suggestion)](#chapter-8--the-build-order-law-not-suggestion)
10. [What Good Training Looks Like](#chapter-9--what-good-training-looks-like)

---

## Chapter 0 — Mental Model Before Anything Else

Before writing a single line of code, internalise this completely:

> **Every deep learning project is the same three steps, repeated forever:**
>
> 1. **Data** → Shape it into numbers a computer can digest
> 2. **Model** → A mathematical function that transforms those numbers
> 3. **Training** → Adjust the function until it stops being wrong

Everything in this document is detail *inside* those three steps.

---

## Chapter 1 — Understanding the 3D Tensor

### 1.1 Why 3D at All?

Your raw solar data is a flat table — rows are seconds, columns are features:

```
Time(s)   SoLEXS_COUNTS   HEL1OS_energy   rise_rate   ...
0         0.12            0.34            0.001       ...
1         0.13            0.35            0.002       ...
2         0.11            0.33            0.000       ...
...
86400     0.45            0.71            0.021       ...
```

A flat table is **2D**: `(T, F)` — T time rows, F feature columns.

A neural network that processes sequences needs to work on **chunks of time**, not the whole observation at once. You cut this table into overlapping windows of length `L = 512` seconds. Each window is still 2D: `(L, F)`. For `F=32` selected features that's `(512, 32)`.

Stack `N` such windows → **3D**: `(N, L, F)`.

PyTorch's `Conv1d` wants channels first → swap the last two dims → `(N, F, L)`.

That is the entire reason it is 3D. Nothing more mysterious than that.

---

### 1.2 Visualising Your Actual Tensor

> **Verified on EC2** by running `python scripts/demo.py --path /opt/helioforge-ai/data/windows/train_feat32_w512.pt`

The active production `train.pt` is the multivariate 32-feature tensor:

```
tensor.shape = (1874, 32, 512)

Axis 0  →  N = 1,874   windows      (independent training examples)
Axis 1  →  F = 32      features     (physics-informed selected feature channels)
Axis 2  →  L = 512     timesteps    (512 timesteps × 32s stride ≈ 4.5 hours per window)
```

**Verified Split Shapes & Tensors:**

| Split | File | Shape | Size |
|-------|------|-------|------|
| Train | `train_feat32_w512.pt` | `torch.Size([1874, 32, 512])` | 117.1 MB |
| Val   | `val_feat32_w512.pt`   | `torch.Size([389, 32, 512])`  | 24.3 MB  |
| Test  | `test_feat32_w512.pt`  | `torch.Size([389, 32, 512])`  | 24.3 MB  |
| Scaler | `scaler_f32_w512.json` | — (fitted on train split only) | — |

Picture it as a stack of 1,874 sheets of graph paper:

```
┌────────────────────────────────────────────────────┐
│  Sheet 0   (Window at t=0)                         │
│  ┌──────────────────────────────────────────────┐  │
│  │ Feature 0  (soft_mean)  ──────────────────── │  │  min=0.00 max=1.00 mean=0.23
│  │ Feature 1  (soft_std)   ──────────────────── │  │  min=0.00 max=1.00 mean=0.25
│  │ ...                                          │  │
│  │ Feature 31 (soft_hard_ratio) ─────────────── │  │  min=0.00 max=1.00 mean=0.01
│  └──────────────────────────────────────────────┘  │
│              ← 512 timesteps wide →                │
└────────────────────────────────────────────────────┘
          ↓ repeated 1,873 more times ↓
```

**Global statistics confirmed from `demo.py`:**

```
dtype   : torch.float32
min     : 0.000000   (normalised — 0% of range)
max     : 1.000000   (normalised — 100% of range)
mean    : 0.232685
std     : 0.253854
NaNs    : 0          ← clean
Infs    : 0          ← clean
Memory  : 117.1 MB   (1874 × 32 × 512 × 4 bytes)
```

Each "sheet" is one training example. The model sees one sheet at a time (or a batch of sheets). It never sees all 1,874 at once.

---

### 1.3 The Sliding Window Mechanism (How Windows Are Made)

```
Full observation  T = 86,368 timesteps across 32 paired observations
Window size       L = 512
Stride            S = 32

Window 0:   timesteps [0    → 511]
Window 1:   timesteps [32   → 543]
Window 2:   timesteps [64   → 575]
...
Window k:   timesteps [k*32 → k*32 + 511]
```

With stride 32 on 86,368 timesteps split 70/15/15:

```
N_train = 1,874 windows
N_val   = 389 windows
N_test  = 389 windows
```

**Stride controls overlap.** Stride=32 means each window shares 480 of its 512 timesteps with the next one. High overlap = more training data, but neighbouring windows are highly correlated. This is intentional — solar flare onset is rare, so you sample densely.

---

### 1.4 What Each Number in the Tensor Means

```
tensor[window_idx, feature_idx, timestep_idx]
```

Real values verified from `demo.py` on EC2 (`train_feat32_w512.pt`):

```
tensor[   0,  0,   0]  =  0.00188   ← first window, soft_mean, first second
tensor[   0,  0, 511]  =  0.00293   ← first window, soft_mean, last second
tensor[   0, 31,   0]  =  0.00009   ← first window, soft_hard_ratio, first second
tensor[1873,  0,   0]  =  0.00802   ← last window,  soft_mean, first second
tensor[ 937, 16, 100]  =  0.00000   ← middle window #937, feature #16, second 100
```

All values are normalised to `[0, 1]` per channel using per-observation min/max. So `0.44` means "SoLEXS flux was at 44% of its observed range at timestep 511 of window 0."

---

### 1.5 Strict Evaluation Protocol — Observation-Level Splitting

> ⚠️ **CRITICAL DATA LEAKAGE WARNING**  
> You **must** split full observations **FIRST** before extracting sliding windows.

**Correct Protocol** (implemented in `datasets.py`):

1. Split observations into Train (70%), Val (15%), Test (15%) **by observation ID/date**.
2. Perform window extraction strictly within each split independently.
3. Fit scalers **only on the Train split** observations and transform Val/Test with those fixed bounds.

**Flawed Protocol (DO NOT USE):**  
Concatenating all feature rows from all observations into one table, extracting windows, and then splitting windows. Because overlapping windows share up to 480 timesteps with adjacent windows, window-level splitting causes **massive data leakage** where test windows overlap with training windows.

---

## Chapter 2 — What is a TCN?

### 2.1 The Family Tree

```
Neural Networks
└── Sequence Models (process ordered data over time)
    ├── RNN  (Recurrent Neural Network) — 1986
    │   └── LSTM / GRU — 1997 / 2014
    └── TCN  (Temporal Convolutional Network) — 2018
```

**RNNs** process sequences one step at a time, carrying a "memory state" forward. This is slow, hard to parallelise, and suffers from vanishing gradients (the model forgets things that happened long ago).

**TCN** processes the entire sequence at once using convolutions. It is faster, parallelisable, and through dilated convolutions, it can see very long history without being computationally expensive.

---

### 2.2 The Core Idea — Convolution Over Time

A regular 1D convolution kernel of size `k=3` applied to a single feature:

```
Feature signal:   [0.1,  0.3,  0.2,  0.5,  0.4,  0.7,  0.6]
Kernel weights:   [w0,   w1,   w2]

At position t=3:
    output = w0 * 0.2  +  w1 * 0.5  +  w2 * 0.4
           = learned weighted combination of 3 consecutive timesteps
```

Do this across all 512 timesteps, across all 32 features simultaneously → you get a new sequence of learned representations.

Stack multiple such layers → each layer sees patterns in the layer below → progressively more abstract temporal features.

---

### 2.3 The Causal Constraint — The Most Critical Rule

A normal `Conv1d` kernel at position `t` can see `t-1, t, t+1` — past **and future**.

This is **data leakage**. At deployment time, `t+1` doesn't exist yet. The model would be cheating during training and fail in the real world.

**Causal convolution:** only look at `t-1, t` (the past). Never the future.

**Implementation:** left-pad the input by `kernel_size - 1` zeros before convolution:

```
Normal padding (pad both sides):
  [0, input[0], input[1], ..., input[511], 0]

Causal padding (pad left only):
  [0, 0, input[0], input[1], ..., input[511]]
       ↑ these zeros represent "before the window began"
```

After convolution, trim the right-side overhang so output length = input length.

---

### 2.4 Dilated Convolutions — Seeing Far Without Being Expensive

A kernel of size 3 sees 3 timesteps. To see 512 timesteps, you'd need `kernel_size=512` — computationally brutal.

**Dilation** is the fix. Dilation `d` means the kernel skips `d-1` positions between each element:

```
Dilation 1  (normal):    looks at t-2,  t-1,  t        → receptive field = 3
Dilation 2:              looks at t-4,  t-2,  t        → receptive field = 5
Dilation 4:              looks at t-8,  t-4,  t        → receptive field = 9
Dilation 8:              looks at t-16, t-8,  t        → receptive field = 17
Dilation 16:             looks at t-32, t-16, t        → receptive field = 33
```

**Standard TCN Receptive Field Formula:**

```
RF = 1 + (k − 1) × Σ(i=0 to K-1) d_i
```

Where `k` is kernel size (`k=3`) and `d_i = 2^i` for `K=8` layers (`d = [1,2,4,8,16,32,64,128]`):

```
RF = 1 + (3 − 1) × (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128)
   = 1 + 2 × 255
   = 511 timesteps
```

With 8 layers and `kernel_size=3`, the receptive field is **511 timesteps** — covering 99.8% of our `L=512` window size.

**Visualised:**

```
Layer 1 (d=1):   ●───●───●                                  sees 3 steps
Layer 2 (d=2):   ●───────●───────●                          sees 5 steps
Layer 3 (d=4):   ●───────────────●───────────────●          sees 9 steps
Layer 4 (d=8):   ●───────────────────────────────●─────...  sees 17 steps
...
Layer 8 (d=128): ●──────────────────────────────────────... sees 511 steps
```

---

### 2.5 Residual Connections — How Deep Networks Don't Break

When you stack many layers, gradients shrink as they travel backwards through each layer. By 10 layers deep, the gradient is essentially zero — the early layers stop learning. This is called **vanishing gradient**.

The fix: **skip connections** (residual connections). Route the input directly around the conv layers and add it to the output:

```
        ┌────────────────────────────────────────┐
        │           Residual Block               │
input ──┼──► Conv1d ──► Norm ──► ReLU ──► Conv1d ──► + ──► output
        │                                         ▲
        └─────────────────────────────────────────┘
                      skip connection: input added directly
```

The gradient can now flow directly through the skip path without passing through any layers. Early layers always get a strong gradient signal.

---

## Chapter 3 — Full TCN Architecture for HelioForge

### 3.1 The Complete Picture (Baseline TCN Classifier)

> 📌 **Architecture Note:** The architecture below (`Input → TCN → Classifier`) is the **Baseline TCN Model**. Future iterations of HPINA will expand beyond this baseline by integrating multi-branch spectral processing, temporal attention mechanisms, and physics-informed differential constraints.

```
Input Tensor                        (Batch, 32, 512)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  TCN Encoder (Progressive Channel Widening)         │
│                                                     │
│  ResidualBlock 1 (d=1)    (Batch, 128, 512)         │
│  ResidualBlock 2 (d=2)    (Batch, 256, 512)         │
│  ResidualBlock 3 (d=4)    (Batch, 256, 512)         │
│  ResidualBlock 4 (d=8)    (Batch, 512, 512)         │
│  ResidualBlock 5 (d=16)   (Batch, 512, 512)         │
│  ResidualBlock 6 (d=32)   (Batch, 512, 512)         │
│  ResidualBlock 7 (d=64)   (Batch, 512, 512)         │
│  ResidualBlock 8 (d=128)  (Batch, 512, 512)         │
│                                                     │
│  Total Parameters: ~2,150,000  (2.1M+ capacity)     │
└─────────────────────────────────────────────────────┘
        │
        ▼
  Global Average Pool     (Batch, 512)       ← collapse time dimension
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Classifier Head                                    │
│  Linear(512 → 256) → ReLU → Dropout(0.3)           │
│  Linear(256 → 128) → ReLU → Dropout(0.3)           │
│  Linear(128 → n_classes)                            │
└─────────────────────────────────────────────────────┘
        │
        ▼
Output Logits               (Batch, n_classes)
```

---

### 3.2 Each ResidualBlock in Detail & Normalisation Options

> 💡 **Normalisation Strategy Note:** While `BatchNorm1d` is shown in the block diagram, TCNs are frequently trained with varying batch sizes or non-stationary time-series statistics. Benchmark these three alternatives during hyperparameter tuning:
>
> - **Weight Normalisation (`weight_norm`):** Used in the original Bai et al. (2018) paper; decouples weight vector magnitude from direction.
> - **Layer Normalisation (`LayerNorm` / `GroupNorm`):** Normalises across channels per sample — invariant to batch size.
> - **Batch Normalisation (`BatchNorm1d`):** Standard for fixed, larger batch sizes.

```
input: (Batch, C_in, L)
        │
        ├──────────────────────────────────────────────┐  skip path
        │                                              │
        ▼                                              │
  CausalConv1d(C_in → C_out, kernel=3, dilation=d)    │
        ▼                                              │
  Norm (WeightNorm / LayerNorm / BatchNorm1d)          │
        ▼                                              │
  ReLU                                                 │
        ▼                                              │
  Dropout(p=0.2)                                       │
        ▼                                              │
  CausalConv1d(C_out → C_out, kernel=3, dilation=d)   │
        ▼                                              │
  Norm (WeightNorm / LayerNorm / BatchNorm1d)          │
        ▼                                              │
        + ◄────────────────────────────────────────────┘
        │    if C_in != C_out: skip uses a 1×1 conv to match channels
        ▼
  ReLU
        │
        ▼
output: (Batch, C_out, L)    ← same length, different channels
```

---

### 3.3 CausalConv1d Step by Step

```
input:  (Batch, C_in, L)

Step 1: left-pad by (kernel_size - 1) * dilation zeros
        padded: (Batch, C_in, L + pad)

Step 2: apply standard Conv1d
        output: (Batch, C_out, L + 1)    ← one extra on the right

Step 3: trim the last element
        output: (Batch, C_out, L)        ← same length as input ✓

Result: position t in output saw only t, t-d, t-2d, ... (all in the past) ✓
```

---

## Chapter 4 — How the TCN Learns

### 4.1 The Forward Pass (Prediction Direction)

```
1. Feed one batch: tensor of shape (32, 32, 512)
   32 windows, 32 physics feature channels, 512 timesteps (~4.5h window at stride=32)

2. Progressive Channel Widening (32 → 128 → 256 → 512 channels across 8 blocks):
   - Layer 1 (d=1,   ch=128):  finds 3-step local spike patterns
   - Layer 3 (d=4,   ch=256):  finds 9-step medium-term trends
   - Layer 8 (d=128, ch=512):  finds 257-step long-term flux evolution (~2.1M total parameters)

3. Global Average Pool: collapses 512 timesteps → 1 summary vector (512-dim)
   This is the model's high-capacity representation of this ~4.5-hour window

4. Classifier Head: maps 512-dim → 256 → 128 → class probabilities
   e.g. [0.02, 0.05, 0.73, 0.15, 0.05]
         B     C     M     X     No-flare

5. argmax → predicted class: M-class flare
```

---

### 4.2 The Backward Pass (Learning Direction)

```
1. Prediction:   [0.02, 0.05, 0.73, 0.15, 0.05]
   Ground truth: [0,    0,    0,    1,    0   ]   ← actually X-class

2. Compute loss (Cross-Entropy):
   loss = -log(0.15) = 1.90    ← large = very wrong

3. Backpropagate:
   PyTorch computes dLoss/dWeight for every single parameter automatically

4. Optimizer step (AdamW):
   weight = weight - learning_rate * gradient
   All ~2 million weights nudge slightly toward the correct answer

5. Repeat for next batch → loss gradually decreases → model learns
```

---

### 4.3 Why Batches and Not the Whole Dataset?

```
Dataset: 1,874 training windows
Batch size: 32

One epoch = 1874 / 32 = ~59 gradient update steps
50 epochs  = 59 × 50  = 2,950 weight updates
```

**Reasons:**

- **Memory** — 1,874 tensors at once won't fit in GPU RAM efficiently
- **Noise is good** — each batch is a slightly different sample, preventing memorisation
- **Speed** — GPU parallelism is optimised for fixed-size blocks

---

### 4.4 Overfitting vs Underfitting — The Core Training Tension

```
Underfitting                  Overfitting                  Just right
─────────────────────────────────────────────────────────────────────
train loss: high              train loss: low ✓            train loss: low ✓
val loss:   high              val loss:   high ↑           val loss:   low ✓

Model too simple              Memorised training data       Generalised
→ increase model capacity     → add dropout, reduce size    → stop here, ship it
```

This is exactly why you have `val.pt`. Every epoch you check val loss. If train loss keeps dropping but val loss stops → overfitting → stop (early stopping).

---

## Chapter 5 — The Complete File Structure

```
src/HPINA/models/baseline_tcn/
│
├── __init__.py              ← exports HelioForgeTCN
├── causal_conv.py           ← Step 1: single causal 1D conv layer
├── residual_block.py        ← Step 2: CausalConv1d × 2 + skip + norm
├── tcn_encoder.py           ← Step 3: 8 ResidualBlocks, dilation 1→128
├── classifier.py            ← Step 4: GAP + linear head → class logits
└── model.py                 ← Step 5: wires Encoder + Classifier

scripts/
└── train_tcn.py             ← Steps 6+7: DataLoader, train loop, checkpointing
```

---

## Chapter 6 — Code Skeleton

### `causal_conv.py`

```python
import torch.nn as nn


class CausalConv1d(nn.Module):
    """
    1D convolution that only looks at the past.

    Left-pads by (kernel_size - 1) * dilation zeros.
    Output length == input length. Always.

    Args:
        in_channels  (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        kernel_size  (int): Convolution kernel size (typically 3).
        dilation     (int): Dilation factor; set to 2^layer_index.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

    def forward(self, x):
        # x: (Batch, C_in, L)
        out = self.conv(x)
        return out[:, :, : -self.padding]  # trim right overhang → causal
        # out: (Batch, C_out, L) — length preserved ✓
```

---

### `residual_block.py`

```python
import torch.nn as nn
from .causal_conv import CausalConv1d


class ResidualBlock(nn.Module):
    """
    Two causal convolutions with normalisation, ReLU, Dropout,
    and a skip connection (with 1×1 projection if channels change).

    Args:
        in_channels  (int):   Input channel count.
        out_channels (int):   Output channel count.
        kernel_size  (int):   Convolution kernel size (typically 3).
        dilation     (int):   Dilation factor for this block.
        dropout      (float): Dropout probability (default 0.2).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels,  out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.drop  = nn.Dropout(dropout)
        self.relu  = nn.ReLU()
        # 1×1 projection only when channel dimensions differ
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        # x: (Batch, C_in, L)
        residual = self.skip(x)                      # skip path
        out = self.relu(self.norm1(self.conv1(x)))   # conv → norm → relu
        out = self.drop(out)                         # regularise
        out = self.norm2(self.conv2(out))            # conv → norm (no relu yet)
        return self.relu(out + residual)             # add skip, then relu
        # out: (Batch, C_out, L) ✓
```

---

### `tcn_encoder.py`

```python
import torch.nn as nn
from .residual_block import ResidualBlock


class TCNEncoder(nn.Module):
    """
    Stacks ResidualBlocks with exponentially growing dilation:
      dilations = [1, 2, 4, 8, 16, 32, 64, 128]

    Progressive Channel Widening Schedule (32 → 128 → 256 → 512):
      channel_schedule = [128, 256, 256, 512, 512, 512, 512, 512]
      ~2.1M parameters for high-capacity production learning.

    Receptive Field (kernel=3, 8 layers):
      RF = 1 + (3 - 1) × (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128) = 511 timesteps
      Covers 99.8% of L=512 window.

    Args:
        in_channels      (int):       Number of input feature channels (32 for HelioForge).
        channel_schedule (list[int]): Output channel count per residual block.
        kernel_size      (int):       Convolution kernel size (default 3).
        dropout          (float):     Dropout probability per block (default 0.2).
    """

    def __init__(
        self,
        in_channels: int = 32,
        channel_schedule: list = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if channel_schedule is None:
            channel_schedule = [128, 256, 256, 512, 512, 512, 512, 512]

        layers = []
        prev_channels = in_channels

        for i, out_channels in enumerate(channel_schedule):
            dilation = 2 ** i
            layers.append(
                ResidualBlock(prev_channels, out_channels, kernel_size, dilation, dropout)
            )
            prev_channels = out_channels

        self.network    = nn.Sequential(*layers)
        self.out_channels = prev_channels  # expose for ClassifierHead

    def forward(self, x):
        # x: (Batch, 32, 512)
        return self.network(x)
        # out: (Batch, 512, 512)
```

---

### `classifier.py`

```python
import torch.nn as nn


class ClassifierHead(nn.Module):
    """
    Global Average Pool collapses the time dimension (L timesteps → 1 vector),
    then linear layers map the compact representation to class logits.

    Args:
        in_features (int):   Channel count from TCNEncoder (512 for HelioForge).
        n_classes   (int):   Number of output classes (5 for flare classification).
        dropout     (float): Dropout probability in the MLP (default 0.3).
    """

    def __init__(
        self,
        in_features: int = 512,
        n_classes: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.gap  = nn.AdaptiveAvgPool1d(1)  # (B, 512, L) → (B, 512, 1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),         nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        # x: (Batch, 512, L)
        return self.head(self.gap(x))
        # out: (Batch, n_classes) — raw logits
```

---

### `model.py`

```python
import torch.nn as nn
from .tcn_encoder import TCNEncoder
from .classifier  import ClassifierHead


class HelioForgeTCN(nn.Module):
    """
    Full High-Capacity Production TCN = TCNEncoder (2.1M params) + ClassifierHead.

    Input:  (Batch, F=32, L=512)   e.g. (32, 32, 512)
    Output: (Batch, n_classes)     raw logits — apply softmax for probabilities

    Args:
        in_channels      (int):       Input feature channels (32 for HelioForge).
        n_classes        (int):       Number of output classes (5 for flare types).
        channel_schedule (list[int]): Per-block output channels for TCNEncoder.
        kernel_size      (int):       Convolution kernel size (default 3).
        dropout          (float):     Dropout probability (default 0.2).
    """

    def __init__(
        self,
        in_channels: int = 32,
        n_classes: int = 5,
        channel_schedule: list = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder    = TCNEncoder(in_channels, channel_schedule, kernel_size, dropout)
        self.classifier = ClassifierHead(self.encoder.out_channels, n_classes, dropout)

    def forward(self, x):
        # x:   (Batch, 32, 512)
        # out: (Batch, n_classes)
        return self.classifier(self.encoder(x))
```

---

### `train_tcn.py` — The Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.HPINA.models.baseline_tcn import HelioForgeTCN

# ── Device & model ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = HelioForgeTCN(in_channels=32, n_classes=5).to(device)

# ── Optimizer + loss + scheduler ───────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

# ── DataLoaders ────────────────────────────────────────────────────────────────
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=4)

best_val_loss = float("inf")

# ── Epoch loop ─────────────────────────────────────────────────────────────────
for epoch in range(n_epochs):

    # ── Train ──────────────────────────────────────────────────────────────────
    model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()                                        # 1. clear old gradients
        logits = model(X_batch)                                      # 2. forward pass
        loss   = criterion(logits, y_batch)                          # 3. compute loss
        loss.backward()                                              # 4. backpropagate
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 5. clip gradients
        optimizer.step()                                             # 6. update weights
        train_loss += loss.item()

    # ── Validate ────────────────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for X_val, y_val in val_loader:
            logits    = model(X_val.to(device))
            val_loss += criterion(logits, y_val.to(device)).item()

    train_loss /= len(train_loader)
    val_loss   /= len(val_loader)
    scheduler.step(val_loss)

    print(
        f"Epoch {epoch + 1:3d}/{n_epochs}  "
        f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}"
    )

    # ── Checkpoint best model ───────────────────────────────────────────────────
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "checkpoints/best_model.pt")
        print("  ✓ Checkpoint saved")
```

---

## Chapter 7 — Key Terms Cheat Sheet

| Term | What It Means |
|------|---------------|
| **Tensor** | Multi-dimensional array. 3D = `(N, Channels, Time)` |
| **Forward pass** | Running data through the model to get predictions |
| **Backward pass** | Computing gradients — how wrong each weight was |
| **Loss** | Single number measuring wrongness. Goal: minimise it |
| **Gradient** | Direction and magnitude to adjust each weight |
| **Optimizer** | Updates weights using gradients. AdamW is the best default |
| **Learning rate** | Size of each weight update. Too high → diverges. Too low → never converges |
| **Epoch** | One full pass through the entire training dataset |
| **Batch** | Small subset processed together. Typically 16–128 samples |
| **Overfitting** | Memorises training data, fails on new data |
| **Dropout** | Randomly zeros neurons during training — prevents overfitting |
| **BatchNorm** | Normalises layer inputs — faster, more stable training |
| **Dilation** | Kernel element spacing — wider receptive field, no extra parameters |
| **Causal** | Only looks at the past. No future leakage |
| **Receptive field** | How far back in time the model can see |
| **Skip connection** | Direct path around convolutions — prevents vanishing gradients |
| **Global Avg Pool** | Collapses time dimension to a fixed-size vector |
| **Logits** | Raw output before softmax. Unnormalised class scores |
| **Softmax** | Converts logits to probabilities that sum to 1.0 |
| **Cross-Entropy** | Classification loss. Penalises confident wrong predictions heavily |

---

## Chapter 8 — The Build Order (Law, Not Suggestion)

> **Rule:** Test every single layer in isolation before connecting it to anything.  
> A bug in `causal_conv.py` will silently corrupt every downstream result.  
> **Always build and verify bottom-up.**

| Day | File | Acceptance Test |
|-----|------|-----------------|
| Day 1 | `causal_conv.py` | `(2, 32, 512)` in → `(2, 128, 512)` out. Length preserved? |
| Day 1 | `residual_block.py` | Skip works, channels match, output shape unchanged |
| Day 2 | `tcn_encoder.py` | 8 blocks stack cleanly, receptive field == 511 |
| Day 2 | `classifier.py` | `(2, 512, 512)` → `(2, 5)` logits |
| Day 3 | `model.py` | `(2, 32, 512)` → `(2, 5)` end-to-end in one line |
| Day 3 | `train_tcn.py` | 1 epoch runs, loss is a finite number and is decreasing |
| Day 4+ | Tune | Adjust `lr`, `batch_size`, `channel_schedule`, normalisation |

---

## Chapter 9 — What Good Training Looks Like

```
Epoch  1/50  |  train_loss: 1.612  val_loss: 1.598   random, ≈ log(5 classes)
Epoch  5/50  |  train_loss: 1.201  val_loss: 1.189   learning
Epoch 10/50  |  train_loss: 0.887  val_loss: 0.901   good — val close to train
Epoch 20/50  |  train_loss: 0.623  val_loss: 0.641   converging
Epoch 30/50  |  train_loss: 0.521  val_loss: 0.539   near convergence
Epoch 40/50  |  train_loss: 0.498  val_loss: 0.541   val plateau → lr decay kicks in
Epoch 50/50  |  train_loss: 0.489  val_loss: 0.536   done
```

### Red Flags to Watch For

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| `val_loss` rising while `train_loss` falls | Overfitting | Add dropout, reduce model size |
| Both losses stuck high after 10 epochs | Underfitting or data bug | Increase `lr` or check data pipeline |
| Loss → `nan` | Gradient explosion | Clip gradients, lower learning rate |
| Both losses high after 30+ epochs | Model too small | Increase `hidden_channels` or `n_layers` |
| `val_loss ≈ train_loss`, both low | **Perfect** | Ship it |

---

## Summary

You have clean 3D tensors of shape `(N=1874, F=32, L=512)`.

The TCN reads each tensor left-to-right using **causal convolutions** (only the past, never the future), builds progressively abstract temporal patterns through **8 dilated residual blocks** (seeing up to 511 timesteps of history), collapses the 512-timestep sequence into a single 512-dim vector via **global average pooling**, and maps that vector to a flare class prediction via **linear layers**.

Training adjusts every weight across **2,950+ gradient steps** to minimise the gap between prediction and ground truth.

Build bottom-up. Test each piece in isolation. Connect them in order.

---

*HelioForge AI — TCN Engineering Guide v1.0*