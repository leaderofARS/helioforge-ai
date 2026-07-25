# Baseline TCN Experimentation Directory

This directory hosts baseline experimentation assets, checkpoints, runs, and ablations for the Temporal Convolutional Network (TCN) solar flare forecasting model.

---

## Current Progress & Milestones

### 1. Data Engineering & Preprocessing (Completed & Frozen)
* **Feature Engineering (Stage 3):** Engineered 45 candidate features across 32 paired HEL1OS/SoLEXS observations (86,368 continuous timesteps). Variance and correlation filters selected 32 non-redundant physics features ($F' = 32$), exported to `selected_features.csv`.
* **Stratified Split & Window Generation (Stage 4):**
  * Sliced features into 3D sequence tensors of shape `(N, 32, 512)`.
  * Implemented **Stratified Observation-Level Splitting** grouped by peak flare intensity class (Quiet, B, C, M, X) to preserve observation boundaries (0 sequence leakage) while guaranteeing representation of rare M and X-class events in all splits.
  * **Dataset frozen** under versioned subdirectory `data/windows_fifth/`:
    * **Train:** `train_feat32_w512.pt` shape `torch.Size([1840, 32, 512])` (includes 16 X-class windows)
    * **Val:** `val_feat32_w512.pt` shape `torch.Size([406, 32, 512])` (includes 18 X-class windows)
    * **Test:** `test_feat32_w512.pt` shape `torch.Size([406, 32, 512])` (includes 18 X-class windows)

### 2. Pre-Training Diagnostics (Completed & Verified)
* Implemented `scripts/verify_pre_training.py` which executes:
  1. Per-feature normalization stats (confirming clean $[0, 1]$ MinMax bounds with 0 NaNs / Infs).
  2. Split integrity verification (checking observation-level split boundary overlap).
  3. Class imbalance analysis (reconstructing SoLEXS count rates to recommend weighted Cross-Entropy Loss).
  4. PyTorch DataLoader verification.

### 3. Baseline TCN Architecture (Completed Components)
Implemented baseline modules under `src/HPINA/models/baseline_tcn/`:
* **Causal 1D Convolution (`causal_conv.py`):** Ensures temporal causality by applying left-padding of size $P = (\text{kernel\_size} - 1) \times \text{dilation}$ using `F.pad` before standard conv.
* **Temporal Residual Block (`residual_block.py`):** Features two dilated causal convolutions, 1D normalization (`BatchNorm1d`, `LayerNorm1d`, or `Identity`), ReLU activations, dropout, and a $1 \times 1$ conv shortcut projection to align channels if dimensions change.
* **TCN Encoder (`tcn_encoder.py`):** Integrates 8 sequential residual blocks following a progressive channel widening schedule:
  $$32 \to 128 \to 256 \to 256 \to 512 \to 512 \to 512 \to 512 \to 512$$
  with exponential dilations $[1, 2, 4, 8, 16, 32, 64, 128]$.
  * **Receptive Field:** covers exactly 511 timesteps (99.8% of a 512-timestep window).
  * **Capacity:** exposes $\approx 8.4$ million parameters to model multi-scale temporal dependencies.

---

## Verification & testing
Run the test script to verify dimensions, shape mapping, and class properties of the TCN modules:
```bash
python scripts/test_baseline_tcn.py
```

## Next Steps
1. **Classifier Head (`classifier.py`):** Implement the Global Average Pooling layer and Linear mappings ($512 \to 256 \to 128 \to n\_classes$).
2. **Model Wrapper (`model.py`):** Compose the `TCNEncoder` and `ClassifierHead` under the unified `HelioForgeTCN` model container.
3. **Training Script (`scripts/train.py`):** Write the training loop supporting class-weighted Loss, AdamW optimizer, and learning rate schedulers.
