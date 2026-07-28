"""
losses.py — Loss functions for HelioForge TCN training
=======================================================

Provides class-weighted Cross-Entropy loss to handle severe flare class imbalance.

Class Distribution (train split, windows_fifth dataset):
    Class 0 — Quiet  :  most frequent
    Class 1 — B-class:  common
    Class 2 — C-class:  moderate
    Class 3 — M-class:  rare
    Class 4 — X-class:  very rare (only ~16 training windows)

Strategy:
    Weight_c = N_total / (n_classes × N_c)
    This up-weights rare classes and down-weights majority classes.
    The model is penalised heavily for misclassifying X-class events.
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, List


def compute_class_weights(
    labels: torch.Tensor,
    n_classes: int = 5,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute balanced class weights from a label tensor.

    Formula: weight_c = N_total / (n_classes * N_c)

    Parameters
    ----------
    labels : torch.Tensor
        1D integer tensor of class labels. Shape: (N,).
    n_classes : int
        Total number of classes. Default: 5.
    device : torch.device, optional
        Device to place the weight tensor on. Defaults to labels.device.

    Returns
    -------
    torch.Tensor
        Float tensor of shape (n_classes,) with per-class weights.
    """
    if device is None:
        device = labels.device

    counts = torch.zeros(n_classes, dtype=torch.float32)
    for c in range(n_classes):
        counts[c] = (labels == c).sum().float()

    # Replace zero counts with 1 to avoid division by zero for absent classes
    counts = counts.clamp(min=1.0)
    n_total = labels.shape[0]
    weights = n_total / (n_classes * counts)

    return weights.to(device)


def build_weighted_criterion(
    train_labels: torch.Tensor,
    n_classes: int = 5,
    device: Optional[torch.device] = None,
    label_smoothing: float = 0.0,
) -> nn.CrossEntropyLoss:
    """
    Build a CrossEntropyLoss criterion with balanced class weights.

    Parameters
    ----------
    train_labels : torch.Tensor
        1D integer tensor of all training labels. Used to compute weights.
    n_classes : int
        Number of output classes. Default: 5.
    device : torch.device, optional
        Target device. Defaults to train_labels.device.
    label_smoothing : float
        Label smoothing epsilon (0.0 = no smoothing). Default: 0.0.

    Returns
    -------
    nn.CrossEntropyLoss
        Weighted cross entropy criterion ready for training.
    """
    weights = compute_class_weights(train_labels, n_classes=n_classes, device=device)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)


def load_class_weights_from_scaler(
    scaler_path: str,
    n_classes: int = 5,
    device: Optional[torch.device] = None,
    manual_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Load or construct class weights.

    If manual_weights are provided, they override all other sources.
    Otherwise falls back to uniform weights (used only if label tensor unavailable).

    Parameters
    ----------
    scaler_path : str
        Path to scaler_f32_w512.json (used for reference; weights not stored here).
    n_classes : int
        Number of classes. Default: 5.
    device : torch.device, optional
        Target device.
    manual_weights : list[float], optional
        Explicit per-class weights [w0, w1, w2, w3, w4].

    Returns
    -------
    torch.Tensor
        Float tensor of shape (n_classes,).
    """
    if manual_weights is not None:
        weights = torch.tensor(manual_weights, dtype=torch.float32)
    else:
        # Uniform weights as fallback
        weights = torch.ones(n_classes, dtype=torch.float32)

    if device is not None:
        weights = weights.to(device)

    return weights
