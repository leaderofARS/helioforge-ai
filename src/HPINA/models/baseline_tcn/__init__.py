"""
baseline_tcn — HelioForge AI Baseline TCN Package
================================================

Exports the full HelioForgeTCN model and its constituent components
for use in training scripts and experiments.

Architecture:
    Input  (Batch, 32, 512)
      ↓
    TCNEncoder        — 8 dilated residual blocks (RF = 511 timesteps, ~8.4M params)
      ↓
    ClassifierHead    — GAP + Linear (512→256→128→n_classes)
      ↓
    Output (Batch, n_classes)     raw logits
"""

from .causal_conv      import CausalConv1d
from .residual_block   import TemporalResidualBlock, LayerNorm1d
from .tcn_encoder      import TCNEncoder
from .classifier       import ClassifierHead
from .model            import HelioForgeTCN
from .losses           import compute_class_weights, build_weighted_criterion
from .metrics          import compute_metrics, evaluate, confusion_matrix_str, format_metrics_table, CLASS_NAMES

__all__ = [
    "CausalConv1d",
    "LayerNorm1d",
    "TemporalResidualBlock",
    "TCNEncoder",
    "ClassifierHead",
    "HelioForgeTCN",
    "compute_class_weights",
    "build_weighted_criterion",
    "compute_metrics",
    "evaluate",
    "confusion_matrix_str",
    "format_metrics_table",
    "CLASS_NAMES",
]

