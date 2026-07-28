"""
metrics.py — Evaluation metrics for HelioForge TCN
====================================================

Provides per-epoch and per-evaluation metric computation.
Designed for 5-class imbalanced flare classification:
    0 — Quiet
    1 — B-class
    2 — C-class
    3 — M-class
    4 — X-class

Key metrics:
    - Overall accuracy
    - Macro-averaged F1 score  (treats all classes equally, penalises X-class misses)
    - Per-class precision, recall, F1
    - Confusion matrix

Rationale:
    Accuracy alone is misleading on imbalanced data. A model that predicts
    "Quiet" 100% of the time can achieve >50% accuracy while having zero
    recall on M and X-class flares. Macro F1 is the primary metric.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


CLASS_NAMES: List[str] = ["Quiet", "B", "C", "M", "X"]


def compute_metrics(
    all_preds: List[int],
    all_labels: List[int],
    n_classes: int = 5,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics from prediction and label lists.

    Parameters
    ----------
    all_preds : list[int]
        Predicted class indices for the entire evaluation split.
    all_labels : list[int]
        Ground-truth class indices for the entire evaluation split.
    n_classes : int
        Number of classes. Default: 5.
    class_names : list[str], optional
        Human-readable class names. Defaults to CLASS_NAMES.

    Returns
    -------
    dict
        Dictionary containing:
            - "accuracy"       : float
            - "macro_f1"       : float
            - "macro_precision": float
            - "macro_recall"   : float
            - per_class dict with "precision_<cls>", "recall_<cls>", "f1_<cls>"
    """
    if class_names is None:
        class_names = CLASS_NAMES[:n_classes]

    preds  = np.array(all_preds,  dtype=np.int64)
    labels = np.array(all_labels, dtype=np.int64)

    # Confusion matrix (rows = true, cols = predicted)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    # Per-class precision, recall, F1
    per_class_precision = np.zeros(n_classes)
    per_class_recall    = np.zeros(n_classes)
    per_class_f1        = np.zeros(n_classes)

    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp   # predicted c but not actually c
        fn = cm[c, :].sum() - tp   # actually c but predicted otherwise

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class_precision[c] = prec
        per_class_recall[c]    = rec
        per_class_f1[c]        = f1

    accuracy        = (preds == labels).mean()
    macro_precision = per_class_precision.mean()
    macro_recall    = per_class_recall.mean()
    macro_f1        = per_class_f1.mean()

    result: Dict[str, float] = {
        "accuracy"       : float(accuracy),
        "macro_f1"       : float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall"   : float(macro_recall),
    }

    for c, name in enumerate(class_names):
        result[f"precision_{name}"] = float(per_class_precision[c])
        result[f"recall_{name}"]    = float(per_class_recall[c])
        result[f"f1_{name}"]        = float(per_class_f1[c])

    return result


def confusion_matrix_str(
    all_preds: List[int],
    all_labels: List[int],
    n_classes: int = 5,
    class_names: Optional[List[str]] = None,
) -> str:
    """
    Return a human-readable confusion matrix string for logging.

    Parameters
    ----------
    all_preds : list[int]
        Predicted class indices.
    all_labels : list[int]
        Ground-truth class indices.
    n_classes : int
        Number of classes. Default: 5.
    class_names : list[str], optional
        Human-readable class names.

    Returns
    -------
    str
        Formatted confusion matrix string.
    """
    if class_names is None:
        class_names = CLASS_NAMES[:n_classes]

    preds  = np.array(all_preds,  dtype=np.int64)
    labels = np.array(all_labels, dtype=np.int64)

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    col_width = max(max(len(n) for n in class_names), 5) + 2
    header = "True\\Pred".ljust(col_width) + "".join(n.rjust(col_width) for n in class_names)
    lines  = [header, "-" * len(header)]
    for i, row_name in enumerate(class_names):
        row = row_name.ljust(col_width) + "".join(str(cm[i, j]).rjust(col_width) for j in range(n_classes))
        lines.append(row)

    return "\n".join(lines)


def format_metrics_table(metrics: Dict[str, float], class_names: Optional[List[str]] = None) -> str:
    """
    Format a metrics dict into a human-readable table string for console/logging.

    Parameters
    ----------
    metrics : dict
        Output of compute_metrics().
    class_names : list[str], optional
        Class names to show per-class rows for.

    Returns
    -------
    str
        Formatted metrics table.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    lines = [
        f"  Accuracy        : {metrics['accuracy']:.4f}",
        f"  Macro F1        : {metrics['macro_f1']:.4f}",
        f"  Macro Precision : {metrics['macro_precision']:.4f}",
        f"  Macro Recall    : {metrics['macro_recall']:.4f}",
        "",
        f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}",
        f"  {'-'*40}",
    ]
    for name in class_names:
        p = metrics.get(f"precision_{name}", 0.0)
        r = metrics.get(f"recall_{name}",    0.0)
        f = metrics.get(f"f1_{name}",        0.0)
        lines.append(f"  {name:<10} {p:>10.4f} {r:>10.4f} {f:>10.4f}")

    return "\n".join(lines)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    n_classes: int = 5,
    class_names: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Run a full evaluation pass and return loss + metrics.

    Parameters
    ----------
    model : nn.Module
        The HelioForgeTCN model in eval mode.
    loader : DataLoader
        Evaluation DataLoader (val or test).
    criterion : nn.Module
        Loss function (CrossEntropyLoss).
    device : torch.device
        Compute device.
    n_classes : int
        Number of output classes. Default: 5.
    class_names : list[str], optional
        Human-readable class names.

    Returns
    -------
    Tuple[float, dict]
        (mean_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    all_preds:  List[int] = []
    all_labels: List[int] = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        total_loss += loss.item()

        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().tolist())

    mean_loss = total_loss / len(loader)
    metrics   = compute_metrics(all_preds, all_labels, n_classes=n_classes, class_names=class_names)
    return mean_loss, metrics
