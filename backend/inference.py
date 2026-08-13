from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from src.HPINA.models.baseline_tcn.model import HelioForgeTCN
from src.HPINA.configs.paths import PathConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATHS = PathConfig.from_yaml(PROJECT_ROOT / "configs/data_paths.yaml")
CLASS_NAMES = ["Quiet", "B", "C", "M", "X"]
RISK_LEVELS = {0: "LOW", 1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "EXTREME"}


def _first_existing(*paths: Path) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def _find_checkpoint() -> Path | None:
    """Resolve model artefacts strictly from configs/data_paths.yaml."""
    direct = _first_existing(
        PATHS.experiments.baseline_tcn.checkpoints / "best_macro_f1.pt",
        PATHS.models.baseline_tcn / "best_macro_f1.pt",
    )
    if direct:
        return direct
    for root in (PATHS.experiments.baseline_tcn.runs, PATHS.models.baseline_tcn):
        if root.is_dir():
            found = next(root.rglob("best_macro_f1.pt"), None)
            if found:
                return found
    return None


class HelioForgeInference:
    """Loads the trained model once, using local paths before deployment paths."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.checkpoint: dict[str, Any] = {}
        self.sequences: torch.Tensor | None = None
        self.error: str | None = None
        checkpoint = _find_checkpoint()
        if checkpoint is None:
            self.error = f"Model checkpoint is not installed under {PATHS.experiments.baseline_tcn.root} or {PATHS.models.baseline_tcn}."
            return
        try:
            self.checkpoint = torch.load(checkpoint, map_location=self.device, weights_only=False)
            args = self.checkpoint["args"]
            self.model = HelioForgeTCN(
                in_channels=args["in_channels"], n_classes=args["n_classes"],
                dropout=args.get("dropout", 0.2), norm_type=args.get("norm_type", "batch"),
                head_dims=args.get("head_dims", [256, 128]),
            )
            self.model.load_state_dict(self.checkpoint["model_state"])
            self.model.eval()
            data_path = _first_existing(PATHS.windows.test)
            if data_path:
                loaded = torch.load(data_path, map_location=self.device, weights_only=False)
                self.sequences = loaded.get("sequences") if isinstance(loaded, dict) else loaded
        except Exception as exc:  # Keep health and documentation endpoints available.
            self.error = f"Model initialization failed: {exc}"

    @property
    def ready(self) -> bool:
        return self.error is None and hasattr(self, "model")

    def predict_tensor(self, tensor: torch.Tensor) -> dict[str, Any]:
        if tuple(tensor.shape[1:]) != (32, 512):
            raise ValueError("Expected an input tensor with shape (batch, 32, 512).")
        if not self.ready:
            raise RuntimeError(self.error or "Inference engine is unavailable.")
        started = time.perf_counter()
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor.to(self.device, dtype=torch.float32)), dim=1)[0]
        class_id = int(probabilities.argmax().item())
        return {
            "predicted_class": class_id, "predicted_label": CLASS_NAMES[class_id],
            "confidence": float(probabilities[class_id].item()), "risk_level": RISK_LEVELS[class_id],
            "probabilities": {name: float(probabilities[i].item()) for i, name in enumerate(CLASS_NAMES)},
            "processing_time_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def predict_demo(self, index: int = 0) -> dict[str, Any]:
        if self.sequences is None:
            raise RuntimeError("Demo windows are not installed.")
        if not 0 <= index < len(self.sequences):
            raise IndexError(f"Demo index must be between 0 and {len(self.sequences) - 1}.")
        sample = self.sequences[index:index + 1]
        result = self.predict_tensor(sample)
        signal = sample[0, 0].detach().cpu().tolist()
        result.update({"observation_id": f"DEMO_{index:04d}", "sample_index": index,
                       "input_shape": list(sample.shape), "signal": signal,
                       "features": {f"feature_{i + 1:02d}": round(float(sample[0, i].mean()), 5) for i in range(32)},
                       "rgb_intensity": {"red": 182, "green": 147, "blue": 103},
                       "active_regions": []})
        return result


inference_engine = HelioForgeInference()
