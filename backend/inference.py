from pathlib import Path

import torch

from src.HPINA.models.baseline_tcn.model import HelioForgeTCN


ROOT = Path("/opt/helioforge-ai")

CHECKPOINT_PATH = (
    ROOT
    / "experiments"
    / "baseline_tcn"
    / "runs"
    / "tcn_batch_lr0.001_20260728_123229"
    / "checkpoints"
    / "best_macro_f1.pt"
)

TEST_DATA_PATH = (
    ROOT
    / "data"
    / "windows"
    / "test_feat32_w512.pt"
)


CLASS_NAMES = [
    "Quiet",
    "B",
    "C",
    "M",
    "X",
]


RISK_LEVELS = {
    0: "LOW",
    1: "LOW",
    2: "MEDIUM",
    3: "HIGH",
    4: "EXTREME",
}


class HelioForgeInference:

    def __init__(self):
        print("Loading HelioForgeTCN checkpoint...")

        self.device = torch.device("cpu")

        self.checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=self.device,
            weights_only=False,
        )

        args = self.checkpoint["args"]

        self.model = HelioForgeTCN(
            in_channels=args["in_channels"],
            n_classes=args["n_classes"],
            dropout=args["dropout"],
            norm_type=args["norm_type"],
            head_dims=args["head_dims"],
        )

        self.model.load_state_dict(
            self.checkpoint["model_state"]
        )

        self.model.to(self.device)
        self.model.eval()

        print("✓ HelioForgeTCN loaded")

        # Load demo dataset
        obj = torch.load(
            TEST_DATA_PATH,
            map_location=self.device,
            weights_only=False,
        )

        if not isinstance(obj, dict):
            raise ValueError(
                "Expected test dataset to be a dictionary."
            )

        self.sequences = obj["sequences"]

        if self.sequences.ndim != 3:
            raise ValueError(
                f"Expected 3D tensor, got {self.sequences.shape}"
            )

        if self.sequences.shape[1] != 32:
            raise ValueError(
                f"Expected 32 features, got {self.sequences.shape[1]}"
            )

        if self.sequences.shape[2] != 512:
            raise ValueError(
                f"Expected 512 timesteps, got {self.sequences.shape[2]}"
            )

        print(
            f"✓ Demo dataset loaded: "
            f"{tuple(self.sequences.shape)}"
        )

    def predict_tensor(self, tensor: torch.Tensor):

        if tensor.ndim != 3:
            raise ValueError(
                f"Expected input shape (batch, 32, 512), "
                f"got {tuple(tensor.shape)}"
            )

        if tensor.shape[1] != 32:
            raise ValueError(
                f"Expected 32 features, got {tensor.shape[1]}"
            )

        if tensor.shape[2] != 512:
            raise ValueError(
                f"Expected 512 timesteps, got {tensor.shape[2]}"
            )

        tensor = tensor.to(
            device=self.device,
            dtype=torch.float32,
        )

        with torch.no_grad():

            logits = self.model(tensor)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predicted_class = torch.argmax(
                probabilities,
                dim=1,
            )

        class_id = predicted_class[0].item()

        confidence = probabilities[
            0,
            class_id,
        ].item()

        probability_dict = {
            CLASS_NAMES[i]: float(
                probabilities[0, i].item()
            )
            for i in range(len(CLASS_NAMES))
        }

        return {
            "predicted_class": class_id,
            "predicted_label": CLASS_NAMES[class_id],
            "confidence": float(confidence),
            "risk_level": RISK_LEVELS[class_id],
            "probabilities": probability_dict,
        }

    def predict_demo(self, index: int = 0):

        if index < 0 or index >= len(self.sequences):
            raise IndexError(
                f"Demo index must be between "
                f"0 and {len(self.sequences) - 1}"
            )

        sample = self.sequences[
            index:index + 1
        ]

        result = self.predict_tensor(sample)

        result["observation_id"] = (
            f"DEMO_{index:04d}"
        )

        result["sample_index"] = index

        result["input_shape"] = list(
            sample.shape
        )

        return result


# Load model once when backend starts.
inference_engine = HelioForgeInference()