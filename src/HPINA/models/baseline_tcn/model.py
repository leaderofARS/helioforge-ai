import torch
import torch.nn as nn
from typing import List, Optional
from .tcn_encoder import TCNEncoder
from .classifier  import ClassifierHead

class HelioForgeTCN(nn.Module):
    """
    Full High-Capacity Production TCN = TCNEncoder (8.57M params) + ClassifierHead.

    Input:  (Batch, F=32, L=512)
    Output: (Batch, n_classes)     raw logits — apply softmax for probabilities

    Args:
        in_channels      (int):       Input feature channels (32 for HelioForge).
        n_classes        (int):       Number of output classes (5 for flare types).
        channel_schedule (list[int]): Per-block output channels for TCNEncoder.
        kernel_size      (int):       Convolution kernel size (default 3).
        dropout          (float):     Dropout probability (default 0.2).
        norm_type        (str):       Normalization type: "batch", "layer", or "none" (default "batch").
        head_dims        (list[int]): Classifier MLP hidden dimensions (default [256, 128]).
        head_dropout     (float):     Classifier MLP dropout probability (default 0.3).
    """

    def __init__(
        self,
        in_channels: int = 32,
        n_classes: int = 5,
        channel_schedule: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        norm_type: str = "batch",
        head_dims: Optional[List[int]] = None,
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # Instantiate encoder with configured normalization and schedule
        self.encoder = TCNEncoder(
            in_channels=in_channels,
            channel_schedule=channel_schedule,
            kernel_size=kernel_size,
            dropout=dropout,
            norm_type=norm_type,
        )
        # Instantiate classifier head using out_channels from the encoder
        self.classifier = ClassifierHead(
            in_features=self.encoder.out_channels,
            n_classes=n_classes,
            dropout=head_dropout,
            head_dims=head_dims,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns
        -------
        torch.Tensor
            Class logits of shape (batch_size, n_classes).
        """
        return self.classifier(self.encoder(x))
