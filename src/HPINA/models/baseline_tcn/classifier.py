import torch
import torch.nn as nn
from typing import List, Optional

class ClassifierHead(nn.Module):
    """
    Global Average Pool collapses the time dimension (L timesteps → 1 vector),
    then linear layers map the compact representation to class logits.

    Args:
        in_features (int):         Channel count from TCNEncoder (512 for HelioForge).
        n_classes   (int):         Number of output classes (5 for flare classification).
        dropout     (float):       Dropout probability in the MLP (default 0.3).
        head_dims   (list[int]):   Hidden layer dimensions for the MLP head.
                                   Default: [256, 128]. Allows easy ablation of classifier capacity.
    """

    def __init__(
        self,
        in_features: int = 512,
        n_classes: int = 5,
        dropout: float = 0.3,
        head_dims: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if head_dims is None:
            head_dims = [256, 128]

        self.in_features = in_features
        self.n_classes   = n_classes
        self.dropout     = dropout
        self.head_dims   = head_dims

        self.gap = nn.AdaptiveAvgPool1d(1)  # (B, in_features, L) → (B, in_features, 1)

        layers: List[nn.Module] = [nn.Flatten()]
        curr_dim = in_features

        for h_dim in head_dims:
            layers.extend([
                nn.Linear(curr_dim, h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            curr_dim = h_dim

        layers.append(nn.Linear(curr_dim, n_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features, sequence_length).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, n_classes).
        """
        return self.head(self.gap(x))
