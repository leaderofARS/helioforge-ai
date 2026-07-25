import torch
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
        # x: (Batch, in_features, L) -> gap(x): (Batch, in_features, 1) -> head: (Batch, n_classes)
        return self.head(self.gap(x))
