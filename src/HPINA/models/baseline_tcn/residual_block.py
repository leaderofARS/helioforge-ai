import torch
import torch.nn as nn
from src.HPINA.models.baseline_tcn.causal_conv import CausalConv1d

class LayerNorm1d(nn.Module):
    """
    1D Layer Normalization wrapper.
    Transposes (B, C, L) to (B, L, C), applies standard nn.LayerNorm over channels,
    and transposes back to (B, C, L).
    """
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class TemporalResidualBlock(nn.Module):
    """
    Dilated Temporal Residual Block for TCN.
    Consists of two dilated causal convolutions, normalization layers, ReLU activations,
    dropout, and a residual shortcut projection.
    
    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output feature channels.
    kernel_size : int
        Convolution kernel size (default 3).
    dilation : int
        Dilation rate (default 1).
    dropout : float
        Dropout rate (default 0.2).
    norm_type : str
        Normalization type: "batch" (BatchNorm1d), "layer" (LayerNorm1d), or "none".
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
        norm_type: str = "batch"
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = dropout
        self.norm_type = norm_type

        # First convolutional block
        self.conv1 = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self.norm1 = self._get_norm_layer(out_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        # Second convolutional block
        self.conv2 = CausalConv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self.norm2 = self._get_norm_layer(out_channels)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        # Residual shortcut connection (1x1 convolution if channels change, else identity)
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        # Final activation after adding residual
        self.relu_out = nn.ReLU()

    def _get_norm_layer(self, num_features: int) -> nn.Module:
        if self.norm_type == "batch":
            return nn.BatchNorm1d(num_features)
        elif self.norm_type == "layer":
            return LayerNorm1d(num_features)
        elif self.norm_type == "none":
            return nn.Identity()
        else:
            raise ValueError(f"Unknown norm_type: {self.norm_type}")

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
            Residual block output of shape (batch_size, out_channels, sequence_length).
        """
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        return self.relu_out(out + residual)
