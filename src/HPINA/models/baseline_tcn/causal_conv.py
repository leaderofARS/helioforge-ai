import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    """
    1D Causal Convolutional Layer.
    Guarantees strict temporal causality (no future information leakage) by applying
    left-padding of size P = (kernel_size - 1) * dilation before standard 1D convolution.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        
        self.padding = (self.kernel_size - 1) * self.dilation
        
        # We manually left-pad in forward(), so we use padding=0 here.
        self.conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            bias=self.bias
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
            Causal output tensor of shape (batch_size, out_channels, sequence_length).
        """
        # Left-pad the sequence dimension (last dimension)
        x_padded = F.pad(x, (self.padding, 0))
        return self.conv(x_padded)
