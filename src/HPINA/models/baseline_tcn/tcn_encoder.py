import torch
import torch.nn as nn
from typing import List, Optional
from src.HPINA.models.baseline_tcn.residual_block import TemporalResidualBlock

class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network (TCN) Encoder.
    Composes multiple dilated causal temporal residual blocks in series to extract 
    highly contextualized representations from multivariate time series.
    
    Adheres to the progressive widening channel schedule:
    32 -> 128 -> 256 -> 256 -> 512 -> 512 -> 512 -> 512 -> 512
    with exponential dilations [1, 2, 4, 8, 16, 32, 64, 128].
    
    Receptive Field:
    RF = 1 + (k - 1) * sum(d_i) = 1 + (3 - 1) * (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128)
       = 1 + 2 * 255 = 511 timesteps.
    This covers 99.8% of a 512-timestep input window.
    """
    def __init__(
        self,
        in_channels: int = 32,
        channel_schedule: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        norm_type: str = "batch",
        dilations: Optional[List[int]] = None,
        num_channels: Optional[List[int]] = None
    ) -> None:
        super().__init__()
        
        # Support both channel_schedule and num_channels for maximum compatibility
        if channel_schedule is None:
            if num_channels is not None:
                channel_schedule = num_channels
            else:
                channel_schedule = [128, 256, 256, 512, 512, 512, 512, 512]
                
        if dilations is None:
            dilations = [2 ** i for i in range(len(channel_schedule))]
            
        if len(channel_schedule) != len(dilations):
            raise ValueError("channel_schedule and dilations list lengths must match.")
            
        self.in_channels = in_channels
        self.channel_schedule = channel_schedule
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.norm_type = norm_type

        # Stack the residual blocks sequentially
        layers = []
        current_channels = in_channels
        
        for out_ch, dil in zip(channel_schedule, dilations):
            layers.append(
                TemporalResidualBlock(
                    in_channels=current_channels,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dil,
                    dropout=dropout,
                    norm_type=norm_type
                )
            )
            current_channels = out_ch
            
        self.network = nn.Sequential(*layers)
        self.out_channels = current_channels  # Expose for ClassifierHead compatibility

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
            Encoded feature representation of shape (batch_size, out_channels, sequence_length).
        """
        return self.network(x)
