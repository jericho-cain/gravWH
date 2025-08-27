"""
WaveNet model for gravitational wave detection.

This module implements a WaveNet architecture with dilated convolutions
for multi-scale pattern recognition in gravitational wave data.
"""

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseGWModel


class WaveNet(BaseGWModel):
    """
    WaveNet model for gravitational wave detection.
    
    WaveNet uses dilated convolutions to capture patterns at multiple
    time scales efficiently. The exponentially increasing dilation rates
    allow the model to have a very large receptive field while maintaining
    computational efficiency.
    
    Args:
        input_length: Length of input time series
        num_layers: Number of convolutional layers
        num_channels: Number of channels in each layer
        num_classes: Number of output classes
        kernel_size: Size of convolutional kernels
        dropout_rate: Dropout rate for regularization
        use_skip_connections: Whether to use skip connections
        
    Example:
        >>> model = WaveNet(input_length=32768, num_layers=10, num_channels=32)
        >>> x = torch.randn(32, 32768)
        >>> output = model(x)
    """
    
    def __init__(
        self,
        input_length: int,
        num_layers: int = 10,
        num_channels: int = 32,
        num_classes: int = 1,
        kernel_size: int = 3,
        dropout_rate: float = 0.1,
        use_skip_connections: bool = True,
    ) -> None:
        """Initialize the WaveNet model."""
        super().__init__(input_length, num_classes, dropout_rate)
        
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.use_skip_connections = use_skip_connections
        
        # Input convolution
        self.input_conv = nn.Conv1d(1, num_channels, kernel_size=1)
        
        # Dilated convolution layers
        self.dilated_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        for i in range(num_layers):
            dilation = 2 ** i
            
            # Dilated convolution
            conv = nn.Conv1d(
                num_channels,
                num_channels * 2,  # For gated activation
                kernel_size=kernel_size,
                dilation=dilation,
                padding=dilation * (kernel_size - 1) // 2,
            )
            self.dilated_convs.append(conv)
            
            # Skip connection convolution
            if use_skip_connections:
                skip_conv = nn.Conv1d(num_channels, num_channels, kernel_size=1)
                self.skip_convs.append(skip_conv)
        
        # Output layers
        self.output_conv1 = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.output_conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(num_channels, num_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_channels // 2, num_classes),
        )
        
        self.init_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the WaveNet model."""
        self.validate_input(x)
        
        # Add channel dimension
        x = x.unsqueeze(1)
        
        # Input convolution
        x = self.input_conv(x)
        
        # Skip connections accumulator
        skip_connections = []
        
        # Dilated convolution layers
        for i, (dilated_conv, skip_conv) in enumerate(
            zip(self.dilated_convs, self.skip_convs if self.use_skip_connections else [None] * self.num_layers)
        ):
            # Dilated convolution with gated activation
            conv_out = dilated_conv(x)
            
            # Split for gated activation
            filter_out, gate_out = torch.split(conv_out, self.num_channels, dim=1)
            
            # Gated activation: tanh(filter) * sigmoid(gate)
            gated_out = torch.tanh(filter_out) * torch.sigmoid(gate_out)
            
            # Skip connection
            if self.use_skip_connections:
                skip = skip_conv(gated_out)
                skip_connections.append(skip)
            
            # Residual connection
            x = x + gated_out
        
        # Combine skip connections
        if self.use_skip_connections:
            x = sum(skip_connections)
        
        # Output convolutions
        x = F.relu(x)
        x = self.output_conv1(x)
        x = F.relu(x)
        x = self.output_conv2(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)
        
        # Classification
        output = self.classifier(x)
        
        return output


class TemporalConvNet(BaseGWModel):
    """
    Temporal Convolutional Network for gravitational wave detection.
    
    A simplified version of WaveNet with residual connections and
    causal convolutions for sequence modeling.
    """
    
    def __init__(
        self,
        input_length: int,
        num_channels: List[int] = [25, 50, 100],
        num_classes: int = 1,
        kernel_size: int = 3,
        dropout_rate: float = 0.1,
    ) -> None:
        """Initialize the Temporal Convolutional Network."""
        super().__init__(input_length, num_classes, dropout_rate)
        
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        
        # Build TCN layers
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = 1 if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size-1) * dilation_size,
                    dropout=dropout_rate,
                )
            )
        
        self.network = nn.Sequential(*layers)
        
        # Output layer
        self.classifier = nn.Linear(num_channels[-1], num_classes)
        
        self.init_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the TCN."""
        self.validate_input(x)
        
        # Add channel dimension
        x = x.unsqueeze(1)
        
        # TCN layers
        y = self.network(x)
        
        # Global average pooling
        y = F.adaptive_avg_pool1d(y, 1).squeeze(-1)
        
        # Classification
        output = self.classifier(y)
        
        return output


class TemporalBlock(nn.Module):
    """Temporal block for TCN."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize temporal block."""
        super().__init__()
        
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(
                out_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of temporal block."""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove extra padding from convolution output."""
    
    def __init__(self, chomp_size: int) -> None:
        """Initialize chomp layer."""
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Remove padding."""
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x
