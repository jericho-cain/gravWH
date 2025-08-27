"""
Transformer model for gravitational wave detection.

This module implements a Transformer architecture with self-attention
mechanisms for capturing long-range dependencies in gravitational wave signals.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseGWModel


class GWTransformer(BaseGWModel):
    """
    Transformer model for gravitational wave detection.
    
    This model uses self-attention mechanisms to capture long-range
    dependencies in the input time series, making it effective for
    detecting gravitational wave patterns that span multiple time scales.
    
    Args:
        input_length: Length of input time series
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        num_classes: Number of output classes
        d_ff: Feed-forward network dimension
        dropout_rate: Dropout rate
        max_seq_length: Maximum sequence length for positional encoding
        
    Example:
        >>> model = GWTransformer(input_length=32768, d_model=512, num_heads=8)
        >>> x = torch.randn(32, 32768)
        >>> output = model(x)
    """
    
    def __init__(
        self,
        input_length: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        num_classes: int = 1,
        d_ff: Optional[int] = None,
        dropout_rate: float = 0.1,
        max_seq_length: int = 50000,
    ) -> None:
        """Initialize the Transformer model."""
        super().__init__(input_length, num_classes, dropout_rate)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff or 4 * d_model
        
        # Input embedding and positional encoding
        self.input_projection = nn.Linear(1, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout_rate)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=self.d_ff,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 4, num_classes),
        )
        
        self.init_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Transformer model."""
        self.validate_input(x)
        
        # Add feature dimension: (batch_size, seq_length, 1)
        x = x.unsqueeze(-1)
        
        # Input projection: (batch_size, seq_length, d_model)
        x = self.input_projection(x)
        
        # Scale by sqrt(d_model) as in the original paper
        x = x * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling over sequence dimension
        x = torch.mean(x, dim=1)
        
        # Classification
        output = self.classifier(x)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> list:
        """
        Extract attention weights from all transformer layers.
        
        Args:
            x: Input tensor
            
        Returns:
            List of attention weight tensors
        """
        self.validate_input(x)
        
        # Prepare input
        x = x.unsqueeze(-1)
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        attention_weights = []
        
        # Forward through each layer and collect attention weights
        for layer in self.transformer_encoder.layers:
            # Get attention weights from multi-head attention
            attn_output, attn_weights = layer.self_attn(x, x, x, need_weights=True)
            attention_weights.append(attn_weights)
            
            # Continue forward pass through this layer
            x = layer.norm1(x + layer.dropout1(attn_output))
            ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = layer.norm2(x + layer.dropout2(ff_output))
        
        return attention_weights


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    
    Adds positional information to input embeddings using sinusoidal functions.
    """
    
    def __init__(self, d_model: int, max_length: int = 50000, dropout: float = 0.1) -> None:
        """Initialize positional encoding."""
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class ConvTransformer(BaseGWModel):
    """
    Hybrid CNN-Transformer model for gravitational wave detection.
    
    This model combines convolutional layers for local feature extraction
    with transformer layers for capturing long-range dependencies.
    """
    
    def __init__(
        self,
        input_length: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        num_classes: int = 1,
        num_conv_layers: int = 3,
        conv_channels: int = 64,
        dropout_rate: float = 0.1,
    ) -> None:
        """Initialize the Conv-Transformer model."""
        super().__init__(input_length, num_classes, dropout_rate)
        
        self.d_model = d_model
        self.num_conv_layers = num_conv_layers
        
        # Convolutional feature extractor
        conv_layers = []
        in_channels = 1
        
        for i in range(num_conv_layers):
            out_channels = conv_channels * (2 ** i)
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout1d(dropout_rate),
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate sequence length after convolutions
        self.conv_output_length = input_length // (2 ** num_conv_layers)
        
        # Project to transformer dimension
        self.conv_to_transformer = nn.Linear(in_channels, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, self.conv_output_length, dropout_rate)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, num_classes),
        )
        
        self.init_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Conv-Transformer model."""
        self.validate_input(x)
        
        batch_size = x.shape[0]
        
        # Add channel dimension
        x = x.unsqueeze(1)
        
        # Convolutional feature extraction
        x = self.conv_layers(x)
        
        # Reshape for transformer: (batch_size, seq_length, features)
        x = x.transpose(1, 2)
        
        # Project to transformer dimension
        x = self.conv_to_transformer(x)
        
        # Scale and add positional encoding
        x = x * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # Transformer processing
        x = self.transformer(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Classification
        output = self.classifier(x)
        
        return output


class MultiScaleTransformer(BaseGWModel):
    """
    Multi-scale Transformer that processes the input at different resolutions.
    
    This model creates multiple pathways that downsample the input to different
    scales before applying transformer processing, then combines the results.
    """
    
    def __init__(
        self,
        input_length: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        num_classes: int = 1,
        scales: list = [1, 2, 4, 8],
        dropout_rate: float = 0.1,
    ) -> None:
        """Initialize the Multi-scale Transformer model."""
        super().__init__(input_length, num_classes, dropout_rate)
        
        self.scales = scales
        self.d_model = d_model
        
        # Create transformer for each scale
        self.scale_transformers = nn.ModuleList()
        
        for scale in scales:
            # Downsampling layer
            if scale > 1:
                downsample = nn.AvgPool1d(kernel_size=scale, stride=scale)
            else:
                downsample = nn.Identity()
            
            # Sequence length at this scale
            seq_length = input_length // scale
            
            # Input projection
            input_proj = nn.Linear(1, d_model)
            
            # Positional encoding
            pos_enc = PositionalEncoding(d_model, seq_length, dropout_rate)
            
            # Transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout_rate,
                batch_first=True,
            )
            transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            scale_module = nn.ModuleDict({
                'downsample': downsample,
                'input_proj': input_proj,
                'pos_enc': pos_enc,
                'transformer': transformer,
            })
            
            self.scale_transformers.append(scale_module)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * len(scales), d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, num_classes),
        )
        
        self.init_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Multi-scale Transformer."""
        self.validate_input(x)
        
        scale_outputs = []
        
        for scale_module in self.scale_transformers:
            # Downsample
            x_scale = scale_module['downsample'](x)
            
            # Add feature dimension
            x_scale = x_scale.unsqueeze(-1)
            
            # Input projection
            x_scale = scale_module['input_proj'](x_scale)
            x_scale = x_scale * math.sqrt(self.d_model)
            
            # Positional encoding
            x_scale = scale_module['pos_enc'](x_scale)
            
            # Transformer
            x_scale = scale_module['transformer'](x_scale)
            
            # Global average pooling
            x_scale = torch.mean(x_scale, dim=1)
            
            scale_outputs.append(x_scale)
        
        # Concatenate outputs from all scales
        x = torch.cat(scale_outputs, dim=1)
        
        # Fusion
        x = self.fusion(x)
        
        # Classification
        output = self.classifier(x)
        
        return output
