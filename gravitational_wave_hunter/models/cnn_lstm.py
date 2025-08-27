"""
CNN-LSTM model for gravitational wave detection.

This module implements a hybrid CNN-LSTM architecture that combines
convolutional layers for local feature extraction with LSTM layers
for temporal sequence modeling.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseGWModel


class CNNLSTM(BaseGWModel):
    """
    CNN-LSTM model for gravitational wave detection.
    
    This model uses convolutional layers to extract local features from
    the strain data, followed by LSTM layers to model temporal dependencies.
    The architecture is particularly effective for detecting time-series
    patterns characteristic of gravitational wave signals.
    
    Architecture:
        1. Input normalization
        2. Multiple 1D convolutional blocks with pooling
        3. Bidirectional LSTM layers
        4. Fully connected output layers with dropout
    
    Args:
        input_length: Length of input time series
        num_filters: Number of filters in convolutional layers
        lstm_hidden_size: Hidden size of LSTM layers
        num_classes: Number of output classes (1 for binary, 2+ for multi-class)
        num_conv_layers: Number of convolutional layers
        num_lstm_layers: Number of LSTM layers
        dropout_rate: Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization
        use_bidirectional_lstm: Whether to use bidirectional LSTM
        
    Example:
        >>> model = CNNLSTM(input_length=32768, num_filters=64, lstm_hidden_size=128)
        >>> x = torch.randn(32, 32768)  # batch_size=32
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([32, 1])
    """
    
    def __init__(
        self,
        input_length: int,
        num_filters: int = 64,
        lstm_hidden_size: int = 128,
        num_classes: int = 1,
        num_conv_layers: int = 4,
        num_lstm_layers: int = 2,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_bidirectional_lstm: bool = True,
    ) -> None:
        """Initialize the CNN-LSTM model."""
        super().__init__(input_length, num_classes, dropout_rate)
        
        self.num_filters = num_filters
        self.lstm_hidden_size = lstm_hidden_size
        self.num_conv_layers = num_conv_layers
        self.num_lstm_layers = num_lstm_layers
        self.use_batch_norm = use_batch_norm
        self.use_bidirectional_lstm = use_bidirectional_lstm
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(1) if use_batch_norm else nn.Identity()
        
        # Convolutional layers
        self.conv_layers = self._build_conv_layers()
        
        # Calculate the output size after convolutions
        self.conv_output_size = self._calculate_conv_output_size()
        
        # LSTM layers
        lstm_input_size = self.num_filters * (2 ** (self.num_conv_layers - 1))
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0,
            bidirectional=use_bidirectional_lstm,
        )
        
        # Calculate LSTM output size
        lstm_output_size = lstm_hidden_size * (2 if use_bidirectional_lstm else 1)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_size // 2, num_classes),
        )
        
        # Initialize weights
        self.init_weights()
    
    def _build_conv_layers(self) -> nn.ModuleList:
        """Build the convolutional layers."""
        layers = nn.ModuleList()
        
        in_channels = 1
        for i in range(self.num_conv_layers):
            out_channels = self.num_filters * (2 ** i)
            
            conv_block = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=7,
                    stride=1,
                    padding=3,
                ),
                nn.BatchNorm1d(out_channels) if self.use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout1d(self.dropout_rate),
            )
            
            layers.append(conv_block)
            in_channels = out_channels
        
        return layers
    
    def _calculate_conv_output_size(self) -> int:
        """Calculate the output sequence length after convolutions."""
        length = self.input_length
        for _ in range(self.num_conv_layers):
            # Conv1d with padding preserves length
            # MaxPool1d with kernel_size=2 halves the length
            length = length // 2
        return length
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN-LSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, input_length)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Validate input
        self.validate_input(x)
        
        batch_size = x.shape[0]
        
        # Add channel dimension: (batch_size, 1, input_length)
        x = x.unsqueeze(1)
        
        # Input normalization
        x = self.input_norm(x)
        
        # Convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Reshape for LSTM: (batch_size, sequence_length, features)
        x = x.transpose(1, 2)  # (batch_size, sequence_length, channels)
        
        # LSTM layers
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state for classification
        if self.use_bidirectional_lstm:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            hidden = h_n[-1]
        
        # Fully connected layers
        output = self.fc_layers(hidden)
        
        return output
    
    def init_weights(self) -> None:
        """Initialize model weights."""
        super().init_weights()
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
    
    def get_feature_maps(self, x: torch.Tensor) -> list:
        """
        Extract feature maps from convolutional layers.
        
        Args:
            x: Input tensor of shape (batch_size, input_length)
            
        Returns:
            List of feature maps from each convolutional layer
        """
        self.validate_input(x)
        
        # Add channel dimension
        x = x.unsqueeze(1)
        
        # Input normalization
        x = self.input_norm(x)
        
        feature_maps = []
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            feature_maps.append(x.clone())
        
        return feature_maps
    
    def get_lstm_states(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get LSTM hidden states and output.
        
        Args:
            x: Input tensor of shape (batch_size, input_length)
            
        Returns:
            Tuple of (lstm_output, hidden_state, cell_state)
        """
        self.validate_input(x)
        
        # Forward through conv layers
        x = x.unsqueeze(1)
        x = self.input_norm(x)
        
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Reshape for LSTM
        x = x.transpose(1, 2)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        return lstm_out, h_n, c_n
    
    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights over the sequence.
        
        This is a simple attention mechanism that can help identify
        which parts of the input are most important for the prediction.
        
        Args:
            x: Input tensor of shape (batch_size, input_length)
            
        Returns:
            Attention weights of shape (batch_size, sequence_length)
        """
        # Get LSTM outputs
        lstm_out, _, _ = self.get_lstm_states(x)
        
        # Simple attention: linear layer + softmax
        batch_size, seq_len, hidden_size = lstm_out.shape
        
        # Flatten for linear layer
        lstm_flat = lstm_out.view(-1, hidden_size)
        
        # Attention scores
        attention_layer = nn.Linear(hidden_size, 1).to(lstm_out.device)
        attention_scores = attention_layer(lstm_flat).view(batch_size, seq_len)
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=1)
        
        return attention_weights


class ResidualCNNLSTM(CNNLSTM):
    """
    CNN-LSTM model with residual connections.
    
    This variant adds residual connections to the convolutional layers,
    which can help with training deeper networks and gradient flow.
    
    Args:
        Same as CNNLSTM plus:
        use_residual: Whether to use residual connections
    """
    
    def __init__(
        self,
        input_length: int,
        num_filters: int = 64,
        lstm_hidden_size: int = 128,
        num_classes: int = 1,
        num_conv_layers: int = 4,
        num_lstm_layers: int = 2,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_bidirectional_lstm: bool = True,
        use_residual: bool = True,
    ) -> None:
        """Initialize the residual CNN-LSTM model."""
        self.use_residual = use_residual
        super().__init__(
            input_length=input_length,
            num_filters=num_filters,
            lstm_hidden_size=lstm_hidden_size,
            num_classes=num_classes,
            num_conv_layers=num_conv_layers,
            num_lstm_layers=num_lstm_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_bidirectional_lstm=use_bidirectional_lstm,
        )
    
    def _build_conv_layers(self) -> nn.ModuleList:
        """Build convolutional layers with optional residual connections."""
        if not self.use_residual:
            return super()._build_conv_layers()
        
        layers = nn.ModuleList()
        
        in_channels = 1
        for i in range(self.num_conv_layers):
            out_channels = self.num_filters * (2 ** i)
            
            # Main convolution
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                stride=1,
                padding=3,
            )
            
            # Residual connection projection (if needed)
            if in_channels != out_channels:
                residual_proj = nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                )
            else:
                residual_proj = nn.Identity()
            
            conv_block = ResidualBlock(
                conv=conv,
                residual_proj=residual_proj,
                batch_norm=nn.BatchNorm1d(out_channels) if self.use_batch_norm else nn.Identity(),
                activation=nn.ReLU(),
                pool=nn.MaxPool1d(kernel_size=2, stride=2),
                dropout=nn.Dropout1d(self.dropout_rate),
            )
            
            layers.append(conv_block)
            in_channels = out_channels
        
        return layers


class ResidualBlock(nn.Module):
    """Residual block for CNN layers."""
    
    def __init__(
        self,
        conv: nn.Module,
        residual_proj: nn.Module,
        batch_norm: nn.Module,
        activation: nn.Module,
        pool: nn.Module,
        dropout: nn.Module,
    ) -> None:
        """Initialize residual block."""
        super().__init__()
        self.conv = conv
        self.residual_proj = residual_proj
        self.batch_norm = batch_norm
        self.activation = activation
        self.pool = pool
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        # Store input for residual connection
        residual = self.residual_proj(x)
        
        # Main path
        out = self.conv(x)
        out = self.batch_norm(out)
        
        # Add residual connection
        out = out + residual
        out = self.activation(out)
        out = self.pool(out)
        out = self.dropout(out)
        
        return out
