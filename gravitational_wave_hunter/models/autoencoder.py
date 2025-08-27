"""
Autoencoder model for gravitational wave detection.

This module implements autoencoder architectures for unsupervised
gravitational wave detection using reconstruction error as an anomaly score.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseGWModel


class GWAutoencoder(BaseGWModel):
    """
    Autoencoder for gravitational wave detection.
    
    This model learns to reconstruct normal detector noise and identifies
    gravitational wave signals as anomalies with high reconstruction error.
    The architecture uses convolutional layers for efficient processing
    of time series data.
    
    Args:
        input_length: Length of input time series
        encoding_dim: Dimension of the encoded representation
        num_layers: Number of encoder/decoder layers
        num_filters: Base number of filters
        dropout_rate: Dropout rate for regularization
        use_variational: Whether to use variational autoencoder
        
    Example:
        >>> model = GWAutoencoder(input_length=32768, encoding_dim=128)
        >>> x = torch.randn(32, 32768)
        >>> reconstructed = model(x)
        >>> reconstruction_error = F.mse_loss(reconstructed, x)
    """
    
    def __init__(
        self,
        input_length: int,
        encoding_dim: int = 128,
        num_layers: int = 4,
        num_filters: int = 32,
        dropout_rate: float = 0.1,
        use_variational: bool = False,
    ) -> None:
        """Initialize the autoencoder model."""
        # For autoencoders, num_classes is not relevant
        super().__init__(input_length, num_classes=1, dropout_rate=dropout_rate)
        
        self.encoding_dim = encoding_dim
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.use_variational = use_variational
        
        # Calculate intermediate dimensions
        self.encoded_length = input_length // (2 ** num_layers)
        
        # Encoder
        self.encoder = self._build_encoder()
        
        # Latent space layers
        if use_variational:
            # For VAE, we need mu and logvar
            self.fc_mu = nn.Linear(self.num_filters * (2 ** (num_layers - 1)) * self.encoded_length, encoding_dim)
            self.fc_logvar = nn.Linear(self.num_filters * (2 ** (num_layers - 1)) * self.encoded_length, encoding_dim)
            self.fc_decode = nn.Linear(encoding_dim, self.num_filters * (2 ** (num_layers - 1)) * self.encoded_length)
        else:
            # Standard autoencoder
            self.fc_encode = nn.Linear(
                self.num_filters * (2 ** (num_layers - 1)) * self.encoded_length, 
                encoding_dim
            )
            self.fc_decode = nn.Linear(
                encoding_dim, 
                self.num_filters * (2 ** (num_layers - 1)) * self.encoded_length
            )
        
        # Decoder
        self.decoder = self._build_decoder()
        
        self.init_weights()
    
    def _build_encoder(self) -> nn.Sequential:
        """Build the encoder network."""
        layers = []
        in_channels = 1
        
        for i in range(self.num_layers):
            out_channels = self.num_filters * (2 ** i)
            
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout1d(self.dropout_rate),
            ])
            
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self) -> nn.Sequential:
        """Build the decoder network."""
        layers = []
        
        # Start with the largest number of channels
        channels = [self.num_filters * (2 ** i) for i in range(self.num_layers)]
        channels.reverse()
        
        for i in range(self.num_layers):
            in_channels = channels[i]
            out_channels = channels[i + 1] if i < self.num_layers - 1 else 1
            
            layers.extend([
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(out_channels) if out_channels > 1 else nn.Identity(),
                nn.ReLU() if out_channels > 1 else nn.Tanh(),
            ])
            
            if out_channels > 1:
                layers.append(nn.Dropout1d(self.dropout_rate))
        
        return nn.Sequential(*layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor of shape (batch_size, input_length)
            
        Returns:
            Encoded representation
        """
        self.validate_input(x)
        
        # Add channel dimension
        x = x.unsqueeze(1)
        
        # Encoder forward pass
        x = self.encoder(x)
        
        # Flatten for fully connected layer
        x = x.view(x.size(0), -1)
        
        if self.use_variational:
            # Return mu and logvar for VAE
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)
            return mu, logvar
        else:
            # Standard autoencoder
            encoded = self.fc_encode(x)
            return encoded
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to output.
        
        Args:
            z: Latent representation
            
        Returns:
            Decoded output
        """
        # Fully connected decode
        x = self.fc_decode(z)
        
        # Reshape for convolutional decoder
        batch_size = x.size(0)
        channels = self.num_filters * (2 ** (self.num_layers - 1))
        x = x.view(batch_size, channels, self.encoded_length)
        
        # Decoder forward pass
        x = self.decoder(x)
        
        # Remove channel dimension
        x = x.squeeze(1)
        
        return x
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the autoencoder.
        
        Args:
            x: Input tensor
            
        Returns:
            Reconstructed output (and VAE components if variational)
        """
        if self.use_variational:
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            reconstructed = self.decode(z)
            return reconstructed, mu, logvar
        else:
            encoded = self.encode(x)
            reconstructed = self.decode(encoded)
            return reconstructed
    
    def compute_loss(self, x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """
        Compute reconstruction loss (and KL divergence for VAE).
        
        Args:
            x: Input tensor
            reduction: Loss reduction method
            
        Returns:
            Total loss
        """
        if self.use_variational:
            reconstructed, mu, logvar = self.forward(x)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(reconstructed, x, reduction=reduction)
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            if reduction == 'mean':
                kl_loss = kl_loss / x.size(0)
            
            return recon_loss + 0.1 * kl_loss  # Beta = 0.1
        else:
            reconstructed = self.forward(x)
            return F.mse_loss(reconstructed, x, reduction=reduction)
    
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly scores for input samples.
        
        Args:
            x: Input tensor
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        self.eval()
        with torch.no_grad():
            if self.use_variational:
                reconstructed, _, _ = self.forward(x)
            else:
                reconstructed = self.forward(x)
            
            # Compute reconstruction error per sample
            errors = torch.mean((x - reconstructed) ** 2, dim=1)
            
        return errors


class ConvolutionalAutoencoder(GWAutoencoder):
    """
    Convolutional autoencoder with skip connections.
    
    This variant adds skip connections between encoder and decoder layers
    to preserve fine-grained details in the reconstruction.
    """
    
    def __init__(
        self,
        input_length: int,
        encoding_dim: int = 128,
        num_layers: int = 4,
        num_filters: int = 32,
        dropout_rate: float = 0.1,
        use_skip_connections: bool = True,
    ) -> None:
        """Initialize the convolutional autoencoder."""
        self.use_skip_connections = use_skip_connections
        super().__init__(
            input_length=input_length,
            encoding_dim=encoding_dim,
            num_layers=num_layers,
            num_filters=num_filters,
            dropout_rate=dropout_rate,
            use_variational=False,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional skip connections."""
        self.validate_input(x)
        
        # Add channel dimension
        x = x.unsqueeze(1)
        
        # Store skip connections
        skip_connections = []
        
        # Encoder with skip connection storage
        current = x
        for i, layer in enumerate(self.encoder):
            current = layer(current)
            if self.use_skip_connections and isinstance(layer, nn.ReLU):
                skip_connections.append(current.clone())
        
        # Flatten and encode
        encoded_flat = current.view(current.size(0), -1)
        latent = self.fc_encode(encoded_flat)
        
        # Decode
        decoded_flat = self.fc_decode(latent)
        batch_size = decoded_flat.size(0)
        channels = self.num_filters * (2 ** (self.num_layers - 1))
        current = decoded_flat.view(batch_size, channels, self.encoded_length)
        
        # Decoder with skip connections
        skip_idx = len(skip_connections) - 1
        
        for i, layer in enumerate(self.decoder):
            current = layer(current)
            
            # Add skip connection if available and appropriate
            if (self.use_skip_connections and 
                isinstance(layer, nn.ReLU) and 
                skip_idx >= 0 and 
                current.shape == skip_connections[skip_idx].shape):
                current = current + skip_connections[skip_idx]
                skip_idx -= 1
        
        # Remove channel dimension
        output = current.squeeze(1)
        
        return output


class DenoisingAutoencoder(GWAutoencoder):
    """
    Denoising autoencoder for gravitational wave detection.
    
    This model is trained to reconstruct clean signals from noisy inputs,
    making it robust to detector noise while sensitive to signal anomalies.
    """
    
    def __init__(
        self,
        input_length: int,
        encoding_dim: int = 128,
        num_layers: int = 4,
        num_filters: int = 32,
        dropout_rate: float = 0.1,
        noise_factor: float = 0.1,
    ) -> None:
        """Initialize the denoising autoencoder."""
        super().__init__(
            input_length=input_length,
            encoding_dim=encoding_dim,
            num_layers=num_layers,
            num_filters=num_filters,
            dropout_rate=dropout_rate,
            use_variational=False,
        )
        
        self.noise_factor = noise_factor
    
    def add_noise(self, x: torch.Tensor, noise_factor: Optional[float] = None) -> torch.Tensor:
        """
        Add noise to input for denoising training.
        
        Args:
            x: Clean input tensor
            noise_factor: Noise level (uses self.noise_factor if None)
            
        Returns:
            Noisy input tensor
        """
        if noise_factor is None:
            noise_factor = self.noise_factor
        
        noise = torch.randn_like(x) * noise_factor
        return x + noise
    
    def forward(self, x: torch.Tensor, add_noise: bool = False) -> torch.Tensor:
        """
        Forward pass with optional noise addition.
        
        Args:
            x: Input tensor
            add_noise: Whether to add noise to input
            
        Returns:
            Reconstructed output
        """
        if add_noise and self.training:
            x_noisy = self.add_noise(x)
        else:
            x_noisy = x
        
        return super().forward(x_noisy)
