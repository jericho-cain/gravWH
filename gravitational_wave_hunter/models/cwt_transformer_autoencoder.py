#!/usr/bin/env python3
"""
CWT-Transformer Autoencoder for Gravitational Wave Detection
Combines Continuous Wavelet Transform with Transformer autoencoder for unsupervised anomaly detection

TODO: Test on real LIGO data - synthetic data produces suspiciously perfect results
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_curve, roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import pywt
import logging
from typing import Optional, Tuple, List
import math

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import signal
import pywt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import warnings

warnings.filterwarnings('ignore')
device = torch.device('cpu')

def generate_realistic_chirp(
    t: np.ndarray, 
    m1: float = 30, 
    m2: float = 30, 
    distance: float = 400, 
    noise_level: float = 1e-23
) -> np.ndarray:
    """
    Generate realistic gravitational wave chirp signal.
    
    Simulates the gravitational wave signal from a binary black hole merger
    using post-Newtonian approximations. The signal includes frequency evolution
    (chirp), amplitude scaling with mass and distance, and realistic envelope.
    
    Parameters
    ----------
    t : np.ndarray
        Time array in seconds.
    m1 : float, optional
        Mass of first black hole in solar masses, by default 30.
    m2 : float, optional
        Mass of second black hole in solar masses, by default 30.
    distance : float, optional
        Distance to the source in megaparsecs, by default 400.
    noise_level : float, optional
        Noise level (not currently used), by default 1e-23.
    
    Returns
    -------
    np.ndarray
        Gravitational wave strain signal as a function of time.
    
    Notes
    -----
    The signal includes both plus and cross polarizations combined.
    Frequency evolution follows post-Newtonian approximation.
    Amplitude scales with chirp mass and inverse distance.
    """
    # Physical parameters
    M_total = m1 + m2
    M_chirp = (m1 * m2)**(3/5) / (M_total)**(1/5)
    
    # Time to coalescence
    tc = t[-1]
    tau = tc - t
    tau[tau <= 0] = 1e-10  # Avoid division by zero
    
    # Frequency evolution (post-Newtonian approximation)
    f_0 = 35
    f = f_0 * (tau / tau[0])**(-3/8)
    f = np.clip(f, f_0, 512)
    
    # Phase evolution
    phi = 2 * np.pi * np.cumsum(f) * (t[1] - t[0])
    
    # Amplitude with realistic scaling
    # Include mass and distance dependence
    amplitude = 1e-21 * (M_chirp / 30)**(5/6) * (400 / distance)
    
    # Envelope (signal becomes stronger near coalescence)
    envelope = np.sqrt(f / f_0) * np.exp(-((t - tc) / (tc/8))**2)
    
    # Generate the signal
    signal = amplitude * envelope * np.cos(phi)
    
    return signal

def generate_noise(
    t: np.ndarray, 
    noise_level: float = 1e-23,
    sample_rate: int = 2048
) -> np.ndarray:
    """
    Generate realistic colored noise for gravitational wave detectors.
    
    Creates noise with power spectral density similar to Advanced LIGO,
    including the characteristic shape with low-frequency roll-off and
    high-frequency noise floor.
    
    Parameters
    ----------
    t : np.ndarray
        Time array in seconds.
    noise_level : float, optional
        Base noise level, by default 1e-23.
    sample_rate : int, optional
        Sampling rate in Hz, by default 2048.
    
    Returns
    -------
    np.ndarray
        Colored noise time series with LIGO-like spectral characteristics.
    
    Notes
    -----
    The noise spectrum follows the approximate Advanced LIGO sensitivity curve
    with frequency-dependent noise levels.
    """
    # Generate white noise
    white_noise = np.random.normal(0, 1, len(t))
    
    # Create frequency-dependent filter to mimic LIGO noise
    freqs = np.fft.fftfreq(len(t), 1/sample_rate)
    
    # LIGO-like noise spectrum (approximate)
    # Low frequency roll-off and high frequency noise floor
    noise_spectrum = np.ones_like(freqs)
    noise_spectrum[freqs > 0] = 1 + (freqs[freqs > 0] / 100)**2  # Low freq roll-off
    noise_spectrum[freqs > 200] = 10  # High freq noise floor
    
    # Apply filter
    filtered_noise = np.real(np.fft.ifft(np.fft.fft(white_noise) * np.sqrt(noise_spectrum)))
    
    # Scale to realistic amplitude
    noise = noise_level * filtered_noise
    
    return noise

def continuous_wavelet_transform(
    signal: np.ndarray, 
    sample_rate: int, 
    scales: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Continuous Wavelet Transform of a signal.
    
    Performs CWT using Morlet wavelets, which are optimal for gravitational
    wave analysis due to their good time-frequency localization properties.
    
    Parameters
    ----------
    signal : np.ndarray
        Input time series signal.
    sample_rate : int
        Sampling rate of the signal in Hz.
    scales : np.ndarray, optional
        Wavelet scales to use. If None, automatically chosen for GW frequency range.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - scalogram: 2D array of wavelet coefficients (magnitude)
        - frequencies: Corresponding frequency values in Hz
    
    Notes
    -----
    Uses Morlet wavelet ('morl') which is optimal for gravitational wave
    analysis. Scales are chosen to cover the typical GW frequency range
    of 20-512 Hz if not specified.
    """
    if scales is None:
        # Choose scales to cover gravitational wave frequency range (20-512 Hz)
        freqs = np.logspace(np.log10(20), np.log10(512), 64)
        scales = sample_rate / freqs
    
    # Use Morlet wavelet (good for GW chirps)
    wavelet = 'morl'
    
    # Compute CWT
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet, sampling_period=1/sample_rate)
    
    # Return magnitude (scalogram)
    scalogram = np.abs(coefficients)
    
    return scalogram, frequencies

def preprocess_with_cwt(
    strain_data: np.ndarray, 
    sample_rate: int, 
    target_height: int = 64
) -> np.ndarray:
    """
    Preprocess gravitational wave strain data using Continuous Wavelet Transform.
    
    Applies a complete preprocessing pipeline to strain data including high-pass
    filtering, whitening, CWT computation, and normalization. This prepares the
    data for neural network analysis by converting time series to time-frequency
    representations.
    
    Parameters
    ----------
    strain_data : np.ndarray
        Array of strain time series data, shape (n_samples, n_time_points).
    sample_rate : int
        Sampling rate of the data in Hz.
    target_height : int, optional
        Target height for the CWT scalograms, by default 64.
    
    Returns
    -------
    np.ndarray
        Preprocessed CWT data with shape (n_samples, target_height, n_time_points).
        Each sample is a time-frequency representation (scalogram).
    
    Notes
    -----
    The preprocessing pipeline includes:
    1. High-pass filtering (20 Hz cutoff)
    2. Whitening (zero mean, unit variance)
    3. CWT computation with Morlet wavelets
    4. Resizing to target dimensions
    5. Log transformation and normalization
    """
    cwt_data = []
    
    logger.info(f"Computing CWT for {len(strain_data)} samples...")
    
    for i, strain in enumerate(strain_data):
        # Apply basic preprocessing
        # High-pass filter to remove low-frequency noise
        sos = signal.butter(4, 20, btype='highpass', fs=sample_rate, output='sos')
        filtered = signal.sosfilt(sos, strain)
        
        # Whiten the data
        whitened = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-10)
        
        # Compute CWT
        scalogram, freqs = continuous_wavelet_transform(whitened, sample_rate)
        
        # Resize to target height if needed
        if scalogram.shape[0] != target_height:
            # Simple interpolation to target size
            from scipy.ndimage import zoom
            zoom_factor = target_height / scalogram.shape[0]
            scalogram = zoom(scalogram, (zoom_factor, 1), order=1)
        
        # Log transform and normalize (crucial for neural networks)
        log_scalogram = np.log10(scalogram + 1e-10)
        normalized = (log_scalogram - np.mean(log_scalogram)) / (np.std(log_scalogram) + 1e-10)
        
        cwt_data.append(normalized)
        
        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i+1}/{len(strain_data)} samples")
    
    return np.array(cwt_data)

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer architecture.
    
    Adds positional information to the input embeddings using sine and cosine
    functions of different frequencies.
    
    Parameters
    ----------
    d_model : int
        Dimension of the model embeddings.
    max_len : int
        Maximum sequence length.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Parameters
        ----------
        x : torch.Tensor
            Input embeddings of shape (seq_len, batch_size, d_model).
        
        Returns
        -------
        torch.Tensor
            Embeddings with positional encoding added.
        """
        return x + self.pe[:x.size(0), :]

class TransformerEncoder(nn.Module):
    """
    Transformer encoder block for temporal modeling.
    
    Implements a transformer encoder with multi-head self-attention and
    feed-forward networks for processing temporal sequences.
    
    Parameters
    ----------
    d_model : int
        Dimension of the model embeddings.
    nhead : int
        Number of attention heads.
    dim_feedforward : int
        Dimension of the feed-forward network.
    dropout : float
        Dropout rate.
    """
    
    def __init__(self, d_model: int, nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_model).
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed-forward
        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x

class CWT_Transformer_Autoencoder(nn.Module):
    """
    Transformer Autoencoder for gravitational wave detection using CWT scalograms.
    
    A hybrid neural network architecture that combines 2D convolutional layers
    for spatial feature extraction with transformer layers for temporal modeling.
    Designed for unsupervised anomaly detection in gravitational wave data.
    
    The model learns to reconstruct normal noise patterns and identifies
    anomalies (potential GW signals) through high reconstruction error.
    
    Attributes
    ----------
    input_height : int
        Height of input CWT scalograms.
    input_width : int
        Width of input CWT scalograms.
    latent_dim : int
        Dimension of the latent space representation.
    spatial_encoder : nn.Sequential
        CNN encoder for spatial feature extraction.
    temporal_encoder : TransformerEncoder
        Transformer encoder for temporal sequence modeling.
    to_latent : nn.Linear
        Linear layer mapping to latent space.
    from_latent : nn.Linear
        Linear layer mapping from latent space.
    temporal_decoder : TransformerEncoder
        Transformer decoder for temporal sequence generation.
    spatial_decoder : nn.Sequential
        CNN decoder for spatial feature reconstruction.
    """
    
    def __init__(
        self, 
        input_height: int, 
        input_width: int, 
        latent_dim: int = 32, 
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2
    ) -> None:
        super().__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        self.latent_dim = latent_dim
        self.d_model = d_model
        
        # Encoder: 2D CNN to extract spatial features + Transformer for temporal modeling
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, input_width//4))  # Reduce spatial dimensions
        )
        
        # Transformer encoder for temporal evolution
        self.temporal_encoder = nn.ModuleList([
            TransformerEncoder(d_model, nhead, d_model * 4, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_width//4)
        
        # Projection to transformer dimension
        self.spatial_to_temporal = nn.Linear(32 * 8, d_model)
        
        # Latent space
        self.to_latent = nn.Linear(d_model, latent_dim)
        
        # Decoder
        self.from_latent = nn.Linear(latent_dim, d_model)
        
        # Transformer decoder
        self.temporal_decoder = nn.ModuleList([
            TransformerEncoder(d_model, nhead, d_model * 4, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # Projection back to spatial features
        self.temporal_to_spatial = nn.Linear(d_model, 32 * 8 * (input_width // 4))
        
        # Spatial decoder - simplified to match the encoder structure
        self.spatial_decoder = nn.Sequential(
            nn.Linear(32 * 8 * (input_width // 4), 32 * 8 * (input_width // 4)),  # Expand to spatial features
            nn.ReLU(),
            nn.Unflatten(1, (32, 8, input_width // 4)),  # Reshape to spatial dimensions
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ConvTranspose2d(16, 1, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((input_height, input_width)),  # Ensure exact output dimensions
            nn.Tanh()  # Output in [-1, 1] range
        )
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input scalogram to latent representation.
        
        Processes the input through spatial and temporal encoders to produce
        a compact latent representation of the time-frequency patterns.
        
        Parameters
        ----------
        x : torch.Tensor
            Input CWT scalogram tensor of shape (batch_size, 1, height, width).
        
        Returns
        -------
        torch.Tensor
            Latent representation of shape (batch_size, latent_dim).
        """
        batch_size, channels, height, width = x.size()
        
        # Spatial encoding (treat each time step as separate image)
        # Reshape: (batch, 1, height, width) -> (batch*time, 1, height, width)
        x_reshaped = x.view(-1, 1, height, width)
        spatial_features = self.spatial_encoder(x_reshaped)  # (batch*time, 32, 8, width//4)
        
        # Reshape back for temporal modeling
        spatial_flat = spatial_features.view(batch_size, width//4, -1)  # (batch, time, features)
        
        # Project to transformer dimension
        temporal_input = self.spatial_to_temporal(spatial_flat)  # (batch, time, d_model)
        
        # Add positional encoding
        temporal_input = temporal_input.transpose(0, 1)  # (time, batch, d_model)
        temporal_input = self.pos_encoder(temporal_input)
        temporal_input = temporal_input.transpose(0, 1)  # (batch, time, d_model)
        
        # Temporal encoding with transformer
        temporal_out = temporal_input
        for transformer_layer in self.temporal_encoder:
            temporal_out = transformer_layer(temporal_out)
        
        # Use mean pooling over time dimension
        temporal_pooled = temporal_out.mean(dim=1)  # (batch, d_model)
        
        # Map to latent space
        latent = self.to_latent(temporal_pooled)  # (batch, latent_dim)
        
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to scalogram.
        
        Reconstructs the original CWT scalogram from the latent representation
        using temporal and spatial decoders.
        
        Parameters
        ----------
        latent : torch.Tensor
            Latent representation of shape (batch_size, latent_dim).
        
        Returns
        -------
        torch.Tensor
            Reconstructed scalogram of shape (batch_size, 1, height, width).
        """
        batch_size = latent.size(0)
        
        # Expand latent to sequence
        decoded_features = self.from_latent(latent)  # (batch, d_model)
        
        # Create sequence by repeating latent
        sequence_length = self.input_width // 4
        sequence = decoded_features.unsqueeze(1).repeat(1, sequence_length, 1)  # (batch, time, d_model)
        
        # Add positional encoding
        sequence = sequence.transpose(0, 1)  # (time, batch, d_model)
        sequence = self.pos_encoder(sequence)
        sequence = sequence.transpose(0, 1)  # (batch, time, d_model)
        
        # Temporal decoding with transformer
        temporal_out = sequence
        for transformer_layer in self.temporal_decoder:
            temporal_out = transformer_layer(temporal_out)
        
        # Project back to spatial features
        spatial_features = self.temporal_to_spatial(temporal_out)  # (batch, time, spatial_features)
        
        # Use the last output from the transformer for reconstruction
        last_output = spatial_features[:, -1, :]  # (batch, spatial_features)
        
        # Spatial decoding
        reconstructed = self.spatial_decoder(last_output)  # (batch, 1, height, width)
        
        return reconstructed
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Encodes the input to latent space and decodes back to reconstruction.
        
        Parameters
        ----------
        x : torch.Tensor
            Input CWT scalogram tensor of shape (batch_size, 1, height, width).
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - reconstructed: Reconstructed scalogram
            - latent: Latent representation
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent

class SimpleCWTTransformerAutoencoder(nn.Module):
    """
    Simplified CWT Transformer Autoencoder for gravitational wave detection.
    
    A streamlined version of the CWT-Transformer autoencoder that uses only
    convolutional layers and a simple transformer for easier training and understanding.
    This model is more stable to train and provides a good baseline for comparison.
    
    Attributes
    ----------
    height : int
        Height of input CWT scalograms.
    width : int
        Width of input CWT scalograms.
    encoder : nn.Sequential
        Convolutional encoder network.
    decoder : nn.Sequential
        Convolutional decoder network.
    transformer : TransformerEncoder
        Simple transformer for temporal modeling.
    """
    
    def __init__(self, height: int, width: int, latent_dim: int = 64, d_model: int = 128):
        super().__init__()
        
        self.height = height
        self.width = width
        self.d_model = d_model
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),  # Fixed size output
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, d_model),
            nn.ReLU()
        )
        
        # Simple transformer
        self.transformer = TransformerEncoder(d_model, nhead=4, dim_feedforward=d_model*2, dropout=0.1)
        self.pos_encoder = PositionalEncoding(d_model, max_len=100)
        
        # Latent projection
        self.to_latent = nn.Linear(d_model, latent_dim)
        self.from_latent = nn.Linear(latent_dim, d_model)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 64 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the simplified transformer autoencoder.
        
        Encodes the input to latent space and decodes back to reconstruction.
        
        Parameters
        ----------
        x : torch.Tensor
            Input CWT scalogram tensor of shape (batch_size, 1, height, width).
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - reconstructed: Reconstructed scalogram
            - latent: Latent representation
        """
        batch_size = x.size(0)
        
        # Encode
        encoded = self.encoder(x)  # (batch, d_model)
        
        # Add transformer processing
        encoded = encoded.unsqueeze(1)  # (batch, 1, d_model)
        encoded = encoded.transpose(0, 1)  # (1, batch, d_model)
        encoded = self.pos_encoder(encoded)
        encoded = self.transformer(encoded)
        encoded = encoded.transpose(0, 1).squeeze(1)  # (batch, d_model)
        
        # Latent space
        latent = self.to_latent(encoded)
        
        # Decode
        decoded_features = self.from_latent(latent)
        reconstructed = self.decoder(decoded_features)
        
        # Resize to original dimensions if needed
        if reconstructed.shape[-2:] != (self.height, self.width):
            reconstructed = torch.nn.functional.interpolate(
                reconstructed, size=(self.height, self.width), mode='bilinear', align_corners=False
            )
        
        return reconstructed, latent

def train_autoencoder(
    model: nn.Module, 
    noise_loader: DataLoader, 
    num_epochs: int = 50, 
    lr: float = 0.001
) -> List[float]:
    """
    Train autoencoder on noise-only data for anomaly detection.
    
    Trains the autoencoder to reconstruct normal noise patterns. The model
    learns to minimize reconstruction error for noise data, so that when
    presented with gravitational wave signals (anomalies), it will have
    higher reconstruction error.
    
    Parameters
    ----------
    model : nn.Module
        The autoencoder model to train.
    noise_loader : DataLoader
        DataLoader containing only noise data for training.
    num_epochs : int, optional
        Number of training epochs, by default 50.
    lr : float, optional
        Learning rate, by default 0.001.
    
    Returns
    -------
    List[float]
        List of training losses for each epoch.
    
    Notes
    -----
    Uses MSE loss for reconstruction error. The model learns to reconstruct
    noise patterns well, so gravitational wave signals will have higher
    reconstruction error and can be detected as anomalies.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    
    logger.info(f"Training {model.__class__.__name__} for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, _) in enumerate(noise_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, _ = model(data)
            
            # Compute loss
            loss = criterion(reconstructed, data)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed. Average loss: {avg_loss:.6f}")
    
    return losses

def evaluate_autoencoder(
    model: nn.Module, 
    test_loader: DataLoader
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate autoencoder on test data to compute reconstruction errors.
    
    Computes reconstruction errors for all test samples. This is used to
    determine the threshold for anomaly detection.
    
    Parameters
    ----------
    model : nn.Module
        The trained autoencoder model.
    test_loader : DataLoader
        DataLoader containing test data.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - reconstruction_errors: Array of reconstruction errors for each sample
        - labels: Array of true labels (0 for noise, 1 for signal)
    
    Notes
    -----
    Reconstruction error is computed as MSE between input and output.
    Higher reconstruction error indicates potential gravitational wave signals.
    """
    model.eval()
    reconstruction_errors = []
    labels = []
    
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            
            # Forward pass
            reconstructed, _ = model(data)
            
            # Compute reconstruction error (MSE)
            error = torch.mean((data - reconstructed) ** 2, dim=[1, 2, 3])
            
            reconstruction_errors.extend(error.cpu().numpy())
            labels.extend(label.numpy())
    
    return np.array(reconstruction_errors), np.array(labels)

def compute_optimal_threshold(
    reconstruction_errors: np.ndarray, 
    labels: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute optimal threshold for anomaly detection using precision-recall analysis.
    
    Finds the threshold that maximizes F1-score, balancing precision and recall
    for gravitational wave detection.
    
    Parameters
    ----------
    reconstruction_errors : np.ndarray
        Array of reconstruction errors for all test samples.
    labels : np.ndarray
        Array of true labels (0 for noise, 1 for signal).
    
    Returns
    -------
    Tuple[float, float, float]
        A tuple containing:
        - optimal_threshold: Threshold that maximizes F1-score
        - optimal_precision: Precision at optimal threshold
        - optimal_recall: Recall at optimal threshold
    
    Notes
    -----
    Uses sklearn's precision_recall_curve to find the optimal threshold.
    The threshold is chosen to maximize F1-score, which balances precision
    and recall for gravitational wave detection.
    """
    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(labels, reconstruction_errors)
    
    # Compute F1-score for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Find threshold that maximizes F1-score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
    optimal_precision = precision[optimal_idx]
    optimal_recall = recall[optimal_idx]
    
    return optimal_threshold, optimal_precision, optimal_recall

def main():
    """
    Main function to run the CWT-Transformer autoencoder analysis.
    
    This function demonstrates the complete pipeline for gravitational wave
    detection using the CWT-Transformer autoencoder approach.
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Parameters - More realistic for testing
    sample_rate = 2048
    duration = 1.0
    num_samples = 1000
    num_signals = 100
    
    # Add more realistic noise levels
    noise_level = 1e-22  # Increased noise level
    
    logger.info("Generating synthetic gravitational wave data...")
    
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate noise samples
    noise_samples = []
    for _ in range(num_samples):
        noise = generate_noise(t, noise_level=noise_level, sample_rate=sample_rate)
        noise_samples.append(noise)
    
    # Generate signal samples
    signal_samples = []
    for _ in range(num_signals):
        # Random parameters for variety
        m1 = np.random.uniform(10, 80)
        m2 = np.random.uniform(10, 80)
        distance = np.random.uniform(200, 800)
        
        signal = generate_realistic_chirp(t, m1, m2, distance, noise_level=noise_level)
        signal_samples.append(signal)
    
    # Combine and preprocess
    all_samples = noise_samples + signal_samples
    labels = [0] * len(noise_samples) + [1] * len(signal_samples)
    
    logger.info("Preprocessing data with CWT...")
    cwt_data = preprocess_with_cwt(np.array(all_samples), sample_rate)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        cwt_data, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # Separate noise data for training (unsupervised)
    noise_indices = np.where(np.array(y_train) == 0)[0]
    X_train_noise = X_train[noise_indices]
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_noise).unsqueeze(1),  # Add channel dimension
        torch.LongTensor([0] * len(X_train_noise))  # Dummy labels
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test).unsqueeze(1),  # Add channel dimension
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    input_height, input_width = X_train.shape[1], X_train.shape[2]
    
    logger.info("Initializing CWT-Transformer Autoencoder...")
    model = SimpleCWTTransformerAutoencoder(
        height=input_height,
        width=input_width,
        latent_dim=32,
        d_model=64
    ).to(device)
    
    # Train model
    losses = train_autoencoder(model, train_loader, num_epochs=20, lr=0.001)
    
    # Quick validation: Test on a few samples to see if reconstruction is working
    logger.info("Validating model reconstruction capability...")
    model.eval()
    with torch.no_grad():
        # Get a few samples from test set
        test_batch = next(iter(test_loader))
        test_data, test_labels = test_batch
        test_data = test_data.to(device)
        
        # Reconstruct
        reconstructed, _ = model(test_data)
        
        # Check reconstruction quality
        mse_loss = torch.mean((test_data - reconstructed) ** 2)
        logger.info(f"  Test reconstruction MSE: {mse_loss.item():.6f}")
        
        # Check if noise samples have lower error than signal samples
        noise_mask = test_labels == 0
        signal_mask = test_labels == 1
        
        if torch.any(noise_mask) and torch.any(signal_mask):
            noise_errors = torch.mean((test_data[noise_mask] - reconstructed[noise_mask]) ** 2, dim=[1,2,3])
            signal_errors = torch.mean((test_data[signal_mask] - reconstructed[signal_mask]) ** 2, dim=[1,2,3])
            
            logger.info(f"  Noise reconstruction error: {torch.mean(noise_errors).item():.6f}")
            logger.info(f"  Signal reconstruction error: {torch.mean(signal_errors).item():.6f}")
            logger.info(f"  Error ratio (signal/noise): {torch.mean(signal_errors).item() / torch.mean(noise_errors).item():.3f}")
    
    # Evaluate model
    logger.info("Evaluating model performance...")
    reconstruction_errors, labels = evaluate_autoencoder(model, test_loader)
    
    # Compute optimal threshold
    optimal_threshold, optimal_precision, optimal_recall = compute_optimal_threshold(
        reconstruction_errors, labels
    )
    
    # Compute additional metrics
    predictions = (reconstruction_errors > optimal_threshold).astype(int)
    
    # Calculate metrics
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    
    # ROC and AUC
    fpr, tpr, _ = roc_curve(labels, reconstruction_errors)
    auc_score = auc(fpr, tpr)
    
    # Average precision
    avg_precision = precision_score(labels, predictions, average='weighted')
    
    logger.info("Results:")
    logger.info(f"Optimal Threshold: {optimal_threshold:.6f}")
    logger.info(f"Precision: {precision:.3f}")
    logger.info(f"Recall: {recall:.3f}")
    logger.info(f"F1-Score: {f1:.3f}")
    logger.info(f"Accuracy: {accuracy:.3f}")
    logger.info(f"AUC: {auc_score:.3f}")
    logger.info(f"Average Precision: {avg_precision:.3f}")
    
    # Debug: Print detailed statistics
    logger.info("DEBUG: Reconstruction Error Statistics:")
    logger.info(f"  Mean reconstruction error: {np.mean(reconstruction_errors):.6f}")
    logger.info(f"  Std reconstruction error: {np.std(reconstruction_errors):.6f}")
    logger.info(f"  Min reconstruction error: {np.min(reconstruction_errors):.6f}")
    logger.info(f"  Max reconstruction error: {np.max(reconstruction_errors):.6f}")
    
    # Check if errors are different between classes
    noise_errors = reconstruction_errors[labels == 0]
    signal_errors = reconstruction_errors[labels == 1]
    logger.info(f"  Noise errors - Mean: {np.mean(noise_errors):.6f}, Std: {np.std(noise_errors):.6f}")
    logger.info(f"  Signal errors - Mean: {np.mean(signal_errors):.6f}, Std: {np.std(signal_errors):.6f}")
    
    # Plot results
    plt.figure(figsize=(20, 5))
    
    # Training loss
    plt.subplot(1, 4, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # ROC curve
    plt.subplot(1, 4, 2)
    plt.plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    
    # Precision-Recall curve
    plt.subplot(1, 4, 3)
    precision_curve, recall_curve, _ = precision_recall_curve(labels, reconstruction_errors)
    plt.plot(recall_curve, precision_curve, label=f'PR (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    
    # Reconstruction error distribution
    plt.subplot(1, 4, 4)
    plt.hist(noise_errors, alpha=0.7, label='Noise', bins=20, density=True)
    plt.hist(signal_errors, alpha=0.7, label='Signal', bins=20, density=True)
    plt.axvline(optimal_threshold, color='red', linestyle='--', label=f'Threshold: {optimal_threshold:.6f}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Error Distribution by Class')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('cwt_transformer_autoencoder_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("Analysis complete! Results saved to 'cwt_transformer_autoencoder_results.png'")

if __name__ == "__main__":
    main()
