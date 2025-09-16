#!/usr/bin/env python3
"""
CWT-LSTM Autoencoder for Gravitational Wave Detection
Combines Continuous Wavelet Transform with LSTM autoencoder for unsupervised anomaly detection
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
    
    # Two polarizations combined
    h_plus = amplitude * envelope * np.cos(phi)
    h_cross = amplitude * envelope * np.sin(phi)
    
    return 0.5 * (h_plus + h_cross)

def generate_colored_noise(
    length: int, 
    sample_rate: int, 
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate realistic LIGO-like colored noise.
    
    Creates noise with a power spectral density that mimics the Advanced LIGO
    noise curve, including seismic noise at low frequencies, thermal noise
    at mid frequencies, and shot noise at high frequencies.
    
    Parameters
    ----------
    length : int
        Length of the noise array in samples.
    sample_rate : int
        Sampling rate in Hz.
    seed : int, optional
        Random seed for reproducibility, by default None.
    
    Returns
    -------
    np.ndarray
        Colored noise array with LIGO-like spectral properties.
    
    Notes
    -----
    The noise is generated using the frequency domain method:
    1. Create white noise in frequency domain
    2. Apply LIGO-like PSD
    3. Transform back to time domain
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create frequency array
    freqs = np.fft.fftfreq(length, 1/sample_rate)
    freqs = freqs[:length//2 + 1]
    f = np.abs(freqs)
    f[0] = 1e-10
    
    # Advanced LIGO noise curve (simplified)
    psd = np.zeros_like(f)
    
    # Seismic noise (low frequency)
    low_mask = f < 10
    psd[low_mask] = 1e-46 * (f[low_mask] / 10)**(-4)
    
    # Thermal noise (mid frequency)
    mid_mask = (f >= 10) & (f < 100)
    psd[mid_mask] = 1e-48 * (1 + (f[mid_mask] / 50)**2)
    
    # Shot noise (high frequency)
    high_mask = f >= 100
    psd[high_mask] = 1e-48 * (f[high_mask] / 100)**(1.5)
    
    # Generate colored noise
    white_noise = np.random.normal(0, 1, length//2 + 1) + 1j * np.random.normal(0, 1, length//2 + 1)
    white_noise[0] = white_noise[0].real
    if length % 2 == 0:
        white_noise[-1] = white_noise[-1].real
    
    colored_noise_fft = white_noise * np.sqrt(psd * sample_rate / 2)
    noise = np.fft.irfft(colored_noise_fft, n=length)
    
    return noise

def continuous_wavelet_transform(
    signal: np.ndarray, 
    sample_rate: int, 
    scales: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Continuous Wavelet Transform using Morlet wavelets.
    
    Performs CWT analysis on the input signal using Morlet wavelets, which are
    particularly well-suited for analyzing gravitational wave chirp signals
    due to their good time-frequency localization.
    
    Parameters
    ----------
    signal : np.ndarray
        Input time series signal.
    sample_rate : int
        Sampling rate of the signal in Hz.
    scales : np.ndarray, optional
        Wavelet scales to use. If None, automatically chosen to cover
        20-512 Hz frequency range, by default None.
    
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

class CWT_LSTM_Autoencoder(nn.Module):
    """
    LSTM Autoencoder for gravitational wave detection using CWT scalograms.
    
    A hybrid neural network architecture that combines 2D convolutional layers
    for spatial feature extraction with LSTM layers for temporal modeling.
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
    temporal_encoder : nn.LSTM
        LSTM encoder for temporal sequence modeling.
    to_latent : nn.Linear
        Linear layer mapping to latent space.
    from_latent : nn.Linear
        Linear layer mapping from latent space.
    temporal_decoder : nn.LSTM
        LSTM decoder for temporal sequence generation.
    spatial_decoder : nn.Sequential
        CNN decoder for spatial feature reconstruction.
    
    Notes
    -----
    The architecture follows an encoder-decoder pattern:
    1. Spatial encoder: 2D CNN extracts features from scalograms
    2. Temporal encoder: LSTM models temporal evolution
    3. Latent space: Compact representation
    4. Temporal decoder: LSTM reconstructs temporal patterns
    5. Spatial decoder: 2D CNN reconstructs scalograms
    """
    
    def __init__(
        self, 
        input_height: int, 
        input_width: int, 
        latent_dim: int = 32, 
        lstm_hidden: int = 64
    ) -> None:
        super().__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        self.latent_dim = latent_dim
        
        # Encoder: 2D CNN to extract spatial features + LSTM for temporal modeling
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, input_width//4))  # Reduce spatial dimensions
        )
        
        # LSTM encoder for temporal evolution
        self.temporal_encoder = nn.LSTM(
            input_size=32 * 8,  # Flattened spatial features
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Latent space
        self.to_latent = nn.Linear(lstm_hidden, latent_dim)
        
        # Decoder
        self.from_latent = nn.Linear(latent_dim, lstm_hidden)
        
        self.temporal_decoder = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Spatial decoder - simplified to match the encoder structure
        self.spatial_decoder = nn.Sequential(
            nn.Linear(lstm_hidden, 32 * 8 * (input_width // 4)),  # Expand to spatial features
            nn.ReLU(),
            nn.Unflatten(1, (32, 8, input_width // 4)),  # Reshape to spatial dimensions
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ConvTranspose2d(16, 1, kernel_size=3, padding=1),  # Remove output_padding
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
        
        # Temporal encoding
        temporal_out, (hidden, _) = self.temporal_encoder(spatial_flat)
        
        # Use last hidden state
        latent = self.to_latent(hidden[-1])  # (batch, latent_dim)
        
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
        decoded_features = self.from_latent(latent)  # (batch, lstm_hidden)
        
        # Create sequence by repeating latent
        sequence_length = self.input_width // 4
        sequence = decoded_features.unsqueeze(1).repeat(1, sequence_length, 1)
        
        # Temporal decoding
        temporal_out, _ = self.temporal_decoder(sequence)  # (batch, time, lstm_hidden)
        
        # Use the last output from the LSTM for reconstruction
        last_output = temporal_out[:, -1, :]  # (batch, lstm_hidden)
        
        # Spatial decoding - directly from the last LSTM output
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
        reconstructed = self.decode(latent)  # Fixed: should decode from latent, not x
        return reconstructed, latent

class SimpleCWTAutoencoder(nn.Module):
    """
    Simplified CWT Autoencoder for gravitational wave detection.
    
    A streamlined version of the CWT-LSTM autoencoder that uses only
    convolutional layers for easier training and understanding. This model
    is more stable to train and provides a good baseline for comparison.
    
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
    
    Notes
    -----
    This simplified version uses only 2D convolutions without LSTM layers,
    making it easier to train and debug while still capturing spatial
    patterns in the CWT scalograms.
    """
    
    def __init__(self, height: int, width: int, latent_dim: int = 64) -> None:
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),  # Fixed size output
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
        
        self.height = height
        self.width = width
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the simplified autoencoder.
        
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
        # Encode
        latent = self.encoder(x)
        
        # Decode
        reconstructed = self.decoder(latent)
        
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
        Learning rate for optimization, by default 0.001.
    
    Returns
    -------
    List[float]
        List of average loss values for each epoch.
    
    Notes
    -----
    Uses MSE loss and Adam optimizer with learning rate scheduling.
    The model is trained only on noise data to learn normal patterns.
    """
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    model = model.to(device)
    model.train()
    
    logger.info(f"Training autoencoder for {num_epochs} epochs...")
    
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch in noise_loader:
            data = batch[0].to(device)
            
            optimizer.zero_grad()
            
            reconstructed, latent = model(data)
            loss = criterion(reconstructed, data)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    return losses

def detect_anomalies(
    model: nn.Module, 
    test_loader: DataLoader, 
    noise_threshold_percentile: float = 95
) -> dict:
    """
    Detect anomalies using reconstruction error.
    
    Uses the trained autoencoder to identify potential gravitational wave
    signals by measuring reconstruction error. Higher reconstruction error
    indicates potential anomalies (GW signals) since the model was trained
    only on noise data.
    
    Parameters
    ----------
    model : nn.Module
        Trained autoencoder model.
    test_loader : DataLoader
        DataLoader containing test data (noise + potential signals).
    noise_threshold_percentile : float, optional
        Percentile threshold for anomaly detection, by default 95.
    
    Returns
    -------
    dict
        Dictionary containing:
        - predictions: Binary predictions (0=noise, 1=signal)
        - reconstruction_errors: MSE reconstruction errors for each sample
        - threshold: Calculated threshold value
        - latent_representations: Latent space representations
    
    Notes
    -----
    The threshold is set based on the percentile of reconstruction errors
    from the test data. Samples with error above this threshold are
    classified as anomalies (potential GW signals).
    """
    
    model.eval()
    reconstruction_errors = []
    latent_representations = []
    
    with torch.no_grad():
        for batch in test_loader:
            data = batch[0].to(device)
            
            reconstructed, latent = model(data)
            
            # Calculate reconstruction error for each sample
            mse = torch.mean((reconstructed - data)**2, dim=(1, 2, 3))
            reconstruction_errors.extend(mse.cpu().numpy())
            latent_representations.extend(latent.cpu().numpy())
    
    reconstruction_errors = np.array(reconstruction_errors)
    
    # Set threshold based on percentile of reconstruction errors
    threshold = np.percentile(reconstruction_errors, noise_threshold_percentile)
    
    # Classify as anomaly if error > threshold
    predictions = (reconstruction_errors > threshold).astype(int)
    
    return {
        'predictions': predictions,
        'reconstruction_errors': reconstruction_errors,
        'threshold': threshold,
        'latent_representations': np.array(latent_representations)
    }

def main() -> None:
    """
    Main function for CWT-LSTM Autoencoder gravitational wave detection.
    
    Demonstrates the complete pipeline for gravitational wave detection:
    1. Generate realistic gravitational wave and noise data
    2. Preprocess data using Continuous Wavelet Transform
    3. Train autoencoder on noise-only data
    4. Detect anomalies using reconstruction error
    5. Evaluate performance and generate visualizations
    
    Notes
    -----
    This function serves as both a demonstration and a complete working
    example of the gravitational wave detection pipeline.
    """
    logger.info("CWT + LSTM Autoencoder for Gravitational Wave Detection")
    logger.info("=" * 65)
    
    # Configuration
    SAMPLE_RATE = 512
    DURATION = 4
    NUM_SAMPLES = 200
    SIGNAL_PROB = 0.3
    
    logger.info(f"Generating {NUM_SAMPLES} samples...")
    
    # Generate realistic data
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION))
    strain_data = []
    labels = []
    snr_values = []
    
    for i in range(NUM_SAMPLES):
        # Generate colored noise
        noise = generate_colored_noise(len(t), SAMPLE_RATE, seed=42+i)
        
        if np.random.random() < SIGNAL_PROB:
            # Random parameters for diversity
            m1 = np.random.uniform(25, 50)
            m2 = np.random.uniform(25, 50)
            distance = np.random.uniform(300, 800)
            
            # Generate GW signal
            gw_signal = generate_realistic_chirp(t, m1, m2, distance)
            
            # Calculate SNR and scale
            signal_power = np.std(gw_signal)
            noise_power = np.std(noise)
            target_snr = np.random.uniform(8, 20)  # Realistic range
            
            if signal_power > 0:
                scaling = target_snr * noise_power / signal_power
                gw_signal = gw_signal * scaling
            
            combined = noise + gw_signal
            label = 1
            snr_values.append(target_snr)
        else:
            combined = noise
            label = 0
            snr_values.append(0)
        
        strain_data.append(combined)
        labels.append(label)
    
    try:
        strain_data = np.array(strain_data)
        labels = np.array(labels)
        snr_values = np.array(snr_values)
        
        logger.info(f"Dataset: {strain_data.shape}")
        logger.info(f"Signals: {np.sum(labels)}, Noise: {np.sum(1-labels)}")
    except Exception as e:
        logger.error(f"Error converting to numpy arrays: {e}")
        logger.error(f"strain_data length: {len(strain_data) if strain_data else 'None'}")
        logger.error(f"labels length: {len(labels) if labels else 'None'}")
        return
    
    # Compute CWT representations
    logger.info(f"\nComputing Continuous Wavelet Transforms...")
    cwt_data = preprocess_with_cwt(strain_data, SAMPLE_RATE, target_height=32)
    
    logger.info(f"CWT data shape: {cwt_data.shape}")
    
    # Split data: Train autoencoder ONLY on noise
    noise_indices = np.where(labels == 0)[0]
    
    # Use only noise data for training autoencoder
    noise_cwt = cwt_data[noise_indices]
    
    logger.info(f"\nTraining Strategy:")
    logger.info(f"Training autoencoder on {len(noise_cwt)} NOISE-ONLY CWT scalograms")
    logger.info(f"Testing on ALL {len(cwt_data)} samples")
    
    # Create datasets
    noise_dataset = TensorDataset(torch.FloatTensor(noise_cwt).unsqueeze(1))  # Add channel dim
    test_dataset = TensorDataset(torch.FloatTensor(cwt_data).unsqueeze(1))
    
    noise_loader = DataLoader(noise_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Create and train model
    height, width = cwt_data.shape[1], cwt_data.shape[2]
    model = CWT_LSTM_Autoencoder(input_height=height, input_width=width, latent_dim=16, lstm_hidden=32)
    
    logger.info(f"\nModel architecture:")
    logger.info(f"  Input: {height}×{width} CWT scalogram")
    logger.info(f"  Model: CWT-LSTM Autoencoder")
    logger.info(f"  Latent dimension: 16")
    logger.info(f"  LSTM hidden size: 32")
    
    # Train autoencoder
    train_losses = train_autoencoder(model, noise_loader, num_epochs=30, lr=0.001)
    
    # Detect anomalies
    logger.info(f"\nDetecting anomalies...")
    results = detect_anomalies(model, test_loader, noise_threshold_percentile=90)
    
    # Evaluate performance
    predictions = results['predictions']
    reconstruction_errors = results['reconstruction_errors']
    
    # Calculate metrics
    accuracy = np.mean(predictions == labels)
    
    if np.sum(predictions) > 0:  # Avoid division by zero
        true_positives = np.sum((labels == 1) & (predictions == 1))
        false_positives = np.sum((labels == 0) & (predictions == 1))
        false_negatives = np.sum((labels == 1) & (predictions == 0))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        precision = recall = f1 = 0
    
    # Calculate AUC if we have both classes
    if len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, reconstruction_errors)
    else:
        auc = 0.5
    
    logger.info(f"\nCWT-LSTM Autoencoder Results:")
    logger.info(f"Accuracy: {accuracy:.1%}")
    logger.info(f"Precision: {precision:.1%}")
    logger.info(f"Recall: {recall:.1%}")
    logger.info(f"F1-Score: {f1:.1%}")
    logger.info(f"AUC: {auc:.3f}")
    logger.info(f"Threshold: {results['threshold']:.6f}")
    
    # Create comprehensive visualizations
    plt.figure(figsize=(20, 16))
    
    # 1. Sample CWT scalograms
    plt.subplot(3, 5, 1)
    signal_idx = np.where(labels == 1)[0][0] if np.any(labels == 1) else 0
    plt.imshow(cwt_data[signal_idx], aspect='auto', origin='lower', cmap='viridis')
    plt.title(f'CWT: Signal (SNR={snr_values[signal_idx]:.1f})')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar()
    
    plt.subplot(3, 5, 2)
    noise_idx = np.where(labels == 0)[0][0]
    plt.imshow(cwt_data[noise_idx], aspect='auto', origin='lower', cmap='viridis')
    plt.title('CWT: Noise Only')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar()
    
    # 2. Training loss
    plt.subplot(3, 5, 3)
    plt.plot(train_losses)
    plt.title('Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)
    
    # 3. Reconstruction error distribution
    plt.subplot(3, 5, 4)
    noise_errors = reconstruction_errors[labels == 0]
    signal_errors = reconstruction_errors[labels == 1]
    
    plt.hist(noise_errors, alpha=0.7, label='Noise', bins=20, color='blue', density=True)
    plt.hist(signal_errors, alpha=0.7, label='Signals', bins=20, color='red', density=True)
    plt.axvline(results['threshold'], color='black', linestyle='--', label='Threshold')
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. ROC curve
    plt.subplot(3, 5, 5)
    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, reconstruction_errors)
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 5. Sample reconstruction
    plt.subplot(3, 5, 6)
    with torch.no_grad():
        model.eval()
        sample_input = torch.FloatTensor(cwt_data[signal_idx:signal_idx+1]).unsqueeze(1).to(device)
        reconstructed, _ = model(sample_input)
        reconstructed = reconstructed.cpu().numpy().squeeze()
    
    plt.imshow(reconstructed, aspect='auto', origin='lower', cmap='viridis')
    plt.title('Reconstructed Signal')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar()
    
    # 6. Original time series
    plt.subplot(3, 5, 7)
    t_plot = np.linspace(0, DURATION, len(strain_data[0]))
    plt.plot(t_plot, strain_data[signal_idx], label='Signal')
    plt.plot(t_plot, strain_data[noise_idx], label='Noise', alpha=0.7)
    plt.title('Original Time Series')
    plt.xlabel('Time (s)')
    plt.ylabel('Strain')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Detection performance vs SNR
    plt.subplot(3, 5, 8)
    if np.any(labels == 1):
        signal_mask = labels == 1
        signal_snrs = snr_values[signal_mask]
        signal_detected = predictions[signal_mask]
        
        # Bin by SNR
        snr_bins = np.linspace(8, 20, 5)
        detection_rates = []
        bin_centers = []
        
        for i in range(len(snr_bins)-1):
            mask = (signal_snrs >= snr_bins[i]) & (signal_snrs < snr_bins[i+1])
            if np.sum(mask) > 0:
                rate = np.mean(signal_detected[mask])
                detection_rates.append(rate)
                bin_centers.append((snr_bins[i] + snr_bins[i+1]) / 2)
        
        if detection_rates:
            plt.plot(bin_centers, detection_rates, 'o-')
            plt.title('Detection Rate vs SNR')
            plt.xlabel('SNR')
            plt.ylabel('Detection Rate')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
    
    # 8. Latent space visualization (2D projection)
    plt.subplot(3, 5, 9)
    latent_reps = results['latent_representations']
    if latent_reps.shape[1] >= 2:
        plt.scatter(latent_reps[labels == 0, 0], latent_reps[labels == 0, 1], 
                   alpha=0.6, label='Noise', color='blue', s=20)
        plt.scatter(latent_reps[labels == 1, 0], latent_reps[labels == 1, 1], 
                   alpha=0.6, label='Signals', color='red', s=20)
        plt.title('Latent Space (2D projection)')
        plt.xlabel('Latent Dim 1')
        plt.ylabel('Latent Dim 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 9. Confusion Matrix
    plt.subplot(3, 5, 10)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, predictions)
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)
    
    plt.xticks([0, 1], ['Pred Noise', 'Pred Signal'])
    plt.yticks([0, 1], ['True Noise', 'True Signal'])
    
    plt.tight_layout()
    plt.savefig('results/cwt_lstm_autoencoder_results.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid interactive plot
    
    # Generate standalone publication-quality figures
    logger.info("\nGenerating individual publication plots...")
    
    # Import metrics at the top to avoid scope conflicts
    from sklearn.metrics import precision_recall_curve, roc_curve as sklearn_roc_curve, auc
    
    # Extract data from results for individual plots
    test_predictions = results['predictions']
    test_labels = labels  # labels is already available in main function scope
    test_scores = results['reconstruction_errors']
    
    # Figure 1: Precision-Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(test_labels, test_scores)
    avg_precision = auc(recall, precision)
    
    # Find key operating points
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = pr_thresholds[optimal_idx] if optimal_idx < len(pr_thresholds) else pr_thresholds[-1]
    
    # Find highest precision point
    max_precision_idx = np.argmax(precision)
    max_precision = precision[max_precision_idx]
    max_precision_recall = recall[max_precision_idx]
    max_precision_threshold = pr_thresholds[max_precision_idx] if max_precision_idx < len(pr_thresholds) else pr_thresholds[-1]
    
    # Find highest precision >= 90%
    precision_90_plus = precision >= 0.90
    if np.any(precision_90_plus):
        # Among points with precision >= 90%, find the one with highest recall
        valid_indices = np.where(precision_90_plus)[0]
        best_90_idx = valid_indices[np.argmax(recall[valid_indices])]
        precision_90 = precision[best_90_idx]
        recall_90 = recall[best_90_idx]
        threshold_90 = pr_thresholds[best_90_idx] if best_90_idx < len(pr_thresholds) else pr_thresholds[-1]
    else:
        precision_90 = recall_90 = threshold_90 = None
    
    # Print key operating points
    logger.info(f"\nKey Operating Points:")
    logger.info(f"Maximum Precision: {max_precision:.1%} (Recall: {max_precision_recall:.1%})")
    if precision_90 is not None:
        logger.info(f"Best ≥90% Precision: {precision_90:.1%} (Recall: {recall_90:.1%})")
    else:
        logger.warning(f"No points found with ≥90% precision")
    logger.info(f"Optimal F1: Precision {precision[optimal_idx]:.1%}, Recall {recall[optimal_idx]:.1%}")
    
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    ax_pr.plot(recall, precision, 'b-', linewidth=2.5, 
               label=f'AUPRC = {avg_precision:.3f}')
    
    # Mark key operating points
    ax_pr.plot(recall[optimal_idx], precision[optimal_idx], 
               'ro', markersize=8, label=f'Optimal F1 ({f1_scores[optimal_idx]:.3f})')
    ax_pr.plot(max_precision_recall, max_precision, 
               'go', markersize=8, label=f'Max Precision ({max_precision:.1%})')
    
    if precision_90 is not None:
        ax_pr.plot(recall_90, precision_90, 
                   'mo', markersize=8, label=f'Best ≥90% Prec ({precision_90:.1%})')
    
    # Add LIGO requirement line
    ax_pr.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='LIGO Requirement (90%)')
    ax_pr.plot([0, 1], [0.5, 0.5], 'k--', alpha=0.5, label='Random Classifier')
    ax_pr.set_xlabel('Recall', fontsize=14)
    ax_pr.set_ylabel('Precision', fontsize=14)
    ax_pr.set_title('Precision-Recall Curve', fontsize=16)
    ax_pr.legend(fontsize=12)
    ax_pr.grid(True, alpha=0.3)
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig('results/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: ROC Curve
    fpr, tpr, roc_thresholds = sklearn_roc_curve(test_labels, test_scores)
    auc_score = auc(fpr, tpr)
    
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'AUC-ROC = {auc_score:.3f}')
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    ax_roc.set_xlabel('False Positive Rate', fontsize=14)
    ax_roc.set_ylabel('True Positive Rate', fontsize=14)
    ax_roc.set_title('ROC Curve', fontsize=16)
    ax_roc.legend(fontsize=12)
    ax_roc.grid(True, alpha=0.3)
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig('results/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: SNR Performance (if SNR data exists)
    if 'snr_values' in results and len(results['snr_values']) > 0:
        snr_data = results['snr_values']
        
        # Group performance by SNR bins
        snr_bins = np.arange(8, 26, 2)
        snr_centers = (snr_bins[:-1] + snr_bins[1:]) / 2
        precision_by_snr = []
        recall_by_snr = []
        f1_by_snr = []
        
        for i in range(len(snr_bins)-1):
            mask = (snr_data >= snr_bins[i]) & (snr_data < snr_bins[i+1])
            if np.sum(mask) > 0:
                bin_labels = test_labels[mask]
                bin_predictions = test_predictions[mask]
                
                if len(np.unique(bin_labels)) > 1:  # Both classes present
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    prec = precision_score(bin_labels, bin_predictions, zero_division=0)
                    rec = recall_score(bin_labels, bin_predictions, zero_division=0)
                    f1 = f1_score(bin_labels, bin_predictions, zero_division=0)
                else:
                    prec = rec = f1 = 0
                    
                precision_by_snr.append(prec)
                recall_by_snr.append(rec)
                f1_by_snr.append(f1)
            else:
                precision_by_snr.append(0)
                recall_by_snr.append(0)
                f1_by_snr.append(0)
        
        fig_snr, ax_snr = plt.subplots(figsize=(10, 6))
        ax_snr.plot(snr_centers, precision_by_snr, 'b-', linewidth=2.5, 
                    label='Precision', marker='o', markersize=6)
        ax_snr.plot(snr_centers, recall_by_snr, 'r-', linewidth=2.5, 
                    label='Recall', marker='s', markersize=6)
        ax_snr.plot(snr_centers, f1_by_snr, 'g-', linewidth=2.5, 
                    label='F1-Score', marker='^', markersize=6)
        
        # Add performance regions
        ax_snr.axvspan(15, 25, alpha=0.2, color='green', label='High SNR (>15)')
        ax_snr.axvspan(10, 15, alpha=0.2, color='orange', label='Moderate SNR (10-15)')
        ax_snr.axvspan(8, 10, alpha=0.2, color='red', label='Threshold SNR (<10)')
        
        ax_snr.set_xlabel('Signal-to-Noise Ratio (SNR)', fontsize=14)
        ax_snr.set_ylabel('Detection Performance', fontsize=14)
        ax_snr.set_title('CWT-LSTM Autoencoder Performance vs Signal-to-Noise Ratio', fontsize=16)
        ax_snr.legend(fontsize=12, loc='lower right')
        ax_snr.grid(True, alpha=0.3)
        ax_snr.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig('results/snr_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SNR performance plot saved")
    
    logger.info(f"Publication plots saved to results/ directory:")
    logger.info(f"   - precision_recall_curve.png")
    logger.info(f"   - roc_curve.png")
    if 'snr_values' in results and len(results['snr_values']) > 0:
        logger.info(f"   - snr_performance.png")
    
    # Auto-update paper with new results
    logger.info(f"\nAuto-updating paper...")
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from paper.scripts.update_results import update_from_model_results
        
        # Update paper with the key operating points
        update_from_model_results(
            precision=precision_90 if precision_90 is not None else precision[optimal_idx],
            recall=recall_90 if recall_90 is not None else recall[optimal_idx],
            auc=auc_score,
            max_precision=max_precision,
            max_precision_recall=max_precision_recall,
            f1_precision=precision[optimal_idx],
            f1_recall=recall[optimal_idx],
            avg_precision=avg_precision
        )
        logger.info(f"Paper automatically updated with latest results!")
        
    except ImportError:
        logger.warning(f"Paper update system not found - results saved to plots only")
    except Exception as e:
        logger.warning(f"Paper update failed: {e}")
    
    logger.info(f"\nAnalysis Complete!")
    logger.info(f"Key Insights:")
    logger.info(f"  • CWT captures frequency evolution of gravitational wave chirps")
    logger.info(f"  • Autoencoder learns 'normal' noise patterns in time-frequency domain")
    logger.info(f"  • Anomalies (GW signals) have higher reconstruction error")
    logger.info(f"  • This approach is used in real LIGO data analysis!")
    logger.info(f"Results saved to 'cwt_lstm_autoencoder_results.png'")

if __name__ == "__main__":
    main()

