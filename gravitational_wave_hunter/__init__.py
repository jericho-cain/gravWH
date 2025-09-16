"""
Gravitational Wave Hunter: Deep Learning Framework for GW Detection.

A comprehensive deep learning framework designed for gravitational wave detection
in open astronomical data. Combines Continuous Wavelet Transform (CWT) with
LSTM autoencoders for unsupervised anomaly detection in gravitational wave data.

The framework provides tools for:
- Realistic gravitational wave signal generation
- LIGO-like colored noise simulation
- CWT preprocessing of strain data
- Autoencoder-based anomaly detection
- Comprehensive evaluation and visualization

Main Components
--------------
- CWT_LSTM_Autoencoder: Hybrid CNN-LSTM autoencoder for GW detection
- SimpleCWTAutoencoder: Simplified convolutional autoencoder
- generate_realistic_chirp: GW signal generation from binary mergers
- generate_colored_noise: LIGO-like noise generation

Examples
--------
>>> from gravitational_wave_hunter import CWT_LSTM_Autoencoder
>>> model = CWT_LSTM_Autoencoder(input_height=64, input_width=128)
>>> # Train and use for anomaly detection

Notes
-----
This framework is designed for research and educational purposes in
gravitational wave astronomy and machine learning applications.
"""

__version__ = "0.1.2"
__author__ = "Gravitational Wave Research Team"
__email__ = "contact@gravitational-wave-hunter.org"

# Import main modules
from .models.cwt_lstm_autoencoder import (
    CWT_LSTM_Autoencoder, 
    SimpleCWTAutoencoder,
    generate_realistic_chirp, 
    generate_colored_noise
)

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "CWT_LSTM_Autoencoder",
    "SimpleCWTAutoencoder",
    "generate_realistic_chirp",
    "generate_colored_noise",
]