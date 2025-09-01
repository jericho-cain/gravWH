"""
Gravitational Wave Hunter

A deep learning framework for gravitational wave detection in open astronomical data.
"""

__version__ = "0.1.2"
__author__ = "Gravitational Wave Research Team"
__email__ = "contact@gravitational-wave-hunter.org"

# Import main modules
from .models.cwt_lstm_autoencoder import CWT_LSTM_Autoencoder, SimpleCWTAutoencoder
from .data.generation import generate_realistic_chirp, generate_colored_noise

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "CWT_LSTM_Autoencoder",
    "SimpleCWTAutoencoder",
    "generate_realistic_chirp",
    "generate_colored_noise",
]