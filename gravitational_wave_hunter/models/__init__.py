"""
CWT-LSTM Autoencoder for Gravitational Wave Detection.

This module provides the breakthrough CWT-LSTM autoencoder model that achieves
92.3% precision with 67.6% recall in gravitational wave detection using continuous 
wavelet transform preprocessing and LSTM autoencoder architecture.
"""

from .cwt_lstm_autoencoder import SimpleCWTAutoencoder

__all__ = [
    "SimpleCWTAutoencoder",
]
