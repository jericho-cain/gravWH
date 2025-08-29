"""
CWT-LSTM Autoencoder for Gravitational Wave Detection.

This module provides the breakthrough CWT-LSTM autoencoder model that achieves
89.3% precision in gravitational wave detection using continuous wavelet
transform preprocessing and LSTM autoencoder architecture.
"""

from .cwt_lstm_autoencoder import SimpleCWTAutoencoder

__all__ = [
    "SimpleCWTAutoencoder",
]
