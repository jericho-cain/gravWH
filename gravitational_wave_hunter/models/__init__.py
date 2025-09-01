"""
Neural Network Models for Gravitational Wave Detection.

This module provides advanced neural network architectures specifically designed
for gravitational wave detection using time-frequency analysis and deep learning.

Available Models
---------------
- CWT_LSTM_Autoencoder: Hybrid CNN-LSTM autoencoder for anomaly detection
- SimpleCWTAutoencoder: Simplified convolutional autoencoder for baseline comparison

Architecture Overview
-------------------
The models use Continuous Wavelet Transform (CWT) preprocessing to convert
time series strain data into time-frequency representations (scalograms),
which are then processed by neural networks for anomaly detection.

Key Features
-----------
- Unsupervised learning: Trained only on noise data
- Anomaly detection: Identifies GW signals through reconstruction error
- Scalable: Handles variable-length time series
- Interpretable: Provides reconstruction error as confidence measure

Notes
-----
These models are designed for research in gravitational wave astronomy
and provide state-of-the-art performance in GW signal detection.
"""

from .cwt_lstm_autoencoder import (
    CWT_LSTM_Autoencoder,
    SimpleCWTAutoencoder,
    generate_realistic_chirp,
    generate_colored_noise,
    continuous_wavelet_transform,
    preprocess_with_cwt,
    train_autoencoder,
    detect_anomalies
)

__all__ = [
    "CWT_LSTM_Autoencoder",
    "SimpleCWTAutoencoder",
    "generate_realistic_chirp",
    "generate_colored_noise",
    "continuous_wavelet_transform",
    "preprocess_with_cwt",
    "train_autoencoder",
    "detect_anomalies",
]
