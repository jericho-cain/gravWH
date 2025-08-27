"""
Gravitational Wave Hunter: Deep Learning for Gravitational Wave Detection

A comprehensive framework for detecting gravitational waves in open astronomical data
using state-of-the-art deep learning techniques with PyTorch.

This package provides:
- Neural network models optimized for gravitational wave detection
- Data loading and preprocessing for LIGO/Virgo open data
- Signal processing utilities for time-series analysis
- Visualization tools for results interpretation
- Pre-trained models and training utilities

Example:
    >>> from gravitational_wave_hunter import GWDetector, load_ligo_data
    >>> data = load_ligo_data('H1', start_time=1126259446, duration=4096)
    >>> detector = GWDetector(model_type='cnn_lstm')
    >>> detections = detector.detect(data)
"""

__version__ = "0.1.0"
__author__ = "Gravitational Wave Research Team"
__email__ = "contact@gw-hunter.org"
__license__ = "MIT"

# Core imports
from .detector import GWDetector
from .data.loader import load_ligo_data, load_virgo_data, GWDataset
from .models.cnn_lstm import CNNLSTM
from .models.wavenet import WaveNet
from .models.transformer import GWTransformer
from .models.autoencoder import GWAutoencoder

# Utility imports
from .signal_processing.preprocessing import (
    whiten_data,
    bandpass_filter,
    remove_glitches,
)
from .visualization.plotting import (
    plot_strain_data,
    plot_spectrogram,
    plot_detection_results,
)

__all__ = [
    # Core classes
    "GWDetector",
    "GWDataset",
    
    # Data loading functions
    "load_ligo_data",
    "load_virgo_data",
    
    # Model classes
    "CNNLSTM",
    "WaveNet", 
    "GWTransformer",
    "GWAutoencoder",
    
    # Signal processing
    "whiten_data",
    "bandpass_filter",
    "remove_glitches",
    
    # Visualization
    "plot_strain_data",
    "plot_spectrogram",
    "plot_detection_results",
]
