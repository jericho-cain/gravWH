"""
Data loading and preprocessing module for gravitational wave detection.

This module provides functionality for loading and preprocessing gravitational wave
data from various sources including LIGO, Virgo, and synthetic data generation.
"""

from .loader import (
    load_ligo_data,
    load_virgo_data,
    load_event_data,
    download_open_data,
    GWDataset,
)
from .preprocessing import (
    preprocess_strain_data,
    create_training_segments,
    augment_data,
)
from .synthetic import (
    generate_synthetic_gw,
    generate_noise,
    inject_signal,
)

__all__ = [
    # Data loading
    "load_ligo_data",
    "load_virgo_data", 
    "load_event_data",
    "download_open_data",
    "GWDataset",
    
    # Preprocessing
    "preprocess_strain_data",
    "create_training_segments",
    "augment_data",
    
    # Synthetic data
    "generate_synthetic_gw",
    "generate_noise",
    "inject_signal",
]
