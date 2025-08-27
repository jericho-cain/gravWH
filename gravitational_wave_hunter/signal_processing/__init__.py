"""
Signal processing utilities for gravitational wave data.

This module provides comprehensive signal processing capabilities including
filtering, whitening, feature extraction, and time-frequency analysis.
"""

from .preprocessing import (
    preprocess_strain_data,
    whiten_data,
    bandpass_filter,
    remove_glitches,
    resample_data,
)
from .features import (
    extract_features,
    compute_psd,
    compute_spectrogram,
    compute_q_transform,
)
from .filtering import (
    apply_notch_filter,
    apply_highpass_filter,
    apply_lowpass_filter,
    butter_bandpass,
)
from .utils import (
    find_segments,
    merge_segments,
    calculate_snr,
)

__all__ = [
    # Preprocessing
    "preprocess_strain_data",
    "whiten_data",
    "bandpass_filter",
    "remove_glitches",
    "resample_data",
    
    # Feature extraction
    "extract_features",
    "compute_psd",
    "compute_spectrogram",
    "compute_q_transform",
    
    # Filtering
    "apply_notch_filter",
    "apply_highpass_filter", 
    "apply_lowpass_filter",
    "butter_bandpass",
    
    # Utilities
    "find_segments",
    "merge_segments",
    "calculate_snr",
]
