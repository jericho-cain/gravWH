"""
Signal preprocessing utilities for gravitational wave data.

This module provides functions for cleaning, filtering, and preparing
gravitational wave strain data for machine learning analysis.
"""

from typing import Optional, Tuple, Union
import warnings
import logging

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, welch
from scipy.interpolate import interp1d

from ..utils.config import Config

# Set up logging
logger = logging.getLogger(__name__)


def preprocess_strain_data(
    strain_data: np.ndarray,
    sample_rate: int = 4096,
    config: Optional[Config] = None,
) -> np.ndarray:
    """
    Apply comprehensive preprocessing pipeline to strain data.
    
    This function applies a series of preprocessing steps commonly used
    in gravitational wave data analysis including filtering, whitening,
    and normalization.
    
    Args:
        strain_data: Input strain data as 1D numpy array
        sample_rate: Sample rate of the data in Hz
        config: Configuration object with preprocessing parameters
        
    Returns:
        Preprocessed strain data
        
    Example:
        >>> data = np.random.randn(4096 * 10)  # 10 seconds at 4096 Hz
        >>> processed = preprocess_strain_data(data, sample_rate=4096)
    """
    if config is None:
        config = Config()
    
    if strain_data.ndim != 1:
        raise ValueError("Input strain_data must be 1D array")
    
    processed_data = strain_data.copy()
    
    logger.debug(f"Starting preprocessing of {len(processed_data)} samples")
    
    # Step 1: Remove DC offset
    if config.remove_dc_offset:
        processed_data = processed_data - np.mean(processed_data)
        logger.debug("Removed DC offset")
    
    # Step 2: Apply bandpass filter
    if config.apply_bandpass:
        processed_data = bandpass_filter(
            processed_data,
            lowcut=config.bandpass_low,
            highcut=config.bandpass_high,
            sample_rate=sample_rate,
            order=config.filter_order,
        )
        logger.debug(f"Applied bandpass filter: {config.bandpass_low}-{config.bandpass_high} Hz")
    
    # Step 3: Remove glitches
    if config.remove_glitches:
        processed_data = remove_glitches(
            processed_data,
            threshold=config.glitch_threshold,
            window_size=config.glitch_window_size,
        )
        logger.debug("Removed glitches")
    
    # Step 4: Whiten the data
    if config.apply_whitening:
        processed_data = whiten_data(
            processed_data,
            sample_rate=sample_rate,
            segment_length=config.whitening_segment_length,
            overlap=config.whitening_overlap,
        )
        logger.debug("Applied whitening")
    
    # Step 5: Normalize
    if config.normalize:
        processed_data = normalize_data(
            processed_data,
            method=config.normalization_method,
        )
        logger.debug(f"Applied {config.normalization_method} normalization")
    
    logger.debug("Preprocessing completed")
    return processed_data


def bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    sample_rate: int,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to the data.
    
    Args:
        data: Input data array
        lowcut: Low cutoff frequency in Hz
        highcut: High cutoff frequency in Hz
        sample_rate: Sample rate in Hz
        order: Filter order
        
    Returns:
        Filtered data array
        
    Example:
        >>> filtered = bandpass_filter(data, 20, 2000, 4096, order=6)
    """
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    
    if high >= 1.0:
        warnings.warn(
            f"High cutoff frequency {highcut} Hz is too close to Nyquist frequency "
            f"{nyquist} Hz. Adjusting to {0.95 * nyquist} Hz"
        )
        high = 0.95
    
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data


def whiten_data(
    data: np.ndarray,
    sample_rate: int,
    segment_length: int = 4,
    overlap: float = 0.5,
    fftlength: Optional[int] = None,
) -> np.ndarray:
    """
    Whiten gravitational wave strain data by normalizing by the power spectral density.
    
    Whitening is a crucial step in gravitational wave data analysis that
    flattens the frequency spectrum, making weak signals more detectable.
    
    Args:
        data: Input strain data
        sample_rate: Sample rate in Hz
        segment_length: Length of segments for PSD estimation in seconds
        overlap: Overlap fraction between segments
        fftlength: FFT length for PSD estimation
        
    Returns:
        Whitened data array
        
    Example:
        >>> whitened = whiten_data(strain_data, sample_rate=4096)
    """
    if fftlength is None:
        fftlength = int(segment_length * sample_rate)
    
    # Estimate power spectral density
    frequencies, psd = welch(
        data,
        fs=sample_rate,
        nperseg=fftlength,
        noverlap=int(fftlength * overlap),
        window='hann',
    )
    
    # Interpolate PSD to match FFT frequencies
    data_fft = np.fft.rfft(data)
    fft_freqs = np.fft.rfftfreq(len(data), 1/sample_rate)
    
    # Interpolate PSD values
    psd_interp = interp1d(
        frequencies, 
        psd, 
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'
    )(fft_freqs)
    
    # Avoid division by very small numbers
    psd_interp = np.maximum(psd_interp, np.max(psd_interp) * 1e-10)
    
    # Whiten in frequency domain
    whitened_fft = data_fft / np.sqrt(psd_interp * sample_rate / 2)
    
    # Transform back to time domain
    whitened_data = np.fft.irfft(whitened_fft, n=len(data))
    
    return whitened_data


def remove_glitches(
    data: np.ndarray,
    threshold: float = 20.0,
    window_size: int = 1024,
) -> np.ndarray:
    """
    Remove transient glitches from strain data.
    
    This function identifies and removes glitches (transient noise artifacts)
    by detecting outliers and replacing them with interpolated values.
    
    Args:
        data: Input strain data
        threshold: Threshold for glitch detection (in standard deviations)
        window_size: Size of window for local statistics
        
    Returns:
        Data with glitches removed
        
    Example:
        >>> clean_data = remove_glitches(noisy_data, threshold=15.0)
    """
    cleaned_data = data.copy()
    
    # Calculate local statistics using rolling window
    half_window = window_size // 2
    
    for i in range(half_window, len(data) - half_window):
        # Extract local window
        window = data[i - half_window:i + half_window]
        
        # Calculate local statistics
        local_mean = np.mean(window)
        local_std = np.std(window)
        
        # Check if current point is an outlier
        if local_std > 0:  # Avoid division by zero
            z_score = abs(data[i] - local_mean) / local_std
            
            if z_score > threshold:
                # Replace glitch with interpolated value
                left_idx = max(0, i - 10)
                right_idx = min(len(data), i + 10)
                
                # Linear interpolation
                x_points = [left_idx, right_idx]
                y_points = [data[left_idx], data[right_idx]]
                
                cleaned_data[i] = np.interp(i, x_points, y_points)
    
    return cleaned_data


def normalize_data(
    data: np.ndarray,
    method: str = 'standard',
) -> np.ndarray:
    """
    Normalize strain data using specified method.
    
    Args:
        data: Input data array
        method: Normalization method ('standard', 'minmax', 'robust')
        
    Returns:
        Normalized data array
        
    Example:
        >>> normalized = normalize_data(data, method='standard')
    """
    if method == 'standard':
        # Z-score normalization
        mean = np.mean(data)
        std = np.std(data)
        if std > 0:
            return (data - mean) / std
        else:
            return data - mean
            
    elif method == 'minmax':
        # Min-max normalization to [0, 1]
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val > min_val:
            return (data - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(data)
            
    elif method == 'robust':
        # Robust normalization using median and MAD
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad > 0:
            return (data - median) / (1.4826 * mad)  # 1.4826 for normal distribution
        else:
            return data - median
            
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def resample_data(
    data: np.ndarray,
    original_rate: int,
    target_rate: int,
    method: str = 'linear',
) -> np.ndarray:
    """
    Resample data to a different sample rate.
    
    Args:
        data: Input data array
        original_rate: Original sample rate in Hz
        target_rate: Target sample rate in Hz
        method: Interpolation method ('linear', 'cubic')
        
    Returns:
        Resampled data array
        
    Example:
        >>> resampled = resample_data(data, 16384, 4096)
    """
    if original_rate == target_rate:
        return data.copy()
    
    # Calculate new length
    ratio = target_rate / original_rate
    new_length = int(len(data) * ratio)
    
    # Create time arrays
    original_time = np.arange(len(data)) / original_rate
    target_time = np.arange(new_length) / target_rate
    
    # Ensure target time doesn't exceed original time range
    target_time = target_time[target_time <= original_time[-1]]
    
    # Interpolate
    if method == 'linear':
        kind = 'linear'
    elif method == 'cubic':
        kind = 'cubic'
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    interpolator = interp1d(
        original_time, 
        data, 
        kind=kind,
        bounds_error=False,
        fill_value='extrapolate'
    )
    
    resampled_data = interpolator(target_time)
    
    return resampled_data


def compute_running_statistics(
    data: np.ndarray,
    window_size: int,
    statistic: str = 'mean',
) -> np.ndarray:
    """
    Compute running statistics over a sliding window.
    
    Args:
        data: Input data array
        window_size: Size of the sliding window
        statistic: Type of statistic ('mean', 'std', 'median')
        
    Returns:
        Array of running statistics
    """
    if statistic not in ['mean', 'std', 'median']:
        raise ValueError(f"Unknown statistic: {statistic}")
    
    result = np.zeros(len(data))
    half_window = window_size // 2
    
    for i in range(len(data)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(data), i + half_window + 1)
        window = data[start_idx:end_idx]
        
        if statistic == 'mean':
            result[i] = np.mean(window)
        elif statistic == 'std':
            result[i] = np.std(window)
        elif statistic == 'median':
            result[i] = np.median(window)
    
    return result


def find_data_quality_segments(
    data: np.ndarray,
    sample_rate: int,
    min_length: float = 1.0,
    noise_threshold: float = 3.0,
) -> List[Tuple[int, int]]:
    """
    Identify segments of good data quality.
    
    Args:
        data: Input strain data
        sample_rate: Sample rate in Hz
        min_length: Minimum segment length in seconds
        noise_threshold: Threshold for noise detection
        
    Returns:
        List of (start_index, end_index) tuples for good segments
    """
    # Calculate local noise level
    window_size = int(sample_rate)  # 1 second window
    local_std = compute_running_statistics(data, window_size, 'std')
    
    # Identify noisy regions
    median_noise = np.median(local_std)
    noise_mask = local_std > noise_threshold * median_noise
    
    # Find continuous good segments
    good_segments = []
    in_segment = False
    segment_start = 0
    
    min_samples = int(min_length * sample_rate)
    
    for i, is_noisy in enumerate(noise_mask):
        if not is_noisy and not in_segment:
            # Start of good segment
            segment_start = i
            in_segment = True
        elif is_noisy and in_segment:
            # End of good segment
            if i - segment_start >= min_samples:
                good_segments.append((segment_start, i))
            in_segment = False
    
    # Handle case where data ends during a good segment
    if in_segment and len(data) - segment_start >= min_samples:
        good_segments.append((segment_start, len(data)))
    
    return good_segments
