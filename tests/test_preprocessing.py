"""
Tests for signal preprocessing functionality.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from gravitational_wave_hunter.signal_processing.preprocessing import (
    preprocess_strain_data,
    bandpass_filter,
    whiten_data,
    remove_glitches,
    downsample_data,
    normalize_strain
)


class TestBandpassFilter:
    """Test bandpass filtering functionality."""
    
    def test_bandpass_filter_basic(self, sample_strain_data, sample_rate):
        """Test basic bandpass filtering."""
        try:
            filtered_data = bandpass_filter(
                sample_strain_data,
                lowcut=20,
                highcut=1000,
                sample_rate=sample_rate,
                order=4
            )
            
            # Check output properties
            assert len(filtered_data) == len(sample_strain_data)
            assert np.all(np.isfinite(filtered_data))
            assert filtered_data.dtype in [np.float32, np.float64]
            
        except NotImplementedError:
            pytest.skip("bandpass_filter not implemented")
    
    def test_bandpass_filter_frequency_response(self, sample_rate):
        """Test that bandpass filter removes frequencies outside the band."""
        duration = 2.0
        n_samples = int(sample_rate * duration)
        
        # Create test signal with known frequencies
        time = np.linspace(0, duration, n_samples)
        low_freq_signal = np.sin(2 * np.pi * 10 * time)  # Below cutoff
        pass_freq_signal = np.sin(2 * np.pi * 100 * time)  # In passband
        high_freq_signal = np.sin(2 * np.pi * 2000 * time)  # Above cutoff
        
        test_signal = low_freq_signal + pass_freq_signal + high_freq_signal
        
        try:
            filtered_signal = bandpass_filter(
                test_signal,
                lowcut=50,
                highcut=1500,
                sample_rate=sample_rate,
                order=4
            )
            
            # Analyze frequency content
            fft_original = np.fft.fft(test_signal)
            fft_filtered = np.fft.fft(filtered_signal)
            freqs = np.fft.fftfreq(n_samples, 1/sample_rate)
            
            # Find power at test frequencies
            freq_10_idx = np.argmin(np.abs(freqs - 10))
            freq_100_idx = np.argmin(np.abs(freqs - 100))
            freq_2000_idx = np.argmin(np.abs(freqs - 2000))
            
            power_original_10 = np.abs(fft_original[freq_10_idx])**2
            power_filtered_10 = np.abs(fft_filtered[freq_10_idx])**2
            
            power_original_100 = np.abs(fft_original[freq_100_idx])**2
            power_filtered_100 = np.abs(fft_filtered[freq_100_idx])**2
            
            # Low frequency should be attenuated more than passband frequency
            if power_original_10 > 0:
                attenuation_10 = power_filtered_10 / power_original_10
                attenuation_100 = power_filtered_100 / power_original_100
                assert attenuation_10 < attenuation_100
            
        except NotImplementedError:
            pytest.skip("bandpass_filter not implemented")
    
    def test_bandpass_filter_invalid_params(self, sample_strain_data, sample_rate):
        """Test bandpass filter with invalid parameters."""
        try:
            # Test invalid frequency order
            with pytest.raises((ValueError, AssertionError)):
                bandpass_filter(
                    sample_strain_data,
                    lowcut=1000,
                    highcut=100,  # highcut < lowcut
                    sample_rate=sample_rate
                )
            
            # Test frequencies above Nyquist
            with pytest.raises((ValueError, AssertionError)):
                bandpass_filter(
                    sample_strain_data,
                    lowcut=100,
                    highcut=sample_rate,  # At Nyquist frequency
                    sample_rate=sample_rate
                )
                
        except NotImplementedError:
            pytest.skip("bandpass_filter not implemented")


class TestWhitenData:
    """Test data whitening functionality."""
    
    def test_whiten_data_basic(self, sample_strain_data, sample_rate):
        """Test basic data whitening."""
        try:
            whitened_data = whiten_data(sample_strain_data, sample_rate)
            
            # Check output properties
            assert len(whitened_data) == len(sample_strain_data)
            assert np.all(np.isfinite(whitened_data))
            
            # Whitened data should have approximately unit variance
            # (allowing for edge effects and finite data length)
            assert 0.5 < np.std(whitened_data) < 2.0
            
        except NotImplementedError:
            pytest.skip("whiten_data not implemented")
    
    def test_whiten_data_flat_spectrum(self, sample_rate):
        """Test that whitening flattens the power spectrum."""
        duration = 4.0
        n_samples = int(sample_rate * duration)
        
        # Create colored noise with known spectral shape
        np.random.seed(42)
        white_noise = np.random.normal(0, 1, n_samples)
        
        # Apply simple coloring (low-pass filter effect)
        from scipy import signal
        sos = signal.butter(2, 200, btype='low', fs=sample_rate, output='sos')
        colored_noise = signal.sosfilt(sos, white_noise)
        
        try:
            whitened_data = whiten_data(colored_noise, sample_rate)
            
            # Compare power spectra
            freqs = np.fft.fftfreq(n_samples, 1/sample_rate)
            psd_original = np.abs(np.fft.fft(colored_noise))**2
            psd_whitened = np.abs(np.fft.fft(whitened_data))**2
            
            # In frequency band of interest, whitened PSD should be flatter
            freq_mask = (np.abs(freqs) > 50) & (np.abs(freqs) < 500)
            
            if np.sum(freq_mask) > 10:  # Enough frequency bins
                psd_var_original = np.var(psd_original[freq_mask])
                psd_var_whitened = np.var(psd_whitened[freq_mask])
                
                # Whitened spectrum should be flatter (less variance)
                assert psd_var_whitened < psd_var_original
            
        except NotImplementedError:
            pytest.skip("whiten_data not implemented")


class TestGlitchRemoval:
    """Test glitch removal functionality."""
    
    def test_remove_glitches_basic(self, sample_strain_data):
        """Test basic glitch removal."""
        try:
            # Add artificial glitches
            glitchy_data = sample_strain_data.copy()
            glitch_indices = [len(glitchy_data)//4, 3*len(glitchy_data)//4]
            for idx in glitch_indices:
                glitchy_data[idx:idx+10] += 100 * np.std(sample_strain_data)  # Large spike
            
            cleaned_data = remove_glitches(glitchy_data)
            
            # Check output properties
            assert len(cleaned_data) == len(glitchy_data)
            assert np.all(np.isfinite(cleaned_data))
            
            # Glitches should be reduced
            for idx in glitch_indices:
                original_max = np.max(np.abs(sample_strain_data[idx:idx+10]))
                glitchy_max = np.max(np.abs(glitchy_data[idx:idx+10]))
                cleaned_max = np.max(np.abs(cleaned_data[idx:idx+10]))
                
                # Cleaned data should be closer to original than glitchy data
                assert cleaned_max < glitchy_max
            
        except NotImplementedError:
            pytest.skip("remove_glitches not implemented")
    
    def test_remove_glitches_threshold(self, sample_strain_data):
        """Test glitch removal with different thresholds."""
        try:
            # Add artificial glitch
            glitchy_data = sample_strain_data.copy()
            glitch_idx = len(glitchy_data) // 2
            glitch_amplitude = 50 * np.std(sample_strain_data)
            glitchy_data[glitch_idx] += glitch_amplitude
            
            # Test different thresholds
            cleaned_low_thresh = remove_glitches(glitchy_data, threshold=5.0)
            cleaned_high_thresh = remove_glitches(glitchy_data, threshold=100.0)
            
            # Lower threshold should remove more
            assert np.var(cleaned_low_thresh) <= np.var(cleaned_high_thresh)
            
        except NotImplementedError:
            pytest.skip("remove_glitches not implemented")


class TestDownsampling:
    """Test downsampling functionality."""
    
    def test_downsample_data_basic(self, sample_strain_data, sample_rate):
        """Test basic downsampling."""
        try:
            target_rate = sample_rate // 2
            downsampled_data = downsample_data(sample_strain_data, sample_rate, target_rate)
            
            # Check output length
            expected_length = len(sample_strain_data) * target_rate // sample_rate
            assert abs(len(downsampled_data) - expected_length) <= 1  # Allow for rounding
            
            # Check data properties
            assert np.all(np.isfinite(downsampled_data))
            
        except NotImplementedError:
            pytest.skip("downsample_data not implemented")
    
    def test_downsample_data_nyquist(self, sample_rate):
        """Test that downsampling respects Nyquist frequency."""
        duration = 2.0
        n_samples = int(sample_rate * duration)
        time = np.linspace(0, duration, n_samples)
        
        # Create signal with frequency near new Nyquist
        test_freq = sample_rate // 8  # Should be preserved after 4x downsampling
        test_signal = np.sin(2 * np.pi * test_freq * time)
        
        try:
            target_rate = sample_rate // 4
            downsampled_signal = downsample_data(test_signal, sample_rate, target_rate)
            
            # Signal should still contain the test frequency
            fft_downsampled = np.fft.fft(downsampled_signal)
            freqs_downsampled = np.fft.fftfreq(len(downsampled_signal), 1/target_rate)
            
            # Find peak frequency
            peak_idx = np.argmax(np.abs(fft_downsampled[:len(fft_downsampled)//2]))
            peak_freq = freqs_downsampled[peak_idx]
            
            # Should be close to test frequency
            assert abs(peak_freq - test_freq) < target_rate / len(downsampled_signal)
            
        except NotImplementedError:
            pytest.skip("downsample_data not implemented")


class TestNormalization:
    """Test strain data normalization."""
    
    def test_normalize_strain_basic(self, sample_strain_data):
        """Test basic strain normalization."""
        try:
            normalized_data = normalize_strain(sample_strain_data)
            
            # Check output properties
            assert len(normalized_data) == len(sample_strain_data)
            assert np.all(np.isfinite(normalized_data))
            
            # Should be normalized (zero mean, unit variance)
            assert abs(np.mean(normalized_data)) < 1e-10
            assert abs(np.std(normalized_data) - 1.0) < 1e-6
            
        except NotImplementedError:
            pytest.skip("normalize_strain not implemented")
    
    def test_normalize_strain_methods(self, sample_strain_data):
        """Test different normalization methods."""
        methods = ['zscore', 'minmax', 'robust']
        
        for method in methods:
            try:
                normalized_data = normalize_strain(sample_strain_data, method=method)
                
                assert len(normalized_data) == len(sample_strain_data)
                assert np.all(np.isfinite(normalized_data))
                
                if method == 'zscore':
                    assert abs(np.mean(normalized_data)) < 1e-10
                    assert abs(np.std(normalized_data) - 1.0) < 1e-6
                elif method == 'minmax':
                    assert abs(np.min(normalized_data) - 0.0) < 1e-10
                    assert abs(np.max(normalized_data) - 1.0) < 1e-10
                    
            except (NotImplementedError, ValueError):
                # Method might not be implemented
                continue


class TestPreprocessStrainData:
    """Test the main preprocessing pipeline."""
    
    def test_preprocess_strain_data_basic(self, sample_strain_data, sample_rate):
        """Test basic preprocessing pipeline."""
        try:
            processed_data = preprocess_strain_data(
                sample_strain_data,
                sample_rate=sample_rate,
                highpass_freq=20,
                lowpass_freq=1000,
                whiten=True
            )
            
            # Check output properties
            assert len(processed_data) == len(sample_strain_data)
            assert np.all(np.isfinite(processed_data))
            
            # Should be approximately normalized if whitening is applied
            if abs(np.std(processed_data) - 1.0) > 0.5:
                # If not whitened, at least check it's reasonable
                assert 0.1 < np.std(processed_data) < 100
            
        except NotImplementedError:
            pytest.skip("preprocess_strain_data not implemented")
    
    def test_preprocess_strain_data_options(self, sample_strain_data, sample_rate):
        """Test preprocessing with different options."""
        try:
            # Test with minimal processing
            processed_minimal = preprocess_strain_data(
                sample_strain_data,
                sample_rate=sample_rate,
                whiten=False,
                remove_glitches=False
            )
            
            # Test with full processing
            processed_full = preprocess_strain_data(
                sample_strain_data,
                sample_rate=sample_rate,
                highpass_freq=20,
                lowpass_freq=1000,
                notch_freqs=[60, 120],
                whiten=True,
                remove_glitches=True
            )
            
            # Both should produce valid output
            assert np.all(np.isfinite(processed_minimal))
            assert np.all(np.isfinite(processed_full))
            
            # Full processing should change the data more
            diff_minimal = np.sum((processed_minimal - sample_strain_data)**2)
            diff_full = np.sum((processed_full - sample_strain_data)**2)
            
            assert diff_full >= diff_minimal
            
        except NotImplementedError:
            pytest.skip("preprocess_strain_data not implemented")
    
    def test_preprocess_strain_data_batch(self, sample_strain_batch, sample_rate):
        """Test preprocessing with batch data."""
        try:
            # Test batch processing
            processed_batch = []
            for strain in sample_strain_batch:
                processed = preprocess_strain_data(
                    strain,
                    sample_rate=sample_rate,
                    highpass_freq=20,
                    lowpass_freq=1000
                )
                processed_batch.append(processed)
            
            processed_batch = np.array(processed_batch)
            
            # Check batch output
            assert processed_batch.shape == sample_strain_batch.shape
            assert np.all(np.isfinite(processed_batch))
            
        except NotImplementedError:
            pytest.skip("preprocess_strain_data not implemented")
    
    @patch('gravitational_wave_hunter.signal_processing.preprocessing.scipy')
    def test_preprocess_fallback_without_scipy(self, mock_scipy, sample_strain_data, sample_rate):
        """Test preprocessing fallback when scipy is not available."""
        # Mock scipy to not be available
        mock_scipy.side_effect = ImportError("No module named 'scipy'")
        
        try:
            # Should still work with basic processing
            processed_data = preprocess_strain_data(
                sample_strain_data,
                sample_rate=sample_rate
            )
            
            assert len(processed_data) == len(sample_strain_data)
            assert np.all(np.isfinite(processed_data))
            
        except (NotImplementedError, ImportError):
            pytest.skip("Fallback preprocessing not implemented")


class TestPreprocessingEdgeCases:
    """Test edge cases and error handling in preprocessing."""
    
    def test_empty_data(self, sample_rate):
        """Test preprocessing with empty data."""
        empty_data = np.array([])
        
        try:
            with pytest.raises((ValueError, IndexError)):
                preprocess_strain_data(empty_data, sample_rate=sample_rate)
        except NotImplementedError:
            pytest.skip("preprocess_strain_data not implemented")
    
    def test_very_short_data(self, sample_rate):
        """Test preprocessing with very short data."""
        short_data = np.random.normal(0, 1e-23, 10)  # Only 10 samples
        
        try:
            # Should either work or raise appropriate error
            processed_data = preprocess_strain_data(short_data, sample_rate=sample_rate)
            assert len(processed_data) == len(short_data)
            
        except (ValueError, NotImplementedError):
            # Either not implemented or appropriately rejected
            pass
    
    def test_nan_data(self, sample_rate):
        """Test preprocessing with NaN values."""
        nan_data = np.array([1e-23, 2e-23, np.nan, 1e-23, 2e-23] * 100)
        
        try:
            # Should either handle NaNs or raise appropriate error
            processed_data = preprocess_strain_data(nan_data, sample_rate=sample_rate)
            
            # If it succeeds, should not contain NaNs
            assert np.all(np.isfinite(processed_data))
            
        except (ValueError, NotImplementedError):
            # Either not implemented or appropriately rejected
            pass
    
    def test_constant_data(self, sample_rate):
        """Test preprocessing with constant data."""
        constant_data = np.ones(1000) * 1e-23
        
        try:
            processed_data = preprocess_strain_data(constant_data, sample_rate=sample_rate)
            
            # Should handle constant data gracefully
            assert len(processed_data) == len(constant_data)
            assert np.all(np.isfinite(processed_data))
            
        except (ValueError, NotImplementedError):
            # Either not implemented or appropriately rejected
            pass
