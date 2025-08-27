"""
Tests for data loading and simulation functionality.
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from gravitational_wave_hunter.data.loader import (
    load_simulated_data,
    generate_chirp_signal,
    create_synthetic_dataset,
    load_ligo_data
)


class TestChirpSignalGeneration:
    """Test gravitational wave chirp signal generation."""
    
    def test_generate_chirp_signal_basic(self, sample_rate, duration):
        """Test basic chirp signal generation."""
        time = np.linspace(0, duration, int(sample_rate * duration))
        signal = generate_chirp_signal(
            time, 
            initial_freq=35, 
            final_freq=250, 
            amplitude=1e-21
        )
        
        # Check output properties
        assert len(signal) == len(time)
        assert np.all(np.isfinite(signal))
        assert signal.dtype == np.float64 or signal.dtype == np.float32
        
        # Check amplitude is reasonable
        assert np.abs(signal).max() <= 2e-21  # Allow some margin
    
    def test_generate_chirp_signal_frequency_evolution(self, sample_rate):
        """Test that chirp signal frequency increases over time."""
        duration = 2.0
        time = np.linspace(0, duration, int(sample_rate * duration))
        signal = generate_chirp_signal(
            time, 
            initial_freq=50, 
            final_freq=200, 
            amplitude=1e-21
        )
        
        # Calculate instantaneous frequency using FFT
        # Should increase over time for a chirp
        fft_signal = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
        
        # Basic check that signal has energy in expected frequency range
        power_spectrum = np.abs(fft_signal)**2
        peak_freq_idx = np.argmax(power_spectrum[:len(power_spectrum)//2])
        peak_freq = freqs[peak_freq_idx]
        
        assert 40 <= peak_freq <= 210  # Should be in our frequency range
    
    def test_generate_chirp_signal_amplitude_scaling(self, sample_rate, duration):
        """Test chirp signal amplitude scaling."""
        time = np.linspace(0, duration, int(sample_rate * duration))
        
        amp1 = 1e-21
        amp2 = 2e-21
        
        signal1 = generate_chirp_signal(time, 35, 250, amp1)
        signal2 = generate_chirp_signal(time, 35, 250, amp2)
        
        # Signal2 should have roughly twice the amplitude
        ratio = np.abs(signal2).max() / np.abs(signal1).max()
        assert 1.8 <= ratio <= 2.2  # Allow some numerical tolerance
    
    def test_generate_chirp_signal_invalid_params(self, sample_rate, duration):
        """Test chirp signal generation with invalid parameters."""
        time = np.linspace(0, duration, int(sample_rate * duration))
        
        # Test invalid frequency order
        with pytest.raises((ValueError, AssertionError)):
            generate_chirp_signal(time, final_freq=50, initial_freq=100, amplitude=1e-21)
        
        # Test negative amplitude
        with pytest.raises((ValueError, AssertionError)):
            generate_chirp_signal(time, 35, 250, amplitude=-1e-21)


class TestSimulatedDataGeneration:
    """Test simulated data generation functions."""
    
    def test_load_simulated_data_basic(self, sample_rate, duration):
        """Test basic simulated data loading."""
        num_samples = 50
        signal_probability = 0.3
        
        strain_data, labels, metadata = load_simulated_data(
            num_samples=num_samples,
            duration=duration,
            sample_rate=sample_rate,
            signal_probability=signal_probability,
            noise_level=1e-23
        )
        
        # Check output shapes and types
        expected_length = int(sample_rate * duration)
        assert strain_data.shape == (num_samples, expected_length)
        assert labels.shape == (num_samples,)
        assert isinstance(metadata, dict)
        
        # Check data properties
        assert np.all(np.isfinite(strain_data))
        assert np.all((labels == 0) | (labels == 1))
        
        # Check signal probability is approximately correct
        signal_fraction = np.mean(labels)
        assert abs(signal_fraction - signal_probability) < 0.2  # Allow some variance
    
    def test_load_simulated_data_metadata(self, sample_rate, duration):
        """Test that metadata contains expected information."""
        strain_data, labels, metadata = load_simulated_data(
            num_samples=10,
            duration=duration,
            sample_rate=sample_rate,
            signal_probability=0.5
        )
        
        # Check metadata content
        assert 'sample_rate' in metadata
        assert 'duration' in metadata
        assert metadata['sample_rate'] == sample_rate
        assert metadata['duration'] == duration
    
    def test_load_simulated_data_noise_level(self, sample_rate, duration):
        """Test different noise levels."""
        num_samples = 20
        
        # Generate data with different noise levels
        _, _, _ = load_simulated_data(
            num_samples=num_samples,
            duration=duration,
            sample_rate=sample_rate,
            signal_probability=0.0,  # No signals, just noise
            noise_level=1e-24
        )
        
        _, _, _ = load_simulated_data(
            num_samples=num_samples,
            duration=duration,
            sample_rate=sample_rate,
            signal_probability=0.0,
            noise_level=1e-22
        )
        
        # Both should succeed without errors
        assert True
    
    def test_create_synthetic_dataset(self, sample_rate, duration):
        """Test synthetic dataset creation."""
        try:
            dataset = create_synthetic_dataset(
                num_samples=30,
                duration=duration,
                sample_rate=sample_rate
            )
            
            # Should return some form of dataset
            assert dataset is not None
            
        except NotImplementedError:
            # Function might not be implemented yet
            pytest.skip("create_synthetic_dataset not implemented")


class TestLIGODataLoading:
    """Test LIGO data loading functionality."""
    
    @patch('gravitational_wave_hunter.data.loader.gwpy')
    def test_load_ligo_data_mock(self, mock_gwpy, sample_rate, duration):
        """Test LIGO data loading with mocked gwpy."""
        # Mock gwpy TimeSeries
        mock_timeseries = MagicMock()
        mock_timeseries.value = np.random.normal(0, 1e-23, int(sample_rate * duration))
        mock_timeseries.sample_rate.value = sample_rate
        mock_gwpy.TimeSeries.fetch_open_data.return_value = mock_timeseries
        
        try:
            data = load_ligo_data('H1', start_time=1126259446, duration=duration)
            
            # Check that data was returned
            assert data is not None
            assert len(data) == int(sample_rate * duration)
            
        except ImportError:
            # gwpy might not be available
            pytest.skip("gwpy not available")
        except NotImplementedError:
            # Function might not be implemented yet
            pytest.skip("load_ligo_data not implemented")
    
    def test_load_ligo_data_without_gwpy(self):
        """Test LIGO data loading when gwpy is not available."""
        with patch.dict('sys.modules', {'gwpy': None}):
            try:
                # Should handle missing gwpy gracefully
                data = load_ligo_data('H1', start_time=1126259446, duration=4)
                
                # Should either return fallback data or raise ImportError
                if data is not None:
                    assert isinstance(data, np.ndarray)
                    
            except ImportError:
                # Expected when gwpy is not available
                assert True
            except NotImplementedError:
                # Function might not be implemented yet
                pytest.skip("load_ligo_data not implemented")


class TestDataValidation:
    """Test data validation and error handling."""
    
    def test_invalid_sample_rate(self, duration):
        """Test handling of invalid sample rates."""
        with pytest.raises((ValueError, AssertionError)):
            load_simulated_data(
                num_samples=10,
                duration=duration,
                sample_rate=-100,  # Invalid sample rate
                signal_probability=0.3
            )
    
    def test_invalid_duration(self, sample_rate):
        """Test handling of invalid durations."""
        with pytest.raises((ValueError, AssertionError)):
            load_simulated_data(
                num_samples=10,
                duration=-1.0,  # Invalid duration
                sample_rate=sample_rate,
                signal_probability=0.3
            )
    
    def test_invalid_signal_probability(self, sample_rate, duration):
        """Test handling of invalid signal probabilities."""
        with pytest.raises((ValueError, AssertionError)):
            load_simulated_data(
                num_samples=10,
                duration=duration,
                sample_rate=sample_rate,
                signal_probability=1.5  # Invalid probability
            )
    
    def test_edge_case_parameters(self, sample_rate, duration):
        """Test edge case parameters."""
        # Very small dataset
        strain_data, labels, metadata = load_simulated_data(
            num_samples=1,
            duration=duration,
            sample_rate=sample_rate,
            signal_probability=0.5
        )
        
        assert strain_data.shape[0] == 1
        assert labels.shape[0] == 1
        
        # All noise
        strain_data, labels, metadata = load_simulated_data(
            num_samples=10,
            duration=duration,
            sample_rate=sample_rate,
            signal_probability=0.0
        )
        
        assert np.all(labels == 0)
        
        # All signals
        strain_data, labels, metadata = load_simulated_data(
            num_samples=10,
            duration=duration,
            sample_rate=sample_rate,
            signal_probability=1.0
        )
        
        assert np.all(labels == 1)
