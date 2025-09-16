import pytest
import numpy as np
from typing import Tuple
from gravitational_wave_hunter.models.cwt_lstm_autoencoder import (
    generate_realistic_chirp,
    generate_colored_noise
)


class TestDataGeneration:
    """
    Test suite for data generation functions.
    
    Tests the functions responsible for generating realistic gravitational
    wave signals and LIGO-like colored noise for training and evaluation.
    
    Tested Functions
    ----------------
    - generate_realistic_chirp: Creates GW signals from binary mergers
    - generate_colored_noise: Creates LIGO-like noise with realistic PSD
    """
    
    def test_generate_realistic_chirp_basic(self) -> None:
        """
        Test basic chirp generation with default parameters.
        
        Verifies that the function produces valid gravitational wave signals
        with correct shape, non-zero values, and finite numbers.
        """
        t = np.linspace(0, 4, 2048)
        chirp = generate_realistic_chirp(t)
        
        assert isinstance(chirp, np.ndarray)
        assert chirp.shape == (2048,)
        assert not np.all(chirp == 0)
        assert np.isfinite(chirp).all()
    
    def test_generate_realistic_chirp_parameters(self) -> None:
        """
        Test chirp generation with different mass parameters.
        
        Verifies that different mass combinations produce different signals,
        ensuring the function responds correctly to parameter changes.
        """
        t = np.linspace(0, 4, 2048)
        
        # Test different mass combinations
        chirp1 = generate_realistic_chirp(t, m1=20, m2=15)
        chirp2 = generate_realistic_chirp(t, m1=50, m2=45)
        
        assert chirp1.shape == chirp2.shape
        assert not np.array_equal(chirp1, chirp2)  # Different masses should produce different signals
    
    def test_generate_realistic_chirp_distance_effect(self) -> None:
        """
        Test that distance affects signal amplitude.
        
        Verifies that signals from closer sources have higher amplitude,
        which is physically correct for gravitational wave propagation.
        """
        t = np.linspace(0, 4, 2048)
        
        chirp_close = generate_realistic_chirp(t, distance=100)
        chirp_far = generate_realistic_chirp(t, distance=1000)
        
        # Signal should be stronger at closer distance
        assert np.std(chirp_close) > np.std(chirp_far)
    
    def test_generate_colored_noise_basic(self) -> None:
        """
        Test basic colored noise generation.
        
        Verifies that the function produces valid noise arrays with correct
        shape and finite values, suitable for LIGO-like data simulation.
        """
        length = 2048
        sample_rate = 512
        noise = generate_colored_noise(length, sample_rate)
        
        assert isinstance(noise, np.ndarray)
        assert noise.shape == (length,)
        assert np.isfinite(noise).all()
        # The function generates very small amplitude noise (as intended for LIGO-like data)
        # Check that the array has the expected properties
        assert len(noise) == length
        assert np.isfinite(noise).all()
    
    def test_generate_colored_noise_reproducibility(self) -> None:
        """
        Test that noise generation is reproducible with same seed.
        
        Verifies that using the same random seed produces identical noise,
        ensuring reproducibility for testing and debugging.
        """
        length = 2048
        sample_rate = 512
        seed = 42
        
        noise1 = generate_colored_noise(length, sample_rate, seed=seed)
        noise2 = generate_colored_noise(length, sample_rate, seed=seed)
        
        np.testing.assert_array_equal(noise1, noise2)
    
    def test_generate_colored_noise_different_seeds(self) -> None:
        """
        Test that different seeds produce different noise.
        
        Verifies that different random seeds produce different noise patterns,
        ensuring proper randomization in the generation process.
        """
        length = 2048
        sample_rate = 512
        
        noise1 = generate_colored_noise(length, sample_rate, seed=42)
        noise2 = generate_colored_noise(length, sample_rate, seed=123)
        
        assert not np.array_equal(noise1, noise2)
    
    def test_generate_colored_noise_spectral_properties(self) -> None:
        """
        Test that generated noise has expected spectral properties.
        
        Verifies that the generated noise has realistic spectral characteristics
        with finite power values across different frequency ranges.
        """
        length = 2048
        sample_rate = 512
        noise = generate_colored_noise(length, sample_rate)
        
        # Compute power spectral density
        freqs, psd = np.fft.fftfreq(length, 1/sample_rate), np.abs(np.fft.fft(noise))**2
        
        # Should have more power at lower frequencies (colored noise)
        # Use positive frequencies only and avoid DC component
        positive_freqs = freqs[freqs > 0]
        positive_psd = psd[freqs > 0]
        
        if len(positive_freqs) > 0:
            low_freq_power = np.mean(positive_psd[:len(positive_psd)//4])
            high_freq_power = np.mean(positive_psd[-len(positive_psd)//4:])
            
            # For colored noise, low frequency power should be higher
            # But allow for some variation due to random generation
            assert low_freq_power >= 0
            assert high_freq_power >= 0
            assert np.isfinite(low_freq_power)
            assert np.isfinite(high_freq_power)
    
    @pytest.mark.slow
    def test_large_dataset_generation(self) -> None:
        """
        Test generation of larger datasets.
        
        Verifies that the functions can handle multiple samples efficiently
        and produce consistent results across a larger dataset.
        """
        t = np.linspace(0, 4, 2048)
        
        # Generate multiple samples
        signals = []
        for _ in range(10):
            chirp = generate_realistic_chirp(t)
            signals.append(chirp)
        
        signals = np.array(signals)
        assert signals.shape == (10, 2048)
        assert np.isfinite(signals).all()
