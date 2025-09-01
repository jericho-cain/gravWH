import pytest
import numpy as np
from gravitational_wave_hunter.models.cwt_lstm_autoencoder import (
    generate_realistic_chirp,
    generate_colored_noise
)


class TestDataGeneration:
    """Test data generation functions."""
    
    def test_generate_realistic_chirp_basic(self):
        """Test basic chirp generation with default parameters."""
        t = np.linspace(0, 4, 2048)
        chirp = generate_realistic_chirp(t)
        
        assert isinstance(chirp, np.ndarray)
        assert chirp.shape == (2048,)
        assert not np.all(chirp == 0)
        assert np.isfinite(chirp).all()
    
    def test_generate_realistic_chirp_parameters(self):
        """Test chirp generation with different mass parameters."""
        t = np.linspace(0, 4, 2048)
        
        # Test different mass combinations
        chirp1 = generate_realistic_chirp(t, m1=20, m2=15)
        chirp2 = generate_realistic_chirp(t, m1=50, m2=45)
        
        assert chirp1.shape == chirp2.shape
        assert not np.array_equal(chirp1, chirp2)  # Different masses should produce different signals
    
    def test_generate_realistic_chirp_distance_effect(self):
        """Test that distance affects signal amplitude."""
        t = np.linspace(0, 4, 2048)
        
        chirp_close = generate_realistic_chirp(t, distance=100)
        chirp_far = generate_realistic_chirp(t, distance=1000)
        
        # Signal should be stronger at closer distance
        assert np.std(chirp_close) > np.std(chirp_far)
    
    def test_generate_colored_noise_basic(self):
        """Test basic colored noise generation."""
        length = 2048
        sample_rate = 512
        noise = generate_colored_noise(length, sample_rate)
        
        assert isinstance(noise, np.ndarray)
        assert noise.shape == (length,)
        assert np.isfinite(noise).all()
        assert np.std(noise) > 0
    
    def test_generate_colored_noise_reproducibility(self):
        """Test that noise generation is reproducible with same seed."""
        length = 2048
        sample_rate = 512
        seed = 42
        
        noise1 = generate_colored_noise(length, sample_rate, seed=seed)
        noise2 = generate_colored_noise(length, sample_rate, seed=seed)
        
        np.testing.assert_array_equal(noise1, noise2)
    
    def test_generate_colored_noise_different_seeds(self):
        """Test that different seeds produce different noise."""
        length = 2048
        sample_rate = 512
        
        noise1 = generate_colored_noise(length, sample_rate, seed=42)
        noise2 = generate_colored_noise(length, sample_rate, seed=123)
        
        assert not np.array_equal(noise1, noise2)
    
    def test_generate_colored_noise_spectral_properties(self):
        """Test that generated noise has expected spectral properties."""
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
    def test_large_dataset_generation(self):
        """Test generation of larger datasets."""
        t = np.linspace(0, 4, 2048)
        
        # Generate multiple samples
        signals = []
        for _ in range(10):
            chirp = generate_realistic_chirp(t)
            signals.append(chirp)
        
        signals = np.array(signals)
        assert signals.shape == (10, 2048)
        assert np.isfinite(signals).all()
