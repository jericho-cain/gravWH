import pytest
import numpy as np
import pywt
from gravitational_wave_hunter.models.cwt_lstm_autoencoder import (
    generate_realistic_chirp,
    generate_colored_noise
)


class TestCWTPreprocessing:
    """Test CWT preprocessing functionality."""
    
    def test_cwt_basic_functionality(self):
        """Test basic CWT computation."""
        # Generate test signal
        t = np.linspace(0, 4, 2048)
        signal = generate_realistic_chirp(t)
        
        # Compute CWT
        scales = np.logspace(1, 3, 64)
        cwt_coeffs, freqs = pywt.cwt(signal, scales, 'morl', sampling_period=1/512)
        
        assert isinstance(cwt_coeffs, np.ndarray)
        assert cwt_coeffs.shape == (len(scales), len(signal))
        assert np.isfinite(cwt_coeffs).all()
        assert not np.all(cwt_coeffs == 0)
    
    def test_cwt_scales_parameter(self):
        """Test CWT with different scale parameters."""
        t = np.linspace(0, 4, 2048)
        signal = generate_realistic_chirp(t)
        
        # Test different scale ranges
        scales1 = np.logspace(1, 2, 32)
        scales2 = np.logspace(1, 3, 64)
        
        cwt1, _ = pywt.cwt(signal, scales1, 'morl', sampling_period=1/512)
        cwt2, _ = pywt.cwt(signal, scales2, 'morl', sampling_period=1/512)
        
        assert cwt1.shape[0] == len(scales1)
        assert cwt2.shape[0] == len(scales2)
        assert cwt1.shape[1] == cwt2.shape[1] == len(signal)
    
    def test_cwt_noise_vs_signal(self):
        """Test that CWT coefficients differ between noise and signal."""
        t = np.linspace(0, 4, 2048)
        signal = generate_realistic_chirp(t)
        noise = generate_colored_noise(2048, 512)
        
        scales = np.logspace(1, 3, 64)
        cwt_signal, _ = pywt.cwt(signal, scales, 'morl', sampling_period=1/512)
        cwt_noise, _ = pywt.cwt(noise, scales, 'morl', sampling_period=1/512)
        
        # Signal should have different CWT characteristics than noise
        assert not np.array_equal(cwt_signal, cwt_noise)
        
        # Both should have finite energy
        signal_energy = np.sum(np.abs(cwt_signal)**2)
        noise_energy = np.sum(np.abs(cwt_noise)**2)
        assert np.isfinite(signal_energy)
        assert np.isfinite(noise_energy)
        assert signal_energy >= 0
        assert noise_energy >= 0
    
    def test_cwt_frequency_mapping(self):
        """Test that CWT frequency mapping is correct."""
        t = np.linspace(0, 4, 2048)
        signal = generate_realistic_chirp(t)
        
        scales = np.logspace(1, 3, 64)
        cwt_coeffs, freqs = pywt.cwt(signal, scales, 'morl', sampling_period=1/512)
        
        assert len(freqs) == len(scales)
        assert np.all(freqs > 0)  # All frequencies should be positive
        assert np.all(np.diff(freqs) < 0)  # Frequencies should decrease with scale
    
    def test_cwt_morlet_wavelet(self):
        """Test CWT with Morlet wavelet specifically."""
        t = np.linspace(0, 4, 2048)
        signal = generate_realistic_chirp(t)
        
        scales = np.logspace(1, 3, 64)
        cwt_coeffs, freqs = pywt.cwt(signal, scales, 'cmor1.5-1.0', sampling_period=1/512)
        
        assert isinstance(cwt_coeffs, np.ndarray)
        assert cwt_coeffs.shape == (len(scales), len(signal))
        assert np.isfinite(cwt_coeffs).all()
    
    def test_cwt_scalogram_computation(self):
        """Test computation of scalogram (magnitude squared of CWT)."""
        t = np.linspace(0, 4, 2048)
        signal = generate_realistic_chirp(t)
        
        scales = np.logspace(1, 3, 64)
        cwt_coeffs, freqs = pywt.cwt(signal, scales, 'morl', sampling_period=1/512)
        
        # Compute scalogram
        scalogram = np.abs(cwt_coeffs)**2
        
        assert scalogram.shape == cwt_coeffs.shape
        assert np.all(scalogram >= 0)  # All values should be non-negative
        assert np.isfinite(scalogram).all()
    
    def test_cwt_edge_effects(self):
        """Test CWT behavior at signal edges."""
        t = np.linspace(0, 4, 2048)
        signal = generate_realistic_chirp(t)
        
        scales = np.logspace(1, 3, 64)
        cwt_coeffs, freqs = pywt.cwt(signal, scales, 'morl', sampling_period=1/512)
        
        # Check that edge effects don't produce infinite or NaN values
        assert np.isfinite(cwt_coeffs[:, 0]).all()  # First sample
        assert np.isfinite(cwt_coeffs[:, -1]).all()  # Last sample
    
    @pytest.mark.slow
    def test_cwt_large_dataset(self):
        """Test CWT computation on larger dataset."""
        t = np.linspace(0, 4, 2048)
        scales = np.logspace(1, 3, 64)
        
        # Generate multiple signals and compute CWT
        cwt_results = []
        for _ in range(5):
            signal = generate_realistic_chirp(t)
            cwt_coeffs, _ = pywt.cwt(signal, scales, 'morl', sampling_period=1/512)
            cwt_results.append(cwt_coeffs)
        
        cwt_results = np.array(cwt_results)
        assert cwt_results.shape == (5, len(scales), len(t))
        assert np.isfinite(cwt_results).all()
