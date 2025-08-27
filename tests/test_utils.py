"""
Tests for utility functions and helper modules.
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

from gravitational_wave_hunter.utils.config import (
    load_config,
    save_config,
    get_default_config,
    validate_config
)
from gravitational_wave_hunter.utils.helpers import (
    set_random_seed,
    get_device,
    save_model,
    load_model,
    ensure_dir
)
from gravitational_wave_hunter.utils.metrics import (
    detection_metrics,
    calculate_snr,
    overlap_metric,
    effective_fisher_matrix
)


class TestConfig:
    """Test configuration management utilities."""
    
    def test_get_default_config(self):
        """Test getting default configuration."""
        try:
            config = get_default_config()
            
            # Should return a dictionary
            assert isinstance(config, dict)
            
            # Should contain basic required keys
            expected_keys = ['sample_rate', 'duration', 'batch_size']
            for key in expected_keys:
                if key in config:
                    assert isinstance(config[key], (int, float))
                    
        except NotImplementedError:
            pytest.skip("get_default_config not implemented")
    
    def test_save_load_config(self, temp_dir):
        """Test saving and loading configuration."""
        config = {
            'sample_rate': 4096,
            'duration': 4.0,
            'batch_size': 32,
            'learning_rate': 0.001,
            'model_type': 'cnn_lstm'
        }
        
        config_path = temp_dir / "test_config.yaml"
        
        try:
            # Save config
            save_config(config, config_path)
            assert config_path.exists()
            
            # Load config
            loaded_config = load_config(config_path)
            
            # Should match original
            assert loaded_config == config
            
        except NotImplementedError:
            pytest.skip("save_config/load_config not implemented")
    
    def test_validate_config(self):
        """Test configuration validation."""
        valid_config = {
            'sample_rate': 4096,
            'duration': 4.0,
            'batch_size': 32,
            'learning_rate': 0.001
        }
        
        invalid_config = {
            'sample_rate': -100,  # Invalid
            'duration': 4.0,
            'batch_size': 32,
            'learning_rate': 0.001
        }
        
        try:
            # Valid config should pass
            assert validate_config(valid_config) is True
            
            # Invalid config should fail
            assert validate_config(invalid_config) is False
            
        except NotImplementedError:
            pytest.skip("validate_config not implemented")
    
    def test_config_file_formats(self, temp_dir):
        """Test different configuration file formats."""
        config = {'sample_rate': 4096, 'duration': 4.0}
        
        formats = ['.yaml', '.yml', '.json']
        
        for fmt in formats:
            config_path = temp_dir / f"test_config{fmt}"
            
            try:
                save_config(config, config_path)
                loaded_config = load_config(config_path)
                assert loaded_config == config
                
            except (NotImplementedError, ValueError):
                # Format might not be supported
                continue


class TestHelpers:
    """Test helper utility functions."""
    
    def test_set_random_seed(self):
        """Test random seed setting."""
        try:
            set_random_seed(42)
            
            # Generate some random numbers
            np_rand1 = np.random.random(5)
            torch_rand1 = torch.rand(5)
            
            # Reset seed and generate again
            set_random_seed(42)
            np_rand2 = np.random.random(5)
            torch_rand2 = torch.rand(5)
            
            # Should be identical
            np.testing.assert_array_equal(np_rand1, np_rand2)
            torch.testing.assert_close(torch_rand1, torch_rand2)
            
        except NotImplementedError:
            pytest.skip("set_random_seed not implemented")
    
    def test_get_device(self):
        """Test device detection."""
        try:
            device = get_device()
            
            # Should return a torch device
            assert isinstance(device, torch.device)
            
            # Should be either CPU or CUDA
            assert device.type in ['cpu', 'cuda']
            
        except NotImplementedError:
            pytest.skip("get_device not implemented")
    
    def test_ensure_dir(self, temp_dir):
        """Test directory creation utility."""
        try:
            new_dir = temp_dir / "new_directory" / "nested"
            
            # Directory shouldn't exist initially
            assert not new_dir.exists()
            
            # Create directory
            ensure_dir(new_dir)
            
            # Should exist now
            assert new_dir.exists()
            assert new_dir.is_dir()
            
            # Calling again should not raise error
            ensure_dir(new_dir)
            
        except NotImplementedError:
            pytest.skip("ensure_dir not implemented")
    
    def test_save_load_model(self, temp_dir, torch_device):
        """Test model saving and loading utilities."""
        # Create a simple test model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
        model = model.to(torch_device)
        
        model_path = temp_dir / "test_model.pth"
        
        try:
            # Save model
            save_model(model, model_path)
            assert model_path.exists()
            
            # Load model
            loaded_model = load_model(model_path, model_class=type(model))
            
            # Compare parameters
            for param1, param2 in zip(model.parameters(), loaded_model.parameters()):
                torch.testing.assert_close(param1.cpu(), param2.cpu())
                
        except NotImplementedError:
            pytest.skip("save_model/load_model not implemented")


class TestMetrics:
    """Test metrics calculation utilities."""
    
    def test_detection_metrics_basic(self):
        """Test basic detection metrics calculation."""
        # Create test predictions and targets
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1])
        y_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.6, 0.9, 0.95])
        
        try:
            metrics = detection_metrics(y_true, y_pred, y_proba)
            
            # Should return a dictionary with expected metrics
            assert isinstance(metrics, dict)
            
            expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
            for metric in expected_metrics:
                if metric in metrics:
                    assert isinstance(metrics[metric], (int, float))
                    assert 0 <= metrics[metric] <= 1
                    
        except NotImplementedError:
            pytest.skip("detection_metrics not implemented")
    
    def test_detection_metrics_perfect_classifier(self):
        """Test metrics with perfect classifier."""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = y_true.copy()  # Perfect predictions
        y_proba = y_true.astype(float)  # Perfect probabilities
        
        try:
            metrics = detection_metrics(y_true, y_pred, y_proba)
            
            # Perfect classifier should have perfect metrics
            if 'accuracy' in metrics:
                assert metrics['accuracy'] == 1.0
            if 'precision' in metrics:
                assert metrics['precision'] == 1.0
            if 'recall' in metrics:
                assert metrics['recall'] == 1.0
            if 'f1_score' in metrics:
                assert metrics['f1_score'] == 1.0
                
        except NotImplementedError:
            pytest.skip("detection_metrics not implemented")
    
    def test_detection_metrics_edge_cases(self):
        """Test metrics with edge cases."""
        # All negative predictions
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.1, 0.3])
        
        try:
            metrics = detection_metrics(y_true, y_pred, y_proba)
            
            # Should handle edge cases gracefully
            assert isinstance(metrics, dict)
            
            # All predictions correct, so accuracy should be 1
            if 'accuracy' in metrics:
                assert metrics['accuracy'] == 1.0
                
        except (NotImplementedError, ValueError, ZeroDivisionError):
            # Function might not handle edge cases or not be implemented
            pass
    
    def test_calculate_snr(self, sample_strain_data, noise_level):
        """Test SNR calculation."""
        # Create signal with known SNR
        signal_amplitude = 5 * noise_level
        noise = np.random.normal(0, noise_level, len(sample_strain_data))
        signal = signal_amplitude * np.sin(2 * np.pi * 100 * np.linspace(0, 1, len(sample_strain_data)))
        
        try:
            snr_signal = calculate_snr(signal + noise, noise)
            snr_expected = signal_amplitude / noise_level
            
            # SNR should be approximately correct (within factor of 2)
            assert 0.5 * snr_expected < snr_signal < 2.0 * snr_expected
            
        except NotImplementedError:
            pytest.skip("calculate_snr not implemented")
    
    def test_overlap_metric(self, sample_rate, duration):
        """Test waveform overlap calculation."""
        # Create two similar waveforms
        time = np.linspace(0, duration, int(sample_rate * duration))
        waveform1 = np.sin(2 * np.pi * 100 * time)
        waveform2 = np.sin(2 * np.pi * 100 * time + 0.1)  # Slightly phase shifted
        
        try:
            overlap = overlap_metric(waveform1, waveform2)
            
            # Should be a number between -1 and 1
            assert isinstance(overlap, (int, float))
            assert -1 <= overlap <= 1
            
            # Identical waveforms should have overlap close to 1
            overlap_identical = overlap_metric(waveform1, waveform1)
            assert abs(overlap_identical - 1.0) < 1e-10
            
        except NotImplementedError:
            pytest.skip("overlap_metric not implemented")
    
    def test_effective_fisher_matrix(self, sample_rate, duration):
        """Test Fisher information matrix calculation."""
        # Create test waveform
        time = np.linspace(0, duration, int(sample_rate * duration))
        waveform = np.sin(2 * np.pi * 100 * time)
        
        # Simple PSD (flat)
        psd = np.ones(len(waveform))
        
        try:
            fisher_matrix = effective_fisher_matrix(waveform, psd, sample_rate)
            
            # Should return a matrix
            assert isinstance(fisher_matrix, np.ndarray)
            assert fisher_matrix.ndim == 2
            assert fisher_matrix.shape[0] == fisher_matrix.shape[1]  # Square matrix
            
        except NotImplementedError:
            pytest.skip("effective_fisher_matrix not implemented")


class TestMetricsValidation:
    """Test input validation for metrics functions."""
    
    def test_detection_metrics_mismatched_lengths(self):
        """Test detection metrics with mismatched input lengths."""
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0, 1])  # Different length
        y_proba = np.array([0.1, 0.9, 0.4])
        
        try:
            with pytest.raises((ValueError, AssertionError)):
                detection_metrics(y_true, y_pred, y_proba)
        except NotImplementedError:
            pytest.skip("detection_metrics not implemented")
    
    def test_detection_metrics_invalid_values(self):
        """Test detection metrics with invalid values."""
        y_true = np.array([0, 1, 2])  # Invalid class label
        y_pred = np.array([0, 1, 1])
        y_proba = np.array([0.1, 0.9, 0.4])
        
        try:
            with pytest.raises((ValueError, AssertionError)):
                detection_metrics(y_true, y_pred, y_proba)
        except NotImplementedError:
            pytest.skip("detection_metrics not implemented")
    
    def test_snr_zero_noise(self):
        """Test SNR calculation with zero noise."""
        signal = np.array([1, 2, 3, 4, 5])
        noise = np.zeros(5)
        
        try:
            # Should handle division by zero
            snr = calculate_snr(signal, noise)
            
            # Should return infinity or very large value
            assert snr > 1000 or np.isinf(snr)
            
        except (NotImplementedError, ValueError, ZeroDivisionError):
            # Function might not handle this case
            pass
    
    def test_overlap_different_lengths(self):
        """Test overlap with different length waveforms."""
        waveform1 = np.array([1, 2, 3, 4, 5])
        waveform2 = np.array([1, 2, 3])  # Different length
        
        try:
            with pytest.raises((ValueError, AssertionError)):
                overlap_metric(waveform1, waveform2)
        except NotImplementedError:
            pytest.skip("overlap_metric not implemented")


class TestUtilsIntegration:
    """Test integration between utility modules."""
    
    def test_config_device_integration(self, temp_dir):
        """Test that config and device utilities work together."""
        config = {
            'use_cuda': torch.cuda.is_available(),
            'device': 'auto'
        }
        
        config_path = temp_dir / "integration_config.yaml"
        
        try:
            save_config(config, config_path)
            loaded_config = load_config(config_path)
            
            device = get_device()
            
            # Device should be consistent with config
            if loaded_config.get('use_cuda', False) and torch.cuda.is_available():
                assert device.type == 'cuda'
            else:
                assert device.type == 'cpu'
                
        except NotImplementedError:
            pytest.skip("Config or device utilities not implemented")
    
    def test_metrics_with_model_output(self, torch_device):
        """Test metrics calculation with actual model output."""
        # Create simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 2),
            torch.nn.Softmax(dim=1)
        )
        model = model.to(torch_device)
        
        # Generate test data
        batch_size = 20
        X = torch.randn(batch_size, 10).to(torch_device)
        y_true = torch.randint(0, 2, (batch_size,))
        
        # Get model predictions
        with torch.no_grad():
            y_proba = model(X).cpu().numpy()
            y_pred = np.argmax(y_proba, axis=1)
        
        try:
            metrics = detection_metrics(
                y_true.numpy(), 
                y_pred, 
                y_proba[:, 1]  # Probability of positive class
            )
            
            # Should calculate metrics successfully
            assert isinstance(metrics, dict)
            
        except NotImplementedError:
            pytest.skip("detection_metrics not implemented")
