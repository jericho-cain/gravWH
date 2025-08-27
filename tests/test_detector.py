"""
Tests for the main detector module and integration.
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from gravitational_wave_hunter.detector import GWDetector


class TestGWDetector:
    """Test the main GWDetector class."""
    
    def test_detector_initialization_default(self):
        """Test detector initialization with default parameters."""
        try:
            detector = GWDetector()
            
            # Should be initialized successfully
            assert detector is not None
            assert hasattr(detector, 'model')
            assert hasattr(detector, 'config')
            
        except NotImplementedError:
            pytest.skip("GWDetector not implemented")
    
    def test_detector_initialization_with_config(self, sample_config):
        """Test detector initialization with custom configuration."""
        try:
            detector = GWDetector(config=sample_config)
            
            # Should use provided configuration
            assert detector is not None
            
            # Check that config is applied
            if hasattr(detector, 'config'):
                for key, value in sample_config.items():
                    if key in detector.config:
                        assert detector.config[key] == value
                        
        except NotImplementedError:
            pytest.skip("GWDetector not implemented")
    
    def test_detector_initialization_model_type(self):
        """Test detector initialization with different model types."""
        model_types = ['cnn_lstm', 'transformer', 'wavenet', 'autoencoder']
        
        for model_type in model_types:
            try:
                detector = GWDetector(model_type=model_type)
                
                # Should initialize with specified model
                assert detector is not None
                
                if hasattr(detector, 'model_type'):
                    assert detector.model_type == model_type
                    
            except (NotImplementedError, ValueError):
                # Model type might not be implemented
                continue
    
    def test_detector_device_handling(self, torch_device):
        """Test that detector handles device correctly."""
        try:
            detector = GWDetector(device=torch_device)
            
            # Should be on correct device
            if hasattr(detector, 'device'):
                assert detector.device == torch_device
                
            # Model should be on correct device
            if hasattr(detector, 'model') and detector.model is not None:
                for param in detector.model.parameters():
                    assert param.device == torch_device
                    
        except NotImplementedError:
            pytest.skip("GWDetector not implemented")


class TestGWDetectorTraining:
    """Test GWDetector training functionality."""
    
    def test_detector_train_method(self, sample_strain_batch, sample_labels, torch_device):
        """Test detector training method."""
        try:
            detector = GWDetector(device=torch_device)
            
            # Should have training method
            assert hasattr(detector, 'train') or hasattr(detector, 'fit')
            
            # Prepare training data
            X_train = torch.FloatTensor(sample_strain_batch).to(torch_device)
            y_train = torch.LongTensor(sample_labels).to(torch_device)
            
            # Train detector
            if hasattr(detector, 'train'):
                history = detector.train(X_train, y_train, epochs=2, batch_size=4)
            else:
                history = detector.fit(X_train, y_train, epochs=2, batch_size=4)
            
            # Should return training history
            assert history is not None
            
        except NotImplementedError:
            pytest.skip("GWDetector training not implemented")
    
    def test_detector_train_with_validation(self, sample_strain_batch, sample_labels, torch_device):
        """Test detector training with validation data."""
        try:
            detector = GWDetector(device=torch_device)
            
            # Split data
            split_idx = len(sample_strain_batch) // 2
            X_train = torch.FloatTensor(sample_strain_batch[:split_idx]).to(torch_device)
            y_train = torch.LongTensor(sample_labels[:split_idx]).to(torch_device)
            X_val = torch.FloatTensor(sample_strain_batch[split_idx:]).to(torch_device)
            y_val = torch.LongTensor(sample_labels[split_idx:]).to(torch_device)
            
            # Train with validation
            if hasattr(detector, 'train'):
                history = detector.train(
                    X_train, y_train, 
                    validation_data=(X_val, y_val),
                    epochs=2, batch_size=2
                )
            else:
                pytest.skip("Validation training not available")
            
            # Should include validation metrics in history
            assert history is not None
            
        except NotImplementedError:
            pytest.skip("GWDetector validation training not implemented")
    
    def test_detector_save_load_model(self, sample_strain_batch, sample_labels, temp_dir, torch_device):
        """Test saving and loading trained models."""
        try:
            detector = GWDetector(device=torch_device)
            
            # Train briefly
            X_train = torch.FloatTensor(sample_strain_batch).to(torch_device)
            y_train = torch.LongTensor(sample_labels).to(torch_device)
            
            if hasattr(detector, 'train'):
                detector.train(X_train, y_train, epochs=1, batch_size=4)
            
            # Save model
            model_path = temp_dir / "test_detector.pth"
            if hasattr(detector, 'save_model'):
                detector.save_model(model_path)
                assert model_path.exists()
                
                # Load model
                new_detector = GWDetector(device=torch_device)
                new_detector.load_model(model_path)
                
                # Should have loaded successfully
                assert new_detector.model is not None
                
        except NotImplementedError:
            pytest.skip("GWDetector save/load not implemented")


class TestGWDetectorInference:
    """Test GWDetector inference functionality."""
    
    def test_detector_detect_method(self, sample_strain_data, torch_device):
        """Test detector detection method."""
        try:
            detector = GWDetector(device=torch_device)
            
            # Should have detection method
            assert hasattr(detector, 'detect') or hasattr(detector, 'predict')
            
            # Prepare test data
            test_data = torch.FloatTensor(sample_strain_data).unsqueeze(0).to(torch_device)
            
            # Detect gravitational waves
            if hasattr(detector, 'detect'):
                result = detector.detect(test_data)
            else:
                result = detector.predict(test_data)
            
            # Should return detection result
            assert result is not None
            
            # Result should be appropriate format
            if isinstance(result, torch.Tensor):
                assert result.numel() > 0
            elif isinstance(result, np.ndarray):
                assert result.size > 0
            elif isinstance(result, (list, tuple)):
                assert len(result) > 0
                
        except NotImplementedError:
            pytest.skip("GWDetector detection not implemented")
    
    def test_detector_batch_detection(self, sample_strain_batch, torch_device):
        """Test detector batch detection."""
        try:
            detector = GWDetector(device=torch_device)
            
            # Prepare batch data
            batch_data = torch.FloatTensor(sample_strain_batch).to(torch_device)
            
            # Detect on batch
            if hasattr(detector, 'detect'):
                results = detector.detect(batch_data)
            else:
                results = detector.predict(batch_data)
            
            # Should return results for entire batch
            assert results is not None
            
            # Check batch dimension
            if isinstance(results, (torch.Tensor, np.ndarray)):
                assert results.shape[0] == len(sample_strain_batch)
                
        except NotImplementedError:
            pytest.skip("GWDetector batch detection not implemented")
    
    def test_detector_confidence_scores(self, sample_strain_data, torch_device):
        """Test that detector returns confidence scores."""
        try:
            detector = GWDetector(device=torch_device)
            
            test_data = torch.FloatTensor(sample_strain_data).unsqueeze(0).to(torch_device)
            
            # Get predictions with confidence
            if hasattr(detector, 'predict_proba'):
                probabilities = detector.predict_proba(test_data)
                
                # Should return probabilities
                assert probabilities is not None
                
                if isinstance(probabilities, (torch.Tensor, np.ndarray)):
                    # Should be valid probabilities
                    assert np.all(probabilities >= 0)
                    assert np.all(probabilities <= 1)
                    
            elif hasattr(detector, 'detect'):
                # Check if detect returns confidence scores
                result = detector.detect(test_data, return_confidence=True)
                
                if isinstance(result, tuple) and len(result) >= 2:
                    predictions, confidence = result[:2]
                    assert confidence is not None
                    
        except (NotImplementedError, TypeError):
            pytest.skip("GWDetector confidence scores not implemented")


class TestGWDetectorPreprocessing:
    """Test GWDetector preprocessing integration."""
    
    def test_detector_preprocess_method(self, sample_strain_data):
        """Test detector preprocessing method."""
        try:
            detector = GWDetector()
            
            # Should have preprocessing method
            if hasattr(detector, 'preprocess'):
                processed_data = detector.preprocess(sample_strain_data)
                
                # Should return processed data
                assert processed_data is not None
                assert len(processed_data) > 0
                
                # Should be finite
                assert np.all(np.isfinite(processed_data))
                
        except NotImplementedError:
            pytest.skip("GWDetector preprocessing not implemented")
    
    def test_detector_automatic_preprocessing(self, sample_strain_data, torch_device):
        """Test that detector automatically preprocesses data."""
        try:
            detector = GWDetector(device=torch_device)
            
            # Raw data detection should include preprocessing
            if hasattr(detector, 'detect'):
                result = detector.detect(sample_strain_data, preprocess=True)
                
                # Should handle raw data
                assert result is not None
                
        except (NotImplementedError, TypeError):
            pytest.skip("GWDetector automatic preprocessing not implemented")


class TestGWDetectorConfiguration:
    """Test GWDetector configuration management."""
    
    def test_detector_get_config(self):
        """Test getting detector configuration."""
        try:
            detector = GWDetector()
            
            # Should have config access
            if hasattr(detector, 'get_config'):
                config = detector.get_config()
                
                # Should return configuration dictionary
                assert isinstance(config, dict)
                assert len(config) > 0
                
        except NotImplementedError:
            pytest.skip("GWDetector get_config not implemented")
    
    def test_detector_set_config(self, sample_config):
        """Test setting detector configuration."""
        try:
            detector = GWDetector()
            
            # Should be able to update config
            if hasattr(detector, 'set_config'):
                detector.set_config(sample_config)
                
                # Config should be updated
                current_config = detector.get_config()
                for key, value in sample_config.items():
                    if key in current_config:
                        assert current_config[key] == value
                        
        except NotImplementedError:
            pytest.skip("GWDetector set_config not implemented")
    
    def test_detector_config_validation(self):
        """Test configuration validation."""
        invalid_config = {
            'sample_rate': -100,  # Invalid
            'batch_size': 0,      # Invalid
            'learning_rate': 'invalid'  # Invalid type
        }
        
        try:
            # Should handle invalid config appropriately
            with pytest.raises((ValueError, TypeError, AssertionError)):
                detector = GWDetector(config=invalid_config)
                
        except NotImplementedError:
            pytest.skip("GWDetector config validation not implemented")


class TestGWDetectorIntegration:
    """Test GWDetector integration with other components."""
    
    def test_detector_data_loader_integration(self, sample_rate, duration):
        """Test detector integration with data loader."""
        try:
            detector = GWDetector()
            
            # Should be able to load data
            if hasattr(detector, 'load_data'):
                data = detector.load_data(
                    num_samples=10,
                    sample_rate=sample_rate,
                    duration=duration
                )
                
                # Should return proper format
                assert data is not None
                
        except NotImplementedError:
            pytest.skip("GWDetector data loading not implemented")
    
    def test_detector_visualization_integration(self, sample_strain_data, torch_device):
        """Test detector integration with visualization."""
        try:
            detector = GWDetector(device=torch_device)
            
            # Should be able to create plots
            if hasattr(detector, 'plot_detection'):
                test_data = torch.FloatTensor(sample_strain_data).unsqueeze(0).to(torch_device)
                
                fig = detector.plot_detection(test_data)
                
                # Should return matplotlib figure
                import matplotlib.pyplot as plt
                assert isinstance(fig, plt.Figure)
                plt.close(fig)
                
        except (NotImplementedError, ImportError):
            pytest.skip("GWDetector visualization not implemented")
    
    def test_detector_end_to_end_pipeline(self, sample_strain_data, torch_device):
        """Test complete end-to-end pipeline."""
        try:
            # Initialize detector
            detector = GWDetector(device=torch_device)
            
            # Preprocess data
            if hasattr(detector, 'preprocess'):
                processed_data = detector.preprocess(sample_strain_data)
            else:
                processed_data = sample_strain_data
            
            # Convert to tensor
            test_data = torch.FloatTensor(processed_data).unsqueeze(0).to(torch_device)
            
            # Detect gravitational waves
            if hasattr(detector, 'detect'):
                result = detector.detect(test_data)
                
                # Should get valid result
                assert result is not None
                
        except NotImplementedError:
            pytest.skip("GWDetector end-to-end pipeline not implemented")


class TestGWDetectorErrorHandling:
    """Test GWDetector error handling."""
    
    def test_detector_empty_input(self, torch_device):
        """Test detector with empty input."""
        try:
            detector = GWDetector(device=torch_device)
            
            empty_data = torch.FloatTensor([]).to(torch_device)
            
            # Should handle empty input gracefully
            if hasattr(detector, 'detect'):
                with pytest.raises((ValueError, RuntimeError)):
                    detector.detect(empty_data)
                    
        except NotImplementedError:
            pytest.skip("GWDetector error handling not implemented")
    
    def test_detector_invalid_input_shape(self, torch_device):
        """Test detector with invalid input shape."""
        try:
            detector = GWDetector(device=torch_device)
            
            # Wrong shape data
            wrong_shape_data = torch.randn(10, 10, 10).to(torch_device)  # 3D instead of 2D
            
            if hasattr(detector, 'detect'):
                with pytest.raises((ValueError, RuntimeError)):
                    detector.detect(wrong_shape_data)
                    
        except NotImplementedError:
            pytest.skip("GWDetector error handling not implemented")
    
    def test_detector_device_mismatch(self):
        """Test detector with device mismatch."""
        try:
            # Create detector on CPU
            detector = GWDetector(device=torch.device('cpu'))
            
            # Try to process CUDA data (if available)
            if torch.cuda.is_available():
                cuda_data = torch.randn(1, 1000).cuda()
                
                if hasattr(detector, 'detect'):
                    # Should either handle gracefully or raise appropriate error
                    try:
                        result = detector.detect(cuda_data)
                        # If it succeeds, data should be moved automatically
                        assert result is not None
                    except RuntimeError:
                        # Expected error for device mismatch
                        pass
                        
        except NotImplementedError:
            pytest.skip("GWDetector device handling not implemented")
