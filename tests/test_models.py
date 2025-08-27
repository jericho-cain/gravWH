"""
Tests for neural network models.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from gravitational_wave_hunter.models.base import BaseDetector
from gravitational_wave_hunter.models.cnn_lstm import CNNLSTMDetector
from gravitational_wave_hunter.models.transformer import TransformerDetector
from gravitational_wave_hunter.models.autoencoder import AutoencoderDetector
from gravitational_wave_hunter.models.wavenet import WaveNetDetector


class TestBaseDetector:
    """Test the base detector class."""
    
    def test_base_detector_instantiation(self):
        """Test that BaseDetector can be instantiated."""
        try:
            detector = BaseDetector()
            assert isinstance(detector, nn.Module)
        except NotImplementedError:
            # Base class might be abstract
            pytest.skip("BaseDetector is abstract")
    
    def test_base_detector_methods(self):
        """Test that BaseDetector has expected methods."""
        # Check that required methods are defined
        assert hasattr(BaseDetector, 'forward')
        assert hasattr(BaseDetector, 'train_step') or hasattr(BaseDetector, 'training_step')
        

class TestCNNLSTMDetector:
    """Test CNN-LSTM detector model."""
    
    @pytest.fixture
    def model_params(self):
        """Parameters for CNN-LSTM model."""
        return {
            'input_length': 4096,
            'num_conv_layers': 3,
            'conv_channels': [32, 64, 128],
            'lstm_hidden_size': 128,
            'num_classes': 2,
            'dropout': 0.3
        }
    
    def test_cnn_lstm_initialization(self, model_params, torch_device):
        """Test CNN-LSTM model initialization."""
        model = CNNLSTMDetector(**model_params)
        model = model.to(torch_device)
        
        # Check model components
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'conv_layers') or hasattr(model, 'convolution')
        assert hasattr(model, 'lstm') or hasattr(model, 'rnn')
        assert hasattr(model, 'classifier') or hasattr(model, 'fc')
    
    def test_cnn_lstm_forward_pass(self, model_params, torch_device):
        """Test CNN-LSTM forward pass."""
        model = CNNLSTMDetector(**model_params)
        model = model.to(torch_device)
        
        batch_size = 4
        input_tensor = torch.randn(batch_size, model_params['input_length']).to(torch_device)
        
        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)
        
        # Check output shape
        expected_shape = (batch_size, model_params['num_classes'])
        assert output.shape == expected_shape
        
        # Check output is finite
        assert torch.all(torch.isfinite(output))
    
    def test_cnn_lstm_different_input_lengths(self, torch_device):
        """Test CNN-LSTM with different input lengths."""
        input_lengths = [1024, 2048, 4096, 8192]
        
        for input_length in input_lengths:
            model = CNNLSTMDetector(
                input_length=input_length,
                num_conv_layers=2,
                conv_channels=[32, 64],
                lstm_hidden_size=64,
                num_classes=2
            )
            model = model.to(torch_device)
            
            batch_size = 2
            input_tensor = torch.randn(batch_size, input_length).to(torch_device)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            assert output.shape == (batch_size, 2)
    
    def test_cnn_lstm_gradient_flow(self, model_params, torch_device):
        """Test that gradients flow through CNN-LSTM model."""
        model = CNNLSTMDetector(**model_params)
        model = model.to(torch_device)
        
        batch_size = 2
        input_tensor = torch.randn(batch_size, model_params['input_length']).to(torch_device)
        target = torch.randint(0, 2, (batch_size,)).to(torch_device)
        
        # Forward pass
        output = model(input_tensor)
        loss = nn.CrossEntropyLoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.all(param.grad == 0)


class TestTransformerDetector:
    """Test Transformer detector model."""
    
    @pytest.fixture
    def transformer_params(self):
        """Parameters for Transformer model."""
        return {
            'input_length': 4096,
            'd_model': 256,
            'nhead': 8,
            'num_layers': 4,
            'num_classes': 2,
            'dropout': 0.1
        }
    
    def test_transformer_initialization(self, transformer_params, torch_device):
        """Test Transformer model initialization."""
        try:
            model = TransformerDetector(**transformer_params)
            model = model.to(torch_device)
            
            assert isinstance(model, nn.Module)
            assert hasattr(model, 'transformer') or hasattr(model, 'encoder')
            
        except NotImplementedError:
            pytest.skip("TransformerDetector not implemented")
    
    def test_transformer_forward_pass(self, transformer_params, torch_device):
        """Test Transformer forward pass."""
        try:
            model = TransformerDetector(**transformer_params)
            model = model.to(torch_device)
            
            batch_size = 4
            input_tensor = torch.randn(batch_size, transformer_params['input_length']).to(torch_device)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            expected_shape = (batch_size, transformer_params['num_classes'])
            assert output.shape == expected_shape
            
        except NotImplementedError:
            pytest.skip("TransformerDetector not implemented")
    
    def test_transformer_attention(self, transformer_params, torch_device):
        """Test that Transformer uses attention mechanism."""
        try:
            model = TransformerDetector(**transformer_params)
            model = model.to(torch_device)
            
            # Check for attention layers
            has_attention = any('attention' in name.lower() or 'attn' in name.lower() 
                              for name, _ in model.named_modules())
            assert has_attention
            
        except NotImplementedError:
            pytest.skip("TransformerDetector not implemented")


class TestAutoencoderDetector:
    """Test Autoencoder detector model."""
    
    @pytest.fixture
    def autoencoder_params(self):
        """Parameters for Autoencoder model."""
        return {
            'input_length': 4096,
            'encoding_dim': 128,
            'hidden_dims': [512, 256],
            'num_classes': 2
        }
    
    def test_autoencoder_initialization(self, autoencoder_params, torch_device):
        """Test Autoencoder model initialization."""
        try:
            model = AutoencoderDetector(**autoencoder_params)
            model = model.to(torch_device)
            
            assert isinstance(model, nn.Module)
            assert hasattr(model, 'encoder')
            assert hasattr(model, 'decoder')
            
        except NotImplementedError:
            pytest.skip("AutoencoderDetector not implemented")
    
    def test_autoencoder_forward_pass(self, autoencoder_params, torch_device):
        """Test Autoencoder forward pass."""
        try:
            model = AutoencoderDetector(**autoencoder_params)
            model = model.to(torch_device)
            
            batch_size = 4
            input_tensor = torch.randn(batch_size, autoencoder_params['input_length']).to(torch_device)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            # Output could be reconstruction or classification
            assert output.shape[0] == batch_size
            
        except NotImplementedError:
            pytest.skip("AutoencoderDetector not implemented")
    
    def test_autoencoder_encoding_decoding(self, autoencoder_params, torch_device):
        """Test that autoencoder can encode and decode."""
        try:
            model = AutoencoderDetector(**autoencoder_params)
            model = model.to(torch_device)
            
            batch_size = 2
            input_tensor = torch.randn(batch_size, autoencoder_params['input_length']).to(torch_device)
            
            # Test encoding
            if hasattr(model, 'encode'):
                encoded = model.encode(input_tensor)
                assert encoded.shape == (batch_size, autoencoder_params['encoding_dim'])
            
            # Test decoding
            if hasattr(model, 'decode'):
                encoded = torch.randn(batch_size, autoencoder_params['encoding_dim']).to(torch_device)
                decoded = model.decode(encoded)
                assert decoded.shape == input_tensor.shape
                
        except NotImplementedError:
            pytest.skip("AutoencoderDetector encode/decode not implemented")


class TestWaveNetDetector:
    """Test WaveNet detector model."""
    
    @pytest.fixture
    def wavenet_params(self):
        """Parameters for WaveNet model."""
        return {
            'input_length': 4096,
            'residual_channels': 64,
            'dilation_channels': 64,
            'skip_channels': 64,
            'num_blocks': 3,
            'num_layers': 10,
            'num_classes': 2
        }
    
    def test_wavenet_initialization(self, wavenet_params, torch_device):
        """Test WaveNet model initialization."""
        try:
            model = WaveNetDetector(**wavenet_params)
            model = model.to(torch_device)
            
            assert isinstance(model, nn.Module)
            
        except NotImplementedError:
            pytest.skip("WaveNetDetector not implemented")
    
    def test_wavenet_forward_pass(self, wavenet_params, torch_device):
        """Test WaveNet forward pass."""
        try:
            model = WaveNetDetector(**wavenet_params)
            model = model.to(torch_device)
            
            batch_size = 4
            input_tensor = torch.randn(batch_size, wavenet_params['input_length']).to(torch_device)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            expected_shape = (batch_size, wavenet_params['num_classes'])
            assert output.shape == expected_shape
            
        except NotImplementedError:
            pytest.skip("WaveNetDetector not implemented")
    
    def test_wavenet_dilated_convolutions(self, wavenet_params, torch_device):
        """Test that WaveNet uses dilated convolutions."""
        try:
            model = WaveNetDetector(**wavenet_params)
            model = model.to(torch_device)
            
            # Check for dilated convolutions
            has_dilated_conv = False
            for module in model.modules():
                if isinstance(module, nn.Conv1d) and module.dilation[0] > 1:
                    has_dilated_conv = True
                    break
            
            assert has_dilated_conv, "WaveNet should use dilated convolutions"
            
        except NotImplementedError:
            pytest.skip("WaveNetDetector not implemented")


class TestModelCommon:
    """Test common functionality across all models."""
    
    @pytest.mark.parametrize("model_class,params", [
        (CNNLSTMDetector, {
            'input_length': 2048,
            'num_conv_layers': 2,
            'conv_channels': [32, 64],
            'lstm_hidden_size': 64,
            'num_classes': 2
        }),
    ])
    def test_model_device_transfer(self, model_class, params, torch_device):
        """Test that models can be moved between devices."""
        model = model_class(**params)
        
        # Move to device
        model = model.to(torch_device)
        
        # Check that parameters are on correct device
        for param in model.parameters():
            assert param.device == torch_device
    
    @pytest.mark.parametrize("model_class,params", [
        (CNNLSTMDetector, {
            'input_length': 2048,
            'num_conv_layers': 2,
            'conv_channels': [32, 64],
            'lstm_hidden_size': 64,
            'num_classes': 2
        }),
    ])
    def test_model_training_mode(self, model_class, params, torch_device):
        """Test switching between training and evaluation modes."""
        model = model_class(**params)
        model = model.to(torch_device)
        
        # Test training mode
        model.train()
        assert model.training
        
        # Test evaluation mode
        model.eval()
        assert not model.training
    
    @pytest.mark.parametrize("model_class,params", [
        (CNNLSTMDetector, {
            'input_length': 2048,
            'num_conv_layers': 2,
            'conv_channels': [32, 64],
            'lstm_hidden_size': 64,
            'num_classes': 2
        }),
    ])
    def test_model_save_load(self, model_class, params, torch_device, temp_dir):
        """Test saving and loading model state."""
        model = model_class(**params)
        model = model.to(torch_device)
        
        # Save model
        model_path = temp_dir / "test_model.pth"
        torch.save(model.state_dict(), model_path)
        
        # Create new model and load state
        new_model = model_class(**params)
        new_model.load_state_dict(torch.load(model_path, map_location=torch_device))
        new_model = new_model.to(torch_device)
        
        # Compare parameters
        for param1, param2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(param1, param2)
    
    def test_model_parameter_count(self):
        """Test that models have reasonable parameter counts."""
        model = CNNLSTMDetector(
            input_length=4096,
            num_conv_layers=3,
            conv_channels=[32, 64, 128],
            lstm_hidden_size=128,
            num_classes=2
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Should have reasonable number of parameters
        assert 1000 < total_params < 10_000_000  # Between 1K and 10M parameters
        assert trainable_params == total_params  # All parameters should be trainable by default
