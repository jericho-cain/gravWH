import pytest
import torch
import numpy as np
from gravitational_wave_hunter.models.cwt_lstm_autoencoder import SimpleCWTAutoencoder


class TestModelArchitecture:
    """Test LSTM autoencoder model architecture."""
    
    def test_model_initialization(self):
        """Test model initialization with default parameters."""
        model = SimpleCWTAutoencoder(height=64, width=128)
        
        assert isinstance(model, SimpleCWTAutoencoder)
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'decoder')
    
    def test_model_parameters(self):
        """Test model initialization with custom parameters."""
        model = SimpleCWTAutoencoder(
            height=64,
            width=128,
            latent_dim=16
        )
        
        assert isinstance(model, SimpleCWTAutoencoder)
        assert model.encoder[-2].out_features == 16  # Check latent dimension
    
    def test_forward_pass_basic(self):
        """Test basic forward pass through the model."""
        model = SimpleCWTAutoencoder(height=64, width=128)
        
        # Create dummy input (batch_size, channels, height, width)
        batch_size = 4
        channels = 1
        height = 64
        width = 128
        x = torch.randn(batch_size, channels, height, width)
        
        # Forward pass
        reconstructed, latent = model(x)
        
        assert isinstance(reconstructed, torch.Tensor)
        assert isinstance(latent, torch.Tensor)
        assert reconstructed.shape == x.shape
        assert torch.isfinite(reconstructed).all()
        assert torch.isfinite(latent).all()
    
    def test_encoder_output_shape(self):
        """Test encoder output shape."""
        model = SimpleCWTAutoencoder(height=64, width=128, latent_dim=32)
        
        batch_size = 4
        channels = 1
        height = 64
        width = 128
        x = torch.randn(batch_size, channels, height, width)
        
        # Get encoder output
        encoded = model.encoder(x)
        
        assert encoded.shape == (batch_size, 32)  # latent_dim
        assert torch.isfinite(encoded).all()
    
    def test_latent_compression(self):
        """Test that latent space compresses the representation."""
        model = SimpleCWTAutoencoder(height=64, width=128, latent_dim=16)
        
        batch_size = 4
        channels = 1
        height = 64
        width = 128
        x = torch.randn(batch_size, channels, height, width)
        
        # Get encoded representation
        encoded = model.encoder(x)
        
        # Check compression
        assert encoded.shape[1] == 16  # latent_dim
        assert encoded.shape[0] == batch_size
    
    def test_decoder_reconstruction(self):
        """Test decoder reconstruction capability."""
        model = SimpleCWTAutoencoder(height=64, width=128)
        
        batch_size = 4
        channels = 1
        height = 64
        width = 128
        x = torch.randn(batch_size, channels, height, width)
        
        # Get encoded representation
        encoded = model.encoder(x)
        
        # Decode using the full forward pass to get interpolation
        reconstructed, _ = model(x)
        
        # The decoder should resize to original dimensions via interpolation
        assert reconstructed.shape == x.shape
        assert torch.isfinite(reconstructed).all()
    
    def test_end_to_end_reconstruction(self):
        """Test end-to-end reconstruction."""
        model = SimpleCWTAutoencoder(height=64, width=128)
        
        batch_size = 4
        channels = 1
        height = 64
        width = 128
        x = torch.randn(batch_size, channels, height, width)
        
        # Full forward pass
        reconstructed, latent = model(x)
        
        assert reconstructed.shape == x.shape
        assert torch.isfinite(reconstructed).all()
    
    def test_model_parameters_count(self):
        """Test that model has reasonable number of parameters."""
        model = SimpleCWTAutoencoder(height=64, width=128)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should have parameters (not be empty)
        assert total_params > 0
        
        # Should not be unreasonably large
        assert total_params < 1000000  # Less than 1M parameters
    
    def test_model_gradients(self):
        """Test that model can compute gradients."""
        model = SimpleCWTAutoencoder(height=64, width=128)
        
        batch_size = 4
        channels = 1
        height = 64
        width = 128
        x = torch.randn(batch_size, channels, height, width)
        
        # Forward pass
        reconstructed, latent = model(x)
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(reconstructed, x)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()
    
    def test_model_device_compatibility(self):
        """Test model works on CPU."""
        model = SimpleCWTAutoencoder(height=64, width=128)
        
        batch_size = 4
        channels = 1
        height = 64
        width = 128
        x = torch.randn(batch_size, channels, height, width)
        
        # Ensure model and input are on same device
        model = model.cpu()
        x = x.cpu()
        
        # Forward pass should work
        reconstructed, latent = model(x)
        assert reconstructed.device == x.device
    
    @pytest.mark.gpu
    def test_model_gpu_compatibility(self):
        """Test model works on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = SimpleCWTAutoencoder(height=64, width=128)
        
        batch_size = 4
        channels = 1
        height = 64
        width = 128
        x = torch.randn(batch_size, channels, height, width)
        
        # Move to GPU
        model = model.cuda()
        x = x.cuda()
        
        # Forward pass should work
        reconstructed, latent = model(x)
        assert reconstructed.device == x.device
        assert reconstructed.device.type == 'cuda'
    
    def test_model_save_load(self):
        """Test model can be saved and loaded."""
        model = SimpleCWTAutoencoder(height=64, width=128)
        
        # Save model
        torch.save(model.state_dict(), 'test_model.pth')
        
        # Load model
        new_model = SimpleCWTAutoencoder(height=64, width=128)
        new_model.load_state_dict(torch.load('test_model.pth'))
        
        # Test that loaded model produces same output
        batch_size = 4
        channels = 1
        height = 64
        width = 128
        x = torch.randn(batch_size, channels, height, width)
        
        with torch.no_grad():
            reconstructed1, latent1 = model(x)
            reconstructed2, latent2 = new_model(x)
        
        torch.testing.assert_close(reconstructed1, reconstructed2)
        torch.testing.assert_close(latent1, latent2)
        
        # Clean up
        import os
        os.remove('test_model.pth')
