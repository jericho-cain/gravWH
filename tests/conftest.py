"""
Pytest configuration and shared fixtures for gravitational wave hunter tests.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

# Try to import torch, but handle if it's not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create a dummy torch module for testing
    class DummyTorch:
        def device(self, device_str):
            return device_str
        def manual_seed(self, seed):
            pass
        def cuda_is_available(self):
            return False
        class cuda:
            @staticmethod
            def manual_seed(seed):
                pass
    torch = DummyTorch()


@pytest.fixture
def sample_rate():
    """Standard LIGO sample rate."""
    return 4096


@pytest.fixture
def duration():
    """Standard test signal duration in seconds."""
    return 4.0


@pytest.fixture
def noise_level():
    """Standard noise level for test signals."""
    return 1e-23


@pytest.fixture
def sample_strain_data(sample_rate, duration, noise_level):
    """Generate sample strain data for testing."""
    n_samples = int(sample_rate * duration)
    time = np.linspace(0, duration, n_samples)
    
    # Generate realistic noise
    noise = np.random.normal(0, noise_level, n_samples)
    
    # Add a simple sinusoidal signal
    signal_freq = 100  # Hz
    signal_amplitude = 5e-22
    signal = signal_amplitude * np.sin(2 * np.pi * signal_freq * time)
    
    strain = noise + signal
    return strain.astype(np.float32)


@pytest.fixture
def sample_strain_batch(sample_strain_data):
    """Generate a batch of strain data for testing."""
    batch_size = 8
    return np.stack([sample_strain_data for _ in range(batch_size)])


@pytest.fixture
def sample_labels():
    """Generate sample labels for testing."""
    return np.array([0, 1, 0, 1, 1, 0, 1, 0])


@pytest.fixture
def torch_device():
    """Get the appropriate torch device for testing."""
    if TORCH_AVAILABLE:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return 'cpu'


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'sample_rate': 4096,
        'duration': 4.0,
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 10,
        'model_type': 'cnn_lstm',
        'input_length': 16384,
        'num_classes': 2
    }


@pytest.fixture
def sample_waveform_params():
    """Sample gravitational wave parameters for testing."""
    return {
        'mass1': 30.0,  # Solar masses
        'mass2': 25.0,  # Solar masses
        'distance': 400.0,  # Mpc
        'inclination': 0.0,  # radians
        'polarization': 0.0,  # radians
        'initial_frequency': 35.0,  # Hz
        'final_frequency': 250.0  # Hz
    }


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    np.random.seed(42)
    if TORCH_AVAILABLE:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)


# Helper functions for tests
def assert_array_shape(array, expected_shape):
    """Assert that array has the expected shape."""
    assert array.shape == expected_shape, f"Expected shape {expected_shape}, got {array.shape}"


def assert_array_finite(array):
    """Assert that all values in array are finite."""
    assert np.all(np.isfinite(array)), "Array contains non-finite values"


def assert_tensor_shape(tensor, expected_shape):
    """Assert that tensor has the expected shape."""
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"


def assert_model_output_shape(model, input_tensor, expected_output_shape):
    """Assert that model produces expected output shape."""
    with torch.no_grad():
        output = model(input_tensor)
        assert output.shape == expected_output_shape, f"Expected output shape {expected_output_shape}, got {output.shape}"
