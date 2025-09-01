import pytest
import numpy as np
import torch
from gravitational_wave_hunter.models.cwt_lstm_autoencoder import (
    SimpleCWTAutoencoder,
    generate_realistic_chirp,
    generate_colored_noise
)


@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    t = np.linspace(0, 4, 2048)
    signal = generate_realistic_chirp(t)
    noise = generate_colored_noise(2048, 512)
    return {
        'time': t,
        'signal': signal,
        'noise': noise,
        'signal_plus_noise': signal + noise
    }


@pytest.fixture
def sample_model():
    """Provide a sample model for testing."""
    return SimpleCWTAutoencoder(height=64, width=128)


@pytest.fixture
def sample_batch():
    """Provide a sample batch of data for testing."""
    batch_size = 4
    channels = 1
    height = 64
    width = 128
    return torch.randn(batch_size, channels, height, width)


@pytest.fixture
def sample_labels():
    """Provide sample labels for testing."""
    return np.random.randint(0, 2, 100)


@pytest.fixture
def sample_predictions():
    """Provide sample predictions for testing."""
    return np.random.randint(0, 2, 100)


@pytest.fixture
def sample_scores():
    """Provide sample scores for testing."""
    return np.random.random(100)


@pytest.fixture(scope="session")
def test_data_directory(tmp_path_factory):
    """Create a temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def mock_model_output():
    """Provide mock model output for testing."""
    return {
        'reconstruction_error': np.random.exponential(0.1, 100),
        'predictions': np.random.randint(0, 2, 100),
        'scores': np.random.random(100)
    }


# Markers for different test types
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


# Skip GPU tests if CUDA is not available
def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if CUDA is not available."""
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
