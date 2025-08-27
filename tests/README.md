# Gravitational Wave Hunter Tests

This directory contains comprehensive tests for the Gravitational Wave Hunter framework. The test suite covers all major components of the system including data loading, signal processing, machine learning models, and utilities.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest configuration and fixtures
├── pytest.ini                 # Pytest settings
├── test_data_loader.py         # Data loading and simulation tests
├── test_models.py              # Neural network model tests
├── test_preprocessing.py       # Signal processing tests
├── test_utils.py               # Utility function tests
├── test_visualization.py       # Plotting and visualization tests
├── test_detector.py            # Main detector integration tests
└── README.md                   # This file
```

## Running Tests

### Install Test Dependencies

First, install the development dependencies:

```bash
pip install -e ".[dev]"
# or
pip install -r requirements-dev.txt
```

### Run All Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=gravitational_wave_hunter

# Run tests in parallel
pytest -n auto

# Run tests with verbose output
pytest -v
```

### Run Specific Test Categories

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run only GPU tests (if GPU available)
pytest -m gpu
```

### Run Specific Test Files

```bash
# Test data loading functionality
pytest tests/test_data_loader.py

# Test neural network models
pytest tests/test_models.py

# Test signal preprocessing
pytest tests/test_preprocessing.py

# Test utilities
pytest tests/test_utils.py

# Test visualization
pytest tests/test_visualization.py

# Test main detector
pytest tests/test_detector.py
```

### Run Specific Tests

```bash
# Run a specific test class
pytest tests/test_models.py::TestCNNLSTMDetector

# Run a specific test method
pytest tests/test_data_loader.py::TestChirpSignalGeneration::test_generate_chirp_signal_basic

# Run tests matching a pattern
pytest -k "test_detector"
```

## Test Coverage

The test suite aims for high coverage across all components:

- **Data Loading**: Tests for simulated data generation, LIGO data loading, chirp signal creation
- **Models**: Tests for all neural network architectures (CNN-LSTM, Transformer, WaveNet, Autoencoder)
- **Preprocessing**: Tests for filtering, whitening, glitch removal, normalization
- **Utils**: Tests for configuration, metrics, helpers, device management
- **Visualization**: Tests for plotting functions, figure saving, error handling
- **Integration**: End-to-end tests for the complete detection pipeline

### Coverage Report

Generate a detailed coverage report:

```bash
# Generate HTML coverage report
pytest --cov=gravitational_wave_hunter --cov-report=html

# View coverage in terminal
pytest --cov=gravitational_wave_hunter --cov-report=term-missing

# Generate XML coverage report (for CI)
pytest --cov=gravitational_wave_hunter --cov-report=xml
```

## Test Configuration

### Pytest Configuration

The test suite uses `pytest.ini` for configuration with the following key settings:

- **Test Discovery**: Automatically finds `test_*.py` files
- **Markers**: Support for categorizing tests (`unit`, `integration`, `slow`, `gpu`)
- **Coverage**: Integrated coverage reporting
- **Warnings**: Filters out common warning messages
- **Timeout**: 5-minute timeout per test to prevent hanging

### Fixtures

Common test fixtures are defined in `conftest.py`:

- `sample_rate`: Standard LIGO sample rate (4096 Hz)
- `duration`: Test signal duration (4 seconds)
- `sample_strain_data`: Generated strain data for testing
- `sample_strain_batch`: Batch of strain data
- `sample_labels`: Test labels for classification
- `torch_device`: Appropriate PyTorch device (CPU/CUDA)
- `temp_dir`: Temporary directory for file operations
- `sample_config`: Test configuration dictionary

### Mocking and Dependencies

Tests use mocking for external dependencies:

- **LIGO/gwpy**: Mocked when not available
- **GPU operations**: Graceful fallback to CPU
- **File operations**: Use temporary directories
- **Network calls**: Mocked to avoid external dependencies

## Writing New Tests

### Test Guidelines

1. **Isolation**: Tests should be independent and not rely on external state
2. **Deterministic**: Use fixed random seeds for reproducible results
3. **Fast**: Keep tests fast; mark slow tests with `@pytest.mark.slow`
4. **Clear**: Use descriptive test names and docstrings
5. **Coverage**: Test both success and failure cases

### Test Template

```python
class TestNewComponent:
    """Test the new component functionality."""
    
    def test_basic_functionality(self, fixture_name):
        """Test basic functionality works correctly."""
        # Arrange
        input_data = ...
        expected_output = ...
        
        # Act
        result = new_component.function(input_data)
        
        # Assert
        assert result == expected_output
        assert isinstance(result, expected_type)
    
    def test_error_handling(self):
        """Test that errors are handled appropriately."""
        with pytest.raises(ValueError):
            new_component.function(invalid_input)
    
    @pytest.mark.slow
    def test_performance(self):
        """Test performance with large datasets."""
        # Performance tests here
        pass
```

### Adding Fixtures

Add new fixtures to `conftest.py`:

```python
@pytest.fixture
def new_fixture():
    """Description of what this fixture provides."""
    # Setup
    data = create_test_data()
    
    yield data
    
    # Teardown (if needed)
    cleanup(data)
```

## Continuous Integration

The test suite is designed to work with CI/CD pipelines:

### GitHub Actions

```yaml
- name: Run Tests
  run: |
    pytest --cov=gravitational_wave_hunter --cov-report=xml
    
- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

### Pre-commit Hooks

Install pre-commit hooks to run tests before commits:

```bash
pre-commit install
```

This will run:
- Code formatting (black, isort)
- Linting (flake8)
- Type checking (mypy)
- Basic tests

## Troubleshooting

### Common Issues

1. **ImportError**: Make sure the package is installed in development mode: `pip install -e .`
2. **CUDA errors**: Tests automatically fall back to CPU if CUDA is not available
3. **Slow tests**: Use `pytest -m "not slow"` to skip performance tests
4. **Memory issues**: Use `pytest --maxfail=1` to stop on first failure

### Debug Mode

Run tests in debug mode:

```bash
# Drop into debugger on failure
pytest --pdb

# Show local variables on failure
pytest --tb=long

# Capture print statements
pytest -s
```

### Test Data

Test data is generated dynamically to avoid including large files in the repository. If you need specific test cases, add them to the appropriate fixture in `conftest.py`.

## Contributing

When contributing new features:

1. Write tests for new functionality
2. Ensure tests pass locally
3. Maintain or improve coverage
4. Update this README if adding new test categories
5. Follow the existing test patterns and naming conventions

The test suite is a critical part of ensuring the reliability and correctness of the gravitational wave detection framework. Well-written tests help catch bugs early and provide confidence when making changes to the codebase.
