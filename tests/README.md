# Test Suite for Gravitational Wave Hunter

This directory contains comprehensive tests for the gravitational wave hunter project.

## Test Structure

- `test_data_generation.py` - Tests for synthetic data generation functions
- `test_cwt_preprocessing.py` - Tests for CWT preprocessing functionality
- `test_model_architecture.py` - Tests for LSTM autoencoder model architecture
- `test_training.py` - Tests for training functionality
- `test_evaluation.py` - Tests for evaluation metrics and performance calculation
- `conftest.py` - Pytest configuration and shared fixtures
- `run_tests.py` - Convenient test runner script

## Running Tests

### Basic Test Execution

```bash
# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=gravitational_wave_hunter --cov-report=term-missing
```

### Using the Test Runner

```bash
# Run all tests
python tests/run_tests.py

# Run only fast tests (exclude slow tests)
python tests/run_tests.py --type fast

# Run with coverage
python tests/run_tests.py --coverage

# Run with verbose output
python tests/run_tests.py -v
```

### Test Categories

- **Fast tests** - Quick unit tests that run in seconds
- **Slow tests** - Longer tests that may take minutes
- **GPU tests** - Tests that require CUDA (skipped if not available)
- **Integration tests** - Tests that verify component interactions

## Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.slow` - Marks tests as slow
- `@pytest.mark.gpu` - Marks tests that require GPU
- `@pytest.mark.integration` - Marks integration tests

## Coverage

The test suite aims for comprehensive coverage of:

- Data generation functions
- CWT preprocessing
- Model architecture
- Training pipeline
- Evaluation metrics
- Error handling
- Edge cases

## Adding New Tests

When adding new functionality, please add corresponding tests:

1. Create test functions with descriptive names
2. Use appropriate pytest markers
3. Include both positive and negative test cases
4. Test edge cases and error conditions
5. Use fixtures for common test data

## Example Test Structure

```python
def test_function_name_description():
    """Test description of what is being tested."""
    # Arrange - Set up test data
    input_data = create_test_data()
    
    # Act - Execute the function being tested
    result = function_under_test(input_data)
    
    # Assert - Verify the results
    assert result is not None
    assert result.shape == expected_shape
    assert np.isfinite(result).all()
```

## Continuous Integration

Tests are automatically run in CI/CD pipelines to ensure code quality and prevent regressions.
