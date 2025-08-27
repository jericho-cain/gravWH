# Contributing to Gravitational Wave Hunter

Thank you for your interest in contributing to the Gravitational Wave Hunter project! This document provides guidelines and information for contributors.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Environment](#development-environment)
4. [Contributing Guidelines](#contributing-guidelines)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Pull Request Process](#pull-request-process)
8. [Issue Reporting](#issue-reporting)
9. [Community](#community)

## Code of Conduct

This project follows the [Python Community Code of Conduct](https://www.python.org/psf/conduct/). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

### Our Standards

- **Be respectful**: Treat everyone with respect and professionalism
- **Be inclusive**: Welcome newcomers and help them learn
- **Be collaborative**: Work together to achieve common goals
- **Be constructive**: Provide helpful feedback and suggestions
- **Be patient**: Remember that everyone has different skill levels and backgrounds

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic knowledge of gravitational waves and machine learning
- Familiarity with PyTorch

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/gravitational-wave-hunter.git
   cd gravitational-wave-hunter
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/gravitational-wave-hunter.git
   ```

## Development Environment

### Setting Up the Environment

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

### Development Dependencies

The `[dev]` extra includes:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `black` - Code formatting
- `flake8` - Linting
- `mypy` - Type checking
- `isort` - Import sorting
- `pre-commit` - Pre-commit hooks
- `sphinx` - Documentation generation

### Optional Dependencies

For specific use cases:
```bash
# For GPU acceleration
pip install torch[cuda]

# For LIGO data access
pip install gwpy pycbc

# For interactive plotting
pip install plotly ipywidgets
```

## Contributing Guidelines

### Types of Contributions

We welcome various types of contributions:

1. **Bug Reports**: Help us identify and fix issues
2. **Feature Requests**: Suggest new functionality
3. **Code Contributions**: Bug fixes, new features, improvements
4. **Documentation**: Improve docs, add examples, write tutorials
5. **Testing**: Add tests, improve coverage
6. **Performance**: Optimize algorithms, reduce memory usage

### Development Workflow

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests and checks:**
   ```bash
   # Run tests
   pytest tests/ -v

   # Check code formatting
   black --check gravitational_wave_hunter/
   isort --check-only gravitational_wave_hunter/

   # Run linting
   flake8 gravitational_wave_hunter/

   # Type checking
   mypy gravitational_wave_hunter/
   ```

4. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Add feature: your feature description"
   ```

5. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a pull request** on GitHub

### Code Style Guidelines

#### Python Style

We follow [PEP 8](https://pep8.org/) with some specific guidelines:

- **Line length**: Maximum 88 characters (Black default)
- **Imports**: Use `isort` for import sorting
- **Type hints**: Use type hints for all public functions
- **Docstrings**: Use Google-style docstrings

#### Example Function

```python
def detect_gravitational_waves(
    strain_data: np.ndarray,
    sample_rate: int = 4096,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Detect gravitational waves in strain data.
    
    Args:
        strain_data: Input strain time series
        sample_rate: Sample rate in Hz
        threshold: Detection threshold (0-1)
        
    Returns:
        Dictionary containing detection results
        
    Raises:
        ValueError: If input data is invalid
        
    Example:
        >>> data = np.random.randn(16384)
        >>> results = detect_gravitational_waves(data)
        >>> print(f"Found {len(results['detections'])} signals")
    """
    # Implementation here
    pass
```

#### Neural Network Models

When contributing new models:

1. **Inherit from BaseGWModel:**
   ```python
   from .base import BaseGWModel
   
   class YourModel(BaseGWModel):
       def __init__(self, input_length: int, **kwargs):
           super().__init__(input_length, **kwargs)
   ```

2. **Implement required methods:**
   - `forward()` - Forward pass
   - `init_weights()` - Weight initialization (optional)

3. **Add comprehensive docstrings:**
   - Model description and purpose
   - Architecture details
   - Parameter explanations
   - Usage examples

### Commit Message Guidelines

Use clear, descriptive commit messages:

- **Format**: `type(scope): description`
- **Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- **Scope**: module or component affected
- **Description**: concise explanation of changes

Examples:
```
feat(models): add WaveNet architecture for GW detection
fix(preprocessing): correct whitening frequency interpolation
docs(readme): update installation instructions
test(detector): add unit tests for detection pipeline
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gravitational_wave_hunter --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run tests with specific markers
pytest -m "not slow"  # Skip slow tests
pytest -m "gpu"       # Run only GPU tests
```

### Writing Tests

1. **Test organization:**
   ```
   tests/
   ├── test_models.py         # Model tests
   ├── test_preprocessing.py  # Signal processing tests
   ├── test_detector.py       # Detector tests
   ├── test_data.py          # Data loading tests
   └── conftest.py           # Test configuration
   ```

2. **Test naming:**
   - Use descriptive names: `test_cnn_lstm_forward_pass()`
   - Group related tests in classes: `TestCNNLSTM`

3. **Test structure:**
   ```python
   def test_feature_name():
       # Arrange
       input_data = create_test_data()
       
       # Act
       result = function_under_test(input_data)
       
       # Assert
       assert result.shape == expected_shape
       assert np.allclose(result, expected_output)
   ```

4. **Fixtures for common setup:**
   ```python
   @pytest.fixture
   def sample_strain_data():
       """Create sample strain data for testing."""
       return np.random.randn(4096 * 10)  # 10 seconds at 4096 Hz
   ```

### Test Categories

Mark tests appropriately:

```python
import pytest

@pytest.mark.slow
def test_full_training_pipeline():
    """Test that takes significant time."""
    pass

@pytest.mark.gpu
def test_cuda_acceleration():
    """Test requiring GPU."""
    pass

@pytest.mark.integration
def test_end_to_end_detection():
    """Integration test."""
    pass
```

## Documentation

### Documentation Types

1. **API Documentation**: Automatically generated from docstrings
2. **User Guides**: Tutorials and examples
3. **Developer Documentation**: Architecture and contribution guides
4. **Physics Background**: Gravitational wave theory

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build HTML documentation
cd docs/
make html

# Serve documentation locally
python -m http.server 8000 --directory _build/html/
```

### Writing Documentation

1. **Docstrings**: Use Google style for consistency
2. **Tutorials**: Include complete, runnable examples
3. **API docs**: Document all public functions and classes
4. **Examples**: Show real-world usage patterns

### Documentation Standards

- **Clear explanations**: Assume readers have basic Python knowledge
- **Code examples**: Include working code snippets
- **Mathematical notation**: Use LaTeX for equations
- **Figures**: Include diagrams and plots where helpful

## Pull Request Process

### Before Submitting

1. **Update your branch:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run the full test suite:**
   ```bash
   pytest tests/ --cov=gravitational_wave_hunter
   ```

3. **Check code quality:**
   ```bash
   pre-commit run --all-files
   ```

4. **Update documentation:**
   - Add/update docstrings
   - Update relevant documentation
   - Add examples if appropriate

### Pull Request Template

When creating a pull request, include:

1. **Description**: Clear explanation of changes
2. **Motivation**: Why is this change needed?
3. **Testing**: How was the change tested?
4. **Documentation**: Any documentation updates?
5. **Breaking changes**: Are there any breaking changes?

### Review Process

1. **Automated checks**: All CI checks must pass
2. **Code review**: At least one maintainer review required
3. **Testing**: New features must include tests
4. **Documentation**: Changes must be documented

### Getting Your PR Merged

- Respond to feedback promptly
- Make requested changes
- Keep the PR focused and atomic
- Ensure CI passes
- Be patient - reviews take time

## Issue Reporting

### Bug Reports

When reporting bugs, include:

1. **Environment information:**
   - Python version
   - Package versions
   - Operating system
   - Hardware (CPU/GPU)

2. **Steps to reproduce:**
   - Minimal code example
   - Input data description
   - Expected vs actual behavior

3. **Error messages:**
   - Full traceback
   - Relevant log output

### Feature Requests

For feature requests, describe:

1. **Use case**: What problem does this solve?
2. **Proposed solution**: How should it work?
3. **Alternatives**: Other approaches considered?
4. **Implementation**: Willing to implement it yourself?

### Issue Labels

We use labels to categorize issues:

- `bug`: Something isn't working
- `enhancement`: New feature or improvement
- `documentation`: Documentation related
- `good first issue`: Good for newcomers
- `help wanted`: Community help needed
- `question`: General questions

## Community

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Pull Requests**: Code review and discussion

### Getting Help

1. **Check existing issues**: Someone may have had the same problem
2. **Read the documentation**: Many questions are answered there
3. **Ask questions**: Don't hesitate to ask for help
4. **Provide context**: Include relevant details when asking

### Helping Others

Ways to help the community:

1. **Answer questions**: Help other users
2. **Review pull requests**: Provide feedback on code
3. **Improve documentation**: Fix typos, add examples
4. **Report bugs**: Help identify issues
5. **Share knowledge**: Write tutorials, blog posts

## Acknowledgments

### Recognition

Contributors are recognized through:

- **AUTHORS.md**: List of all contributors
- **Release notes**: Notable contributions mentioned
- **GitHub contributors**: Automatic recognition

### Maintainers

Current maintainers:
- [List of maintainers and their roles]

### Special Thanks

We thank the following organizations and projects:

- **LIGO Scientific Collaboration**: For open data
- **Virgo Collaboration**: For detector data
- **PyTorch**: For the deep learning framework
- **Python Scientific Community**: For excellent tools

## Additional Resources

### Learning Resources

- [LIGO Open Science Center](https://www.gw-openscience.org/)
- [Gravitational Wave Astronomy](https://en.wikipedia.org/wiki/Gravitational-wave_astronomy)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Scientific Python](https://scipy.org/)

### Related Projects

- [GWpy](https://gwpy.github.io/): Python package for gravitational-wave astrophysics
- [PyCBC](https://pycbc.org/): Python toolkit for gravitational wave astronomy
- [LALSuite](https://lscsoft.docs.ligo.org/lalsuite/): LIGO Algorithm Library

### Papers and References

Key papers in gravitational wave machine learning:

1. George, D., & Huerta, E.A. (2018). "Deep Learning for Real-time Gravitational Wave Detection and Parameter Estimation."
2. Gabbard, H., et al. (2018). "Matching matched filtering with deep networks for gravitational-wave astronomy."
3. Cuoco, E., et al. (2020). "Enhancing gravitational-wave science with machine learning."

---

Thank you for contributing to the Gravitational Wave Hunter project! Your contributions help advance gravitational wave astronomy and make this field more accessible to researchers worldwide.
