# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive test suite with pytest
- GitHub Actions CI/CD workflow
- Automated versioning with bump2version
- Branch protection rules for main branch
- Version management scripts

### Changed
- Updated project structure and documentation
- Improved model architecture and testing

### Fixed
- Various test failures and model compatibility issues

## [0.1.0] - 2024-01-XX

### Added
- Initial release of Gravitational Wave Hunter
- CWT-LSTM Autoencoder implementation
- SimpleCWTAutoencoder for testing
- Data generation utilities for synthetic gravitational wave signals
- Continuous Wavelet Transform preprocessing
- Comprehensive evaluation metrics
- PyTorch-based deep learning framework
- Support for LIGO-like colored noise generation
- Anomaly detection pipeline

### Features
- **CWT-LSTM Autoencoder**: Combines Continuous Wavelet Transform with LSTM autoencoder for gravitational wave detection
- **Data Generation**: Realistic chirp signal generation with controlled SNR
- **Preprocessing**: High-pass filtering, whitening, and CWT transformation
- **Evaluation**: Precision, Recall, F1-Score, AUC-ROC, and AUC-PR metrics
- **Modular Design**: Clean separation of data, models, and evaluation components

### Technical Details
- Python 3.8+ compatibility
- PyTorch 2.0+ support
- Comprehensive dependency management with pyproject.toml
- MIT License

---

## Versioning Guidelines

### Semantic Versioning (MAJOR.MINOR.PATCH)

- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

### Release Process

1. **Development**: Work on feature branches
2. **Testing**: Ensure all tests pass
3. **Version Bump**: Use `python scripts/version.py bump <part>`
4. **Release**: Use `python scripts/version.py release`
5. **Automation**: GitHub Actions handles building and publishing

### Commands

```bash
# Show current version
python scripts/version.py show

# Bump version
python scripts/version.py bump patch    # 0.1.0 -> 0.1.1
python scripts/version.py bump minor    # 0.1.0 -> 0.2.0
python scripts/version.py bump major    # 0.1.0 -> 1.0.0

# Create release (bump patch + tag + push)
python scripts/version.py release
```
