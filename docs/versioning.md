# Versioning Guide

This document explains the versioning system used in Gravitational Wave Hunter.

## Overview

We use **Semantic Versioning (SemVer)** with automated tools for version management:

- **Format**: `MAJOR.MINOR.PATCH` (e.g., `0.1.0`)
- **Automation**: `bump2version` for version bumping
- **Releases**: GitHub Actions for automated releases
- **Documentation**: CHANGELOG.md for change tracking

## Current Version

**Version**: `0.1.0` (Alpha)

- **Development Status**: Alpha
- **Stability**: API may change
- **Production Ready**: No

## Version Components

### MAJOR Version
- **When**: Breaking API changes
- **Example**: `0.1.0` → `1.0.0`
- **Impact**: Incompatible changes

### MINOR Version
- **When**: New features (backwards compatible)
- **Example**: `0.1.0` → `0.2.0`
- **Impact**: New functionality added

### PATCH Version
- **When**: Bug fixes (backwards compatible)
- **Example**: `0.1.0` → `0.1.1`
- **Impact**: Bug fixes only

## Quick Commands

### Show Current Version
```bash
python scripts/version.py show
```

### Bump Version
```bash
# Bug fix
python scripts/version.py bump patch

# New feature
python scripts/version.py bump minor

# Breaking change
python scripts/version.py bump major
```

### Create Release
```bash
python scripts/version.py release
```

## Automated Workflow

### 1. Development
- Work on feature branches
- Ensure tests pass
- Update CHANGELOG.md

### 2. Version Bump
```bash
# Choose appropriate bump type
python scripts/version.py bump patch
```

### 3. Release
```bash
# Creates tag and pushes to GitHub
python scripts/version.py release
```

### 4. Automation
- GitHub Actions builds package
- Creates GitHub release
- Uploads to PyPI (if configured)

## Files Updated Automatically

When you bump the version, these files are updated:

1. **`pyproject.toml`**: Package version
2. **`gravitational_wave_hunter/__init__.py`**: Module version
3. **Git**: Commit and tag created

## Manual Version Management

### Using bump2version directly
```bash
# Install if not already installed
pip install bump2version

# Bump version
bump2version patch
bump2version minor
bump2version major

# Show current version
bump2version --current-version
```

### Git Commands
```bash
# Create tag manually
git tag v0.1.1

# Push tag
git push origin v0.1.1

# List tags
git tag -l
```

## Release Process

### Pre-Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No breaking changes (or major version bump)

### Release Steps
1. **Bump version**: `python scripts/version.py bump patch`
2. **Review changes**: `git diff`
3. **Commit**: `git add . && git commit -m "Bump version to X.Y.Z"`
4. **Release**: `python scripts/version.py release`

### Post-Release
- GitHub Actions automatically:
  - Builds the package
  - Creates GitHub release
  - Uploads to PyPI (if configured)
- Monitor the release process in GitHub Actions

## Configuration

### bump2version Configuration
Located in `pyproject.toml`:

```toml
[tool.bumpversion]
current_version = "0.1.0"
commit = true
tag = true
parse = (?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)
serialize = {major}.{minor}.{patch}

[tool.bumpversion.file:pyproject.toml]
search = version = "{current_version}"
replace = version = "{new_version}"

[tool.bumpversion.file:gravitational_wave_hunter/__init__.py]
search = __version__ = "{current_version}"
replace = __version__ = "{new_version}"
```

### GitHub Actions
- **Workflow**: `.github/workflows/release.yml`
- **Trigger**: Push to tags starting with `v`
- **Actions**: Build, release, PyPI upload

## Best Practices

### When to Bump Versions

**PATCH (0.1.0 → 0.1.1)**
- Bug fixes
- Documentation updates
- Minor improvements

**MINOR (0.1.0 → 0.2.0)**
- New features
- Performance improvements
- New dependencies

**MAJOR (0.1.0 → 1.0.0)**
- Breaking API changes
- Major architectural changes
- Incompatible updates

### Commit Messages
- Use conventional commit format
- Reference issues when applicable
- Be descriptive about changes

### CHANGELOG Updates
- Update before releasing
- Use clear, user-friendly language
- Group changes by type (Added, Changed, Fixed, Removed)

## Troubleshooting

### Common Issues

**Version not updating**
- Check `pyproject.toml` configuration
- Ensure bump2version is installed
- Verify file paths in configuration

**Git tag issues**
- Check git permissions
- Ensure remote is configured
- Verify tag format (should start with 'v')

**GitHub Actions failures**
- Check workflow file syntax
- Verify secrets are configured
- Review action logs for errors

### Getting Help
- Check the [bump2version documentation](https://github.com/c4urself/bump2version)
- Review GitHub Actions logs
- Consult the project's issue tracker
