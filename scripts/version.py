#!/usr/bin/env python3
"""
Version management script for Gravitational Wave Hunter.

Usage:
    python scripts/version.py bump patch    # 0.1.0 -> 0.1.1
    python scripts/version.py bump minor    # 0.1.0 -> 0.2.0
    python scripts/version.py bump major    # 0.1.0 -> 1.0.0
    python scripts/version.py show          # Show current version
    python scripts/version.py release       # Create release (bump + tag + push)
"""

import subprocess
import sys
import os
import re
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command and return the result."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result


def get_current_version():
    """Get current version from pyproject.toml using regex."""
    try:
        with open("pyproject.toml", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Find version line
        match = re.search(r'version\s*=\s*"([^"]+)"', content)
        if match:
            return match.group(1)
        else:
            print("Error: Could not find version in pyproject.toml")
            sys.exit(1)
    except Exception as e:
        print(f"Error reading version: {e}")
        sys.exit(1)


def update_version_files(new_version):
    """Update version in all relevant files."""
    files_to_update = [
        ("pyproject.toml", r'version\s*=\s*"[^"]+"', f'version = "{new_version}"'),
        ("gravitational_wave_hunter/__init__.py", r'__version__\s*=\s*"[^"]+"', f'__version__ = "{new_version}"'),
        ("README.md", r'\[!\[Version\]\(https://img\.shields\.io/badge/version-[^)]+\)\]\([^)]+\)', f'[![Version](https://img.shields.io/badge/version-{new_version}-blue.svg)](https://github.com/jericho-cain/gravWH/releases)')
    ]
    
    for file_path, pattern, replacement in files_to_update:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            updated_content = re.sub(pattern, replacement, content)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(updated_content)
            
            print(f"Updated {file_path} to version {new_version}")
        except Exception as e:
            print(f"Error updating {file_path}: {e}")
            sys.exit(1)


def bump_version(part):
    """Bump version manually."""
    current_version = get_current_version()
    major, minor, patch = map(int, current_version.split('.'))
    
    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    else:
        print(f"Invalid part: {part}")
        sys.exit(1)
    
    new_version = f"{major}.{minor}.{patch}"
    print(f"Bumping version from {current_version} to {new_version}")
    
    # Update files
    update_version_files(new_version)
    
    # Commit changes
    run_command("git add pyproject.toml gravitational_wave_hunter/__init__.py")
    run_command(f'git commit -m "Bump version to {new_version}"')
    
    return new_version


def create_release():
    """Create a release by bumping patch version and creating git tag."""
    print("Creating release...")
    
    # Bump patch version
    new_version = bump_version("patch")
    
    # Get the git tag
    tag = f"v{new_version}"
    
    # Create tag
    run_command(f"git tag {tag}")
    
    # Push changes and tag
    print("Pushing changes...")
    run_command("git push origin main")
    
    print(f"Pushing tag: {tag}")
    run_command(f"git push origin {tag}")
    
    print(f"Release {tag} created successfully!")
    print("GitHub Actions will automatically build and publish the release.")


def show_version():
    """Show current version."""
    version = get_current_version()
    print(f"Current version: {version}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "bump":
        if len(sys.argv) < 3:
            print("Usage: python scripts/version.py bump <part>")
            print("Parts: patch, minor, major")
            sys.exit(1)
        part = sys.argv[2]
        if part not in ["patch", "minor", "major"]:
            print("Invalid part. Use: patch, minor, or major")
            sys.exit(1)
        bump_version(part)
    
    elif command == "show":
        show_version()
    
    elif command == "release":
        create_release()
    
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
