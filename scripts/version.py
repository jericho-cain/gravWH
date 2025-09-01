#!/usr/bin/env python3
"""
Version Management Script for Gravitational Wave Hunter.

A comprehensive version management tool that handles version bumping, file updates,
git operations, and release creation for the gravitational wave detection framework.

Usage
-----
    python scripts/version.py bump patch    # 0.1.0 -> 0.1.1
    python scripts/version.py bump minor    # 0.1.0 -> 0.2.0
    python scripts/version.py bump major    # 0.1.0 -> 1.0.0
    python scripts/version.py show          # Show current version
    python scripts/version.py release       # Create release (bump + tag + push)

Features
--------
- Automatic version bumping (patch, minor, major)
- Multi-file version synchronization
- Git integration (commit, tag, push)
- Release automation
- Version validation and error handling

Notes
-----
This script ensures consistent versioning across all project files and
automates the release process for the gravitational wave detection framework.
"""

import subprocess
import sys
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional


def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """
    Run a shell command and return the result.
    
    Executes a shell command and handles errors appropriately based on the
    check parameter.
    
    Parameters
    ----------
    cmd : str
        The shell command to execute.
    check : bool, optional
        If True, exit on error, by default True.
    
    Returns
    -------
    subprocess.CompletedProcess
        The result of the command execution.
    
    Raises
    ------
    SystemExit
        If check=True and command fails.
    """
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result


def get_current_version() -> str:
    """
    Get current version from pyproject.toml using regex.
    
    Extracts the current version number from the pyproject.toml file
    using regular expression matching.
    
    Returns
    -------
    str
        The current version string (e.g., "0.1.2").
    
    Raises
    ------
    SystemExit
        If version cannot be found or file cannot be read.
    """
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


def update_version_files(new_version: str) -> None:
    """
    Update version in all relevant files.
    
    Updates the version number in pyproject.toml, __init__.py, and README.md
    to maintain consistency across the project.
    
    Parameters
    ----------
    new_version : str
        The new version string to set (e.g., "0.1.3").
    
    Raises
    ------
    SystemExit
        If any file cannot be updated.
    """
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


def bump_version(part: str) -> str:
    """
    Bump version manually.
    
    Increments the version number according to semantic versioning rules
    and updates all relevant files.
    
    Parameters
    ----------
    part : str
        Which part to bump: "major", "minor", or "patch".
    
    Returns
    -------
    str
        The new version string.
    
    Raises
    ------
    SystemExit
        If part is invalid or version bumping fails.
    """
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


def create_release() -> None:
    """
    Create a release by bumping patch version and creating git tag.
    
    Automates the release process by bumping the patch version, creating
    a git tag, and pushing changes to the remote repository.
    
    Notes
    -----
    This function performs the following steps:
    1. Bumps the patch version
    2. Commits the changes
    3. Creates a git tag
    4. Pushes changes and tag to remote
    """
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


def show_version() -> None:
    """
    Show current version.
    
    Displays the current version of the gravitational wave detection framework
    as specified in pyproject.toml.
    """
    version = get_current_version()
    print(f"Current version: {version}")


def main() -> None:
    """
    Main function for version management script.
    
    Parses command line arguments and executes the appropriate version
    management function.
    
    Commands
    --------
    bump <part> : Bump version (patch, minor, major)
    show        : Display current version
    release     : Create a new release
    
    Raises
    ------
    SystemExit
        If invalid command or arguments provided.
    """
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
