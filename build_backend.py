"""
Custom build backend for neural_me with automatic C++ extension building.

This module provides a custom build backend that wraps setuptools and
automatically attempts to build the C++ extension during the build
process.
"""

from setuptools import build_meta as _orig
import subprocess
import sys
import os


# Re-export all the standard setuptools.build_meta functions
__all__ = [
    "get_requires_for_build_wheel",
    "prepare_metadata_for_build_wheel",
    "build_wheel",
    "get_requires_for_build_sdist",
    "build_sdist",
    "get_requires_for_build_editable",
    "prepare_metadata_for_build_editable",
    "build_editable",
]


def get_requires_for_build_wheel(config_settings=None):
    """
    Get requirements for building wheel.
    """
    return _orig.get_requires_for_build_wheel(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    """
    Prepare metadata for wheel.
    """
    return _orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """
    Build wheel with automatic C++ extension compilation.
    """

    # Try to build C++ extension before building wheel
    _try_build_cpp_extension()

    # Build the wheel normally
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def get_requires_for_build_editable(config_settings=None):
    """
    Get requirements for building editable install.
    """
    return _orig.get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    """
    Prepare metadata for editable install.
    """
    return _orig.prepare_metadata_for_build_editable(
        metadata_directory, config_settings
    )


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """
    Build editable install with automatic C++ extension compilation.
    """

    # Try to build C++ extension before building editable install
    _try_build_cpp_extension()

    # Build the editable install normally
    return _orig.build_editable(wheel_directory, config_settings, metadata_directory)


def get_requires_for_build_sdist(config_settings=None):
    """
    Get requirements for building sdist.
    """
    return _orig.get_requires_for_build_sdist(config_settings)


def build_sdist(sdist_directory, config_settings=None):
    """
    Build sdist.
    """
    return _orig.build_sdist(sdist_directory, config_settings)


def _try_build_cpp_extension():
    """
    Attempt to build C++ extension.
    """
    try:
        print("\n" + "=" * 60)
        print("Attempting to build C++ extension...")
        print("=" * 60)

        # Get the project root directory
        project_root = os.path.dirname(os.path.abspath(__file__))

        # Run setup_cpp.py build_ext --inplace
        result = subprocess.run(
            [sys.executable, "setup_cpp.py", "build_ext", "--inplace"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✓ C++ extension built successfully!")
            if result.stdout:
                print(result.stdout)
        else:
            print("⚠ C++ extension build failed (PyTorch backend will be used)")
            if result.stderr:
                print(result.stderr)

        print("=" * 60 + "\n")

    except Exception as e:
        print(f"⚠ Could not build C++ extension: {e}")
        print("PyTorch backend will be used instead.")
