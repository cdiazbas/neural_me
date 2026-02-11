"""
Custom setup.py for neural_me with automatic C++ extension building.

This setup script attempts to build the C++ extension automatically
during installation. If compilation fails, the package still installs
with PyTorch backend only.
"""

from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
import subprocess
import sys
import os


class PostInstallCommand(install):
    """
    Post-installation hook to build C++ extension.
    """

    def run(self):
        install.run(self)
        self._build_cpp_extension()

    def _build_cpp_extension(self):
        """
        Attempt to build C++ extension.
        """
        try:
            print("\n" + "=" * 60)
            print("Attempting to build C++ extension...")
            print("=" * 60)

            # Run setup_cpp.py build_ext --inplace
            result = subprocess.run(
                [sys.executable, "setup_cpp.py", "build_ext", "--inplace"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print("✓ C++ extension built successfully!")
                print(result.stdout)
            else:
                print("⚠ C++ extension build failed (PyTorch backend will be used)")
                print(result.stderr)

            print("=" * 60 + "\n")

        except Exception as e:
            print(f"⚠ Could not build C++ extension: {e}")
            print("PyTorch backend will be used instead.")


class PostDevelopCommand(develop):
    """
    Post-development installation hook to build C++ extension.
    """

    def run(self):
        develop.run(self)
        self._build_cpp_extension()

    def _build_cpp_extension(self):
        """
        Attempt to build C++ extension.
        """
        try:
            print("\n" + "=" * 60)
            print("Attempting to build C++ extension...")
            print("=" * 60)

            # Run setup_cpp.py build_ext --inplace
            result = subprocess.run(
                [sys.executable, "setup_cpp.py", "build_ext", "--inplace"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print("✓ C++ extension built successfully!")
                print(result.stdout)
            else:
                print("⚠ C++ extension build failed (PyTorch backend will be used)")
                print(result.stderr)

            print("=" * 60 + "\n")

        except Exception as e:
            print(f"⚠ Could not build C++ extension: {e}")
            print("PyTorch backend will be used instead.")


if __name__ == "__main__":
    setup(
        cmdclass={
            "install": PostInstallCommand,
            "develop": PostDevelopCommand,
        },
    )
