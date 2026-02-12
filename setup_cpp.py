"""
Conditional C++ extension builder for pyMilne backend.

This script attempts to compile the pyMilne C++ extension if
dependencies are available. Falls back gracefully to pure PyTorch if
compilation fails.
"""

import os
import sys
import subprocess
import pathlib
from setuptools import Extension
from setuptools.command.build_ext import build_ext
import numpy
import platform as plt


class CustomBuildExt(build_ext):
    """
    Custom build_ext to copy .so to src/neural_me/physics/
    """

    def run(self):
        """
        Run the build and copy artifacts to physics folder.
        """
        import shutil
        from pathlib import Path

        # Run the standard build
        build_ext.run(self)

        # Copy the compiled .so to src/neural_me/physics/milne_eddington_cpp/
        physics_dir = Path("src/neural_me/physics/milne_eddington_cpp")
        physics_dir.mkdir(parents=True, exist_ok=True)

        # Find the compiled .so file
        for ext in self.extensions:
            # Get the output filename from build
            ext_path = self.get_ext_fullpath(ext.name)
            if os.path.exists(ext_path):
                dest = physics_dir / os.path.basename(ext_path)
                # Only copy if source and dest are different
                if not os.path.samefile(ext_path, dest):
                    print(f"Copying {ext_path} -> {dest}")
                    shutil.copy2(ext_path, dest)
                else:
                    print(f"✓ {dest.name} already in place")

        # Copy MilneEddington.py interface from external/pyMilne/
        milne_interface = Path("external/pyMilne/MilneEddington.py")
        if milne_interface.exists():
            dest = physics_dir / "MilneEddington.py"

            # Read content and patch import
            with open(milne_interface, "r") as f:
                content = f.read()

            # Patch absolute import 'import pyMilne' to relative 'from . import pyMilne'
            # because inside the package, implicit relative imports are forbidden in Py3
            if "import pyMilne" in content and "from . import pyMilne" not in content:
                print(f"Patching {milne_interface} -> relative import")
                content = content.replace("import pyMilne", "from . import pyMilne")

            # Write patched content to destination
            with open(dest, "w") as f:
                f.write(content)
            print(f"✓ Copied and patched {dest}")

        else:
            print(f"Warning: {milne_interface} not found, C++ backend may not work")

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:
            print(f"Failed to build {ext.name}: {e}")


def find_eigen3():
    """
    Try to locate Eigen3 include directory.
    """
    # Priority order:
    # 1. Environment variables (conda/mamba)
    # 2. pkg-config
    # 3. Common system locations

    candidates = []

    # Check conda/mamba environment first
    for env_var in ["CONDA_PREFIX", "MAMBA_PREFIX"]:
        if env_var in os.environ:
            conda_eigen = os.path.join(os.environ[env_var], "include", "eigen3")
            candidates.append(conda_eigen)

    # Add common system locations
    candidates.extend(
        [
            "/usr/include/eigen3",
            "/usr/local/include/eigen3",
            "/opt/local/include/eigen3",
            os.path.expanduser("~/anaconda3/include/eigen3"),
            os.path.expanduser("~/miniconda3/include/eigen3"),
        ]
    )

    # Try pkg-config first
    try:
        result = subprocess.run(
            ["pkg-config", "--cflags", "eigen3"],
            capture_output=True,
            text=True,
            check=True,
        )
        include_flag = result.stdout.strip()
        if include_flag.startswith("-I"):
            pkg_config_path = include_flag[2:]
            if os.path.exists(pkg_config_path):
                return pkg_config_path
    except:
        pass

    # Check candidates in order
    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def get_pymilne_extension():
    """
    Build pyMilne extension if dependencies are available.
    """

    eigen_path = find_eigen3()
    if eigen_path is None:
        print("WARNING: Eigen3 not found. Skipping C++ backend compilation.")
        return None

    print(f"Found Eigen3 at: {eigen_path}")

    # Platform-specific settings
    if plt.system() == "Darwin":
        CC = "clang"
        CXX = "clang++"
        link_opts = [
            "-stdlib=libc++",
            "-bundle",
            "-undefined",
            "dynamic_lookup",
            "-fopenmp",
            "-lgomp",
        ]
    else:
        CC = "gcc"
        CXX = "g++"
        link_opts = ["-shared", "-fopenmp"]

    os.environ["CC"] = CC
    os.environ["CXX"] = CXX

    # Compilation flags (removed -march=native and -mtune=native for CPU compatibility)
    comp_flags = [
        "-O3",
        "-mavx2",
        "-mfma",
        "-fno-math-errno",
        "-fsimd-cost-model=unconditional",
        "-mprefer-vector-width=256",
        "-fopenmp-simd",
        "-ftree-vectorize",
        "-g0",
        "-fstrict-aliasing",
        "-std=c++20",
        "-fPIC",
        "-fopenmp",
        "-I./external/pyMilne/src",
        "-DNPY_NO_DEPRECATED_API",
        "-DNDEBUG",
        "-pedantic",
        "-Wall",
        f"-I{eigen_path}",
    ]

    pymilne_dir = pathlib.Path("external/pyMilne")

    # Define target directory for artifacts
    pathlib.Path("src/neural_me/physics/milne_eddington_cpp")

    extension = Extension(
        name="neural_me.physics.milne_eddington_cpp.pyMilne",
        sources=[
            str(pymilne_dir / "pyMilne.pyx"),
            str(pymilne_dir / "src/wrapper_tools_spatially_coupled.cpp"),
            str(pymilne_dir / "src/lm_sc.cpp"),
            str(pymilne_dir / "src/spatially_coupled_helper.cpp"),
        ],
        include_dirs=[
            str(pymilne_dir),
            numpy.get_include(),
            str(pymilne_dir / "eigen3"),
            eigen_path,
        ],
        language="c++",
        extra_compile_args=comp_flags,
        extra_link_args=comp_flags + link_opts,
        library_dirs=["./", "/usr/lib/", os.path.join(sys.prefix, "lib")],
        libraries=["fftw3"],
    )

    extension.cython_directives = {"language_level": "3"}

    return extension


def get_extensions():
    """
    Return list of extensions to build (may be empty if C++ unavailable).
    """
    ext = get_pymilne_extension()
    return [ext] if ext is not None else []


if __name__ == "__main__":
    from setuptools import setup
    from Cython.Build import cythonize

    # Get extension
    ext = get_pymilne_extension()

    if ext:
        print("✓ C++ extension configuration successful")
        print(f"  Extension name: {ext.name}")
        print(f"  Sources: {len(ext.sources)} files")
        print("\nStarting compilation...")
        print("=" * 60)

        # Run setup with the extension
        setup(
            name="neural_me_cpp",
            package_dir={"": "src"},
            ext_modules=cythonize([ext], compiler_directives={"language_level": "3"}),
            cmdclass={"build_ext": CustomBuildExt},
        )

        print("=" * 60)
        print("✓ Build complete!")
    else:
        print("✗ C++ extension not available")
        sys.exit(1)
