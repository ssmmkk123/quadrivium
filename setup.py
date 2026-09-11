"""Build the required C array core and numerical accelerators together.

Every wheel is platform- and interpreter-specific because the extension uses
CPython's full C API. A source install needs a C compiler and no other language
toolchain; setuptools builds every C source under ``csrc/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from setuptools import Distribution, Extension, setup
from setuptools.command.build_py import build_py as _build_py

HERE = Path(__file__).parent.resolve()
CSRC = HERE / "csrc"
NATIVE_SUFFIXES = {".so", ".pyd", ".dylib", ".dll"}


def core_extension():
    """The array core and numerical accelerators, built as one C extension."""
    sources = sorted(str(path.relative_to(HERE)) for path in CSRC.glob("*.c"))
    if not sources:
        raise SystemExit(
            "quadrivium: the C sources of the array core are missing from "
            f"{CSRC}; the package cannot be built without them"
        )
    if sys.platform == "win32":
        compile_args = ["/O2"]
    else:
        compile_args = [
            "-O3",
            # The core reads one buffer through several element types, which
            # the strict-aliasing rules do not allow the compiler to assume
            # away. This is a correctness flag, not a tuning one.
            "-fno-strict-aliasing",
            # `errno` is never read after a libm call here, and setting it
            # blocks vectorisation of the element-wise loops.
            "-fno-math-errno",
            # Keep embedded ODE error estimates consistent with Python
            # arithmetic, including on architectures with fused multiply-add.
            "-ffp-contract=off",
        ]
    return Extension(
        "quadrivium._qnp",
        sources=sources,
        depends=[str(path.relative_to(HERE)) for path in sorted(CSRC.glob("*.h"))],
        extra_compile_args=compile_args,
    )



class build_py(_build_py):
    """Copy Python sources without recycling native binaries from old builds."""

    def run(self) -> None:
        # A reused build directory can contain an obsolete extension that
        # package-data filtering cannot see. The subsequent build_ext command
        # installs the current extension; never modify an in-place checkout.
        staged = (Path(self.build_lib) / "quadrivium").resolve()
        source = (HERE / "quadrivium").resolve()
        if staged != source and staged.is_dir():
            for path in staged.rglob("*"):
                if path.is_file() and path.suffix in NATIVE_SUFFIXES:
                    path.unlink()
        super().run()

    def find_data_files(self, package, src_dir):
        # Cached egg-info/SOURCES.txt can still list developer-built binaries.
        # Only build_ext is allowed to add a native module to the wheel.
        return [path for path in super().find_data_files(package, src_dir)
                if Path(path).suffix not in NATIVE_SUFFIXES]


class ExtensionAwareDistribution(Distribution):
    """Always a binary distribution: the array core is compiled."""

    def has_ext_modules(self) -> bool:
        return True


setup(
    cmdclass={"build_py": build_py},
    distclass=ExtensionAwareDistribution,
    ext_modules=[core_extension()],
)
