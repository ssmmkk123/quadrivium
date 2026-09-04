"""Build hook that compiles the optional Rust acceleration extension.

The extension is genuinely optional. If Cargo is missing, the build fails, or
``QUADRIVIUM_NO_RUST=1`` is set, installation still succeeds and the package
runs its pure-Python implementations. Set ``QUADRIVIUM_REQUIRE_RUST=1`` to turn
a failed extension build into a hard error instead (used by the wheel CI, where
a silently pure-Python wheel would be a defect).

Whether the extension gets built also decides how the wheel is tagged. A wheel
carrying a compiled object must be platform-specific, or `pip` would hand one
platform's binary to every other; a wheel without one is `py3-none-any` and
installs anywhere. The two cases are distinguished by :data:`WILL_BUILD_RUST`,
which is resolved before any command runs.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py as _build_py

HERE = Path(__file__).parent.resolve()
CRATE = HERE / "rust"

# The oldest CPython the abi3 extension is compatible with; must match the
# `abi3-pyXY` feature selected in rust/Cargo.toml.
ABI3_TAG = "cp39"


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no")


REQUIRE_RUST = _truthy("QUADRIVIUM_REQUIRE_RUST")
WILL_BUILD_RUST = (
    not _truthy("QUADRIVIUM_NO_RUST")
    and CRATE.is_dir()
    and shutil.which("cargo") is not None
)

if REQUIRE_RUST and not WILL_BUILD_RUST:
    raise SystemExit(
        "quadrivium: QUADRIVIUM_REQUIRE_RUST is set but the Rust crate or "
        "cargo is unavailable, so no compiled backend could be built"
    )


def _artifact_name() -> str:
    if sys.platform == "win32":
        return "_quadrivium_rs.dll"
    if sys.platform == "darwin":
        return "lib_quadrivium_rs.dylib"
    return "lib_quadrivium_rs.so"


def _target_name() -> str:
    # abi3 extensions use the plain suffix on POSIX and .pyd on Windows.
    return "_quadrivium_rs.pyd" if sys.platform == "win32" else "_quadrivium_rs.abi3.so"


class build_py(_build_py):
    def run(self) -> None:
        self._build_rust()
        super().run()

    def _build_rust(self) -> None:
        if not WILL_BUILD_RUST:
            self.announce(
                "quadrivium: building without the compiled backend "
                "(set QUADRIVIUM_REQUIRE_RUST=1 to make this an error)",
                level=2,
            )
            return
        cmd = [
            "cargo", "build", "--release",
            "--manifest-path", str(CRATE / "Cargo.toml"),
        ]
        env = dict(os.environ)
        # macOS resolves Python symbols at load time rather than link time.
        if sys.platform == "darwin":
            env.setdefault(
                "RUSTFLAGS", "-C link-arg=-undefined -C link-arg=dynamic_lookup"
            )
            if platform.machine() == "arm64":
                env.setdefault("MACOSX_DEPLOYMENT_TARGET", "11.0")
        try:
            subprocess.run(cmd, check=True, env=env)
        except (subprocess.CalledProcessError, OSError) as exc:
            msg = f"quadrivium: Rust extension build failed ({exc}); continuing without it"
            if REQUIRE_RUST:
                raise SystemExit(msg) from exc
            self.announce(msg, level=3)
            return
        built = CRATE / "target" / "release" / _artifact_name()
        if not built.is_file():
            msg = f"quadrivium: expected {built} after a successful cargo build"
            if REQUIRE_RUST:
                raise SystemExit(msg)
            self.announce(msg, level=3)
            return
        for root in (HERE / "quadrivium", Path(self.build_lib) / "quadrivium"):
            root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(built, root / _target_name())
        self.announce("quadrivium: compiled backend built", level=2)


class ExtensionAwareDistribution(Distribution):
    """Reports a binary distribution exactly when one is being produced."""

    def has_ext_modules(self) -> bool:  # noqa: D102 - setuptools hook
        return WILL_BUILD_RUST


cmdclass = {"build_py": build_py}

# Tag the wheel `cp39-abi3-<platform>` rather than `cp39-cp3XX-<platform>`, so
# one build serves every supported interpreter.
try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # pragma: no cover - older setuptools
    try:
        from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    except ImportError:
        _bdist_wheel = None

if _bdist_wheel is not None:

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self) -> None:
            if WILL_BUILD_RUST:
                self.py_limited_api = ABI3_TAG
                self.root_is_pure = False
            super().finalize_options()

    cmdclass["bdist_wheel"] = bdist_wheel


setup(cmdclass=cmdclass, distclass=ExtensionAwareDistribution)
