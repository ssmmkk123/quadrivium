"""Build hook that compiles the optional Rust acceleration extension.

The extension is genuinely optional. If Cargo is missing, the build fails, or
``QUADRIVIUM_NO_RUST=1`` is set, installation still succeeds and the package
runs its pure-Python implementations. Set ``QUADRIVIUM_REQUIRE_RUST=1`` to turn
a failed extension build into a hard error instead (used by the wheel CI, where
a silently pure-Python wheel would be a defect).

Whether the extension gets built also decides how the wheel is tagged. A wheel
carrying a compiled object must be platform-specific, or `pip` would hand one
platform's binary to every other; a wheel without one is `py3-none-any` and
installs anywhere. Build eligibility is resolved before commands run; normal
wheel tags are updated after Cargo reports whether a backend was produced.
"""

from __future__ import annotations

import json
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


def _target_name(artifact: Path) -> str:
    # abi3 extensions use the plain suffix on POSIX and .pyd on Windows.
    return "_quadrivium_rs.pyd" if artifact.suffix == ".dll" else "_quadrivium_rs.abi3.so"


def _cargo_artifact(output: str):
    """Use Cargo's reported path, including custom target directories/triples."""
    for line in output.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(message, dict) or message.get("reason") != "compiler-artifact":
            continue
        target = message.get("target", {})
        if target.get("name") != "_quadrivium_rs" or "cdylib" not in target.get("crate_types", []):
            continue
        for filename in message.get("filenames", []):
            artifact = Path(filename)
            if not artifact.is_absolute():
                artifact = CRATE / artifact
            if artifact.suffix in (".so", ".dylib", ".dll") and artifact.is_file():
                return artifact
    return None


class build_py(_build_py):
    def run(self) -> None:
        super().run()
        self._build_rust()

    def find_data_files(self, package, src_dir):
        # Package-data and a cached SOURCES.txt can both discover binaries in
        # a developer checkout. Only the current Cargo invocation may add one.
        return [path for path in super().find_data_files(package, src_dir)
                if Path(path).suffix not in (".so", ".pyd", ".dylib")]

    def get_outputs(self, include_bytecode=True):
        outputs = super().get_outputs(include_bytecode)
        backend = getattr(self, "_backend_output", None)
        if backend is not None and not self.editable_mode:
            outputs.append(str(backend))
        return outputs

    def get_output_mapping(self):
        mapping = super().get_output_mapping()
        backend = getattr(self, "_backend_output", None)
        if backend is not None and self.editable_mode:
            output = Path(self.build_lib) / "quadrivium" / backend.name
            mapping[str(output)] = str(backend)
        return mapping

    def _build_rust(self) -> None:
        self._backend_output = None
        self.distribution._rust_build_succeeded = False
        roots = [Path(self.build_lib) / "quadrivium"]
        if self.editable_mode:
            roots.append(HERE / "quadrivium")
        # Reusing a build directory after NO_RUST or a failed compile must not
        # reuse a previous backend. Normal builds never change the checkout.
        for root in roots:
            for path in root.glob("_quadrivium_rs*"):
                if path.suffix in (".so", ".pyd", ".dylib"):
                    path.unlink()
        if not WILL_BUILD_RUST:
            self.announce(
                "quadrivium: building without the compiled backend "
                "(set QUADRIVIUM_REQUIRE_RUST=1 to make this an error)",
                level=2,
            )
            return
        cmd = [
            "cargo", "build", "--release", "--locked",
            "--message-format=json-render-diagnostics",
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
            result = subprocess.run(cmd, check=True, env=env, cwd=CRATE,
                                    stdout=subprocess.PIPE, text=True)
        except (subprocess.CalledProcessError, OSError) as exc:
            msg = f"quadrivium: Rust extension build failed ({exc}); continuing without it"
            if REQUIRE_RUST:
                raise SystemExit(msg) from exc
            self.announce(msg, level=3)
            return
        built = _cargo_artifact(result.stdout)
        if built is None:
            msg = "quadrivium: Cargo did not report a compiled backend artifact"
            if REQUIRE_RUST:
                raise SystemExit(msg)
            self.announce(msg, level=3)
            return
        root = HERE / "quadrivium" if self.editable_mode else roots[0]
        root.mkdir(parents=True, exist_ok=True)
        self._backend_output = root / _target_name(built)
        shutil.copy2(built, self._backend_output)
        self.distribution._rust_build_succeeded = True
        self.announce("quadrivium: compiled backend built", level=2)


class ExtensionAwareDistribution(Distribution):
    """Reports a binary distribution exactly when one is being produced."""

    def has_ext_modules(self) -> bool:  # noqa: D102 - setuptools hook
        return getattr(self, "_rust_build_succeeded", WILL_BUILD_RUST)


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

        def run_command(self, command):
            super().run_command(command)
            if command == "build":
                # bdist_wheel consults this flag before choosing its install
                # layout and tags. A failed optional build is a pure wheel.
                self.root_is_pure = not self.distribution.has_ext_modules()

    cmdclass["bdist_wheel"] = bdist_wheel


setup(cmdclass=cmdclass, distclass=ExtensionAwareDistribution)
