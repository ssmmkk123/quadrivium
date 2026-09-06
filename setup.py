"""Build the two compiled pieces of the package.

The first is required: ``quadrivium._qnp`` is the array core the whole package
computes with, built here from the C sources under ``csrc/``. There is no
fallback for it, so a wheel or an install without it would be useless; the
build simply fails if it cannot be compiled.

The second is the optional Rust extension holding accelerated kernels. If Cargo
is missing, the build fails, or ``QUADRIVIUM_NO_RUST=1`` is set, installation
still succeeds and those routines run their readable Python implementations.
Set ``QUADRIVIUM_REQUIRE_RUST=1`` to turn a failed build into a hard error
instead (used by the wheel CI, where a silently unaccelerated wheel would be a
defect).

Because the array core is always compiled, every wheel is platform- and
interpreter-specific; there is no pure-Python configuration to tag for.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Distribution, Extension, setup
from setuptools.command.build_py import build_py as _build_py

HERE = Path(__file__).parent.resolve()
CRATE = HERE / "rust"
CSRC = HERE / "csrc"


def core_extension():
    """The array core: strided arrays, ufuncs, linear algebra, transforms."""
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
        ]
    return Extension(
        "quadrivium._qnp",
        sources=sources,
        depends=[str(path.relative_to(HERE)) for path in sorted(CSRC.glob("*.h"))],
        extra_compile_args=compile_args,
    )


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
    # The Rust extension is built against the running interpreter, so it takes
    # that interpreter's extension suffix.
    if artifact.suffix == ".dll":
        return "_quadrivium_rs.pyd"
    import sysconfig

    return "_quadrivium_rs" + (sysconfig.get_config_var("EXT_SUFFIX") or ".so")


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
    """Always a binary distribution: the array core is compiled."""

    def has_ext_modules(self) -> bool:  # noqa: D102 - setuptools hook
        return True


setup(
    cmdclass={"build_py": build_py},
    distclass=ExtensionAwareDistribution,
    ext_modules=[core_extension()],
)
