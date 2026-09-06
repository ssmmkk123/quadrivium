"""The optional Rust build must never package a stale native extension.

The array core in `csrc/` is a different matter: it is required, so every
distribution is a binary one and there is no configuration in which its absence
is acceptable. What is tested here is the part that may legitimately be missing.
"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("setuptools"), "setuptools is a build dependency")
class TestOptionalNativeBuild(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        spec = importlib.util.spec_from_file_location(
            "quadrivium_test_setup", Path(__file__).resolve().parents[1] / "setup.py")
        if not spec.origin or not Path(spec.origin).exists():
            self.skipTest("setup.py is unavailable in a wheel-only installation")
        self.module = importlib.util.module_from_spec(spec)
        with patch("setuptools.setup"), patch.dict(os.environ, {"QUADRIVIUM_REQUIRE_RUST": "0"}):
            spec.loader.exec_module(self.module)
        self.module.HERE = self.root
        self.module.CRATE = self.root / "rust"
        self.module.CRATE.mkdir()
        self.module.WILL_BUILD_RUST = True
        self.module.REQUIRE_RUST = False
        self.distribution = self.module.ExtensionAwareDistribution({"name": "build-test"})
        self.command = self.module.build_py(self.distribution)
        self.command.build_lib = str(self.root / "build")
        self.command.editable_mode = False
        suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        self.source = self.root / "quadrivium" / ("_quadrivium_rs" + suffix)
        self.staged = self.root / "build" / "quadrivium" / self.source.name
        for path in (self.source, self.staged):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"stale backend")

    def test_disabled_build_cleans_stage_and_preserves_checkout(self):
        self.module.WILL_BUILD_RUST = False
        with patch.object(self.module.subprocess, "run") as cargo:
            self.command._build_rust()
        cargo.assert_not_called()
        self.assertFalse(self.staged.exists())
        self.assertEqual(self.source.read_bytes(), b"stale backend")
        self.assertFalse(self.distribution._rust_build_succeeded)

    def test_failed_optional_build_does_not_reuse_binary(self):
        with patch.object(self.module.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "cargo")):
            self.command._build_rust()
        self.assertFalse(self.staged.exists())
        self.assertFalse(self.distribution._rust_build_succeeded)
        self.assertTrue(self.source.exists())

    def test_required_build_propagates_failure(self):
        self.module.REQUIRE_RUST = True
        with patch.object(self.module.subprocess, "run", side_effect=OSError("compiler missing")):
            with self.assertRaises(SystemExit):
                self.command._build_rust()
        self.assertFalse(self.staged.exists())

    def test_success_uses_reported_custom_target_and_leaves_source_unchanged(self):
        artifact = self.root / "custom-target" / "cross-target" / "release" / "lib_quadrivium_rs.so"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"fresh backend")
        message = json.dumps({"reason": "compiler-artifact", "target": {
            "name": "_quadrivium_rs", "crate_types": ["cdylib", "rlib"]},
            "filenames": [str(artifact)]})
        result = subprocess.CompletedProcess("cargo", 0, stdout=message)
        with patch.object(self.module.subprocess, "run", return_value=result) as cargo:
            self.command._build_rust()
        self.assertIn("--locked", cargo.call_args.args[0])
        self.assertIn("--message-format=json-render-diagnostics", cargo.call_args.args[0])
        self.assertEqual(self.staged.read_bytes(), b"fresh backend")
        self.assertEqual(self.source.read_bytes(), b"stale backend")
        self.assertTrue(self.distribution._rust_build_succeeded)

    def test_success_without_artifact_leaves_the_accelerator_out(self):
        result = subprocess.CompletedProcess("cargo", 0, stdout='{"reason":"build-finished"}\n')
        with patch.object(self.module.subprocess, "run", return_value=result):
            self.command._build_rust()
        self.assertFalse(self.staged.exists())
        self.assertFalse(self.distribution._rust_build_succeeded)

    def test_the_distribution_is_always_binary(self):
        # The array core is compiled and not optional, so there is no
        # configuration in which a pure-Python wheel would be correct.
        self.module.WILL_BUILD_RUST = False
        with patch.object(self.module.subprocess, "run"):
            self.command._build_rust()
        self.assertTrue(self.distribution.has_ext_modules())

    def test_the_core_extension_lists_every_c_source(self):
        root = Path(__file__).resolve().parents[1]
        sources = sorted(p.name for p in (root / "csrc").glob("*.c"))
        self.assertTrue(sources, "the array core has no C sources")
        with patch.object(self.module, "HERE", root), patch.object(
                self.module, "CSRC", root / "csrc"):
            extension = self.module.core_extension()
        self.assertEqual(sorted(Path(s).name for s in extension.sources), sources)
        self.assertEqual(extension.name, "quadrivium._qnp")

    def test_editable_no_rust_removes_stale_source_backend(self):
        self.module.WILL_BUILD_RUST = False
        self.command.editable_mode = True
        self.command._build_rust()
        self.assertFalse(self.source.exists())
        self.assertFalse(self.staged.exists())

    def test_source_package_data_never_contains_cached_extensions(self):
        files = [str(self.source), "_quadrivium_rs.pyd", "lib_quadrivium_rs.dylib", "data.txt"]
        with patch.object(self.module._build_py, "find_data_files", return_value=files):
            self.assertEqual(self.command.find_data_files("quadrivium", "."), ["data.txt"])


if __name__ == "__main__":
    unittest.main()
