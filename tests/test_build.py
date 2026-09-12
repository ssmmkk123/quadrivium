"""C-only builds include their sources and reject cached native binaries."""

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("setuptools"), "setuptools is a build dependency")
class TestNativeBuild(unittest.TestCase):
    def setUp(self):
        self.project = Path(__file__).resolve().parents[1]
        setup_file = self.project / "setup.py"
        if not setup_file.is_file():
            self.skipTest("setup.py is unavailable in a wheel-only installation")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        spec = importlib.util.spec_from_file_location("quadrivium_test_setup", setup_file)
        self.module = importlib.util.module_from_spec(spec)
        with patch("setuptools.setup") as setup, patch("subprocess.run") as subprocess:
            spec.loader.exec_module(self.module)
        subprocess.assert_not_called()
        self.setup_options = setup.call_args.kwargs
        self.module.HERE = self.root
        self.distribution = self.module.ExtensionAwareDistribution({"name": "build-test"})
        self.command = self.module.build_py(self.distribution)
        self.command.build_lib = str(self.root / "build")
        self.command.editable_mode = False

    def make_file(self, relative, content=b"fixture"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_one_required_interpreter_specific_c_extension(self):
        self.assertTrue(self.distribution.has_ext_modules())
        extensions = self.setup_options["ext_modules"]
        self.assertEqual(len(extensions), 1)
        extension = extensions[0]
        self.assertEqual(extension.name, "quadrivium._qnp")
        self.assertFalse(extension.optional)
        self.assertFalse(extension.py_limited_api)

    def test_extension_lists_every_c_source_and_header(self):
        sources = sorted(str(p.relative_to(self.project))
                         for p in (self.project / "csrc").glob("*.c"))
        headers = sorted(str(p.relative_to(self.project))
                         for p in (self.project / "csrc").glob("*.h"))
        self.assertTrue(sources, "the array core has no C sources")
        self.assertTrue(headers, "the array core has no C headers")
        with patch.object(self.module, "HERE", self.project), patch.object(
                self.module, "CSRC", self.project / "csrc"):
            extension = self.module.core_extension()
        self.assertEqual(extension.sources, sources)
        self.assertEqual(extension.depends, headers)
        self.assertIn(str(Path("csrc") / "accel_ode.c"), extension.sources)
        self.assertIn(str(Path("csrc") / "qaccel.h"), extension.depends)

    def test_missing_required_sources_fails_the_build(self):
        with patch.object(self.module, "CSRC", self.root / "missing"):
            with self.assertRaisesRegex(SystemExit, "cannot be built"):
                self.module.core_extension()

    def test_cached_binaries_are_removed_from_build_directory_only(self):
        staged, source = [], []
        for suffix in (".so", ".pyd", ".dylib", ".dll"):
            staged.append(self.make_file("build/quadrivium/_obsolete" + suffix))
            source.append(self.make_file("quadrivium/_obsolete" + suffix))
        staged.append(self.make_file("build/quadrivium/nested/_obsolete.so"))
        python_file = self.make_file("build/quadrivium/__init__.py", b"# Python")
        with patch.object(self.module._build_py, "run") as parent:
            self.command.run()
        parent.assert_called_once_with()
        self.assertTrue(all(not path.exists() for path in staged))
        self.assertTrue(all(path.read_bytes() == b"fixture" for path in source))
        self.assertEqual(python_file.read_bytes(), b"# Python")

    def test_build_directory_pointing_at_checkout_preserves_source(self):
        source = self.make_file("quadrivium/_qnp.so")
        self.command.build_lib = str(self.root)
        with patch.object(self.module._build_py, "run"):
            self.command.run()
        self.assertEqual(source.read_bytes(), b"fixture")

    def test_staged_package_symlink_cannot_delete_checkout_binaries(self):
        source = self.make_file("quadrivium/_qnp.so")
        staged = self.root / "build" / "quadrivium"
        staged.parent.mkdir()
        try:
            staged.symlink_to(self.root / "quadrivium", target_is_directory=True)
        except (NotImplementedError, OSError):
            self.skipTest("directory symlinks are unavailable")
        with patch.object(self.module._build_py, "run"):
            self.command.run()
        self.assertEqual(source.read_bytes(), b"fixture")

    def test_package_data_never_contains_cached_extensions(self):
        files = ["quadrivium/_obsolete.so", "_qnp.pyd", "nested/lib.dylib", "lib.dll", "data.txt"]
        with patch.object(self.module._build_py, "find_data_files", return_value=files):
            self.assertEqual(self.command.find_data_files("quadrivium", "."), ["data.txt"])

    def test_sdist_manifest_ships_c_and_prunes_cached_obsolete_files(self):
        from setuptools.command.egg_info import FileList

        manifest = (self.project / "MANIFEST.in").read_text()
        fixtures = ["setup.py", "csrc/core.c", "csrc/nested/kernel.c", "csrc/qnp.h",
                    "csrc/nested/kernel.h", "tests/test_sample.py", "rust/Cargo.toml",
                    "rust/Cargo.lock", "rust/src/lib.rs", "quadrivium/_obsolete.so",
                    "quadrivium/_qnp.pyd", "quadrivium/lib.dylib", "quadrivium/lib.dll"]
        for path in fixtures:
            self.make_file(path)
        files = FileList()
        # A previous SOURCES.txt may list files that new include directives
        # no longer mention; the manifest must explicitly prune those entries.
        files.files = fixtures.copy()
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            with patch("setuptools.command.egg_info.log.warn"):
                for line in manifest.splitlines():
                    line = line.split("#", 1)[0].strip()
                    if line:
                        files.process_template_line(line)
        finally:
            os.chdir(previous)
        self.assertTrue(set(fixtures[:6]).issubset(files.files))
        self.assertFalse(any(Path(path).parts[0] == "rust" for path in files.files))
        self.assertFalse(any(Path(path).suffix in self.module.NATIVE_SUFFIXES
                             for path in files.files))


if __name__ == "__main__":
    unittest.main()
