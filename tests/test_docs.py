"""Run the doctests in the package docstrings and in the documentation.

Documentation that is never executed drifts silently out of date, so every
example on the site is a doctest and every one of them runs here.  The two
generated parts of the documentation -- the API reference and the copy of the
changelog -- are regenerated and compared, so they cannot describe an API the
package no longer has.
"""

import doctest
import re
import subprocess
import sys
import unittest
from pathlib import Path

import quadrivium

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

# Doctest compares printed output literally, so the examples round their
# results and convert NumPy scalars; these two flags absorb what is left.
OPTIONFLAGS = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp"}

# The generated reference pages are tables of signatures with no examples in
# them; running doctest over 836 entries would only cost time.
GENERATED_DIRS = {"api"}


def documentation_files():
    """Every hand-written documentation page, plus the README."""
    pages = [p for p in sorted(DOCS.rglob("*.md"))
             if not GENERATED_DIRS.intersection(p.relative_to(DOCS).parts)
             and p.name != "changelog.md"]
    return [ROOT / "README.md"] + pages


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(quadrivium))
    for path in documentation_files():
        tests.addTests(doctest.DocFileSuite(
            str(path), module_relative=False, optionflags=OPTIONFLAGS))
    return tests


class TestDocumentationIsCurrent(unittest.TestCase):
    def test_generated_pages_match_the_package(self):
        """``tools/gen_docs.py`` output is checked in and must be current."""
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "gen_docs.py"), "--check"],
            capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(result.returncode, 0,
                         msg=result.stdout + result.stderr)

    def test_every_page_in_the_navigation_exists(self):
        """A nav entry pointing at a missing file breaks the built site."""
        config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        nav = config.split("nav:", 1)[1].split("\nextra:", 1)[0]
        referenced = re.findall(r"([\w./-]+\.md)", nav)
        self.assertGreater(len(referenced), 30)
        for name in referenced:
            with self.subTest(page=name):
                self.assertTrue((DOCS / name).is_file(), f"missing docs/{name}")

    def test_every_page_is_in_the_navigation(self):
        """And a page missing from the nav is a page nobody can reach."""
        config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        referenced = set(re.findall(r"([\w./-]+\.md)", config))
        for page in DOCS.rglob("*.md"):
            with self.subTest(page=page.name):
                self.assertIn(page.relative_to(DOCS).as_posix(), referenced)

    def test_internal_links_resolve(self):
        """Relative links between pages must point at files that exist."""
        link = re.compile(r"\]\((?!https?://|mailto:|#)([^)\s]+)\)")
        for page in list(DOCS.rglob("*.md")) + [ROOT / "README.md"]:
            text = page.read_text(encoding="utf-8")
            for target in link.findall(text):
                path = target.split("#", 1)[0]
                if not path:
                    continue
                with self.subTest(page=page.name, link=target):
                    self.assertTrue((page.parent / path).resolve().exists(),
                                    f"{page.name} links to missing {target}")

    def test_image_links_have_alt_text_and_resolve(self):
        """Image links should be accessible and point at existing files."""
        image = re.compile(r"!\[([^\]]*)\]\((?!https?://|mailto:|#)([^)\s]+)\)")
        for page in list(DOCS.rglob("*.md")) + [ROOT / "README.md"]:
            text = page.read_text(encoding="utf-8")
            for alt, target in image.findall(text):
                with self.subTest(page=page.name, image=target):
                    self.assertTrue(alt.strip(), f"{page.name} has empty alt text: {target}")
                    path = target.split("#", 1)[0]
                    self.assertTrue((page.parent / path).resolve().exists(),
                                    f"{page.name} image target missing: {target}")

    def test_docs_assets_are_not_orphaned(self):
        """Every image asset under docs/assets should be referenced by docs."""
        assets_root = DOCS / "assets"
        if not assets_root.exists():
            return
        assets = {
            p.resolve()
            for p in assets_root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        }
        image = re.compile(r"!\[[^\]]*\]\((?!https?://|mailto:|#)([^)\s]+)\)")
        referenced = set()
        for page in list(DOCS.rglob("*.md")) + [ROOT / "README.md"]:
            text = page.read_text(encoding="utf-8")
            for target in image.findall(text):
                path = target.split("#", 1)[0]
                resolved = (page.parent / path).resolve()
                if resolved.suffix.lower() in IMAGE_EXTENSIONS:
                    referenced.add(resolved)
        for asset in sorted(assets):
            with self.subTest(asset=asset.name):
                self.assertIn(asset, referenced, f"orphaned docs asset: {asset}")

    def test_the_documented_version_is_the_packaged_one(self):
        """Version numbers quoted in the docs must be the current release."""
        for page in (ROOT / "CHANGELOG.md", DOCS / "installation.md"):
            with self.subTest(page=page.name):
                self.assertIn(quadrivium.__version__,
                              page.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
