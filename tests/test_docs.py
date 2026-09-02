"""Run the doctests in the package docstrings and in the documentation.

Documentation that is never executed drifts silently out of date, so every
example on the site is a doctest and every one of them runs here.  The two
generated parts of the documentation -- the API reference and the copy of the
changelog -- are regenerated and compared, so they cannot describe an API the
package no longer has.

The figures get the same treatment one level down: they are generated from the
library by ``tools/gen_figures.py``, and the checks here tie every file on disk
to the page that shows it and every reference on a page to a file that exists.
Regenerating them needs Matplotlib, so the tests that compare the catalogue
with the directory are skipped when it is not installed; the rest run
everywhere.
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
FIGURES = DOCS / "assets" / "figures"

# ``assets/figures/<name>.svg`` or ``<name>-dark.svg``, wherever it appears.
FIGURE_REFERENCE = re.compile(r"assets/figures/([\w-]+)\.svg")

# A fenced code block showing how to embed a figure is an example, not a
# reference to a file that has to exist.
FENCED_BLOCK = re.compile(r"^```.*?^```", re.M | re.S)

# Doctest compares printed output literally, so the examples round their
# results and convert NumPy scalars; these two flags absorb what is left.
OPTIONFLAGS = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS

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


def markdown_pages():
    """Every page of the site, generated ones included, plus the README."""
    return [ROOT / "README.md"] + sorted(DOCS.rglob("*.md"))


def page_prose(page):
    """A page's text with its code blocks removed."""
    return FENCED_BLOCK.sub("", page.read_text(encoding="utf-8"))


def referenced_figures():
    """Figure stems referenced by the documentation, with the pages using them."""
    used = {}
    for page in markdown_pages():
        for stem in FIGURE_REFERENCE.findall(page_prose(page)):
            used.setdefault(stem, set()).add(page)
    return used


def figure_catalogue():
    """The registered figures, or None when Matplotlib is not installed."""
    try:
        sys.path.insert(0, str(ROOT / "tools"))
        from figures import load_all  # noqa: PLC0415
    except ImportError:  # pragma: no cover - exercised only without Matplotlib
        return None
    return load_all()


class TestFigures(unittest.TestCase):
    """The figures, the pages that show them, and the code that draws them."""

    def test_every_referenced_figure_exists(self):
        """A page must not point at a picture that was never generated."""
        for stem, pages in referenced_figures().items():
            with self.subTest(figure=stem):
                self.assertTrue((FIGURES / f"{stem}.svg").is_file(),
                                f"missing figure, shown on "
                                f"{sorted(p.name for p in pages)}")

    def test_every_figure_has_a_light_and_a_dark_variant(self):
        """An image cannot see the theme, so each one is drawn twice."""
        for path in sorted(FIGURES.glob("*.svg")):
            if path.stem.endswith("-dark"):
                partner = FIGURES / f"{path.stem[:-len('-dark')]}.svg"
            else:
                partner = FIGURES / f"{path.stem}-dark.svg"
            with self.subTest(figure=path.name):
                self.assertTrue(partner.is_file(),
                                f"{path.name} has no {partner.name}")

    def test_every_page_shows_both_variants_of_what_it_shows(self):
        """Referencing only one variant leaves a blank in the other theme."""
        for page in markdown_pages():
            text = page_prose(page)
            stems = {s for s in FIGURE_REFERENCE.findall(text)
                     if not s.endswith("-dark")}
            for stem in stems:
                with self.subTest(page=page.name, figure=stem):
                    self.assertIn(f"{stem}-dark.svg#only-dark", text)
                    self.assertIn(f"{stem}.svg#only-light", text)

    def test_no_figure_is_left_unused(self):
        """A figure nobody shows is a figure nobody maintains."""
        used = set(referenced_figures())
        for path in sorted(FIGURES.glob("*.svg")):
            stem = path.stem[:-len("-dark")] if path.stem.endswith("-dark") \
                else path.stem
            with self.subTest(figure=path.name):
                self.assertIn(stem, used, f"{path.name} is not shown anywhere")

    def test_the_catalogue_matches_the_directory(self):
        """Every registered figure is on disk, and nothing else is."""
        catalogue = figure_catalogue()
        if catalogue is None:
            self.skipTest("Matplotlib is not installed")
        registered = {figure.name for figure in catalogue}
        on_disk = {path.stem for path in FIGURES.glob("*.svg")
                   if not path.stem.endswith("-dark")}
        self.assertEqual(registered, on_disk)

    def test_each_figure_is_shown_on_the_page_it_was_registered_for(self):
        """The page recorded with a figure is the page that has to embed it."""
        catalogue = figure_catalogue()
        if catalogue is None:
            self.skipTest("Matplotlib is not installed")
        for figure in catalogue:
            with self.subTest(figure=figure.name):
                page = DOCS / figure.page
                self.assertTrue(page.is_file(), f"no such page: {figure.page}")
                self.assertIn(f"{figure.name}.svg", page_prose(page))


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
            # Code blocks are read past: a link inside one is a sample of what
            # to write, not a link to somewhere.
            text = page_prose(page)
            for target in link.findall(text):
                path = target.split("#", 1)[0]
                if not path:
                    continue
                with self.subTest(page=page.name, link=target):
                    self.assertTrue((page.parent / path).resolve().exists(),
                                    f"{page.name} links to missing {target}")

    def test_the_documented_version_is_the_packaged_one(self):
        """Version numbers quoted in the docs must be the current release."""
        for page in (ROOT / "CHANGELOG.md", DOCS / "installation.md"):
            with self.subTest(page=page.name):
                self.assertIn(quadrivium.__version__,
                              page.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
