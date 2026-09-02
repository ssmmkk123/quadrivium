#!/usr/bin/env python3
"""Render the documentation figures from the library.

Every figure on the site is computed by quadrivium itself -- the residual
histories are real residual histories, the convergence orders are measured,
the shocks were captured by the solver being described -- and drawn by a
function in ``tools/figures/``.  Nothing here is a hand-drawn illustration of
what the code is supposed to do.

Each figure is written twice, ``<name>.svg`` and ``<name>-dark.svg``, because
an image cannot see the theme the reader picked; MkDocs Material shows one and
hides the other on the ``#only-light`` / ``#only-dark`` fragment.  The files
are checked in, so building the site needs neither Matplotlib nor the time it
takes to solve every problem on the site.

Usage::

    python tools/gen_figures.py                 # rewrite every figure
    python tools/gen_figures.py --only ode-     # just the ones whose name matches
    python tools/gen_figures.py --check         # exit 1 if any figure is stale
    python tools/gen_figures.py --png out/      # PNG copies, for a quick look

Regenerating needs Matplotlib::

    python -m pip install -e ".[figures]"
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import figstyle  # noqa: E402
from figures import load_all  # noqa: E402

FIGURE_DIR = ROOT / "docs" / "assets" / "figures"
DOCS_DIR = ROOT / "docs"

# Matplotlib stamps a creation date and hashes clip-path ids from a salt; both
# would make two runs of the same code differ.  Pinning them keeps the SVGs
# byte-identical, which is what lets --check mean anything.
matplotlib.rcParams["svg.hashsalt"] = "quadrivium"
SVG_METADATA = {"Date": None, "Creator": "tools/gen_figures.py", "Format": None,
                "Type": None}


def render(fig_spec, scheme) -> str:
    """Draw one figure in one colour scheme and return the SVG source."""
    with matplotlib.rc_context(figstyle.rc_params(scheme)):
        fig = plt.figure(figsize=fig_spec.size)
        try:
            fig_spec.draw(fig, scheme)
            buffer = io.StringIO()
            fig.savefig(buffer, format="svg", metadata=SVG_METADATA)
        finally:
            plt.close(fig)
    return buffer.getvalue()


def paths_for(name: str) -> dict:
    return {figstyle.LIGHT.name: FIGURE_DIR / f"{name}.svg",
            figstyle.DARK.name: FIGURE_DIR / f"{name}-dark.svg"}


def selected(pattern: str | None):
    figures = load_all()
    if not pattern:
        return figures
    return [f for f in figures if re.search(pattern, f.name)]


def write_png(fig_spec, out_dir: Path) -> None:
    """A PNG copy of each scheme, for looking at without a browser."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for scheme in figstyle.SCHEMES:
        with matplotlib.rc_context(figstyle.rc_params(scheme)):
            fig = plt.figure(figsize=fig_spec.size)
            try:
                fig_spec.draw(fig, scheme)
                suffix = "" if scheme.name == "light" else "-dark"
                fig.savefig(out_dir / f"{fig_spec.name}{suffix}.png", dpi=110,
                            facecolor=scheme.surface)
            finally:
                plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify the checked-in figures match the code")
    parser.add_argument("--only", metavar="REGEX",
                        help="render only figures whose name matches")
    parser.add_argument("--png", metavar="DIR",
                        help="also write PNG copies to DIR, for inspection")
    args = parser.parse_args()

    figures = selected(args.only)
    if not figures:
        print(f"No figure matches {args.only!r}", file=sys.stderr)
        return 1

    if args.check:
        stale, missing = [], []
        for spec in figures:
            for scheme in figstyle.SCHEMES:
                path = paths_for(spec.name)[scheme.name]
                if not path.exists():
                    missing.append(path.relative_to(ROOT))
                elif path.read_text(encoding="utf-8") != render(spec, scheme):
                    stale.append(path.relative_to(ROOT))
        if missing or stale:
            for label, group in (("Missing", missing), ("Stale", stale)):
                if group:
                    print(f"{label} figures: "
                          + ", ".join(str(p) for p in sorted(group)),
                          file=sys.stderr)
            print("Run: python tools/gen_figures.py", file=sys.stderr)
            return 1
        print(f"{len(figures)} figures ({2 * len(figures)} files) are up to date.")
        return 0

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for spec in figures:
        for scheme in figstyle.SCHEMES:
            paths_for(spec.name)[scheme.name].write_text(render(spec, scheme),
                                                         encoding="utf-8")
        if args.png:
            write_png(spec, Path(args.png))
        print(f"  {spec.name:<44s} {spec.page}")
    elapsed = time.perf_counter() - started
    print(f"Wrote {2 * len(figures)} files to "
          f"{FIGURE_DIR.relative_to(ROOT)}/ in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
