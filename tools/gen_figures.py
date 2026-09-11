#!/usr/bin/env python3
"""Regenerate measured documentation experiments and their provenance manifest.

    .venv/bin/python tools/gen_figures.py
    .venv/bin/python tools/gen_figures.py --check
    .venv/bin/python tools/gen_figures.py --only 'ode|pde' --png /tmp/figures

Full generation removes SVG assets absent from the catalogue after every
selected experiment has rendered successfully. --only is a preview workflow:
it updates matching figures but neither prunes assets nor updates the full
manifest. Run a full generation before committing changes. --check compares
SVG bytes, provenance, and (without --only) stale/unregistered assets.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import quadrivium  # noqa: E402
import figstyle  # noqa: E402
from figures import load_all  # noqa: E402

FIGURE_DIR = ROOT / "docs" / "assets" / "figures"
MANIFEST = FIGURE_DIR / "manifest.json"


def render(spec, scheme, *, png_path=None) -> str:
    """Render one scheme with fixed SVG identifiers and no creation timestamp."""
    with matplotlib.rc_context(figstyle.rc_params(scheme)):
        fig = plt.figure(figsize=spec.size)
        try:
            spec.draw(fig, scheme)
            buffer = io.StringIO()
            fig.savefig(buffer, format="svg", metadata={
                "Date": None, "Creator": "Quadrivium tools/gen_figures.py",
                "Title": spec.name, "Description": spec.summary,
                "Format": None, "Type": None,
            })
            if png_path is not None:
                png_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(png_path, dpi=120, transparent=False,
                            facecolor=scheme.surface)
        finally:
            plt.close(fig)
    return buffer.getvalue()


def paths_for(name):
    return {"light": FIGURE_DIR / f"{name}.svg",
            "dark": FIGURE_DIR / f"{name}-dark.svg"}


def provenance(figures):
    """Stable, reviewable experimental settings; source hashes detect drift."""
    source_paths = [Path(__file__), ROOT / "tools/figstyle.py"]
    source_paths.extend(sorted((ROOT / "tools/figures").glob("*.py")))
    data = {
        "schema_version": 1,
        "generator": "tools/gen_figures.py",
        "command": ".venv/bin/python tools/gen_figures.py",
        "verification": ".venv/bin/python tools/gen_figures.py --check",
        "environment": {"python": platform.python_version(),
                        "quadrivium": quadrivium.__version__,
                        "matplotlib": matplotlib.__version__, "numpy": np.__version__},
        "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source_paths},
        "notes": [
            "Numerical algorithms are executed by Quadrivium; NumPy supplies plot coordinates and diagnostic reductions.",
            "Work axes count callbacks, not wall-clock performance.",
            "Random experiments use the fixed seeds recorded below.",
            "Display floors apply only to logarithmic plots; no error metric is silently modified.",
            "SVG timestamps are omitted and svg.hashsalt is fixed. Exact byte identity requires the recorded rendering environment.",
        ],
        "figures": [{"name": s.name, "page": s.page, "summary": s.summary,
                     "size_inches": list(s.size), "parameters": s.parameters,
                     "files": [p.name for p in paths_for(s.name).values()]}
                    for s in figures],
    }
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check SVG bytes and catalogue provenance")
    parser.add_argument("--only", metavar="REGEX", help="preview/check matching figures without pruning")
    parser.add_argument("--png", metavar="DIR", help="write PNG inspection copies")
    args = parser.parse_args()
    catalogue = load_all()
    selected = [s for s in catalogue if not args.only or re.search(args.only, s.name)]
    if not selected:
        parser.error(f"no figure matches {args.only!r}")
    expected = {p for s in catalogue for p in paths_for(s.name).values()}
    obsolete = set(FIGURE_DIR.glob("*.svg")) - expected
    pending, problems = {}, []
    started = time.perf_counter()
    for spec in selected:
        for scheme in figstyle.SCHEMES:
            path = paths_for(spec.name)[scheme.name]
            png = Path(args.png) / (path.stem + ".png") if args.png else None
            svg = render(spec, scheme, png_path=png)
            if args.check:
                if not path.exists():
                    problems.append(f"Missing: {path.relative_to(ROOT)}")
                elif path.read_text(encoding="utf-8") != svg:
                    problems.append(f"Stale: {path.relative_to(ROOT)}")
            else:
                pending[path] = svg
        if not args.check:
            print(f"  {spec.name:38s} {spec.page}", flush=True)
    manifest = provenance(catalogue)
    if args.check:
        if not args.only:
            problems.extend(f"Unregistered SVG: {p.relative_to(ROOT)}" for p in sorted(obsolete))
            if not MANIFEST.exists() or MANIFEST.read_text(encoding="utf-8") != manifest:
                problems.append("Missing or stale: docs/assets/figures/manifest.json")
        if problems:
            print("\n".join(problems), file=sys.stderr)
            print("Run: .venv/bin/python tools/gen_figures.py", file=sys.stderr)
            return 1
        print(f"{len(selected)} experiments ({2*len(selected)} SVGs) are current; "
              f"checked in {time.perf_counter()-started:.1f}s.")
        return 0
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for path, svg in pending.items():
        path.write_text(svg, encoding="utf-8")
    if not args.only:
        for path in obsolete:
            path.unlink()
        MANIFEST.write_text(manifest, encoding="utf-8")
    print(f"Wrote {len(pending)} SVGs in {time.perf_counter()-started:.1f}s"
          + (f"; removed {len(obsolete)} obsolete SVGs; updated manifest." if not args.only else "."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
