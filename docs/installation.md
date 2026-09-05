# Installation

## Requirements

| | |
| --- | --- |
| Python | 3.9 or newer (tested on 3.9 through 3.14) |
| Dependencies | [NumPy](https://numpy.org) 1.20 or newer — nothing else |
| Platform | any, including Windows, macOS, Linux, and PyPy where NumPy builds |
| Compiler | none: the distribution is a pure-Python `py3-none-any` wheel |

Nothing in the library is compiled, so there are no per-platform wheels, no
build step, and no version of the package that can be subtly mismatched to
your interpreter.

## From PyPI

```bash
pip install quadrivium
```

Into an isolated environment, which is the recommended way to install anything:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install quadrivium
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv add quadrivium
```

Inside a conda environment, install with pip — there is no conda-forge package:

```bash
conda create -n numerics python=3.12 numpy
conda activate numerics
pip install quadrivium
```

## Optional extras

The library itself needs only NumPy. The extras exist for working *on* it:

| Extra | Installs | For |
| --- | --- | --- |
| `test` | pytest | running the test suite |
| `docs` | mkdocs, mkdocs-material | building this documentation site |
| `dev` | both of the above, plus build and twine | full development and release work |

```bash
pip install "quadrivium[test]"
pip install "quadrivium[dev]"
```

## From source

```bash
git clone https://github.com/ssmmkk123/quadrivium.git
cd quadrivium
python -m pip install .
```

For development, install in editable mode so your edits take effect
immediately, then run the suite:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

The suite is 376 tests and takes about half a minute. `pytest` works too, and
is what the `test` extra installs:

```bash
python -m pytest
```

See [Contributing](contributing.md) for the full development workflow.

## Verifying the installation

```bash
python -c "import quadrivium; print(quadrivium.__version__)"
```

A quick end-to-end check that the numerics work, not just the import:

```pycon
>>> import numpy as np
>>> import quadrivium as qd
>>> abs(float(qd.quad(lambda x: np.exp(-x*x), -np.inf, np.inf)) - float(np.sqrt(np.pi))) < 1e-12
True

```

## Upgrading and removing

```bash
pip install --upgrade quadrivium
pip uninstall quadrivium
```

Version numbers follow [semantic versioning](https://semver.org): a patch
release fixes behaviour without changing the API, a minor release adds
methods or optional arguments, and a major release is required for anything
that could break existing calls. The [changelog](changelog.md) records what
changed in each.

## Reproducible environments

Pin the exact version in a requirements file or lock file when results must be
reproducible:

```text
quadrivium==1.1.0
numpy==2.4.3
```

Numerical output can differ in the last few digits between NumPy versions and
BLAS builds, because the library's dense linear algebra rests on NumPy's own
matrix products. Anything sensitive to the last digit should pin both.
