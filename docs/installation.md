# Installation

## Requirements

| | |
| --- | --- |
| Python | 3.9 or newer (tested on 3.9 through 3.14) |
| Dependencies | none at runtime |
| Platform | Windows, macOS, Linux |
| Compiler | a C compiler, when building from source rather than from a wheel |

The package computes on its own array type, `quadrivium.numeric`, compiled from
the C sources in `csrc/`. That extension is not optional — it is the array
core — so wheels are per-platform and per-interpreter, and a source install
needs a C compiler. Nothing else is required: there is no NumPy, no BLAS, and
no LAPACK underneath.

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
conda create -n numerics python=3.12
conda activate numerics
pip install quadrivium
```

## Optional extras

The library itself needs nothing at runtime. The extras exist for working *on*
it -- `test` pulls in NumPy, which the test suite uses to check the compiled
array core against an independent implementation:

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
>>> from quadrivium import numeric as np
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
```

Numerical output can still differ in the last few digits between platforms,
because the array core's matrix kernels use fused multiply-add where the CPU
provides it. Everything else is fixed by the package itself: sums are pairwise
with a fixed block size, and a seeded generator produces the same stream
everywhere, so pinning the version pins the numbers on a given machine.
