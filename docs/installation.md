# Installation

Install Quadrivium into the Python environment that will run your calculations.
The package has no runtime dependencies on NumPy, SciPy, BLAS, or LAPACK. Its
required C extension contains the array engine and native numerical kernels.

## Requirements

| Component | Requirement |
| --- | --- |
| Python | Metadata requires Python 3.9 or newer; CI exercises 3.9–3.14 |
| Binary installation | A wheel matching your interpreter and platform |
| Source installation | A C compiler and the development headers for your Python |
| Test tools | The `test` extra adds pytest and NumPy |
| Documentation build | The `docs` extra adds MkDocs and Material |
| Graph regeneration | The `figures` extra adds Matplotlib |

A source installation compiles the C extension. Disabling numerical
acceleration does not remove that requirement. No Rust toolchain is used by
this checkout.

## Install a published release

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install quadrivium
```

On Windows PowerShell, replace the activation command with:

```powershell
.venv\Scripts\Activate.ps1
```

Using `python -m pip` helps ensure that installation and execution use the same
interpreter. A notebook kernel also needs to use that environment.

These documentation pages describe Quadrivium 1.2.0. Review the migration notes
in the [changelog](changelog.md) when upgrading from 1.1.0. Future changes listed
under **Unreleased** require installation from the corresponding source checkout.

## From source

```bash
git clone https://github.com/ssmmkk123/quadrivium.git
cd quadrivium
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Editable installation makes Python source edits visible immediately. Changes
to `csrc/` require recompilation; rerun the editable installation after editing
C code. A regular source installation uses `python -m pip install .`.

An import from an unbuilt checkout can fail with an error mentioning `_qnp`.
Build the extension with an installation command; simply adding the source
directory to `PYTHONPATH` is insufficient.

## Optional extras

| Extra | Dependencies | Use |
| --- | --- | --- |
| `test` | pytest, NumPy | Run the tests and compare against an independent array implementation |
| `docs` | MkDocs, MkDocs Material | Build and serve the site using checked-in SVGs |
| `figures` | Matplotlib | Compute and render the documentation experiments |
| `dev` | All three extras, build, twine | Development, documentation, and packaging checks |

Install only what your task needs:

```bash
python -m pip install -e ".[test]"
python -m pip install -e ".[docs,figures]"
```

Matplotlib depends on its own plotting stack. Those dependencies belong to
figure generation and are not required to run Quadrivium.

## Verify the installation

First confirm the interpreter and imported package path:

```bash
python -c "import sys, quadrivium as qd; print(sys.executable); print(qd.__file__); print(qd.__version__); print(qd.accel.show_config())"
```

Then check a calculation with a known answer:

```pycon
>>> import math
>>> import quadrivium as qd
>>> from quadrivium import numeric as np
>>> integral = qd.quad(lambda x: math.exp(-x*x), -np.inf, np.inf)
>>> integral.converged and abs(float(integral) - math.sqrt(math.pi)) < 1e-10
True
>>> answer = qd.solve([[3.0, 1.0], [1.0, 2.0]], [9.0, 8.0])
>>> np.allclose(answer, [2.0, 3.0])
True

```

For a development checkout, run `python -m pytest -q`. Test counts and timings
change as the project grows; the current test run is the useful reference.

## Troubleshoot an installation

| Symptom | Likely explanation | Next step |
| --- | --- | --- |
| `_qnp` is missing | The checkout's extension was not built, or belongs to another interpreter | Reinstall using the interpreter that imports the package |
| Compiler or header error during pip installation | Pip is building from source | Install a compatible C toolchain and Python development headers, then retry |
| An example's name cannot be imported | Installed release and documentation differ | Compare `qd.__version__`, `qd.__file__`, and the changelog |
| Pip says installed but Python cannot import it | Pip and Python refer to different environments | Use `python -m pip` and check `sys.executable` |
| A local checkout shadows an installed wheel | The current directory takes precedence on the import path | Verify the wheel from a directory outside the repository |
| Graph generation cannot import Matplotlib | Only runtime or docs dependencies are installed | Install the `figures` extra |

Include the complete build error and interpreter details in an installation
report. The final `ImportError` alone may hide the earlier compiler failure.

## Reproduce an environment

Record the package version and, for unreleased work, the source commit. The
version in this release is **1.2.0**. A modified working tree is not equivalent
to the published artifact.

For published-version experiments, pin the release explicitly:

```text
quadrivium==1.2.0
```

Also record Python, platform, numeric dtype, backend, solver options, and random
seeds. Small floating-point differences can arise from hardware and evaluation
order. A fixed seed supports reproduction in a fixed environment; it is not a
promise that arbitrary code changes preserve all random draws.

## Upgrade or remove

```bash
python -m pip install --upgrade quadrivium
python -m pip uninstall quadrivium
```

Review the [changelog](changelog.md) and rerun your validation cases after an
upgrade. Continue with [getting started](getting-started.md) for array, result,
and tolerance conventions.
