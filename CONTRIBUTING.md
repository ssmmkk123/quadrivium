# Contributing to Quadrivium

Contributions can improve numerical correctness, runtime behavior, examples,
or the clarity of the documentation. Start with a focused problem and a
reproducible result. Discuss substantial new methods or public API changes in
an issue before implementing them.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Source
installation requires a C compiler and Python development headers. Editable
Python changes take effect immediately; reinstall after changes in `csrc/`.

## Numerical work

Use `quadrivium.numeric` for library array operations. NumPy is an optional
independent test reference, not a runtime dependency. Keep the implemented
method visible in the repository's Python or C source.

For a change to numerical behavior, establish an independent expectation:
an analytic result, a reconstruction or conservation identity, a convergence
order, or a separate implementation. Test meaningful failure cases, including
shape, scale, conditioning, and storage ownership where relevant. Explain the
return type, method-specific stopping rule, and limits in the public docstring.

Run focused tests during development and the complete checks before submitting
code changes:

```bash
python -m pytest -q
QUADRIVIUM_NO_ACCEL=1 python -m pytest -q
```

The reference backend still uses the required C array engine. See the
[detailed contributor guide](docs/contributing.md) for the Windows environment
syntax and examples of useful numerical regressions.

## Documentation and graphs

Guides should explain when to use a method, how to form its inputs, how to read
its result, and how to validate it. Write complete, small `pycon` examples with
stable displayed outputs. Doctests execute these examples; ordinary Python and
shell fences are not automatically executed.

The API reference is generated from source signatures, docstrings, and class
interfaces. The site changelog is generated from [CHANGELOG.md](CHANGELOG.md).
Edit those sources before regenerating:

```bash
python tools/gen_docs.py
python -m pytest -q tests/test_docs.py
python -m mkdocs build --strict
```

Graph experiments live in `tools/figures/experiments.py`. They use actual
Quadrivium calculations, explicit references, and recorded parameters. Generate
both theme variants and inspect the result:

```bash
python tools/gen_figures.py --png /tmp/quadrivium-figure-preview
python tools/gen_figures.py --check
```

A full generation removes obsolete SVGs and updates the provenance manifest.
Read [figure methodology](docs/figures.md) before changing a graph. Keep local
benchmark and audit output in the ignored `benchmarks/` directory.

## Submit a pull request

Describe the concrete problem, resulting behavior, validation performed, and
remaining limitations. Keep the change focused and synchronize generated
artifacts. CI validates supported interpreters, backend configurations,
packaging, and documentation.

Follow the [Code of Conduct](CODE_OF_CONDUCT.md). Report vulnerabilities through
[Security](SECURITY.md); use [Support](SUPPORT.md) for help with a calculation.
Contributions are distributed under the project's [MIT license](LICENSE).
