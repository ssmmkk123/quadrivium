# Contributing to Quadrivium

Thank you for helping improve quadrivium. Contributions may be bug fixes,
documentation improvements, new tests, performance work, or carefully scoped
numerical methods.

## Before you start

- Search the existing issues and pull requests before opening a duplicate.
- Open an issue before making a large API change or adding a substantial new
  algorithm so its scope and validation strategy can be discussed first.
- Keep each pull request focused on one coherent change.

## Development setup

Fork and clone the repository, then create an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`.

Run the complete test suite before submitting a pull request:

```bash
python -m unittest discover -s tests
```

That is 370 tests -- 341 covering the numerics, the rest checking that the
documentation still matches the code -- and takes about half a minute.

You can run a single module while developing:

```bash
python -m unittest tests.test_calculus
```

## Numerical changes

Numerical algorithms need evidence beyond a single expected value. Where
applicable, add tests for properties such as convergence order, exactness,
conservation, stability, structural identities, or agreement with an
independent analytic solution. Include difficult boundary cases and document
the method's known limitations.

Avoid delegating the core algorithm to a high-level implementation in another
library. NumPy may be used for array operations and low-level primitives, but
the method itself should remain visible in the source.

## Code and documentation

- Match the surrounding style and use four spaces for Python indentation.
- Add or update docstrings for public APIs.
- Keep public behavior backward compatible unless the change has been agreed
  upon in an issue.
- Update the README, the guides in `docs/`, or the examples when user-facing
  behavior changes.
- Add a regression test for every bug fix when practical.

The documentation is checked by the test suite, so two commands matter after
any change to the public API or to `docs/`:

```bash
python tools/gen_docs.py       # regenerate docs/api/*.md and docs/changelog.md
mkdocs build --strict          # no broken links, anchors, or missing pages
```

Examples in the documentation are written as doctests and executed by
`tests/test_docs.py`. Because doctest compares printed output exactly, convert
NumPy scalars with `float()`, `int()`, or `bool()` before displaying them, and
round to fewer digits than the method actually delivers. Install the tools with
`pip install -e ".[dev]"`.

## Pull requests

In the pull request description, explain the problem, the approach, and how
you validated the result. CI must pass on every supported Python version.
By contributing, you agree that your work will be licensed under the project's
MIT License.
