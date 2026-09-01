# Contributing

The full contributor guide lives in
[CONTRIBUTING.md](https://github.com/ssmmkk123/quadrivium/blob/main/CONTRIBUTING.md)
in the repository. This page is the short version, plus the parts specific to
the documentation.

## Development setup

```bash
git clone https://github.com/ssmmkk123/quadrivium.git
cd quadrivium
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

The `dev` extra installs the test, documentation, and packaging tools
together.

## Running the tests

```bash
python -m unittest discover -s tests      # the whole suite
python -m unittest tests.test_calculus    # one module
python -m pytest                          # same suite under pytest
```

CI runs the suite on Python 3.9 through 3.14 and must pass on every version.

## What a numerical change needs

A single expected value is not evidence that an algorithm is right. Add a test
for a property the theory guarantees:

- a **convergence order** — halve the step, check the error ratio;
- an **exactness** result — a Gauss rule on a polynomial of degree `2n − 1`;
- a **conservation law** or a structural identity;
- a **known failure mode** reproduced deliberately;
- agreement with an **independent analytic solution**.

Include the difficult cases, and document what the method cannot do — see
[Known limitations](limitations.md) for the standard.

Keep the algorithm visible in the source. NumPy may supply array operations
and low-level primitives; it must not supply the method.

## Working on the documentation

The site is built with [MkDocs](https://www.mkdocs.org) and Material:

```bash
mkdocs serve      # live reload at http://127.0.0.1:8000
mkdocs build      # render into site/
```

Three rules keep these pages honest, and the test suite enforces all three:

**Every example is executed.** Examples are written as doctests, and
`tests/test_docs.py` runs every one on every documentation page. An example
that no longer produces the output shown fails the suite.

```pycon
>>> import quadrivium as qd
>>> round(qd.brent(lambda x: x**2 - 2, 0, 2).root, 10)
1.4142135624

```

Because doctest compares printed output exactly, convert NumPy scalars with
`float()`, `int()`, or `bool()` before displaying them, and round to fewer
digits than the method actually delivers.

**The API reference is generated.** `docs/api/*.md` comes from the installed
package. After changing any public signature or docstring, regenerate it:

```bash
python tools/gen_docs.py
```

`python tools/gen_docs.py --check` verifies the checked-in pages match the
code, and the test suite fails if they do not — on every supported Python, so
the generator's output has to be interpreter-independent. That is why it reads
annotations as source text rather than evaluating them: `Optional[float]`
evaluates to a `typing` object whose `repr` changed in 3.14, which would make
the pages depend on the version that generated them.

**The navigation must resolve.** Every page in `mkdocs.yml` must exist, and
every internal link must point at a real file.

## Pull requests

Explain the problem, the approach, and how you validated the result. Keep each
pull request to one coherent change, and add a regression test for every bug
fix. By contributing you agree that your work is licensed under the project's
[MIT License](https://github.com/ssmmkk123/quadrivium/blob/main/LICENSE).
