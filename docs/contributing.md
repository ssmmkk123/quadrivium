# Contributing

A useful contribution makes a calculation more correct, understandable,
reproducible, or efficient. Start with a focused problem and an observable
outcome. For a substantial new algorithm or API change, discuss the scope in an
issue before implementing it.

## Set up a development checkout

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell, use `.venv\Scripts\Activate.ps1`. A source build needs a
C compiler and Python development headers. Python changes in the editable
checkout are visible immediately; rebuild after C changes.

Use the package's array layer in library code:

```python
from quadrivium import numeric as np
```

NumPy belongs in independent-reference tests and plotting tools, not as a new
runtime dependency. Keep the core algorithm inspectable in Python or in the
repository's own C implementation.

## Reproduce a numerical issue

A report should establish the discrepancy, not only state that an output looks
wrong. Include the smallest input that shows it, all nondefault options,
version or source revision, Python version, backend, and the expected answer.
Explain how the expectation was obtained: an analytic solution, an identity,
a refinement law, or an independent implementation.

For a solver report, include its status, message, iteration counts, and residual
where available. For a native crash, include array shapes, dtypes, strides,
view relationships, and a minimal sequence of operations. For performance,
report the workload and environment along with time or memory measurements.
Write local benchmark output under the ignored `benchmarks/` directory.

## Change an algorithm

Preserve the public contract unless an interface change is intentional and
explained. Describe accepted shapes and dtypes, default choices, method-specific
options, outputs, failure behavior, and limitations in the source docstring.
A familiar name does not substitute for a precise contract.

Choose validation that can expose the intended failure. Examples include:

- reconstructing the original matrix from computed factors;
- checking orthogonality in addition to a factorization residual;
- verifying convergence order on several grid sizes;
- comparing a manufactured PDE solution and its boundary conditions;
- checking aliasing and retained callback state for native optimizations;
- confirming a stochastic moment across replications with a justified tolerance.

A single stored output from the same implementation is weak evidence.
For behavior-changing fixes, add a regression case when practical.

## Write explanations and examples

A guide should help readers choose a method, form the input, interpret the
output, and validate the answer. Explain the effect of tolerances and the
conditions under which the method works. Use explicit variable names and avoid
undefined callbacks in examples intended to run.

Hand-written executable examples use `pycon` fences. Show stable outputs or
bounded-error predicates instead of platform-sensitive last digits. Each page
runs in its own doctest namespace, so it needs its own imports. Leave a blank
line before the closing fence after expected output:

````markdown
```pycon
>>> import quadrivium as qd
>>> result = qd.brent(lambda x: x*x - 2, 0, 2)
>>> result.converged and abs(result.root**2 - 2) < 1e-10
True

```
````

Ordinary `python` fences are illustrative code, not automatically executed by
the doctest runner. Shell commands also need manual review or execution as
appropriate. Do not describe every code fence as a tested example.

## Update the reference

`tools/gen_docs.py` generates the API reference from exported names, complete
signatures, source docstrings, and public class interfaces, including inherited
methods implemented by Quadrivium. It also copies the root changelog into the site. Edit the source
rather than patching generated Markdown.

```bash
python tools/gen_docs.py
python tools/gen_docs.py --check
python -m pytest -q tests/test_docs.py
```

The generation check detects stale output; it does not establish that a
source docstring's numerical claim is true. Review descriptions against the
implementation and update the narrative guide when a contract changes.

## Add or revise a graph

Use a plot only when it helps answer a scientific or numerical question. Record
the problem, reference, sampling grid, tolerances, random seeds, and metric.
Draw actual Quadrivium results; identify any analytic guide lines and display
floors. Separate an evaluation count from a wall-clock claim.

The catalogue lives in `tools/figures/experiments.py`. Each registered figure
has one owning documentation page and both theme variants. See
[figure methodology](figures.md) for the catalogue and reproduction details.

```bash
python tools/gen_figures.py --only 'ode-step-budget' --png /tmp/quadrivium-figure-preview
python tools/gen_figures.py
python tools/gen_figures.py --check
```

A full generation removes obsolete SVGs and refreshes the manifest. A subset
preview does neither. Review light and dark renders, captions, labels, scales,
and crowded legends before committing generated assets. Changes in Matplotlib
can change SVG bytes; inspect the changes and record the rendering environment.

## Build and inspect the site

```bash
python -m mkdocs build --strict
python -m mkdocs serve
```

The build checks links, anchors, and navigation. Browser review checks what a
link checker cannot: readable tables, narrow-screen behavior, figure contrast,
code wrapping, and useful search results. The normal build uses checked-in
figures and does not run numerical experiments.

## Run relevant checks

During development, run the tests for the changed area. Before a code contribution,
run the full suite and both backend configurations:

```bash
python -m pytest -q
QUADRIVIUM_NO_ACCEL=1 python -m pytest -q
python tools/gen_docs.py --check
python tools/gen_figures.py --check
python -m mkdocs build --strict
```

The environment variable selects Python reference methods while retaining the C
array engine. On PowerShell, set `$env:QUADRIVIUM_NO_ACCEL = "1"` before the
second pytest command, then remove it afterward.

## Submit a reviewable change

Explain the concrete problem, resulting behavior, validation, and remaining
limitations. Include a before/after numerical example when it clarifies the
fix. Keep generated files synchronized with the source that produces them.
A documentation change should identify the clarified contract or user decision.

Contributions are licensed under the project's MIT license. Follow the root
`CODE_OF_CONDUCT.md`; private security reports use `SECURITY.md`. Maintainers
can continue with the [release process](release-process.md).
