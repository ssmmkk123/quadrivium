# Figure methodology and reproduction

The documentation uses 17 numerical experiments, each rendered as a light and
dark SVG. Each figure answers a practical question: how much error remains,
what refinement buys, why a method struggles, or which diagnostic exposes a
misleading result. Every previous figure has been replaced.

The computations use Quadrivium methods and explicit references. NumPy supplies
plot coordinates and diagnostic reductions in the figure tool; Matplotlib
renders the output. Neither is a runtime requirement for ordinary Quadrivium
calculations. The actual experiment source is `tools/figures/experiments.py`.

## Read a numerical plot

Start with the problem and the quantity on each axis. Error requires a
reference: an analytic solution, an independently evaluated identity, or a
statistical target. A residual is not the same quantity as forward solution
error. A callback count measures one kind of work, not elapsed time.

On log-log axes, a straight reference line indicates a power-law rate within
the shown regime. It does not prove the same rate on rough, stiff, or
ill-conditioned problems. A floor used for plotting zero or very small errors
is a display choice, not a measured nonzero error; those floors are identified
in the source, manifest, or figure annotation.

Each two-panel experiment pairs the primary result with a diagnostic or
comparison. Read the explanation on the owning guide for the assumptions and
the method choice it supports.

## Experiment catalogue

| Experiment | Question it answers | Explanation |
| --- | --- | --- |
| `start-decay-validation` | Does an adaptive trajectory agree with an analytic solution between saved points? | [Getting started](getting-started.md) |
| `core-roundoff-budget` | How does cancellation change a relative error? | [Core](guides/core.md) |
| `numeric-grid-resolution` | Is a sampling grid fine enough to reconstruct a signal? | [Arrays](guides/numeric.md) |
| `linalg-residual-sensitivity` | Can a small residual coexist with a sensitive solution? | [Linear algebra](guides/linalg.md) |
| `rootfind-accuracy-work` | What accuracy do root finders achieve for their evaluation budget? | [Root finding](guides/rootfind.md) |
| `interpolate-node-choice` | How does node placement affect interpolation between observations? | [Interpolation](guides/interpolate.md) |
| `approx-fit-generalization` | Does a fit remain accurate away from its data? | [Approximation](guides/approx.md) |
| `diff-step-selection` | Why does reducing a finite-difference step eventually hurt? | [Differentiation](guides/diff.md) |
| `integrate-accuracy-budget` | How much function-evaluation work buys a given integration error? | [Integration](guides/integrate.md) |
| `ode-step-budget` | How do time-stepping methods trade error against work? | [ODEs](guides/ode.md) |
| `pde-diffusion-refinement` | What changes when a diffusion grid is refined? | [PDEs](guides/pde.md) |
| `optimize-scaling-paths` | How does parameter scaling change an optimizer's path? | [Optimization](guides/optimize.md) |
| `transforms-sampling-spectrum` | How do sampling and window choice affect an observed spectrum? | [Transforms](guides/transforms.md) |
| `stochastic-error-calibration` | How do empirical integration errors compare with their reported uncertainty? | [Stochastic methods](guides/stochastic.md) |
| `special-identity-checks` | Do evaluated special functions satisfy useful reference identities? | [Special functions](guides/special.md) |
| `workflow-parameter-recovery` | Does fitting a dynamical model recover a known parameter? | [Scientific workflows](guides/workflows.md) |
| `validation-refinement-orders` | Do measured error ratios approach the expected order? | [Design and validation](design.md) |

## Reproduce the complete set

From a built source checkout:

```bash
python -m pip install -e ".[figures]"
python tools/gen_figures.py
python tools/gen_figures.py --check
```

Generation runs the numerical experiments, writes both SVG variants, removes
unregistered SVGs, and updates the [provenance manifest](assets/figures/manifest.json).
The check compares the generated SVG bytes, catalogue, manifest, and any
obsolete assets against the files on disk.

The manifest records:

- experiment names, owning pages, summaries, and file names;
- model parameters, grids, tolerances, seeds, and stated display floors;
- Python, Quadrivium, NumPy, and Matplotlib versions;
- SHA-256 hashes of the figure generator, catalogue, and style source.

The source hashes cover the plotting tools, not the entire numerical library.
Record the repository commit separately for a complete experiment provenance.
A working tree can contain unreleased code even when its version string is
unchanged.

## Preview a specific figure

```bash
python tools/gen_figures.py --only 'diff-step-selection' --png /tmp/quadrivium-figure-preview
```

`--only` accepts a regular expression. It updates matching SVGs and can emit
PNG previews for both themes, but it does not prune assets or refresh the full
manifest. Run a complete generation before committing a changed experiment.
Use a temporary or ignored output directory for PNG previews.

## Check numerical and visual quality

Verify that the intended solver actually runs and inspect its status. Confirm
that the plotted metric matches the axis label and that all compared methods
solve the same problem. Count real callback evaluations where work is shown.
For stochastic comparisons, fix seeds and distinguish a repeatable single run
from an uncertainty estimate across replications.

Review the PNG or browser output at a readable size in both themes. Titles,
axis labels, units, legends, markers, and guide lines must be distinguishable.
Series use line styles or markers as well as color. SVG text remains searchable,
and each page supplies descriptive alternative text and an explanatory caption.

No raster image model is used to draw numerical data. The curves are generated
from calculations, so the source can be audited and rerun.

## Understand byte-for-byte checks

SVG creation timestamps are omitted and identifier hashing is fixed. Those
choices remove common incidental differences, but a different Python, NumPy,
Matplotlib, or font environment can still change serialized output. Use the
recorded environment when investigating a failed `--check`.

A changed file can reflect altered numerics, changed plotting behavior, or an
intentional visual revision. Inspect the result and its inputs before replacing
committed assets. The check is a freshness test, not a proof of scientific
validity.

## Build and test the documentation

```bash
python -m pytest -q tests/test_docs.py
python tools/gen_docs.py --check
python -m mkdocs build --strict
```

Documentation tests check referenced files, both theme variants, catalogue
ownership, unused assets, and generated reference freshness. The normal MkDocs
build uses committed SVGs; it does not rerun the experiments. The source
distribution omits generated figure assets, so regenerate them before building
a complete site from an unpacked source archive.
