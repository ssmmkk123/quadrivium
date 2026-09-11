# Runnable examples

The `examples/` directory contains six longer tours of Quadrivium. They print
numerical results and comparisons to the terminal. The guides provide smaller
checked examples and interpretation; the graph generator produces the site's
figures separately.

## Set up and run

Install a checkout first so the required C extension is available:

```bash
python -m pip install -e .
python examples/01_linear_algebra.py
```

Run scripts from the repository root. Each imports `quadrivium.numeric` as
`np`. A comparison with `np.linalg` therefore uses the package's own array
primitive, which is not an independent external implementation.

## Choose a tour

| Script | Main topics | What to examine |
| --- | --- | --- |
| `01_linear_algebra.py` | Factorizations, orthogonality, eigenvalues, Krylov methods, sparse storage, least squares | Reconstruction residuals, loss of orthogonality, method assumptions |
| `02_calculus.py` | Finite differences, automatic differentiation, quadrature, Monte Carlo, special functions | Step-size sensitivity, observed order, callback counts |
| `03_differential_equations.py` | Explicit and implicit ODEs, stiffness, symplectic methods, BVPs, Lorenz system | Stability, error, conserved structure, time horizon |
| `04_optimization.py` | First-order, quasi-Newton, constrained, global, and proximal optimization | Stationarity, feasibility, starting-point dependence |
| `05_pde_and_transforms.py` | Heat, advection, elliptic problems, multigrid, FEM, spectral methods, transforms, sampling | Mesh refinement, numerical diffusion, boundary conditions |
| `06_extended_methods.py` | SDEs, reaction simulation, wavelets, matrix equations, extrapolation, events, WENO, flow | Coupling, reference assumptions, reconstruction, specialized contracts |

Run the other tours individually:

```bash
python examples/02_calculus.py
python examples/03_differential_equations.py
python examples/04_optimization.py
python examples/05_pde_and_transforms.py
python examples/06_extended_methods.py
```

These scripts vary in cost. Long parameter sweeps and PDE examples can take
more time than an introductory doctest. Start with the relevant guide if you
only need one call.

## Read the output critically

A small printed error is useful only when you know the reference and scale.
Check whether a script compares against an analytic expression, a reconstruction
identity, another Quadrivium method, or an external implementation. Those are
different strengths of evidence.

Some examples intentionally show unstable steps, oscillation, slow convergence,
or failed optimization. Read status flags and messages before interpreting a
number. A demonstration's heading expresses its topic; it does not establish a
universal theorem for every input.

Floating-point arithmetic means phrases such as “exact factorization” refer to
an algebraic identity satisfied up to rounding. Automatic differentiation
avoids finite-difference truncation but still uses floating-point operations.
An event location inherits the accuracy of the interpolant used to find it.

## Turn a tour into your own experiment

1. Replace the model or input data while keeping shapes and units explicit.
2. Keep a simple reference case before adding complexity.
3. Record method options, backend, source revision, and seeds.
4. Vary one numerical control at a time: step count, tolerance, mesh, or samples.
5. Compare the downstream quantity of interest and available diagnostics.

For fitting a model or managing long output, use the
[scientific workflow guide](guides/workflows.md). For a numerical method's
signature and return type, use the [API reference](api/index.md).

## Reproduce documentation graphs

```bash
python -m pip install -e ".[figures]"
python tools/gen_figures.py --png /tmp/quadrivium-figure-preview
```

This command runs the graph experiments and writes preview PNGs in addition to
the committed SVGs. The [figure methodology](figures.md) describes their
provenance, scopes, and checks. Running an example script does not refresh those
assets.
