# Partial differential equations

A PDE solver combines a spatial representation, boundary conditions, and either
a time integrator or a stationary linear/nonlinear solve. Choosing the equation
alone is not enough: a periodic advection scheme and a Dirichlet diffusion
scheme represent different physical problems even on the same interval.

This guide develops small problems with known answers, then explains how to
extend the workflow to conservation laws, finite elements, and larger grids.
The [PDE reference](../api/pde.md) contains the complete method catalogue.

## Choose an equation and discretization

| Problem | Starting interface | Main numerical decision |
| --- | --- | --- |
| One-dimensional heat diffusion | `heat_crank_nicolson`, `heat_btcs` | Spatial resolution and temporal damping |
| Explicit heat demonstration | `heat_ftcs` | Diffusion stability limit |
| Custom time-dependent spatial operator | `method_of_lines` | Spatial stencil and ODE stiffness |
| Linear periodic transport | `advection_upwind`, `lax_wendroff`, `tvd_scheme` | Diffusion versus oscillation near fronts |
| Nonlinear conservation law | `fvm_1d_conservation`, `fvm_muscl`, `weno_conservation_law` | Flux, wave speed, and CFL limit |
| Poisson with rectangular boundaries | `poisson_2d_direct`, `poisson_2d_iterative` | Grid size and algebraic tolerance |
| Zero-Dirichlet rectangular Poisson | `poisson_fft` | Transform diagonalization of the stencil |
| Larger structured elliptic problem | `multigrid_solve` | Compatible grid hierarchy |
| Triangular finite-element mesh | `fem_2d_triangular`, `adaptive_fem` | Geometry and local refinement |
| Periodic incompressible flow | `navier_stokes_2d` | Projection consistency and time-step stability |

A method's name is not a complete boundary-condition specification. For
example, `poisson_fft` uses a **sine transform with homogeneous Dirichlet
boundaries**; `poisson_periodic_fft` solves a different periodic problem.
Likewise, Fourier spectral solvers assume periodicity, whereas Chebyshev
methods use a nonperiodic collocation grid.

## Read the solution layout

Structured-grid solvers return `PDESolution` with the field `u`, coordinate
arrays in `grids`, and time levels in `t` for time-dependent problems.
`final` returns the actual last state, or the stationary field when no time
axis exists. `converged`, `iterations`, and `residuals` are meaningful for
solvers that perform iterative solves.

| Solver family | Typical full output shape |
| --- | --- |
| One-dimensional heat, endpoint grid | `(nt + 1, nx + 1)` |
| Periodic linear advection | `(nt + 1, nx)` |
| Stationary rectangular Poisson | `(nx + 1, ny + 1)` |
| Two-dimensional ADI heat | `(nt + 1, nx + 1, ny + 1)` |
| Triangular FEM | One value per mesh vertex |

Some specialized systems use additional component axes. Read the solver's
layout before applying `meshgrid` or plotting: the flow solver uses fields
from `meshgrid(..., indexing="xy")`, whereas rectangular Poisson arrays are
indexed by x and then y. Storage controls can reduce the time dimension.

## Solve a heat equation with a known solution

Consider `u_t = alpha*u_xx` on `[0, 1]`, with zero endpoint values and
`u(x, 0) = sin(pi*x)`. Its exact solution is
`exp(-alpha*pi**2*t)*sin(pi*x)`, making it useful for checking both amplitude
and convergence.

```pycon
>>> import quadrivium as qd
>>> from quadrivium import numeric as np
>>> initial = lambda x: np.sin(np.pi * x)
>>> heat = qd.heat_crank_nicolson(initial, 0.1, (0, 1), (0, 0.5),
...                              nx=40, nt=100)
>>> heat.u.shape, len(heat.grids)
((101, 41), 1)
>>> x = heat.grids[0]
>>> exact = np.exp(-0.1 * np.pi**2 * 0.5) * np.sin(np.pi * x)
>>> float(np.max(np.abs(heat.final - exact))) < 2e-4
True
>>> float(np.max(np.abs(heat.u[:, [0, -1]]))) < 1e-12
True

```

`nx` counts spatial intervals for this solver, so `dx = 1/nx`. `nt` counts
time intervals. `u0` can be a scalar-coordinate callable or a vector of
`nx + 1` values. The `bc=(left, right)` values can be constants or functions
of time; optional `source(x, t)` adds a forcing term. Make the initial endpoint
values consistent with the boundary data to avoid an unintended initial jump.

### Stability and accuracy are different requirements

For heat diffusion define `r = alpha*dt/dx**2`.

| Scheme | Temporal order on smooth solutions | Stability for the standard diffusion problem |
| --- | --- | --- |
| `heat_ftcs` | First | Requires `r <= 1/2` |
| `heat_btcs` | First | Unconditional linear stability |
| `heat_crank_nicolson` | Second | Unconditional linear stability |
| `heat_theta` | Second at `theta=1/2`, otherwise generally first | Unconditional for `theta >= 1/2` |

```pycon
>>> from quadrivium.pde import stability_ratio
>>> round(stability_ratio(alpha=0.1, dt=0.001, dx=0.02), 6)
0.25
>>> from quadrivium.core import DomainError
>>> try:
...     qd.pde.heat_ftcs(initial, 0.1, (0, 1), (0, 0.5), nx=40, nt=10)
... except DomainError:
...     print("Increase nt or choose an implicit diffusion scheme.")
Increase nt or choose an implicit diffusion scheme.

```

The explicit solver checks this limit by default. Do not interpret the absence
of a check in another interface as evidence that every step size is stable.
The general `heat_theta` interface, for example, lets the caller choose theta
and the grid; assess the resulting stability requirement yourself.

An implicit scheme can remain bounded at a large step while being inaccurate.
Crank-Nicolson can retain oscillatory high-frequency components when `r` is
large, especially for rough initial data. Backward Euler damps those modes
more strongly. Refinement and the physical behavior of interest should decide
whether that damping is desirable.

### Measure spatial and temporal error separately

First make temporal error small and refine `nx`; then hold the spatial grid
fine and refine `nt`. Changing both at once can hide which discretization
limits accuracy. A smooth central-difference spatial scheme should approach
second-order behavior until another error source dominates.

```pycon
>>> errors = []
>>> for intervals in (20, 40):
...     run = qd.heat_crank_nicolson(initial, 0.1, (0, 1), (0, 0.5),
...                                  nx=intervals, nt=200, final_only=True)
...     truth = np.exp(-0.1 * np.pi**2 * 0.5) * np.sin(np.pi * run.x)
...     errors.append(float(np.max(np.abs(run.final - truth))))
>>> 3.5 < errors[0] / errors[1] < 4.5
True

```

<figure markdown="span">
  ![Heat profiles compared with an analytic solution and error under joint space and time refinement](../assets/figures/pde-diffusion-refinement.svg#only-light)
  ![Heat profiles compared with an analytic solution and error under joint space and time refinement](../assets/figures/pde-diffusion-refinement-dark.svg#only-dark)
  <figcaption>The profiles show diffusion of a sine mode with alpha=0.1 through time 0.5. The error experiment increases nx from 10 to 80 with nt=2*nx, refining space and time together. Both discretization errors contribute to the measured second-order trend; the separate refinement example above helps identify which contribution controls a particular run.</figcaption>
</figure>

## Retain and restart time-dependent fields

Full field histories can dominate memory. If a grid contains `M` floating-point
values and you save `K` time levels in float64, the retained field payload is
approximately `8*M*K` bytes, before coordinates and solver work arrays.

```pycon
>>> sampled = qd.heat_crank_nicolson(initial, 0.1, (0, 1), (0, 0.5),
...                                 nx=20, nt=100, save_every=25)
>>> sampled.u.shape
(5, 21)
>>> endpoint = qd.heat_crank_nicolson(initial, 0.1, (0, 1), (0, 0.5),
...                                  nx=20, nt=100, final_only=True)
>>> endpoint.u.shape, endpoint.final.shape
((1, 21), (21,))
>>> np.allclose(sampled.final, endpoint.final)
True

```

Time-dependent interfaces accept `save_every`, `save_at`, `final_only`, and
`callback(t, u)` through their output-control wrappers. Requested output times
do not automatically refine the spatial or temporal discretization. A callback
receives a copy and can return `True` to stop.

Checkpoints preserve numerical state; `resume_pde` also needs the original
solver and model/spatial options. Wave and leapfrog restart paths preserve
the previous time level and require the original step. See the
[workflow guide](workflows.md) for a concrete restart.

## Transport, waves, and discontinuities

Linear advection solves `u_t + c*u_x = 0`: a profile translates at speed `c`.
The periodic interfaces store `nx` unique points on an endpoint-excluded grid.
Use the exact translated profile to assess phase error and amplitude loss.

```pycon
>>> from quadrivium.pde import advection_upwind, lax_wendroff
>>> profile = lambda x: np.sin(2 * np.pi * x)
>>> transport = lax_wendroff(profile, 1.0, (0, 1), (0, 0.2), nx=80, nt=80)
>>> transport.u.shape
(81, 80)
>>> translated = np.sin(2 * np.pi * (transport.x - 0.2))
>>> float(np.max(np.abs(transport.final - translated))) < 0.002
True

```

The Courant number is `abs(c)*dt/dx`. For upwind and Lax-Wendroff linear
advection the usual stable range is at most one. The periodic driver does not
automatically reject every unstable user choice, so calculate the number
before running a new grid.

Upwind is robust and diffusive: it smooths fronts and lowers narrow peaks.
Lax-Wendroff improves smooth-solution accuracy but can overshoot near a jump.
`tvd_scheme` uses a flux limiter to reduce such oscillation. Its behavior is
nonlinear even when the PDE is linear; order near extrema and discontinuities
can be lower than order on a smooth wave.

`wave_explicit` solves a second-order wave equation and therefore needs both
initial displacement and initial velocity. Its explicit step is also limited
by a wave CFL condition. A visually plausible wave can still accumulate phase
error over many periods; compare travel time as well as amplitude.

## Conservative nonlinear schemes

For `u_t + F(u)_x = 0`, a finite-volume update changes cell averages through
flux differences. This makes conservation a natural diagnostic: under periodic
boundaries, the discrete mass should remain close to its initial value.

`fvm_1d_conservation` and `fvm_muscl` take the physical flux and a wave-speed
function. An underestimated speed can invalidate the stability limit.
`riemann_solver_burgers` and `godunov_burgers` specialize to Burgers' flux
`F(u)=u**2/2`. WENO methods reconstruct smooth regions at high order and switch
weights near steep gradients; they are not a guarantee that an unresolved
shock will have a visually exact shape.

Validate a conservation-law run with mass balance, front speed, grid
refinement, and minimum/maximum values. Use the actual cell-center grid from
the result when comparing to a solution; do not substitute the endpoint grid
used by the heat example.

## Stationary Poisson problems

The finite-difference Poisson routines use `laplacian(u) = f`. This sign
convention differs from finite-element interfaces written as
`-div(c*grad(u)) + r*u = f`.

```pycon
>>> from quadrivium.pde import poisson_1d, poisson_fft
>>> steady = poisson_1d(lambda x: -2.0, (0, 1), nx=30)
>>> float(np.max(np.abs(steady.final - steady.x * (1 - steady.x)))) < 1e-12
True
>>> steady.t is None
True
>>> source = lambda x, y: -2 * np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)
>>> field = poisson_fft(source, (0, 1), (0, 1), nx=20, ny=20)
>>> truth = np.sin(np.pi * field.x[:, None]) * np.sin(np.pi * field.y[None, :])
>>> float(np.max(np.abs(field.final - truth))) < 0.003
True

```

The transform solver solves the **discrete** system to numerical precision;
its continuum error remains second order for the standard stencil. An
iterative residual near machine precision does not remove discretization
error. Conversely, refining the grid with a loose linear tolerance can hide
the expected order.

`poisson_2d_direct(stencil=9)` includes the right-hand-side correction for the
Mehrstellen scheme and requires equal x/y spacing. `laplacian_matrix` returns
a dense matrix despite the sparse pattern of the stencil; avoid it for grids
where the square matrix would be too large.

Multigrid alternates smoothing with coarse-grid correction. For the supplied
structured hierarchy, use compatible square grids and power-of-two interval
counts. Inspect `converged` and `residuals`; reaching `max_cycles` returns a
result that can still need further work. Periodic and pure-Neumann Poisson
problems also need a compatible source and a convention for the arbitrary
constant mode.

## Method of lines and finite elements

`method_of_lines` supplies `rhs(t, u, x, dx)` with a full state including the
boundary values. Return a **full-length derivative vector**: the current
implementation slices `[1:-1]` before integrating the interior. The boundary
derivatives in that vector are ignored because boundary values are imposed
by the wrapper. Select an ODE solver using `solver=` and pass its tolerances
through the remaining keyword arguments.

Spatial refinement of diffusion makes the resulting ODE increasingly stiff.
An adaptive explicit method may take many small steps even if requested output
times are far apart. An implicit ODE solver can address that stability cost;
it cannot repair an inconsistent spatial stencil.

Finite elements work with basis functions and a weak form of the equation.
`fem_1d_linear` and `fem_1d_quadratic` solve diffusion-reaction equations;
`fem_2d_triangular` works on planar triangular meshes. `TriangularMesh` stores
points, connectivity, and boundary vertices. `adaptive_fem` repeatedly solves,
estimates element error, marks elements, and refines a conforming mesh.

The adaptive result includes `error_estimate`, `element_errors`, and
`refinement_history`. Check `converged` after hitting `max_refinements` or
`max_elements`. The error estimate guides refinement; it is not a certified
pointwise bound. Keep the linear-solve tolerance tighter than the accuracy
being requested from the mesh.

## A practical validation sequence

1. Write down the PDE sign convention, units, and boundary conditions.
2. Confirm the result axes and whether points are endpoints or cell centers.
3. Compute diffusion or Courant numbers for explicit updates.
4. Check a manufactured solution with the same boundary conditions.
5. Refine space and time independently, and monitor a physical balance law.
6. Tighten algebraic tolerances only until discretization error dominates.
7. Reduce retained output after choosing the resolution needed for validation.

For related tools, see [ODE integration](ode.md), [linear algebra](linalg.md),
[transforms](transforms.md), and [scientific workflows](workflows.md).
