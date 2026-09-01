# Ordinary differential equations

```python
from quadrivium.ode import dormand_prince, radau_iia, velocity_verlet
import quadrivium as qd          # qd.solve_ivp, qd.rk4, qd.bdf, qd.shooting, ...
```

76 routines: explicit and implicit one-step methods, multistep families,
symplectic integrators for Hamiltonian systems, exponential integrators for
stiff linear parts, extrapolation, event location, boundary value problems,
index-1 DAEs and delay equations. Full signatures are in the
[`ode` reference](../api/ode.md).

## The default path

```pycon
>>> import numpy as np
>>> import quadrivium as qd
>>> sol = qd.solve_ivp(lambda t, y: -2*y, (0, 1), [1.0], rtol=1e-10)
>>> abs(float(sol.y[-1, 0]) - float(np.exp(-2))) < 1e-9
True
>>> sol.success, sol.method
(True, 'dormand_prince')

```

`solve_ivp` runs adaptive Dormand-Prince by default: fifth order, with an
embedded fourth-order estimate for step control. Pass `method=` for anything
else — `"rk4"`, `"radau"`, `"bdf"`, `"backward_euler"`, `"adams_bashforth"`,
and the rest — with `n=` for fixed-step methods and `rtol`/`atol` for adaptive
ones.

The right-hand side takes `(t, y)` and returns an array; `y0` may be a list.
The result is an `ODESolution` with `t` of shape `(n,)`, `y` of shape
`(n, dim)`, and step statistics:

```pycon
>>> sol.n_accepted > 0, sol.n_rejected >= 0, sol.n_rhs_evals > sol.n_steps
(True, True, True)

```

### Dense output

The solution is callable at any time in the interval, using cubic Hermite
interpolation through the stored states and slopes — accurate to `O(h⁴)`,
which matches a fourth or fifth order integrator rather than degrading it:

```pycon
>>> t_query = np.linspace(0, 1, 11)
>>> y = sol(t_query)
>>> float(np.max(np.abs(y[:, 0] - np.exp(-2*t_query)))) < 1e-8
True

```

## Choosing a method

The first question is whether the problem is stiff — whether the fastest
timescale in the equation is far shorter than the time you want to integrate
over. `detect_stiffness` and `stiffness_ratio` answer it from the Jacobian's
eigenvalues.

| Problem | Method |
| --- | --- |
| non-stiff, general | `solve_ivp` (Dormand-Prince), `dormand_prince`, `cash_karp`, `rkf45` |
| non-stiff, cheap right-hand side, low accuracy | `bogacki_shampine` |
| stiff | `radau_iia` (L-stable, order 5), `bdf`, `tr_bdf2`, `esdirk` |
| very stiff, mildly nonlinear | `rosenbrock` (one linear solve per step, no Newton) |
| Hamiltonian / long-time energy behaviour | `velocity_verlet`, `yoshida4`, `pefrl` |
| second-order `y'' = f(t, y)` | `rk_nystrom`, `stormer_cowell` |
| stiff *linear* part plus a mild nonlinearity | `etd_rk4`, `exponential_rosenbrock` |
| smooth and you want many digits | `gragg_bulirsch_stoer` |
| a fixed step and a specific tableau | `rk_general(f, ..., A, b, c)` |
| cheap right-hand side, high order, dense history | `adams_bashforth`, `variable_step_adams` |
| you need to stop at a condition | `solve_ivp_events` |

### Stiffness, demonstrated

An explicit method on a stiff problem does not merely slow down — it is
unstable unless the step is smaller than the fastest timescale, however smooth
the solution looks:

```pycon
>>> stiff = lambda t, y: [-1000*(y[0] - np.cos(t)) - np.sin(t)]
>>> explicit = qd.rk4(stiff, (0, 1), [1.0], n=100)         # h = 0.01, needs ~0.0028
>>> bool(np.max(np.abs(explicit.y)) > 1e3)                 # blows up
True
>>> implicit = qd.radau_iia(stiff, (0, 1), [1.0], n=100)   # same step, stable
>>> abs(float(implicit.y[-1, 0]) - float(np.cos(1.0))) < 1e-3
True

```

That is the whole argument for implicit methods: the cost per step is higher,
but the step is chosen by the accuracy you want, not by the stability limit.

### Convergence orders

Each family converges at its stated order — halving the step divides the error
by 2ᵖ:

```pycon
>>> exact = float(np.exp(-1.0))
>>> for method, order in ((qd.euler, 1), (qd.ode.heun, 2), (qd.rk4, 4)):
...     e1 = abs(float(method(lambda t, y: -y, (0, 1), [1.0], n=40).y[-1, 0]) - exact)
...     e2 = abs(float(method(lambda t, y: -y, (0, 1), [1.0], n=80).y[-1, 0]) - exact)
...     print(f"{method.__name__:6s} order {order}: error ratio {e1/e2:5.0f}")
euler  order 1: error ratio     2
heun   order 2: error ratio     4
rk4    order 4: error ratio    16

```

## Symplectic integrators

For a Hamiltonian system, what matters over long times is not the error at
each step but whether the energy drifts. A symplectic method keeps it bounded
forever; a general-purpose method of much higher order does not:

```pycon
>>> dHdq = lambda q: q               # harmonic oscillator, H = (p² + q²)/2
>>> dHdp = lambda p: p
>>> sym = qd.ode.leapfrog(dHdq, dHdp, (0, 200), [1.0], [0.0], n=20000)
>>> q, p = sym.y[:, 0], sym.y[:, 1]     # position and momentum in the columns
>>> energy = 0.5 * (p**2 + q**2)
>>> float(np.max(np.abs(energy - 0.5))) < 1e-4      # bounded, not growing
True

Over 200 time units — some thirty oscillations — the energy stays within
`1.2e-5` of its initial value and does not trend. `energy_drift` reports the
relative figure directly:

>>> qd.ode.energy_drift(sym, lambda y: 0.5 * (y[1]**2 + y[0]**2)) < 1e-4
True

```

`velocity_verlet` and `stormer_verlet` take a force and are the natural
interface for molecular dynamics; `ruth3`, `forest_ruth`, `yoshida4`, and
`pefrl` are higher-order compositions. `energy_drift` measures the drift for
any solution.

## Events

`solve_ivp_events` locates the roots of `g(t, y)` on the dense output, so the
event time is as accurate as the solution itself, not as accurate as the step
size. `terminal=True` stops the integration there:

```pycon
>>> throw = lambda t, y: [y[1], -9.81]               # height and velocity
>>> hits_ground = lambda t, y: y[0]
>>> sol, t_events, y_events = qd.solve_ivp_events(
...     throw, (0, 10), [5.0, 10.0], events=hits_ground, terminal=True, rtol=1e-10)
>>> landing = (10 + np.sqrt(100 + 2 * 9.81 * 5)) / 9.81
>>> abs(float(t_events[0][0]) - float(landing)) < 1e-8
True
>>> abs(float(sol.t[-1]) - float(landing)) < 1e-8    # integration stopped there
True

```

`direction=` on `find_events` filters to upward or downward crossings only,
which is how you catch "the ball landing" without also catching the launch.

## Boundary value problems

A BVP fixes conditions at both ends, so it cannot be marched. Three families
solve it:

| Approach | Function | Note |
| --- | --- | --- |
| shooting | `shooting`, `multiple_shooting`, `linear_shooting` | reduces to a root-find on the missing initial slope |
| finite differences | `finite_difference_bvp`, `nonlinear_fd_bvp` | one linear (or Newton) solve on the whole grid |
| weighted residuals | `collocation_bvp`, `galerkin_bvp` | spectral accuracy on smooth problems |

```pycon
>>> # y'' = 6x with y(0) = 0, y(1) = 1: exact solution y = x³
>>> bvp = qd.ode.finite_difference_bvp(lambda x: 0.0, lambda x: 0.0,
...                                    lambda x: 6*x, (0, 1), 0.0, 1.0, n=100)
>>> float(np.max(np.abs(bvp.y[:, 0] - bvp.t**3))) < 1e-13
True

```

`multiple_shooting` splits the interval into segments and matches them
simultaneously, which is what makes shooting work on a problem where a single
trajectory would overflow before reaching the far end.

`sturm_liouville` solves the eigenvalue problem `-(p y')' + q y = λ w y`,
returning the eigenvalues and eigenfunctions — the discrete spectrum that
separation of variables produces.

## Beyond ODEs

**Differential-algebraic equations.** `dae_index1_bdf` integrates
`y' = f(t, y, z)`, `0 = g(t, y, z)` — a differential part coupled to a
constraint. `mass_matrix_ode` handles `M y' = f(t, y)` with `M` singular,
which is the same problem in another form.

**Delay equations.** `dde_method_of_steps` integrates `y'(t) = f(t, y(t),
y(t−τ))` by stepping one delay interval at a time, using the previous
interval's dense output as history.

```pycon
>>> hist = lambda t: np.array([1.0])
>>> sol = qd.dde_method_of_steps(lambda t, y, yd: -yd[0], hist, [1.0], (0, 4), n=400)
>>> sol.y.shape[1], bool(np.all(np.isfinite(sol.y)))
(1, True)

```

**Extrapolation.** `gragg_bulirsch_stoer` combines modified midpoint steps at
several substep counts and extrapolates, choosing its own order as it goes. On
a smooth problem it reaches accuracies a fixed-order method would need
enormously many steps for:

```pycon
>>> gbs = qd.gragg_bulirsch_stoer(lambda t, y: -y, (0, 1), [1.0], rtol=1e-12)
>>> abs(float(gbs.y[-1, 0]) - exact) < 1e-11
True

```

`richardson_ode` applies the same idea to any fixed-step method of known
order.

## Pitfalls

- **`rtol` and `atol` control the local error per step, not the global error.**
  Errors accumulate; over a long integration the final error can be much
  larger than the tolerance. Halve the tolerance and compare.
- **A fixed-step explicit method on a stiff problem produces garbage, not a
  slow answer.** Check `detect_stiffness` when a solution blows up.
- **A symplectic method must be run at a fixed step.** Adapting the step
  destroys the property that keeps the energy bounded.
- **Multistep methods need starting values.** They are bootstrapped with a
  one-step method internally, and the starting error sets a floor on the whole
  integration.
- **Dense output between widely spaced steps is only as good as the
  interpolant.** For a solution with structure between steps, ask for smaller
  steps rather than more query points.
- **Event location finds sign changes.** An event that touches zero without
  crossing is missed.

## See also

- [`ode` API reference](../api/ode.md) — every signature.
- [PDE guide](pde.md) — method of lines, where a PDE becomes a large ODE system.
- [Root finding guide](rootfind.md) — the nonlinear solves inside every
  implicit step.
- [Stochastic guide](stochastic.md) — SDE integrators, the same idea with noise.
- `examples/03_differential_equations.py` — a runnable tour.
