# Ordinary differential equations

An initial value problem specifies a state at one time and a rule for how it
changes: `y' = f(t, y)`, `y(t0) = y0`. Quadrivium provides adaptive solvers for
routine simulation and individual numerical methods for studying accuracy,
stability, conservation, and computational cost. Start with `solve_ivp`; choose
a specialized interface when your equation has additional structure.

The examples use Quadrivium's array implementation. They do not require NumPy.
The [ODE reference](../api/ode.md) lists complete signatures and the available
method families.

## Solve and inspect an initial value problem

```pycon
>>> import quadrivium as qd
>>> from quadrivium import numeric as np
>>> rhs = lambda t, y: -2.0 * y
>>> sol = qd.solve_ivp(rhs, (0.0, 1.0), [1.0], rtol=1e-9, atol=1e-11)
>>> sol.success, sol.method
(True, 'dormand_prince')
>>> abs(float(sol.y_final[0]) - float(np.exp(-2.0))) < 1e-8
True

```

`rhs(t, y)` receives a scalar time and a one-dimensional state vector. Return
one derivative for each state component. A list is acceptable as `y0` or as the
right-hand side result; an array is convenient for vector expressions. The
ordinary solvers use real floating-point states. Represent a complex problem
as coupled real and imaginary components when needed.

`ODESolution` stores time along its **first** axis:

| Attribute | Meaning |
| --- | --- |
| `t` | Stored times, shape `(samples,)` |
| `y` | Stored states, shape `(samples, state_dimension)` |
| `y_final` | Actual last integrated state, shape `(state_dimension,)` |
| `success`, `message` | Completion status and explanation |
| `n_steps` | Step attempts reported by the method |
| `n_accepted`, `n_rejected` | Accepted and rejected step counts |
| `n_rhs_evals` | Counted evaluations of the right-hand side |
| `checkpoint` | Restart data, where the solver supports it |

```pycon
>>> sol.y.shape == (len(sol.t), 1)
True
>>> sol.n_accepted > 0, sol.n_rejected >= 0, sol.n_rhs_evals > 0
(True, True, True)
>>> oscillator = qd.solve_ivp(lambda t, y: [y[1], -y[0]],
...                           (0.0, 1.0), [1.0, 0.0], rtol=1e-9)
>>> oscillator.y.shape[1]
2
>>> np.allclose(oscillator.y_final, [np.cos(1.0), -np.sin(1.0)], atol=1e-7)
True

```

The oscillator stores position in column zero and velocity in column one
because that is the ordering chosen in `y0`. The solver attaches no physical
meaning or units to those columns.

## Choose an integration method

| Need | Starting choice | What to check |
| --- | --- | --- |
| General nonstiff simulation | `solve_ivp` / `dormand_prince` | Tolerance refinement and rejected steps |
| Prescribed uniform steps | `rk4` | Accuracy when `n` is doubled |
| Stiff relaxation | `bdf_adaptive`, `radau_adaptive` | Newton diagnostics and Jacobian quality |
| Fixed-step implicit comparison | `backward_euler`, `bdf`, `radau_iia` | Step and nonlinear-solve error |
| Separable Hamiltonian dynamics | `leapfrog`, `velocity_verlet`, `yoshida4` | Energy behavior and phase error |
| Smooth equations at tight tolerance | `gragg_bulirsch_stoer` | Smoothness and extrapolation cost |
| Known stiff linear term | `etd_rk4`, exponential methods | Linear/nonlinear splitting |
| Second-order position equation | `rk_nystrom`, `stormer_cowell` | Force signature and velocity output |
| Conditions at both endpoints | Boundary value solvers | Boundary residual and spatial refinement |

The dispatcher forwards method-specific arguments. In particular,
`method="bdf"` and `method="radau"` select adaptive stiff integration when
`n` is absent; supplying `n` selects their fixed-step forms. Calling
`bdf_adaptive` or `radau_adaptive` directly makes that distinction explicit.
An argument accepted by one solver is not automatically meaningful to another.

## Separate step accuracy from work

For a fixed-step method, `n` is the number of intervals. A full trajectory has
`n + 1` samples, including the initial state. Different methods use different
numbers of right-hand side evaluations per step, so equal `n` is not equal
computational work.

```pycon
>>> exact = float(np.exp(-1.0))
>>> for method in (qd.euler, qd.ode.heun, qd.rk4):
...     coarse = method(lambda t, y: -y, (0, 1), [1.0], n=20)
...     fine = method(lambda t, y: -y, (0, 1), [1.0], n=40)
...     error1 = abs(float(coarse.y_final[0]) - exact)
...     error2 = abs(float(fine.y_final[0]) - exact)
...     print(method.__name__, round(error1 / error2))
euler 2
heun 4
rk4 16

```

These ratios approach `2**p` for a method of order `p` when truncation error
dominates. A plateau at very small steps can indicate floating-point error;
poor ratios at coarse steps can mean the asymptotic regime has not been reached.
Discontinuities, unstable steps, or inaccurate inner solves can also prevent
the expected order from appearing.

<figure markdown="span">
  ![Decay trajectories and endpoint error against counted right-hand side evaluations](../assets/figures/ode-step-budget.svg#only-light)
  ![Decay trajectories and endpoint error against counted right-hand side evaluations](../assets/figures/ode-step-budget-dark.svg#only-dark)
  <figcaption>Euler, Heun, and RK4 solve the same decay equation. The work plot uses counted right-hand side evaluations so the comparison includes each method's stage cost. Use the error curves to choose a budget; one benchmark does not establish a universal ranking.</figcaption>
</figure>

## Tolerances and dense output

Adaptive methods estimate local error and choose internal steps accordingly.
`atol` sets the absolute scale near zero; `rtol` scales the permitted error with
the state magnitude. A small component can require a much smaller absolute
tolerance than a large component. Adaptive BDF and Radau accept componentwise
`atol` and `rtol` arrays.

A tolerance is a controller setting, not a certified bound on the final error.
Repeat an important solve with tighter tolerances and compare the quantity
that matters: an endpoint, an integral, a peak, or an event time.

```pycon
>>> query = np.linspace(0.0, 1.0, 11)
>>> evaluated = sol(query)
>>> evaluated.shape, sol(0.5).shape
((11, 1), (1,))
>>> float(np.max(np.abs(evaluated[:, 0] - np.exp(-2 * query)))) < 1e-7
True

```

Calling a solution interpolates between recorded states. Where recorded slopes
are available, the generic result uses cubic Hermite interpolation; otherwise
it falls back to linear interpolation. Some solvers supply a dedicated
interpolant. Interpolation error is additional to integration error: cubic
interpolation does not preserve arbitrary high solver order. Querying more
points does not make the trajectory more accurate. Restrict queries to the
integrated interval; extrapolation is not an accuracy guarantee.

## Stiff problems and Jacobians

A stiff equation can have a smooth visible solution and fast perturbations
that force explicit methods to use tiny steps. For example,
`y' = -1000*(y - cos(t)) - sin(t)` has the solution `cos(t)` from `y(0)=1`,
but deviations from that solution decay on a timescale of `0.001`.

```pycon
>>> from quadrivium.ode import radau_adaptive
>>> stiff_rhs = lambda t, y: -1000 * (y - np.cos(t)) - np.sin(t)
>>> stiff_sol = radau_adaptive(stiff_rhs, (0, 1), [1.0],
...                            jac=np.array([[-1000.0]]), rtol=1e-7,
...                            atol=1e-9, final_only=True)
>>> stiff_sol.success
True
>>> abs(float(stiff_sol.y_final[0]) - float(np.cos(1.0))) < 1e-5
True
>>> stiff_sol.n_jac_evals, stiff_sol.n_linear_solves > 0
(1, True)

```

The Jacobian is the derivative of the right-hand side with respect to the
state. A constant dense array avoids repeated Jacobian evaluations. Adaptive
stiff solvers also accept callable Jacobians, sparse matrices,
`LinearOperator` objects, and `BandedJacobian` storage. Structured inputs take
an iterative linear-solve path; they do not become dense matrices simply
because the time integrator is implicit.

Inspect `n_jac_evals`, `n_factorizations`, `n_linear_solves`, and
`n_newton_failures` when diagnosing cost. Rejections can arise from local
error or failed nonlinear iterations. Check signs and scaling in an analytic
Jacobian before interpreting repeated failures as evidence that the equation
itself is difficult. `detect_stiffness` and `stiffness_ratio` provide local
Jacobian-based indicators; they do not certify behavior over an entire run.

## Retain only the output you need

The integration grid and the requested output grid serve different purposes.
Use output controls to avoid keeping an unnecessary trajectory:

```pycon
>>> selected = qd.solve_ivp(rhs, (0, 1), [1.0], save_at=[0.2, 0.6])
>>> selected.t.tolist()
[0.2, 0.6]
>>> selected.checkpoint.t
1.0
>>> abs(float(selected.y_final[0]) - float(np.exp(-2))) < 1e-6
True
>>> compact = qd.rk4(rhs, (0, 1), [1.0], n=100, final_only=True)
>>> compact.y.shape
(1, 1)

```

`save_every=k` retains every kth accepted state and the endpoint.
`save_at` requests ordered times inside the integration interval.
`final_only=True` retains a single state. Output selection occurs during
integration, so discarded states do not accumulate in a hidden full history.
These policies have validation rules; for example, `final_only` and `save_at`
cannot be combined.

A `callback(t, y)` receives an independent state copy. Returning `True` stops
the integration at that state; inspect the status and checkpoint time to
distinguish such a stop from reaching `t_span[1]`. The callback also sees the
initial state. Sparse retention reduces what is available for later dense
interpolation. See [scientific workflows](workflows.md) for restart examples.

## Detect events

An event is a scalar function whose zero identifies a condition of interest.
The following ball reaches the ground when its height changes sign:

```pycon
>>> throw = lambda t, y: [y[1], -9.81]
>>> ground = lambda t, y: y[0]
>>> flight, times, states = qd.solve_ivp_events(
...     throw, (0, 10), [5.0, 10.0], events=ground, terminal=True, rtol=1e-9)
>>> landing = float((10 + np.sqrt(100 + 2 * 9.81 * 5)) / 9.81)
>>> abs(float(times[0][0]) - landing) < 1e-7
True
>>> abs(float(states[0][0, 0])) < 1e-7
True

```

The event interface returns the solution and lists of event-time and
event-state arrays. Root refinement uses the solution interpolant. Therefore,
a small root-search tolerance cannot correct an inaccurate integrated path.
`find_events` searches an existing solution and accepts `direction=1`, `-1`,
or `0` for increasing, decreasing, or either crossing.

Sign-change detection can miss tangential roots or several crossings inside
one step. Keep enough temporal resolution for the event timescale, and avoid
thinning an existing trajectory before searching it for events.

## Geometric and second-order integration

For a separable Hamiltonian, `leapfrog(dHdq, dHdp, t_span, q0, p0, n=...)`
stores position followed by momentum. Fixed-step symplectic methods often
have bounded, oscillatory energy error over useful long intervals. They do
not conserve the exact energy in every problem or eliminate phase error.

```pycon
>>> from quadrivium.ode import leapfrog, energy_drift
>>> sym = leapfrog(lambda q: q, lambda p: p, (0, 20), [1.0], [0.0], n=2000)
>>> sym.y.shape
(2001, 2)
>>> energy_drift(sym, lambda y: 0.5 * (y[0]**2 + y[1]**2)) < 1e-4
True

```

`velocity_verlet` takes a force law and stores position and velocity in the
state. `rk_nystrom` and `stormer_cowell` instead expose a separate `velocity`
trajectory. Check the individual signature: their acceleration callbacks have
different arguments. Arbitrarily adapting a symplectic method's step generally
changes its geometric properties.

## Boundary values, constraints, and delays

A boundary value problem prescribes conditions at both ends of an interval.
For `y'' = p(x)y' + q(x)y + r(x)`, the finite-difference solver uses a whole-grid
linear system:

```pycon
>>> bvp = qd.ode.finite_difference_bvp(lambda x: 0.0, lambda x: 0.0,
...     lambda x: 6*x, (0, 1), 0.0, 1.0, n=40)
>>> float(np.max(np.abs(bvp.y[:, 0] - bvp.t**3))) < 1e-12
True

```

Here the cubic is reproduced particularly accurately by the discrete equation;
that does not make finite differences exact for general smooth functions.
`shooting` adjusts missing initial data with a root solve; `multiple_shooting`
introduces intermediate states to improve conditioning. `collocation_bvp` and
`galerkin_bvp` use alternative whole-interval approximations. Check boundary
residuals and refine the grid or basis independently of nonlinear tolerances.

`dae_index1_bdf` couples differential states with algebraic constraints;
`mass_matrix_ode` handles a mass-matrix formulation. These interfaces target
their documented equation forms and are not general high-index DAE solvers.
`dde_method_of_steps` requires a history function before the initial time and
a list of delays. Its callback receives the delayed states separately from the
current state. Discontinuities inherited from the history can reduce order.

## Differentiate a trajectory

`solve_ivp_sensitivities` integrates variational equations for
`f(t, y, parameters)`. Parameter sensitivities have shape
`(stored_times, state_dimension, parameter_count)`; `sensitivity_final` keeps
the actual endpoint derivative even when the endpoint is not stored.
`initial=True` additionally computes derivatives with respect to initial
state components. Supply `initial_sensitivity` when `y0` depends on parameters.

`adjoint_sensitivity` computes the gradient of a terminal scalar objective
using stored checkpoints and replayed trajectory segments. Replay requires a
deterministic model. Both approaches differentiate the continuous equations;
they do not differentiate adaptive accept/reject decisions. Validate gradients
against perturbation experiments before using them in a calibration problem.

## Continue with related tools

- [Scientific workflows](workflows.md): storage policies, restart, calibration, and sensitivities.
- [PDEs](pde.md): spatial discretization that produces a large ODE system.
- [Stochastic methods](stochastic.md): stochastic integration and its different callback order.
- [Root finding](rootfind.md): nonlinear equations behind implicit and boundary value methods.
- [ODE API](../api/ode.md): individual method signatures and assumptions.
