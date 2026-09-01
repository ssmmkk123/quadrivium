# Getting started

This page covers the conventions shared by all 836 routines: how they are
called, what they return, how tolerances work, and how failure is reported.
Once these are clear, the [guides](guides/linalg.md) and the
[API reference](api/index.md) are enough for the rest.

## The import surface

Every routine lives in one of thirteen subpackages, and the ones reached for
most often are re-exported at the top level.

These pages import the package as `qd`. The natural abbreviation of
*Quadrivium* is `quad`, but `quad` is also the name of the library's
general-purpose integrator, and `quad.quad(f, a, b)` reads like a mistake —
`qd.quad(f, a, b)` does not. Any alias works; the package does not care.


```pycon
>>> import quadrivium as qd
>>> qd.brent is qd.rootfind.brent
True
>>> from quadrivium.linalg import householder_qr      # always available
>>> from quadrivium.ode import dormand_prince

```

There are 146 names at the top level; the other 690 are one import deeper.
When two subpackages have a routine for the same idea, the names differ
(`qd.integrate.gauss_legendre` returns an integral, `qd.approx.gauss_legendre_nodes`
returns nodes and weights), so nothing is shadowed.

## A five-minute tour

```pycon
>>> import numpy as np
>>> import quadrivium as qd

```

Find a root, with a bracket, guaranteed:

```pycon
>>> r = qd.brent(lambda x: x**3 - 2*x - 5, 1, 3)
>>> round(r.root, 12), r.converged
(2.094551481542, True)

```

Integrate over an infinite interval:

```pycon
>>> q = qd.quad(lambda x: np.exp(-x*x), -np.inf, np.inf)
>>> round(float(q), 12) == round(float(np.sqrt(np.pi)), 12)
True

```

Solve an initial value problem to a requested accuracy:

```pycon
>>> sol = qd.solve_ivp(lambda t, y: -2*y, (0, 1), [1.0], rtol=1e-10)
>>> abs(float(sol.y[-1, 0]) - float(np.exp(-2))) < 1e-9
True
>>> abs(float(sol(0.5)[0]) - float(np.exp(-1))) < 1e-8     # dense output
True

```

Minimize a nonconvex function without supplying a gradient:

```pycon
>>> rosen = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
>>> opt = qd.minimize(rosen, [-1.2, 1.0], method="bfgs")
>>> opt.converged, [round(float(v), 6) for v in opt.x]
(True, [1.0, 1.0])

```

Factorize a matrix and check the factorization is exact:

```pycon
>>> A = np.array([[4.0, 3.0], [6.0, 3.0]])
>>> P, L, U = qd.plu_decomposition(A)
>>> float(np.max(np.abs(P @ A - L @ U))) < 1e-15
True

```

Transform a signal:

```pycon
>>> x = np.array([1.0, 2.0, 3.0, 4.0])
>>> X = qd.fft(x)
>>> float(np.max(np.abs(qd.ifft(X) - x))) < 1e-14
True

```

## Calling conventions

**Arrays in, arrays out.** Anything array-shaped may be a list, a tuple, or a
NumPy array; it is converted with `np.asarray(..., dtype=float)` internally.
Results come back as NumPy arrays.

```pycon
>>> qd.solve([[2.0, 1.0], [1.0, 3.0]], [3.0, 5.0]).round(12).tolist()
[0.8, 1.4]

```

**Functions are plain callables.** A scalar problem takes `f(x) -> float`; a
vector problem takes `f(x) -> array`; an ODE right-hand side takes `f(t, y)`.
Nothing needs to be vectorized unless a routine's docstring says so.

**Intervals are tuples.** `t_span=(0, 10)` for an ODE, `(a, b)` as two
positional arguments for quadrature and bracketing methods.

**Derivatives are optional.** Where a method can use a Jacobian, gradient, or
Hessian, the argument (`grad_f`, `jac`, `hess_f`) defaults to `None` and a
finite-difference approximation is used instead. Supplying the exact
derivative is faster and more accurate; not supplying one always works.

```pycon
>>> f = lambda x: x[0]**2 + 3*x[1]**2
>>> a = qd.minimize(f, [1.0, 1.0], method="bfgs")                       # FD gradient
>>> b = qd.minimize(f, [1.0, 1.0], grad_f=lambda x: np.array([2*x[0], 6*x[1]]))
>>> a.converged and b.converged and b.function_calls < a.function_calls
True

```

**Dispatchers take a `method` string.** `qd.solve_ivp`, `qd.minimize`,
`qd.quad`, `qd.linprog`, `qd.polynomial_roots` and friends select an
implementation by name and pass the remaining keywords through; every
underlying routine is also importable and callable directly.

```pycon
>>> qd.solve_ivp(lambda t, y: -y, (0, 1), [1.0], method="rk4", n=50).method
'rk4'
>>> qd.solve_ivp(lambda t, y: -y, (0, 1), [1.0], method="radau", n=50).method
'radau_iia3'

```

## Result records

Results are dataclasses, not tuples: the answer comes with the evidence for
it. Seven records cover the library.

| Record | Answer | Also carries |
| --- | --- | --- |
| `RootResult` | `root` (also as `x`) | `f_root`, `iterations`, `converged`, `function_calls`, `method`, `history`, `message` |
| `IterationResult` | `x` | `iterations`, `converged`, `residuals`, `residual` (the last one), `method`, `message` |
| `QuadratureResult` | `value` | `error_estimate`, `function_calls`, `subintervals`, `converged`, `method` |
| `ODESolution` | `t`, `y`, `y_final` | `n_steps`, `n_accepted`, `n_rejected`, `n_rhs_evals`, `success`, `message`, `dydt`; callable for dense output |
| `OptimizeResult` | `x`, `fun` | `jac`, `hess`, `iterations`, `converged`, `function_calls`, `gradient_calls`, `method`, `history`, `message` |
| `EigenResult` | `eigenvalues`, `eigenvectors` | `iterations`, `converged`, `method`; unpacks as a pair |
| `PDESolution` | `u`, `grids`, `t` | `method`, `iterations`, `converged`, `residuals` |

Two of them behave like the value they carry, so they drop into arithmetic
without ceremony:

```pycon
>>> round(float(qd.quad(lambda x: x**2, 0, 1)), 12)      # QuadratureResult -> float
0.333333333333
>>> values, vectors = qd.jacobi_eigen([[2.0, 1.0], [1.0, 2.0]])   # EigenResult unpacks
>>> sorted(round(float(v), 12) for v in values)
[1.0, 3.0]

```

The extra fields are the point of the records. Cost, convergence, and the
path taken are all inspectable:

```pycon
>>> r = qd.bisection(lambda x: x**2 - 2, 0, 2, tol=1e-10)
>>> r.converged, r.iterations, r.function_calls
(True, 35, 37)
>>> len(r.history) == r.iterations            # every iterate is kept
True

>>> q = qd.adaptive_gauss_kronrod(lambda x: 1/(1 + 25*x**2), -1, 1, tol=1e-12)
>>> q.converged and q.error_estimate < 1e-12
True

>>> sol = qd.solve_ivp(lambda t, y: -50*y, (0, 1), [1.0], rtol=1e-8)
>>> sol.n_accepted > 0 and sol.n_rejected >= 0 and sol.success
True

```

## Tolerances and iteration limits

Iterative routines take `tol` and `max_iter`, with defaults chosen per method
(`1e-12` for bracketing root finders, `1e-10` for optimizers, `1e-14` for
polynomial roots). Adaptive ODE solvers take `rtol` and `atol` instead, and
fixed-step ones take a step count `n`.

```pycon
>>> loose = qd.newton(lambda x: x**2 - 2, 1.0, lambda x: 2*x, tol=1e-6)
>>> tight = qd.newton(lambda x: x**2 - 2, 1.0, lambda x: 2*x, tol=1e-15)
>>> loose.iterations <= tight.iterations
True

```

Asking for more than the arithmetic can deliver is not an error: the method
stops when it stops improving and says so in `message`. A tolerance below
about `1e-16` relative is never achievable in double precision.

## How failure is reported

**Iterative non-convergence is data, not an exception.** A method that runs
out of iterations, stalls, or diverges returns its best answer with
`converged=False` and a `message` explaining what happened. This keeps a
failed solve inspectable instead of unwinding the stack.

```pycon
>>> r = qd.newton(lambda x: x**2 + 1, 1.0, lambda x: 2*x, max_iter=20)
>>> r.converged
False
>>> r.message
'zero derivative encountered'

```

Always check `converged` before trusting a result:

```pycon
>>> r = qd.rootfind.fixed_point(lambda x: 2*x, 1.0, max_iter=50)   # diverges
>>> r.converged
False

```

**Malformed input raises.** Bad arguments, impossible geometry, and singular
matrices raise exceptions from a small hierarchy rooted at `QuadriviumError`,
so one `except` clause catches everything the library throws:

```pycon
>>> from quadrivium.core import QuadriviumError, BracketError, SingularMatrixError
>>> try:
...     qd.bisection(lambda x: x**2 + 1, 0, 1)     # f(0) and f(1) share a sign
... except BracketError as exc:
...     print(exc)
f(a)=1 and f(b)=2 have the same sign: [0.0, 1.0] does not bracket a root
>>> issubclass(BracketError, QuadriviumError) and issubclass(SingularMatrixError, QuadriviumError)
True

```

| Exception | Raised when |
| --- | --- |
| `QuadriviumError` | base class for everything below |
| `ConvergenceError` | a method that cannot return a partial answer failed to converge |
| `SingularMatrixError` | a matrix is singular, or numerically so, for the requested operation |
| `DimensionError` | array shapes are incompatible |
| `DomainError` | an argument lies outside the method's domain of validity |
| `StepSizeError` | an adaptive step size underflowed the minimum allowed |
| `BracketError` | a bracketing method was given an interval that does not bracket a root |

`ConvergenceError` carries `iterations`, `residual`, and `best`, so even the
exception path keeps the partial result.

## Randomness and reproducibility

Every stochastic routine takes `rng=`, passed straight to
`np.random.default_rng`. Give it an integer seed or a `Generator` and the run
is reproducible:

```pycon
>>> a = qd.monte_carlo(lambda x: x**2, 0, 1, n=10_000, rng=0)
>>> b = qd.monte_carlo(lambda x: x**2, 0, 1, n=10_000, rng=0)
>>> float(a) == float(b)
True
>>> abs(float(a) - 1/3) < 4 * a.error_estimate         # within four standard errors
True

```

The historical generators in `quadrivium.stochastic` (`LCG`, `ParkMiller`,
`XorShift`, `MersenneTwister`) are separate: they take a `seed` and reproduce
the classic algorithms exactly, including their defects. They are there to be
studied, not to be used as a source of randomness — use `rng=` for that.

## Performance

The library is written for legibility. The algorithm is in the source at the
level a textbook states it, which costs performance in the places you would
expect: Python-level loops over matrix entries or time steps. Roughly:

- **Dense linear algebra** relies on NumPy's matrix products for its inner
  work, so factorizations are competitive up to a few hundred rows and fall
  behind LAPACK well before a thousand.
- **Krylov and iterative solvers** spend their time in matrix-vector products
  and are close to optimal when the operator is a NumPy array or a
  `matvec` callable.
- **Quadrature, root finding, and optimization** are dominated by your own
  callback. If `f` is a NumPy expression, the library's overhead is small.
- **ODE and PDE time stepping** loops in Python once per step, which is the
  library's slowest pattern: expect roughly 10–100× a compiled integrator on
  the same problem.

For a large sparse solve, a stiff production integrator, or anything in an
inner loop that runs millions of times, reach for a compiled library. For
everything up to moderate size — and for every case where you want to see what
the method did — this is fast enough.

## Interoperating with the rest of the ecosystem

Inputs and outputs are NumPy arrays, so nothing special is needed:

```pycon
>>> import numpy as np
>>> t = np.linspace(0, 1, 5)
>>> sol = qd.solve_ivp(lambda t, y: -y, (0, 1), [1.0], rtol=1e-10)
>>> y = sol(t)                          # dense output on your own grid
>>> y.shape
(5, 1)
>>> float(np.max(np.abs(y[:, 0] - np.exp(-t)))) < 1e-8
True

```

The results are plain dataclasses, so `dataclasses.asdict` serializes them,
and `matplotlib` plots `sol.t` against `sol.y` directly.

## Where next

- The [guides](guides/linalg.md) — one per subpackage, on choosing between
  methods that solve the same problem.
- The [API reference](api/index.md) — every signature, generated from the code.
- [Design and validation](design.md) — how the library decides a method is
  correct.
- [Known limitations](limitations.md) — where a method is genuinely weaker
  than its reputation.
- The [example scripts](examples.md) — six runnable tours of the library.
