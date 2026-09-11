# Finding roots and following solution branches

Root finding solves an equation `f(x)=0` or a system `F(x)=0`. Before choosing
an algorithm, decide whether you know an interval containing a root, a useful
initial guess, a derivative, or the structure of a polynomial. These pieces of
information determine both the method's speed and the reliability of its result.

Most iterative root solvers return `RootResult`. The root is in `.root` (also
`.x`), the evaluated residual in `.f_root`, and termination information in
`.converged` and `.message`. Helper routines such as `find_all_roots` and
`polynomial_roots` instead return arrays. See the
[root-finding reference](../api/rootfind.md) for every signature.

```pycon
>>> from quadrivium import numeric as np
>>> from quadrivium import rootfind as rf
>>> f = lambda x: x*x - 2.0
>>> result = rf.brent(f, 0.0, 2.0)
>>> result.converged, abs(result.root - np.sqrt(2.0)) < 1e-12
(True, True)
>>> abs(result.f_root) < 1e-12
True

```

## Choose the information you can trust

| Information available | Method to try | What to check |
| --- | --- | --- |
| Continuous scalar function and a sign-changing bracket | `brent` | Endpoint signs and continuity |
| Same bracket, simplest predictable interval reduction | `bisection` | Required bracket width |
| Bracket with safeguarded interpolation | `itp`, `ridders`, `illinois`, `pegasus` | Work relative to Brent on your function |
| Good initial guess and derivative | `newton` | Local basin and nonzero derivative |
| Two guesses, derivative expensive | `secant` | Can leave the region of interest |
| First and second derivatives available | `halley` | Extra derivative cost and denominator stability |
| Contractive fixed-point map | `fixed_point`, `aitken_accelerated` | The chosen map actually contracts |
| Vector residual and a reasonable initial state | `damped_newton_system` | Equation scaling and Jacobian quality |
| Polynomial coefficients | `polynomial_roots` | Coefficient order and residuals |
| Parameterized equilibrium branch, possibly with a fold | `pseudo_arclength` | Branch residuals and continuation step size |

There is no general routine that guarantees discovery of every zero of an
arbitrary callable from finitely many samples. Separate root discovery from
root refinement: a good refinement method cannot recover roots never bracketed
or initial guesses that lead to an unintended branch.

## Brackets provide useful structure

A sign change for a continuous real function guarantees at least one zero
between the endpoints. An endpoint that evaluates exactly to zero is accepted.
`BracketError` is raised when both nonzero endpoint values have the same sign.
A sign change at a pole or jump is not evidence of a zero.

```pycon
>>> bracketed = rf.bisection(f, 0.0, 2.0, tol=1e-10)
>>> bracketed.converged and abs(f(bracketed.root)) < 1e-8
True
>>> from quadrivium.core import BracketError
>>> try:
...     rf.brent(lambda x: x*x + 1, -1, 1)
... except BracketError:
...     print("The endpoints do not bracket a real zero.")
The endpoints do not bracket a real zero.

```

Bisection halves the bracket at each step. Brent uses interpolation when it is
safe and falls back to interval reduction. False position can stagnate when
one endpoint stays fixed; Illinois and Pegasus modify the endpoint weighting
to reduce that problem. These methods preserve a bracket but can differ
substantially in the number of function evaluations.

`bracket_root` expands an initial interval until it finds a sign change or
exhausts its budget. The expanded interval may leave the function's valid
domain. Use it only when evaluation outside the initial interval is meaningful.

### Tolerances do not all mean the same thing

Root solvers can stop because the function value is small or because the
location changes little. A residual tolerance is measured in units of `f`;
a location tolerance is measured in units of `x`. Multiplying the equation by
a constant leaves its roots unchanged but changes residual-based stopping.

For a simple root, the local relation is approximately
`location_error = residual / abs(derivative_at_root)`. A tiny residual can
still correspond to an inaccurate location near a flat or multiple root.
Report both the returned location and a freshly evaluated residual. When a
location bound matters, use a trustworthy bracket or problem-specific analysis.

## Newton and secant methods

Newton updates the guess using the tangent line. Supplying `df` avoids finite
difference evaluations and often improves both accuracy and cost. With
`df=None`, Quadrivium approximates the derivative numerically.

```pycon
>>> analytic = rf.newton(f, 1.0, df=lambda x: 2*x)
>>> numerical = rf.newton(f, 1.0)
>>> analytic.converged and numerical.converged
True
>>> abs(analytic.root - numerical.root) < 1e-10
True
>>> analytic.function_calls < numerical.function_calls
True

```

The `damping` parameter multiplies every Newton step by a fixed factor; it is
not an adaptive line search. `multiplicity` applies a known multiplicity
correction for repeated roots. A zero derivative returns a nonconverged
result, and a distant guess can diverge or find a different root.

```pycon
>>> stalled = rf.newton(lambda x: x*x + 1, 0.0, df=lambda x: 2*x)
>>> stalled.converged
False
>>> "zero derivative" in stalled.message
True

```

Near a simple root and with a sufficiently good initial guess, Newton normally
converges quadratically. That is a local asymptotic statement, not a guarantee
for an arbitrary start. A repeated root changes the rate unless multiplicity
is handled. Secant avoids an explicit derivative but also sacrifices the
bracket safeguard. Muller can enter complex arithmetic to locate complex roots.

### Compare work fairly

An iteration may require one or several residual evaluations and additional
derivatives or linear solves. `.function_calls` counts evaluations of the
residual callable on the normal solver path; separately supplied derivative
calls are not added to that count. Wrap expensive derivatives in
`CountedFunction` when comparing total cost.

<figure markdown="span">
  ![Root error plotted against actual function evaluations for several scalar methods](../assets/figures/rootfind-accuracy-work.svg#only-light)
  ![Root error plotted against actual function evaluations for several scalar methods](../assets/figures/rootfind-accuracy-work-dark.svg#only-dark)
  <figcaption>Solvers locate √2 from the bracket [1,2], with Newton starting at 2, over a range of requested tolerances. Actual residual calls and Newton’s additional derivative calls expose work that iteration counts hide. Performance on this smooth problem does not establish robustness for arbitrary starts.</figcaption>
</figure>

## Histories and callbacks

Supported scalar and system solvers expose `store_history`, `history_stride`,
and `callback`. Histories are enabled by default. `store_history=False` avoids
retaining snapshots, while `history_stride=N` retains every Nth appended
snapshot. Algorithms may include an initial state separately, so history length
is not a universal iteration counter.

```pycon
>>> compact = rf.newton(f, 1.0, df=lambda x: 2*x, store_history=False)
>>> compact.history, compact.converged
([], True)
>>> observed = []
>>> def observe(x):
...     observed.append(float(x))
...     return False
>>> monitored = rf.newton(f, 1.0, df=lambda x: 2*x, callback=observe)
>>> monitored.converged and len(observed) > 0
True

```

Callbacks receive a private iterate copy. Returning true or raising
`StopIteration` requests a stop, reported with `converged=False` and a stop
message. The callback runs when the algorithm appends an iterate, not on every
function evaluation. Callback-stop results do not preserve complete ordinary
call accounting, so use your own wrapper if an exact evaluation count matters.

Histories are method-specific. Do not assume that all entries are scalar roots
or that all methods store the same data. Read the API or inspect an entry
before plotting a history. Re-evaluating `f` for a plot is extra diagnostic
work and should not be attributed to the original solver's count.

## Systems of nonlinear equations

For `F: R^n -> R^n`, pass a state vector and return one residual per equation.
An analytic Jacobian has shape `(n, n)`, with `J[i, j] = dF_i/dx_j`.
Finite differences are used when `jac` is omitted.

```pycon
>>> def F(x):
...     return np.array([x[0]**2 + x[1]**2 - 1.0, x[0] - x[1]])
>>> def J(x):
...     return np.array([[2*x[0], 2*x[1]], [1.0, -1.0]])
>>> system = rf.damped_newton_system(F, [0.8, 0.6], jac=J)
>>> system.converged and float(np.linalg.norm(F(system.root))) < 1e-10
True
>>> system.root.shape
(2,)

```

Newton solves a linearized system at each step. Damped Newton searches for a
step reducing the residual, which can improve behavior away from the solution.
Broyden methods update a Jacobian or inverse-Jacobian approximation instead
of rebuilding it. Newton-Krylov methods replace a dense Newton solve with an
iterative product-based calculation where supported.

Scale variables and equations before solving. If one residual is naturally
one million times larger than another, an unscaled norm can effectively ignore
the smaller equation. Validate an analytic Jacobian against
`core.numerical_jacobian` at several representative states, including places
where branches or domain constraints matter.

For fixed-point methods, the residual is associated with `G(x)-x`. Rewriting
an equation as `x=G(x)` changes the iteration even though the fixed points may
be the same. Local contraction requires suitable derivatives; acceleration
cannot turn every divergent map into a reliable solver.

## Polynomial roots

Polynomial root routines use coefficients in **descending powers**. For
example, `[1, 0, -2]` represents `x**2 - 2`. This differs from the ascending
Taylor coefficients used by Padé approximation.

```pycon
>>> coefficients = [1.0, 0.0, -2.0]
>>> roots = rf.polynomial_roots(coefficients)
>>> np.allclose(np.sort(np.real(roots)), [-np.sqrt(2.0), np.sqrt(2.0)])
True
>>> max(abs(rf.horner(coefficients, r)) for r in roots) < 1e-10
True

```

`polynomial_roots` defaults to companion-matrix roots. Durand-Kerner and
Aberth-Ehrlich iterate on all roots; Laguerre targets one root; Bairstow works
with quadratic factors. Horner evaluation, synthetic division, root bounds,
and Sturm-chain helpers support diagnosis and real-root counting.

Root locations can be extremely sensitive to coefficient perturbations,
especially for repeated or clustered roots. A polynomial's degree alone does
not describe its difficulty. Scale the independent variable when coefficients
span a large range, and check residuals relative to
`sum(abs(c[k]) * abs(root)**power_k)` rather than using an arbitrary absolute
threshold. Deflation propagates root errors into subsequent factors.

`find_all_roots` scans a callable on a uniform grid and refines sign changes.
It can miss even-multiplicity roots, multiple roots inside one cell, and
features narrower than the grid. Increase resolution and use polynomial
structure or analytic information when completeness matters.

## Continuation through parameter changes

Solving independently at many parameter values can jump between branches or
fail near a fold. `pseudo_arclength` follows `F(x, parameter)=0` in the combined
state-parameter space, using predictor and corrector steps. The parameter is
allowed to turn around, which is the defining advantage near folds.

```pycon
>>> branch = rf.pseudo_arclength(lambda x, p: [x[0]**2 - p], [1.0], 1.0,
...                             ds=0.05, max_steps=4, direction=-1)
>>> branch.converged
True
>>> float(np.max(branch.residuals)) < 1e-8
True
>>> branch.x.shape, branch.parameters.shape
((5, 1), (5,))

```

`ContinuationResult` stores states, parameters, tangents, residuals, call
counts, and a termination message. Supply `jac` and `parameter_derivative`
when available. Step bounds and the Newton correction budget govern how the
method responds to difficult parts of a branch.

With `stability=True`, the code treats `F` as a dynamical-system right-hand
side and classifies negative real parts of Jacobian eigenvalues as stable.
That interpretation is meaningful only if your residual actually represents
that dynamics. Reported folds and stability changes are interpolated candidate
locations, not certified bifurcation points. Refine and validate interesting
locations independently. See [ODE methods](ode.md) for time evolution and
[differentiation](diff.md) for constructing the required derivatives.
