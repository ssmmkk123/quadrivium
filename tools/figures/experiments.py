"""Small, reproducible experiments that answer one practical question each.

Quadrivium computes every numerical method being illustrated. NumPy is used
only for plot coordinates, analytic references and diagnostic reductions.
Caches share computed results between light, dark and optional PNG renders.
No timing claim is made: work axes count actual callback invocations.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
from functools import lru_cache
import math

import numpy as np
import quadrivium as qd
from quadrivium import numeric as qn
from quadrivium.core import CountedFunction

from . import figure
from figstyle import finish, line, panels, truth

# Only display clipping uses this floor; metrics retain their unmodified values.
FLOOR = 1e-17


def positive(values):
    return np.maximum(np.asarray(values, dtype=float), FLOOR)


def array(values):
    return np.asarray(values, dtype=float)


@lru_cache(maxsize=None)
def decay_runs():
    counts = (4, 8, 16, 32, 64, 128)
    results = []
    for name, method in (("Euler", qd.ode.euler), ("Heun", qd.ode.heun),
                         ("RK4", qd.ode.rk4)):
        errors, calls = [], []
        for n in counts:
            rhs = CountedFunction(lambda t, y: -y)
            sol = method(rhs, (0, 2), [1.0], n=n)
            errors.append(abs(float(sol.y[-1, 0]) - math.exp(-2)))
            calls.append(rhs.calls)
        results.append((name, errors, calls))
    return np.asarray(counts), results


@figure("start-decay-validation", "getting-started.md",
        "Adaptive exponential decay solution and dense-output error against exp(-t).",
        parameters={"equation": "y'=-y; y(0)=1", "t_span": [0, 4],
                    "rtol": 1e-6, "atol": 1e-9, "error_grid_points": 301})
def start_decay(fig, scheme):
    sol = qd.ode.solve_ivp(lambda t, y: -y, (0, 4), [1.0],
                          rtol=1e-6, atol=1e-9)
    t = np.linspace(0, 4, 301)
    computed = array(sol(t))[:, 0]
    exact = np.exp(-t)
    a, b = panels(fig, "A plausible curve still needs an accuracy check",
                  "Adaptive solve of y′ = −y, y(0) = 1; rtol = 10⁻⁶ and atol = 10⁻⁹.")
    truth(a, t, exact, scheme, "exp(−t)")
    line(a, sol.t, sol.y[:, 0], scheme, markers=True,
         label=f"Solver output ({len(sol.t)} times)")
    finish(a, "Solution and retained time points", "Time t", "State y(t)")
    line(b, t, positive(abs(computed - exact)), scheme,
         label="Dense-output absolute error")
    line(b, t, 1e-9 + 1e-6 * abs(exact), scheme, 1,
         label="atol + rtol |exact| (scale)")
    finish(b, "Measure error between saved points too", "Time t", "Absolute error / scale", logy=True)


@figure("core-roundoff-budget", "guides/core.md",
        "Subtractive cancellation in exp(h)-1 compared with expm1 and a 60-digit reference.",
        parameters={"h": "61 log-spaced points, 1e-16 to 1e-1",
                    "reference": "Decimal.exp, precision=60", "display_floor": FLOOR})
def roundoff(fig, scheme):
    h = np.logspace(-16, -1, 61)
    with localcontext() as ctx:
        ctx.prec = 60
        ref = np.array([float(Decimal(str(x)).exp() - 1) for x in h])
    naive = array(qn.exp(qn.array(h))) - 1
    stable = array(qn.expm1(qn.array(h)))
    a, b = panels(fig, "The formula can spend the entire error budget",
                  "Identical mathematics, different subtraction; reference values use 60-digit decimal arithmetic.")
    truth(a, h, ref, scheme, "60-digit reference")
    line(a, h, positive(naive), scheme, 1, label="exp(h) − 1")
    line(a, h, stable, scheme, 0, label="expm1(h)")
    finish(a, "Small increments must survive rounding", "Increment h", "Computed exponential increment", logx=True, logy=True)
    for i, (label, values) in enumerate((("expm1(h)", stable), ("exp(h) − 1", naive))):
        line(b, h, positive(abs(values - ref) / ref), scheme, i, label=label)
    finish(b, "Relative error exposes lost digits", "Increment h", "Relative error (zeros at 10⁻¹⁷)", logx=True, logy=True)


@figure("numeric-grid-resolution", "guides/numeric.md",
        "Linear reconstruction of a sampled sine wave and its dense-grid RMS error under refinement.",
        parameters={"signal": "sin(6*pi*x), x in [0,1]", "point_counts": [9, 17, 33, 65, 129, 257],
                    "validation_points": 2001, "routines": ["numeric.linspace", "numeric.sin", "numeric.interp"]})
def grid_resolution(fig, scheme):
    dense = np.linspace(0, 1, 2001)
    ref = np.sin(6 * np.pi * dense)
    a, b = panels(fig, "A correctly shaped array can still miss the signal",
                  "Three sine-wave cycles on [0, 1]; numeric.interp reconstructs between equally spaced samples.")
    truth(a, dense, ref, scheme, "sin(6πx)")
    counts, errors = [9, 17, 33, 65, 129, 257], []
    for n in counts:
        x = qn.linspace(0, 1, n)
        y = qn.sin(6 * qn.pi * x)
        fitted = array(qn.interp(dense, x, y))
        errors.append(np.sqrt(np.mean((fitted - ref)**2)))
        if n in (9, 33):
            line(a, x, y, scheme, 0 if n == 9 else 1, markers=True,
                 label=f"{n} points / {n - 1} intervals")
    finish(a, "The coarse grid misses the extrema", "Position x", "Signal amplitude")
    line(b, np.asarray(counts) - 1, errors, scheme, markers=True, label="Dense-grid RMS error")
    truth(b, np.asarray(counts) - 1, errors[-1] * (256 / (np.asarray(counts) - 1))**2,
          scheme, "Second-order reference")
    finish(b, "Refine and compare on a separate grid", "Grid intervals", "RMS interpolation error", logx=True, logy=True)


@figure("linalg-residual-sensitivity", "guides/linalg.md",
        "A fixed right-hand-side perturbation produces condition-dependent solution error despite a small residual.",
        parameters={"matrix": "diag(1, 1/kappa)", "exact_solution": [1, 1],
                    "kappa": "1 to 1e10", "rhs_perturbation": [0, 1e-10],
                    "residual_definition": "||A*x_computed-b_original||_2 / ||b_original||_2"})
def residual_sensitivity(fig, scheme):
    condition = np.logspace(0, 10, 21)
    residual, forward, second = [], [], []
    for kappa in condition:
        A = qn.diag([1.0, 1.0 / kappa])
        exact = qn.ones(2)
        rhs = A @ exact
        changed = rhs + qn.array([0.0, 1e-10])
        computed = qd.linalg.solve(A, changed)
        residual.append(float(qn.linalg.norm(A @ computed - rhs) / qn.linalg.norm(rhs)))
        forward.append(float(qn.linalg.norm(computed - exact) / qn.linalg.norm(exact)))
        second.append(float(computed[1]))
    a, b = panels(fig, "A small residual does not establish an accurate solution",
                  "Solve diag(1, 1/κ)x = b after adding 10⁻¹⁰ to b₂; the unperturbed solution is (1, 1).")
    line(a, condition, residual, scheme, label="Residual against original b", markers=True)
    line(a, condition, forward, scheme, 1, label="Relative solution error", markers=True)
    finish(a, "Sensitivity grows with condition number", "2-norm condition number κ", "Relative error", logx=True, logy=True)
    line(b, condition, second, scheme, 1, label="Computed second component", markers=True)
    truth(b, condition, np.ones_like(condition), scheme, "Exact second component = 1")
    finish(b, "The weakly constrained component moves", "2-norm condition number κ", "Solution component x₂", logx=True)


@lru_cache(maxsize=None)
def root_runs():
    tolerances = (1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13)
    runs = []
    for name in ("Bisection", "Brent", "Newton"):
        errors, calls, derivatives = [], [], []
        for tol in tolerances:
            f = CountedFunction(lambda x: x*x - 2)
            df = CountedFunction(lambda x: 2*x)
            if name == "Newton":
                result = qd.rootfind.newton(f, 2.0, df, tol=tol)
            else:
                method = qd.rootfind.bisection if name == "Bisection" else qd.rootfind.brent
                result = method(f, 1.0, 2.0, tol=tol)
            errors.append(abs(result.root - math.sqrt(2)))
            calls.append(f.calls)
            derivatives.append(df.calls)
        runs.append((name, errors, calls, derivatives))
    return tolerances, runs


@figure("rootfind-accuracy-work", "guides/rootfind.md",
        "Measured sqrt(2) root error versus requested tolerance and actual function/derivative calls.",
        parameters={"function": "x*x-2", "bracket": [1, 2], "newton_start": 2,
                    "tolerances": [1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13],
                    "work": "CountedFunction f calls; Newton derivative calls shown separately", "display_floor": FLOOR})
def roots(fig, scheme):
    tolerances, runs = root_runs()
    a, b = panels(fig, "Compare achieved accuracy and the work it required",
                  "Solve x² − 2 = 0 on [1, 2]; Newton starts at 2 and receives the analytic derivative.")
    for i, (name, errors, calls, derivatives) in enumerate(runs):
        line(a, tolerances, positive(errors), scheme, i, markers=True, label=name)
        line(b, calls, positive(errors), scheme, i, markers=True,
             label=name + (" (f calls only)" if name == "Newton" else ""))
    line(b, np.array(runs[-1][2]) + np.array(runs[-1][3]), positive(runs[-1][1]),
         scheme, 3, markers=True, label="Newton (f + derivative calls)")
    finish(a, "Tolerance is a stopping input", "Requested tolerance", "Absolute root error (zeros at 10⁻¹⁷)", logx=True, logy=True)
    finish(b, "Callback counts reveal the cost", "Actual callback calls", "Absolute root error", logy=True)


@figure("interpolate-node-choice", "guides/interpolate.md",
        "Off-node error of barycentric interpolation of the Runge function on uniform and Chebyshev grids.",
        parameters={"function": "1/(1+25*x*x), [-1,1]", "node_counts": [5, 9, 13, 17, 21, 25],
                    "profile_node_count": 17, "validation_points": 2001, "display_floor": FLOOR})
def interpolation(fig, scheme):
    dense = np.linspace(-1, 1, 2001)
    target = lambda x: 1 / (1 + 25*x*x)
    ref = target(dense)
    counts = [5, 9, 13, 17, 21, 25]
    a, b = panels(fig, "Exact values at the nodes can hide large errors between them",
                  "The same barycentric polynomial algorithm fits 1/(1 + 25x²); only the node placement changes.")
    for i, kind in enumerate(("Uniform", "Chebyshev")):
        errors = []
        for n in counts:
            x = qn.linspace(-1, 1, n) if i == 0 else qd.interpolate.chebyshev_nodes(n)
            p = qd.interpolate.barycentric(x, target(x))
            err = abs(array(p(dense)) - ref)
            errors.append(max(err))
            if n == 17:
                line(a, dense, positive(err), scheme, i, label=f"{kind}, 17 nodes")
        line(b, counts, errors, scheme, i, markers=True, label=kind)
    finish(a, "Inspect the entire approximation interval", "Position x", "Absolute error (values below 10⁻⁸ off scale)", logy=True)
    a.set_ylim(1e-8, 30)
    finish(b, "More nodes can make a poor choice worse", "Interpolation nodes", "Maximum error on 2,001-point grid", logy=True)


@figure("approx-fit-generalization", "guides/approx.md",
        "Training residual and independent truth error of polynomial fits to seeded noisy sine data.",
        parameters={"signal": "sin(pi*x), [-1,1]", "training_points": 25,
                    "noise_std": 0.08, "rng": "quadrivium.numeric.default_rng(20260911)",
                    "degrees": "1 through 16", "validation_points": 501})
def approximation(fig, scheme):
    x = qn.linspace(-1, 1, 25)
    y = qn.sin(qn.pi*x) + 0.08 * qn.random.default_rng(20260911).standard_normal(25)
    dense = np.linspace(-1, 1, 501)
    ref = np.sin(np.pi*dense)
    train, check = [], []
    a, b = panels(fig, "A tighter fit to the observations can be a worse model",
                  "25 sine samples with Gaussian noise σ = 0.08; fixed seed 20260911; Chebyshev-basis least squares.")
    truth(a, dense, ref, scheme, "Noise-free sine")
    a.scatter(array(x), array(y), s=17, color=scheme.secondary,
              marker="x", label="Noisy observations", zorder=4)
    for degree in range(1, 17):
        fit = qd.approx.chebyshev_fit(x, y, degree=degree, domain=(-1, 1))
        predicted = array(fit(dense))
        train.append(np.sqrt(np.mean((array(fit(x)) - array(y))**2)))
        check.append(np.sqrt(np.mean((predicted - ref)**2)))
        if degree in (5, 16):
            line(a, dense, predicted, scheme, 0 if degree == 5 else 1,
                 label=f"Degree {degree}")
    finish(a, "Compare predictions with the known signal", "Position x", "Signal / fitted value")
    line(b, range(1, 17), train, scheme, markers=True, label="Training RMS residual")
    line(b, range(1, 17), check, scheme, 1, markers=True, label="Dense-grid RMS truth error")
    finish(b, "Training error alone favors too much complexity", "Polynomial degree", "RMS error", logy=True)


@figure("diff-step-selection", "guides/diff.md",
        "Derivative error for exp at x=1 as the finite-difference or complex step changes, and callback costs.",
        parameters={"function": "exp(x), derivative at x=1", "h": "61 points, 1e-16 to 1e-1",
                    "reference": "math.e", "display_floor": FLOOR})
def differentiation(fig, scheme):
    steps = np.logspace(-16, -1, 61)
    a, b = panels(fig, "Choose the derivative step by measuring the error curve",
                  "Differentiate exp(x) at x = 1; exact derivative e. Zero errors are plotted at 10⁻¹⁷.")
    methods = [("Forward", qd.diff.forward_difference), ("Central", qd.diff.central_difference),
               ("Complex step", qd.diff.complex_step_derivative)]
    calls, best_errors = [], []
    for i, (name, method) in enumerate(methods):
        errors = []
        for h in steps:
            counted = CountedFunction(qn.exp)
            value = method(counted, 1.0, h=float(h))
            errors.append(abs(float(value) - math.e))
        calls.append(counted.calls)
        best_errors.append(min(errors))
        line(a, steps, positive(errors), scheme, i, label=name)
    finish(a, "Shrinking h eventually amplifies rounding", "Derivative step h", "Absolute derivative error", logx=True, logy=True)
    for i, ((name, _), count, error) in enumerate(zip(methods, calls, best_errors)):
        b.barh(i, count, color=scheme.series[i], height=0.58, hatch=("", "//", "..")[i])
        b.text(count + 0.07, i, f"{count} call{'s' if count != 1 else ''}", va="center", fontsize=9)
    b.set_yticks(range(3), [name for name, _ in methods])
    b.invert_yaxis()
    b.set_xlim(0, max(calls) + 1)
    finish(b, "Count work per derivative estimate", "Actual calls to exp", "Method", legend=False)
    b.grid(False, axis="y")


@lru_cache(maxsize=None)
def quadrature_runs():
    data = []
    for name, method, counts in (
        ("Trapezoid", qd.integrate.trapezoid_rule, (2, 4, 8, 16, 32, 64, 128)),
        ("Simpson", qd.integrate.simpson_rule, (2, 4, 8, 16, 32, 64, 128)),
        ("Gauss–Legendre", qd.integrate.gauss_legendre, (2, 3, 4, 5, 6, 8, 10))):
        calls, errors = [], []
        for n in counts:
            f = CountedFunction(math.exp)
            result = method(f, 0, 1, n=n)
            calls.append(f.calls)
            errors.append(abs(float(result.value) - math.expm1(1)))
        data.append((name, calls, errors))
    return data


@figure("integrate-accuracy-budget", "guides/integrate.md",
        "Quadrature of exp(x): analytic area reference and achieved absolute error versus counted integrand calls.",
        parameters={"integrand": "exp(x) on [0,1]", "exact_integral": "e-1",
                    "composite_intervals": [2, 4, 8, 16, 32, 64, 128],
                    "gauss_nodes": [2, 3, 4, 5, 6, 8, 10], "display_floor": FLOOR})
def integration(fig, scheme):
    a, b = panels(fig, "Spend function evaluations where they buy accuracy",
                  "Integrate exp(x) from 0 to 1; the analytic answer is e − 1. Work counts actual integrand calls.")
    x = np.linspace(0, 1, 301)
    truth(a, x, np.exp(x), scheme, "exp(x)")
    grid = np.linspace(0, 1, 5)
    line(a, grid, np.exp(grid), scheme, markers=True, label="4-interval trapezoid interpolant")
    a.fill_between(x, np.exp(x), color=scheme.series[0], alpha=0.12)
    finish(a, "The integrand is smooth across the interval", "Integration coordinate x", "Integrand f(x)")
    for i, (name, calls, errors) in enumerate(quadrature_runs()):
        line(b, calls, positive(errors), scheme, i, markers=True, label=name)
    finish(b, "Achieved error and the rounding plateau", "Actual integrand calls", "Absolute integral error (zeros at 10⁻¹⁷)", logx=True, logy=True)


@figure("ode-step-budget", "guides/ode.md",
        "Euler, Heun and RK4 on exponential decay, showing coarse trajectories and endpoint error per RHS call.",
        parameters={"equation": "y'=-y, y(0)=1, t in [0,2]", "profile_steps": 4,
                    "step_counts": [4, 8, 16, 32, 64, 128], "work": "CountedFunction RHS calls"})
def ode_budget(fig, scheme):
    a, b = panels(fig, "Higher order can buy more accuracy per right-hand-side call",
                  "Same initial-value problem y′ = −y, y(0) = 1, same interval [0, 2]; cost excludes wall-clock time.")
    t = np.linspace(0, 2, 301)
    truth(a, t, np.exp(-t), scheme, "exp(−t)")
    for i, (name, method) in enumerate((("Euler", qd.ode.euler), ("Heun", qd.ode.heun), ("RK4", qd.ode.rk4))):
        sol = method(lambda t, y: -y, (0, 2), [1.0], n=4)
        line(a, sol.t, sol.y[:, 0], scheme, i, markers=True, label=name)
    finish(a, "Four steps make the method differences visible", "Time t", "State y(t)")
    for i, (name, errors, calls) in enumerate(decay_runs()[1]):
        line(b, calls, positive(errors), scheme, i, markers=True, label=name)
    finish(b, "Refine from 4 to 128 steps", "Actual RHS calls", "Absolute endpoint error at t = 2", logx=True, logy=True)


@lru_cache(maxsize=None)
def diffusion_runs():
    results = []
    for n in (10, 20, 40, 80):
        sol = qd.pde.heat_crank_nicolson(lambda x: math.sin(math.pi*x), 0.1,
                                       (0, 1), (0, 0.5), nx=n, nt=2*n)
        x = array(sol.grids[0])
        exact = np.exp(-0.1*np.pi**2*0.5) * np.sin(np.pi*x)
        results.append((n, sol, max(abs(array(sol.u[-1]) - exact))))
    return results


@figure("pde-diffusion-refinement", "guides/pde.md",
        "Crank–Nicolson heat profiles versus the analytic sine mode and measured coupled space/time refinement.",
        parameters={"equation": "u_t=0.1*u_xx; x in [0,1]; t in [0,0.5]",
                    "initial": "sin(pi*x)", "boundary": "zero Dirichlet",
                    "space_intervals": [10, 20, 40, 80], "time_steps": "2*nx"})
def diffusion(fig, scheme):
    runs = diffusion_runs()
    a, b = panels(fig, "Stable diffusion still needs a grid-convergence check",
                  "Crank–Nicolson, uₜ = 0.1uₓₓ, u(x,0) = sin(πx), zero boundary values; exact mode decays exponentially.")
    sol = runs[1][1]
    x = array(sol.grids[0])
    for i, (t, idx) in enumerate(((0.0, 0), (0.25, 20), (0.5, 40))):
        line(a, x, sol.u[idx], scheme, i, markers=True, label=f"Computed t = {t:g}")
        dense = np.linspace(0, 1, 301)
        truth(a, dense, np.exp(-0.1*np.pi**2*t)*np.sin(np.pi*dense), scheme,
              "Analytic reference" if i == 0 else None)
    finish(a, "Profiles on the 20-interval grid", "Position x", "Temperature u(x,t)")
    ns = np.array([r[0] for r in runs])
    errors = np.array([r[2] for r in runs])
    line(b, ns, errors, scheme, markers=True, label="Maximum endpoint error")
    truth(b, ns, errors[0]*(ns[0]/ns)**2, scheme, "Second-order reference")
    finish(b, "Refine space and time together", "Space intervals (time steps = 2 × intervals)", "Maximum error at t = 0.5", logx=True, logy=True)


@figure("optimize-scaling-paths", "guides/optimize.md",
        "Gradient descent and BFGS paths on an anisotropic quadratic with their objective histories.",
        parameters={"objective": "(x*x+20*y*y)/2", "start": [3, 1.5],
                    "analytic_gradient": ["x", "20*y"], "gradient_descent_lr": 0.08,
                    "max_iter": 160, "tol": 1e-9, "display_floor": FLOOR})
def optimization(fig, scheme):
    f = lambda v: 0.5*(v[0]**2 + 20*v[1]**2)
    grad = lambda v: qn.array([v[0], 20*v[1]])
    runs = [("Gradient descent, lr = 0.08", qd.optimize.gradient_descent(
        f, [3, 1.5], grad, lr=0.08, tol=1e-9, max_iter=160)),
        ("BFGS", qd.optimize.bfgs(f, [3, 1.5], grad, tol=1e-9, max_iter=160))]
    a, b = panels(fig, "Unequal curvature makes a single learning rate expensive",
                  "Minimize (x² + 20y²)/2 from (3, 1.5), using analytic gradients; the exact minimizer is (0, 0).")
    xx, yy = np.meshgrid(np.linspace(-0.3, 3.2, 150), np.linspace(-1.1, 1.7, 120))
    contours = a.contour(xx, yy, 0.5*(xx**2 + 20*yy**2), levels=[0.1, 0.5, 2, 5, 10, 20],
                        colors=scheme.secondary, linewidths=0.65, alpha=0.55)
    a.clabel(contours, fontsize=7, fmt="%g")
    for i, (name, result) in enumerate(runs):
        path = array(result.history)
        line(a, path[:, 0], path[:, 1], scheme, i, markers=True, label=name, markersize=3)
        values = [float(f(p)) for p in path]
        line(b, range(len(path)), positive(values), scheme, i,
             label=f"{name.split(',')[0]} ({result.iterations} iterations)")
    a.scatter([0], [0], color=scheme.ink, marker="*", s=85, zorder=5, label="Exact minimum")
    finish(a, "Iterates cross the narrow valley", "Parameter x", "Parameter y")
    finish(b, "Objective gap against iterations", "Iteration (0 is the initial point)", "f(xₖ) − f* (zeros at 10⁻¹⁷)", logy=True)


@figure("transforms-sampling-spectrum", "guides/transforms.md",
        "Sampling a two-tone signal at 64 and 16 Hz demonstrates one-sided FFT amplitude normalization and aliasing.",
        parameters={"signal": "sin(2*pi*5*t)+0.4*sin(2*pi*11*t)", "duration_seconds": 1,
                    "sample_rates_hz": [64, 16], "endpoint": False,
                    "amplitude": "2*abs(rfft(x))/N; DC and Nyquist divided by 2"})
def sampling_spectrum(fig, scheme):
    a, b = panels(fig, "Sampling sets the frequency information an FFT can recover",
                  "A 5 Hz sine plus a 0.4-amplitude 11 Hz sine, observed for one second; no window and no endpoint duplicate.")
    dense = np.linspace(0, 0.5, 1001)
    signal = lambda t: np.sin(2*np.pi*5*t) + 0.4*np.sin(2*np.pi*11*t)
    truth(a, dense, signal(dense), scheme, "Continuous two-tone signal")
    for i, rate in enumerate((64, 16)):
        t = np.arange(rate) / rate
        sampled = signal(t)
        take = t <= 0.5
        line(a, t[take], sampled[take], scheme, i, markers=True, label=f"{rate} samples/s", linewidth=1.1)
        spectrum = np.asarray(qd.transforms.rfft(sampled))
        amplitude = 2*abs(spectrum)/rate
        amplitude[0] /= 2
        amplitude[-1] /= 2
        frequency = np.arange(amplitude.size)
        line(b, frequency, amplitude, scheme, i, markers=True, label=f"{rate} samples/s")
    b.scatter([5, 11], [1, 0.4], marker="x", color=scheme.ink, s=58, zorder=5, label="True tone amplitudes")
    finish(a, "Samples over the first half second", "Time (seconds)", "Signal amplitude")
    finish(b, "At 16 Hz, 11 Hz folds onto −5 Hz", "Frequency (Hz)", "One-sided amplitude")
    b.set_xlim(0, 15)
    b.set_ylim(-0.04, 1.15)


@lru_cache(maxsize=None)
def monte_carlo_runs():
    counts = np.array([64, 256, 1024, 4096])
    errors, estimates = [], []
    for n in counts:
        runs = [qd.integrate.monte_carlo(math.exp, 0, 1, n=int(n), rng=seed)
                for seed in range(40)]
        errors.append(np.array([float(r.value) - math.expm1(1) for r in runs]))
        estimates.append(np.array([float(r.error_estimate) for r in runs]))
    return counts, errors, estimates


@figure("stochastic-error-calibration", "guides/stochastic.md",
        "Repeated seeded Monte Carlo integration compares empirical RMS error with reported standard error and individual error bars.",
        parameters={"integrand": "exp(x) on [0,1]", "sample_counts": [64, 256, 1024, 4096],
                    "seeds": "integers 0 through 39", "ensemble_size": 40,
                    "error_bars": "estimate minus exact, plus/minus 2 reported standard errors; first 12 seeds at N=1024"})
def stochastic(fig, scheme):
    counts, errors, estimates = monte_carlo_runs()
    rms = np.array([np.sqrt(np.mean(e**2)) for e in errors])
    reported = np.array([np.sqrt(np.mean(e**2)) for e in estimates])
    a, b = panels(fig, "Monte Carlo error estimates describe repeated-sample uncertainty",
                  "Integrate exp(x) on [0, 1]; 40 independent fixed seeds (0–39) at each sample count.")
    line(a, counts, rms, scheme, markers=True, label="Empirical RMS actual error")
    line(a, counts, reported, scheme, 1, markers=True, label="RMS reported standard error")
    truth(a, counts, rms[0]*np.sqrt(counts[0]/counts), scheme, "N⁻¹ᐟ² reference")
    finish(a, "Four times the samples roughly halves the error", "Samples / integrand evaluations N", "Integral error", logx=True, logy=True)
    b.errorbar(np.arange(12), errors[2][:12], yerr=2*estimates[2][:12],
               fmt="o", color=scheme.series[0], capsize=3, markersize=4,
               linewidth=1.1, label="Estimate error ± 2 standard errors")
    b.axhline(0, color=scheme.ink, linestyle=":", linewidth=1.2, label="Exact integral")
    finish(b, "Individual estimates fluctuate (N = 1,024)", "Random seed", "Estimated integral − (e − 1)")


@figure("special-identity-checks", "guides/special.md",
        "Spherical Bessel j0 is checked against sin(x)/x, while complementary-error-function formulas are checked against math.erfc.",
        parameters={"bessel_reference": "j0(x)=sin(x)/x, 0.1<=x<=20",
                    "erfc_reference": "Python math.erfc, 0<=x<=8", "display_floor": FLOOR})
def special_checks(fig, scheme):
    a, b = panels(fig, "Check special functions and preserve small tail probabilities",
                  "Spherical Bessel j₀(x) = sin(x)/x; complementary error functions are compared with Python math.erfc.")
    x = np.linspace(0.1, 20, 301)
    ref = np.sin(x)/x
    computed = array(qd.special.spherical_bessel_j(0, qn.array(x)))
    line(a, x, computed, scheme, label="Quadrivium spherical_bessel_j(0, x)")
    truth(a, x[::8], ref[::8], scheme, "sin(x)/x reference")
    maxerr = max(abs(computed - ref))
    finish(a, f"Closed-form check: max error {maxerr:.1e}", "Argument x", "Spherical Bessel j₀(x)")
    x = np.linspace(0, 8, 161)
    ref = np.array([math.erfc(v) for v in x])
    direct = array(qd.special.erfc(qn.array(x)))
    subtracted = 1 - array(qd.special.erf(qn.array(x)))
    line(b, x, positive(abs(direct - ref)/ref), scheme, label="erfc(x)")
    line(b, x, positive(abs(subtracted - ref)/ref), scheme, 1, label="1 − erf(x)")
    finish(b, "Subtracting from one loses the small tail", "Argument x", "Relative tail error (zeros at 10⁻¹⁷)", logy=True)


@lru_cache(maxsize=None)
def recovery_runs():
    exact_rate, final_time = 1.7, 2.0
    target = math.exp(-exact_rate*final_time)
    results = []
    for tolerance in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        def residual(rate):
            sol = qd.ode.solve_ivp(lambda t, y: -rate*y, (0, final_time), [1.0],
                                  rtol=tolerance, atol=tolerance*1e-3)
            return float(sol.y[-1, 0]) - target
        fit = qd.rootfind.brent(residual, 0.2, 3.0, tol=1e-12)
        results.append((tolerance, fit.root, fit.function_calls))
    return results


@figure("workflow-parameter-recovery", "guides/workflows.md",
        "A nested brent/solve_ivp decay calibration demonstrates how inner integration tolerance limits recovered-parameter accuracy.",
        parameters={"equation": "y'=-k*y, y(0)=1", "true_k": 1.7, "target": "y(2)=exp(-3.4)",
                    "outer_bracket": [0.2, 3], "outer_tol": 1e-12,
                    "inner_rtol": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8], "inner_atol": "rtol*1e-3"})
def workflow(fig, scheme):
    results = recovery_runs()
    a, b = panels(fig, "An outer solver cannot recover accuracy lost by an inner solver",
                  "Recover decay rate k from one exact observation y(2) = exp(−3.4); Brent tolerance stays fixed at 10⁻¹².")
    t = np.linspace(0, 2, 301)
    truth(a, t, np.exp(-1.7*t), scheme, "Exact k = 1.7")
    rate = results[0][1]
    sol = qd.ode.solve_ivp(lambda t, y: -rate*y, (0, 2), [1.0], rtol=1e-3, atol=1e-6)
    line(a, sol.t, sol.y[:, 0], scheme, markers=True, label=f"Coarse inner solve: fitted k = {rate:.6f}")
    a.scatter([2], [math.exp(-3.4)], color=scheme.series[1], marker="s", s=48,
              label="Exact observation", zorder=5)
    finish(a, "A visually convincing fit is only the first check", "Time t", "State y(t)")
    tolerances = np.array([r[0] for r in results])
    errors = np.array([abs(r[1] - 1.7) for r in results])
    line(b, tolerances, errors, scheme, markers=True, label="Absolute recovered-rate error")
    finish(b, "Vary the inner IVP tolerance", "Inner solve_ivp relative tolerance", "|Recovered k − 1.7|", logx=True, logy=True)


@figure("validation-refinement-orders", "design.md",
        "Step refinement for Euler and RK4, with observed orders computed from adjacent measured errors.",
        parameters={"equation": "y'=-y, y(0)=1, t in [0,2]", "step_counts": [4, 8, 16, 32, 64, 128],
                    "observed_order": "log2(error_N/error_2N)", "expected_orders": {"Euler": 1, "RK4": 4}})
def validation_orders(fig, scheme):
    counts, runs = decay_runs()
    a, b = panels(fig, "Validate a numerical method by its refinement behavior",
                  "An exact solution provides an error oracle; halving the step should reduce a pth-order method's error by about 2ᵖ.")
    for i, idx in enumerate((0, 2)):
        name, errors, _ = runs[idx]
        errors = np.array(errors)
        line(a, counts, errors, scheme, i, markers=True, label=name)
        measured = np.log2(errors[:-1]/errors[1:])
        line(b, counts[1:], measured, scheme, i, markers=True, label=f"{name}: measured order")
        b.axhline(1 if idx == 0 else 4, color=scheme.series[i], linestyle=":", linewidth=1,
                  label=f"Expected order {1 if idx == 0 else 4}")
    finish(a, "Endpoint errors on successively finer grids", "Time steps N", "|Computed y(2) − exp(−2)|", logx=True, logy=True)
    finish(b, "Estimate order from adjacent refinements", "Fine-grid steps (compared with N/2)", "Observed order log₂(Eₙ/₂ / Eₙ)", logx=True)
    b.set_ylim(0.65, 4.8)
