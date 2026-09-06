#!/usr/bin/env python3
"""Validated end-to-end solver workloads against SciPy, single-threaded.

Optional dependencies: NumPy and SciPy. Matching problems and requested
tolerances do not imply identical stopping rules or iteration counts.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CASES = ("brent", "quad", "ode", "spline", "lbfgs")


def worker(case, provider, repeat):
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from scipy import integrate, interpolate, optimize
    import quadrivium as q
    from quadrivium import numeric as a
    ref = provider == "reference"
    lib = np if ref else a
    info = {}
    if case == "brent":
        f = lambda x: x**3 - 2*x - 5
        op = (lambda: optimize.brentq(f, 1., 3., xtol=1e-12, rtol=1e-12, full_output=True)) if ref else (lambda: q.brent(f, 1., 3., tol=1e-12))
        def validate(out):
            root, ok = (out[0], out[1].converged) if ref else (out.root, out.converged)
            assert ok and abs(f(root)) < 1e-10
        size = "cubic root on [1,3], tol=1e-12"
    elif case == "quad":
        f = lambda x: math.exp(-x*x)
        op = (lambda: integrate.quad(f, -math.inf, math.inf, epsabs=1e-10, epsrel=1e-10)) if ref else (lambda: q.quad(f, -math.inf, math.inf, tol=1e-10))
        def validate(out):
            value = out[0] if ref else out.value
            assert abs(value-math.sqrt(math.pi)) < 1e-9
            if not ref:
                assert out.converged
        size = "Gaussian integral over real line, tol=1e-10"
    elif case == "ode":
        f = lambda t, y: -2*y
        op = (lambda: integrate.solve_ivp(f, (0, 5), [1.], rtol=1e-9, atol=1e-12)) if ref else (lambda: q.solve_ivp(f, (0, 5), [1.], rtol=1e-9, atol=1e-12))
        def validate(out):
            assert out.success
            y = out.y[0, -1] if ref else out.y[-1, 0]
            assert abs(y-math.exp(-10)) < 1e-11
            info["output_steps"] = len(out.t)-1
        size = "y'=-2y, t=[0,5], rtol=1e-9, atol=1e-12"
    elif case == "spline":
        x = lib.linspace(0, 1, 10000)
        y = lib.sin(7*x)
        points = lib.linspace(0, 1, 1001)
        op = (lambda: interpolate.CubicSpline(x, y, bc_type="natural")) if ref else (lambda: q.cubic_spline(x, y, bc="natural"))
        expected = interpolate.CubicSpline(np.linspace(0, 1, 10000), np.sin(7*np.linspace(0, 1, 10000)), bc_type="natural")(np.linspace(0, 1, 1001))
        def validate(out):
            np.testing.assert_allclose(np.asarray(out(points)), expected, atol=1e-10, rtol=1e-10)
        size = "natural cubic spline construction, 10000 knots"
    else:
        weights = lib.linspace(1, 10, 100)
        x = lib.ones(100)*3
        f = lambda x: float(lib.sum(weights*(x-1)**2))
        grad = lambda x: 2*weights*(x-1)
        op = (lambda: optimize.minimize(f, x, jac=grad, method="L-BFGS-B", options={"gtol":1e-8, "ftol":1e-12, "maxiter":1000, "maxcor":10})) if ref else (lambda: q.lbfgs(f, x, grad_f=grad, tol=1e-8, ftol=1e-12, max_iter=1000))
        def validate(out):
            assert out.success if ref else out.converged
            assert out.fun < 1e-10
            info["iterations"] = int(out.nit if ref else out.iterations)
            info["objective"] = float(out.fun)
        size = "100D diagonal quadratic, analytic gradient, gtol=1e-8, ftol=1e-12"
    with q.accel.disabled() if provider == "python" else q.accel.enabled():
        if provider == "rust" and not q.accel.available():
            raise RuntimeError("Rust unavailable")
        validate(op())
        start = time.perf_counter()
        op()
        batch = max(1, min(1000, int(.02/max(time.perf_counter()-start, 1e-9))))
        samples = []
        for _ in range(repeat):
            start = time.perf_counter_ns()
            for _ in range(batch):
                op()
            samples.append((time.perf_counter_ns()-start)/batch/1e6)
    return dict(case=case, provider=provider, size=size, valid=True, batch=batch,
                samples_ms=samples, median_ms=statistics.median(samples),
                p95_ms=float(np.percentile(samples, 95)), **info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=CASES)
    parser.add_argument("--provider", choices=("python", "rust", "reference"))
    parser.add_argument("--repeat", type=int, default=11)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeat < 2:
        parser.error("repeat must be >=2")
    if args.worker:
        print(json.dumps(worker(args.worker, args.provider, args.repeat)))
        return
    if args.output is None:
        parser.error("output is required")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    report = {"threads": 1, "repeat": args.repeat, "results": []}
    for case in CASES:
        for provider in ("python", "rust", "reference"):
            try:
                proc = subprocess.run([sys.executable, __file__, "--worker", case, "--provider", provider, "--repeat", str(args.repeat)], env=env, capture_output=True, text=True, timeout=120)
                result = json.loads(proc.stdout) if proc.returncode == 0 else dict(case=case, provider=provider, error=proc.stderr)
            except subprocess.TimeoutExpired:
                result = dict(case=case, provider=provider, error="timeout 120s")
            report["results"].append(result)
            args.output.write_text(json.dumps(report, indent=2)+"\n")
            print(case, provider, result.get("median_ms", result.get("error")), flush=True)
    return int(any("error" in r for r in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
