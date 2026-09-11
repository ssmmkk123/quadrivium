#!/usr/bin/env python3
"""Seeded, independent numerical audit; optional test deps: numpy scipy psutil.

Run from the checkout with ``python tools/audit_library.py --output report.json``.
Each group runs in a fresh process with a deadline. Failures are reported, not
silenced; the parent exits nonzero if any comparison, worker, or deadline fails.
This is an exploratory audit, not a claim of complete NumPy compatibility.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(os.environ.get("QUADRIVIUM_AUDIT_ROOT", Path(__file__).resolve().parents[1]))
GROUPS = ("arrays", "indexing", "linalg", "special", "special_orders", "transforms", "fft_lengths", "algorithms", "stress")


def worker(group, seed, trials):
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import scipy
    from scipy import special as sp, fft as sf
    import quadrivium as q
    from quadrivium import numeric as a

    rng = np.random.default_rng(seed)
    counts = collections.Counter()
    failures = []
    metrics = {}

    def native(x):
        return np.asarray(memoryview(x)) if isinstance(x, a.ndarray) else np.asarray(x)

    def check(label, fn):
        counts["checks"] += 1
        try:
            fn()
            counts["passed"] += 1
        except Exception as exc:
            counts["failed"] += 1
            if len(failures) < 150:
                failures.append({"case": label, "error": f"{type(exc).__name__}: {exc}"[:2000]})

    def equal(got, expected, tol=1e-11, atol=1e-12):
        got, expected = native(got), np.asarray(expected)
        assert got.shape == expected.shape, (got.shape, expected.shape)
        np.testing.assert_allclose(got, expected, rtol=tol, atol=atol, equal_nan=True)

    if group == "arrays":
        for trial in range(trials):
            shape = tuple(int(x) for x in rng.integers(0, 8, size=int(rng.integers(1, 5))))
            dtype = [np.float64, np.int64, np.complex128, np.bool_][trial % 4]
            x = rng.normal(size=shape)
            if dtype == np.int64:
                x = rng.integers(-10, 11, size=shape)
            elif dtype == np.complex128:
                x = x + 1j * rng.normal(size=shape)
            elif dtype == np.bool_:
                x = x > 0
            x = x.astype(dtype)
            ax = a.array(x)
            if trial % 3 == 0:
                x, ax = x.T, ax.T
            elif trial % 3 == 1:
                x, ax = x[::-1], ax[::-1]
            label = f"trial={trial} shape={shape} dtype={dtype.__name__}"
            check(label + " copy", lambda: equal(ax.copy(), x))
            for name in ("add", "multiply", "subtract", "equal", "maximum", "minimum"):
                if dtype == np.bool_ and name == "subtract":
                    continue
                check(label + " " + name, lambda name=name: equal(getattr(a, name)(ax, ax[-1:] if x.shape[0] else ax), getattr(np, name)(x, x[-1:] if x.shape[0] else x)))
            for name in ("sum", "prod", "mean", "std", "var", "any", "all", "cumsum"):
                for axis in (None, 0, -1):
                    with np.errstate(all="ignore"):
                        check(label + f" {name} axis={axis}", lambda name=name, axis=axis: equal(getattr(a, name)(ax, axis=axis), getattr(np, name)(x, axis=axis)))
        extreme = np.array([0., -0., 1e-300, -1e-300, 1e300, -1e300, np.inf, -np.inf, np.nan])
        for name in ("exp", "expm1", "log", "log1p", "sqrt", "sin", "cos", "tanh", "absolute", "sign", "isfinite"):
            with np.errstate(all="ignore"):
                check("extreme " + name, lambda name=name: equal(getattr(a, name)(a.array(extreme)), getattr(np, name)(extreme)))

    elif group == "indexing":
        for trial in range(trials * 3):
            shape = tuple(int(v) for v in rng.integers(1, 7, size=3))
            x = rng.normal(size=shape)
            ax = a.array(x)
            key = []
            for n in shape:
                choice = int(rng.integers(5))
                if choice == 0:
                    key.append(int(rng.integers(-n, n)))
                elif choice == 1:
                    key.append([int(rng.integers(-n, n)) for _ in range(2)])
                else:
                    key.append(slice(None, None, [-2, -1, 1, 2][int(rng.integers(4))]))
            if trial % 5 == 0:
                key.insert(int(rng.integers(4)), None)
            key = tuple(key)
            label = f"trial={trial} shape={shape} key={key}"
            check(label, lambda: equal(ax[key], x[key]))
            def assignment():
                nx, qx = x.copy(), ax.copy()
                values = rng.normal(size=nx[key].shape)
                nx[key] = values
                qx[key] = a.array(values)
                equal(qx, nx)
            check(label + " assignment", assignment)
        for n in (2, 3, 10, 100):
            for mode in ("shift_right", "shift_left", "reverse", "inplace"):
                def overlap():
                    nx, qx = np.arange(n, dtype=float), a.arange(n, dtype=float)
                    if mode == "shift_right":
                        nx[1:], qx[1:] = nx[:-1], qx[:-1]
                    elif mode == "shift_left":
                        nx[:-1], qx[:-1] = nx[1:], qx[1:]
                    elif mode == "reverse":
                        nx[:], qx[:] = nx[::-1], qx[::-1]
                    else:
                        nx[1:] += nx[:-1]
                        qx[1:] += qx[:-1]
                    equal(qx, nx)
                check(f"overlap {n} {mode}", overlap)

    elif group == "linalg":
        for backend in ("c", "python"):
            with q.accel.enabled() if backend == "c" else q.accel.disabled():
                for trial in range(min(trials, 100)):
                    n = int(rng.integers(2, 25))
                    cond = 10. ** (trial % 13)
                    u, _ = np.linalg.qr(rng.normal(size=(n, n)))
                    A = (u * np.geomspace(1., 1. / cond, n)) @ u.T
                    truth = rng.normal(size=n)
                    b = A @ truth
                    qa, qb = a.array(A), a.array(b)
                    label = f"{backend} trial={trial} n={n} cond={cond}"
                    def solve():
                        x = native(q.linalg.solve(qa, qb))
                        error = np.linalg.norm(A @ x - b) / (np.linalg.norm(A) * np.linalg.norm(x) + np.linalg.norm(b))
                        assert error < 1e-12, error
                    check(label + " solve backward error", solve)
                    def chol():
                        L = native(q.linalg.cholesky(qa))
                        equal(L @ L.T, A, tol=1e-10)
                    check(label + " cholesky reconstruction", chol)
                    def qr():
                        Q, R = map(native, q.linalg.householder_qr(qa))
                        equal(Q @ R, A)
                        equal(Q.T @ Q, np.eye(n))
                    check(label + " QR", qr)
                    def eigen():
                        result = q.linalg.jacobi_eigen(qa)
                        V, w = native(result.eigenvectors), native(result.eigenvalues)
                        equal(A @ V, V * w, tol=1e-8, atol=1e-9)
                        equal(V.T @ V, np.eye(n), tol=1e-9)
                        assert result.converged
                    check(label + " eigen residual", eigen)
                    def least_squares():
                        design = rng.normal(size=(n * 3, n))
                        rhs = rng.normal(size=n * 3)
                        equal(q.linalg.qr_least_squares(a.array(design), a.array(rhs)), np.linalg.lstsq(design, rhs, rcond=None)[0])
                    check(label + " least squares", least_squares)

    elif group == "special":
        domains = {
            # Poles deliberately return +inf in Quadrivium; exclude them from
            # SciPy comparisons (SciPy returns NaN at negative integer poles).
            "gamma": (sp.gamma, np.r_[np.arange(-19, 0) + .123, np.geomspace(.001, 170., 300)]),
            "log_gamma": (sp.gammaln, np.geomspace(.001, 1e5, 400)),
            "digamma": (sp.digamma, np.geomspace(.001, 1e5, 400)),
            "erf": (sp.erf, np.linspace(-30, 30, 500)),
            "erfc": (sp.erfc, np.linspace(-30, 30, 500)),
            "erfinv": (sp.erfinv, np.linspace(-.999999, .999999, 500)),
            "erfcx": (sp.erfcx, np.linspace(-20, 100, 500)),
            "dawson": (sp.dawsn, np.linspace(-100, 100, 500)),
            "bessel_j0": (sp.j0, np.r_[np.linspace(-100, 100, 500), 1e3, 1e6]),
            "bessel_j1": (sp.j1, np.linspace(-100, 100, 500)),
            "bessel_y0": (sp.y0, np.geomspace(1e-8, 1e4, 500)),
            "bessel_y1": (sp.y1, np.geomspace(1e-8, 1e4, 500)),
            "bessel_i0": (sp.i0, np.linspace(-700, 700, 500)),
            "bessel_i1": (sp.i1, np.linspace(-700, 700, 500)),
            "bessel_k0": (sp.k0, np.geomspace(1e-8, 700, 500)),
            "bessel_k1": (sp.k1, np.geomspace(1e-8, 700, 500)),
            "airy_ai": (lambda x: sp.airy(x)[0], np.linspace(-30, 30, 500)),
            "airy_bi": (lambda x: sp.airy(x)[2], np.linspace(-30, 30, 500)),
            "elliptic_k": (sp.ellipk, np.r_[np.linspace(0, .9999, 400), 1 - 1e-14]),
            "elliptic_e": (sp.ellipe, np.r_[np.linspace(0, .9999, 400), 1 - 1e-14]),
            "exponential_integral": (sp.expi, np.r_[-np.geomspace(1e-8, 700, 250), np.geomspace(1e-8, 700, 250)]),
            "sine_integral": (lambda x: sp.sici(x)[0], np.geomspace(1e-8, 1e4, 500)),
            "cosine_integral": (lambda x: sp.sici(x)[1], np.geomspace(1e-8, 1e4, 500)),
            "fresnel_s": (lambda x: sp.fresnel(x)[0], np.linspace(-100, 100, 500)),
            "fresnel_c": (lambda x: sp.fresnel(x)[1], np.linspace(-100, 100, 500)),
            "zeta": (sp.zeta, np.r_[np.linspace(-20, .99, 200), np.linspace(1.01, 100, 200)]),
            "lambert_w": (lambda x: sp.lambertw(x).real, np.r_[np.linspace(-1/math.e+1e-12, 0, 100), np.geomspace(1e-12, 1e100, 300)]),
            "struve_h0": (lambda x: sp.struve(0, x), np.linspace(-100, 100, 500)),
            "logistic": (sp.expit, np.linspace(-1000, 1000, 500)),
        }
        for backend in ("c", "python"):
            with q.accel.enabled() if backend == "c" else q.accel.disabled():
                for name, (reference, x) in domains.items():
                    def compare():
                        expected = reference(x)
                        got = native(getattr(q.special, name)(a.array(x)))
                        scaled = np.abs(got - expected) / (1e-10 + 1e-8 * np.abs(expected))
                        i = int(np.nanargmax(scaled))
                        metrics[f"{backend}.{name}"] = {"points": len(x), "max_tolerance_ratio": float(scaled[i]), "worst_x": float(x[i]), "actual": float(got[i]), "expected": float(expected[i])}
                        equal(got, expected, tol=1e-8, atol=1e-10)
                    check(backend + " " + name, compare)
                for trial in range(trials):
                    alpha, beta = rng.uniform(.1, 50, size=2)
                    x = rng.uniform(0, 100)
                    z = rng.uniform(0, 1)
                    for name, args, reference in (
                        ("regularized_gamma_p", (alpha, x), sp.gammainc),
                        ("regularized_gamma_q", (alpha, x), sp.gammaincc),
                        ("incomplete_beta", (alpha, beta, z), sp.betainc),
                        ("hyp1f1", (alpha, beta, x/10), sp.hyp1f1),
                        ("hyp2f1", (alpha/10, beta/10, 6., z), sp.hyp2f1),
                    ):
                        check(f"{backend} {name}{args}", lambda name=name, args=args, reference=reference: equal(getattr(q.special, name)(*args), reference(*args), tol=1e-8, atol=1e-10))

    elif group == "special_orders":
        grid = np.geomspace(1e-5, 100, 150)
        for order in (0, 1, 2, 5, 10, 20, 50):
            for name, reference in (("bessel_jn", sp.jv), ("bessel_yn", sp.yv),
                                    ("bessel_in", sp.iv), ("bessel_kn", sp.kv),
                                    ("spherical_bessel_j", sp.spherical_jn),
                                    ("spherical_bessel_y", sp.spherical_yn),
                                    ("polygamma", sp.polygamma)):
                def compare():
                    expected = reference(order, grid)
                    got = native(getattr(q.special, name)(order, a.array(grid)))
                    scaled = np.abs(got-expected)/(1e-10+1e-8*np.abs(expected))
                    i = int(np.nanargmax(scaled))
                    metrics[f"{name}.{order}"] = {"points": len(grid), "worst_x": float(grid[i]), "actual": float(got[i]), "expected": float(expected[i]), "max_tolerance_ratio": float(scaled[i])}
                    equal(got, expected, tol=1e-8, atol=1e-10)
                check(f"{name} order={order}", compare)

    elif group == "transforms":
        for backend in ("c", "python"):
            with q.accel.enabled() if backend == "c" else q.accel.disabled():
                for n in list(range(1, 66)) + [97, 127, 257, 1000, 1024, 4093, 8192, 65536]:
                    x = rng.normal(size=n)
                    ax = a.array(x)
                    check(f"{backend} fft {n}", lambda: equal(q.transforms.fft(ax), np.fft.fft(x), tol=1e-10, atol=1e-9))
                    check(f"{backend} roundtrip {n}", lambda: equal(q.transforms.ifft(q.transforms.fft(ax)), x, tol=1e-10))
                    for kind in (1, 2, 3, 4):
                        if kind == 1 and n == 1:
                            continue
                        check(f"{backend} dct {kind} n={n}", lambda kind=kind: equal(q.transforms.dct(ax, kind=kind), sf.dct(x, type=kind), tol=1e-10, atol=1e-9))

    elif group == "fft_lengths":
        for half in (1, 2, 3, 5, 8, 17):
            for length in (1, 2, 3, 7, 16, 32, 65):
                x = rng.normal(size=half) + 1j*rng.normal(size=half)
                for layer in ("numeric", "public"):
                    fn = a.fft.irfft if layer == "numeric" else q.transforms.irfft
                    check(f"{layer} irfft half={half} n={length}", lambda fn=fn: equal(fn(a.array(x), n=length), np.fft.irfft(x, n=length)))
                for axis in (0, 1, -1):
                    shape = (half, 3) if axis == 0 else (3, half)
                    v = rng.normal(size=shape) + 1j*rng.normal(size=shape)
                    check(f"numeric irfft shape={shape} axis={axis} n={length}", lambda: equal(a.fft.irfft(a.array(v), n=length, axis=axis), np.fft.irfft(v, n=length, axis=axis)))

    elif group == "algorithms":
        for trial in range(trials):
            root = float(rng.uniform(-100, 100))
            for name in ("bisection", "brent", "ridders", "itp", "illinois", "pegasus"):
                def find():
                    result = getattr(q.rootfind, name)(lambda x: math.tanh(x - root), root - 2, root + 3)
                    assert result.converged
                    assert abs(result.root - root) < 1e-9
                check(f"root {name} trial={trial}", find)
            rate = float(rng.uniform(.1, 10))
            for name in ("adaptive_simpson", "adaptive_gauss_kronrod", "quad"):
                check(f"integral {name} trial={trial}", lambda name=name: equal(getattr(q.integrate, name)(lambda x: math.exp(-rate*x), 0., 1.).value, -math.expm1(-rate)/rate, tol=1e-8))
        for backend in ("c", "python"):
            with q.accel.enabled() if backend == "c" else q.accel.disabled():
                for method in ("dormand_prince", "rkf45", "cash_karp", "bogacki_shampine"):
                    for rate in (.1, 1., 10., 100.):
                        for backwards in (False, True):
                            def decay():
                                span = (1., 0.) if backwards else (0., 1.)
                                result = q.ode.solve_ivp(lambda t, y: -rate*y, span, [1.], method=method, rtol=1e-9, atol=1e-12)
                                assert result.success
                                equal(result.y[-1], [math.exp(rate if backwards else -rate)], tol=3e-7, atol=1e-10)
                            check(f"{backend} {method} rate={rate} backwards={backwards}", decay)

    elif group == "stress":
        import concurrent.futures
        import gc
        import psutil
        process = psutil.Process()
        def work(i):
            with q.accel.disabled() if i % 2 else q.accel.enabled():
                expected_backend = "python" if i % 2 else "c"
                assert q.accel.backend() == expected_backend
                v = a.arange(257, dtype=float) / 100
                equal(q.transforms.ifft(q.transforms.fft(v)).real, native(v))
                mat = a.eye(12) * (i + 1)
                equal(q.linalg.solve(mat, a.ones(12)), np.ones(12)/(i+1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for i, result in enumerate(pool.map(lambda i: (i, work(i)), range(trials * 10))):
                counts["checks"] += 1
                counts["passed"] += 1
        rss = []
        for block in range(12):
            for _ in range(100):
                x = a.ones(100000)
                y = a.exp(x) + a.sin(x)
                z = y[::-1].copy()
                del x, y, z
            gc.collect()
            rss.append(process.memory_info().rss)
        metrics["rss_blocks_bytes"] = rss
        metrics["retained_growth_bytes"] = max(rss[3:]) - min(rss[3:])
        check("retained RSS after warmup <32 MiB", lambda: np.testing.assert_array_less(metrics["retained_growth_bytes"], 32 * 1024**2))
    return {"group": group, "seed": seed, "trials": trials, "counts": dict(counts), "failures": failures, "metrics": metrics,
            "numpy": np.__version__, "scipy": scipy.__version__, "backend": q.accel.show_config()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--group", choices=GROUPS, action="append")
    parser.add_argument("--worker", choices=GROUPS)
    args = parser.parse_args()
    if args.trials < 1 or args.timeout <= 0:
        parser.error("trials and timeout must be positive")
    if args.worker:
        print(json.dumps(worker(args.worker, args.seed, args.trials)))
        return
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1", PYTHONHASHSEED="0")
    report = {"python": sys.version, "platform": platform.platform(), "threads": 1, "results": []}
    for group in args.group or GROUPS:
        start = time.perf_counter()
        try:
            proc = subprocess.run([sys.executable, "-X", "faulthandler", __file__, "--worker", group, "--seed", str(args.seed), "--trials", str(args.trials)], capture_output=True, text=True, env=env, timeout=args.timeout)
            result = json.loads(proc.stdout) if proc.returncode == 0 else {"group": group, "returncode": proc.returncode, "error": proc.stderr[-10000:]}
            result["stderr"] = proc.stderr[-4000:]
        except subprocess.TimeoutExpired:
            result = {"group": group, "error": f"timeout after {args.timeout}s"}
        except Exception:
            result = {"group": group, "error": traceback.format_exc()}
        result["elapsed_seconds"] = time.perf_counter() - start
        report["results"].append(result)
        print(group, result.get("counts", result.get("error")), flush=True)
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not args.output:
        print(json.dumps(report, indent=2))
    return int(any(r.get("error") or r.get("counts", {}).get("failed", 0) for r in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
