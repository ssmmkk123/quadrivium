#!/usr/bin/env python3
"""Isolated size sweeps with validated outputs and raw timing distributions.

Optional benchmark dependencies: numpy, scipy. Each workload/provider/size has
its own subprocess; all thread pools are limited to one. Input generation and
reference validation are excluded from timing. Samples batch short operations
to at least roughly 10ms, reporting per-call times. RSS includes imports and
validation; tracemalloc is measured separately and excludes native allocations.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
SIZES = {
    "matmul": [64, 256, 1024],
    "solve": [32, 128, 512],
    "cholesky": [32, 128, 512],
    "qr": [32, 128, 512],
    "least_squares": [128, 1024, 8192],
    "eigh": [16, 64, 128],
    "fft": [1024, 10007, 262144],
    "exp": [1000, 100000, 1000000],
    "gamma": [1000, 100000, 1000000],
    "csr_matvec": [1000, 100000, 1000000],
    "sparse_identity": [1000, 100000, 1000000],
    "kde": [1000, 10000, 100000],
}


def measure(case, n, provider, repeat, seed):
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import scipy
    from scipy import sparse, special
    import quadrivium as q
    from quadrivium import numeric as a
    rng = np.random.default_rng(seed)
    is_reference = provider == "reference"
    def native(x):
        return np.asarray(memoryview(x)) if isinstance(x, a.ndarray) else np.asarray(x)
    def close(x, y, tol=1e-9):
        np.testing.assert_allclose(native(x), y, rtol=tol, atol=tol)
    if case in ("matmul", "solve", "cholesky", "qr", "eigh"):
        x = rng.normal(size=(n, n))
        y = rng.normal(size=(n, n))
        spd = x @ x.T + n * np.eye(n)
        b = rng.normal(size=n)
        ax, ay, aspd, ab = map(a.array, (x, y, spd, b))
        if case == "matmul":
            op = (lambda: x @ y) if is_reference else (lambda: ax @ ay)
            expected = x @ y
            validate = lambda out: close(out, expected)
        elif case == "solve":
            op = (lambda: np.linalg.solve(spd, b)) if is_reference else (lambda: q.linalg.solve(aspd, ab))
            validate = lambda out: close(spd @ native(out), b)
        elif case == "cholesky":
            op = (lambda: np.linalg.cholesky(spd)) if is_reference else (lambda: q.linalg.cholesky(aspd))
            validate = lambda out: close(native(out) @ native(out).T, spd)
        elif case == "qr":
            op = (lambda: np.linalg.qr(x)) if is_reference else (lambda: q.linalg.householder_qr(ax))
            def validate(out):
                Q, R = map(native, out)
                close(Q @ R, x)
                close(Q.T @ Q, np.eye(n))
        else:
            sym = x + x.T
            asym = a.array(sym)
            op = (lambda: np.linalg.eigh(sym)) if is_reference else (lambda: q.linalg.jacobi_eigen(asym))
            def validate(out):
                w, v = out if is_reference else (out.eigenvalues, out.eigenvectors)
                w, v = native(w), native(v)
                close(sym @ v, v * w, tol=1e-8)
                if not is_reference:
                    assert out.converged
    elif case == "least_squares":
        x, y = rng.normal(size=(n, 12)), rng.normal(size=n)
        ax, ay = a.array(x), a.array(y)
        op = (lambda: np.linalg.lstsq(x, y, rcond=None)[0]) if is_reference else (lambda: q.linalg.qr_least_squares(ax, ay))
        expected = np.linalg.lstsq(x, y, rcond=None)[0]
        validate = lambda out: close(out, expected)
    elif case == "fft":
        x = rng.normal(size=n) + 1j*rng.normal(size=n)
        ax = a.array(x)
        op = (lambda: np.fft.fft(x)) if is_reference else (lambda: q.transforms.fft(ax))
        expected = np.fft.fft(x)
        validate = lambda out: close(out, expected, tol=1e-7)
    elif case in ("exp", "gamma"):
        x = np.linspace(.5, 20, n)
        ax = a.array(x)
        reference = np.exp if case == "exp" else special.gamma
        ours = a.exp if case == "exp" else q.special.gamma
        op = (lambda: reference(x)) if is_reference else (lambda: ours(ax))
        expected = reference(x)
        validate = lambda out: close(out, expected)
    elif case in ("csr_matvec", "sparse_identity"):
        if case == "csr_matvec":
            mat = sparse.eye(n, format="csr")
            amat = q.linalg.identity_sparse(n)
            x = rng.normal(size=n)
            ax = a.array(x)
            op = (lambda: mat @ x) if is_reference else (lambda: amat.matvec(ax))
            validate = lambda out: close(out, x)
        else:
            op = (lambda: sparse.eye(n, format="csr")) if is_reference else (lambda: q.linalg.identity_sparse(n))
            def validate(out):
                assert out.shape == (n, n)
                close(out.data, np.ones(n))
    else:
        x = rng.normal(size=n)
        points = np.linspace(-4, 4, 200)
        ax, apoints = a.array(x), a.array(points)
        # Same fixed-bandwidth density definition; block reference workspace.
        def reference():
            out = np.zeros(len(points))
            for start in range(0, n, 4096):
                z = (points[:, None] - x[None, start:start+4096])/.3
                out += np.exp(-.5*z*z).sum(axis=1)
            return out/(n*.3*np.sqrt(2*np.pi))
        op = reference if is_reference else (lambda: q.stochastic.kernel_density(ax, apoints, bandwidth=.3))
        expected = reference()
        validate = lambda out: close(out[1] if not is_reference else out, expected)
    with q.accel.disabled() if provider == "python" else q.accel.enabled():
        if provider == "rust" and not q.accel.available():
            raise RuntimeError("Rust provider requested but unavailable")
        out = op()
        validate(out)
        del out
        start = time.perf_counter()
        op()
        calibration = time.perf_counter() - start
        batch = max(1, min(1000, int(.01/max(calibration, 1e-9))))
        samples = []
        for _ in range(repeat):
            start = time.perf_counter_ns()
            for _ in range(batch):
                op()
            samples.append((time.perf_counter_ns()-start)/batch/1e6)
        gc.collect()
        tracemalloc.start()
        op()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return {"case": case, "n": n, "provider": provider, "valid": True, "batch": batch,
            "samples_ms": samples, "median_ms": statistics.median(samples),
            "p95_ms": float(np.percentile(samples, 95)), "min_ms": min(samples),
            "max_ms": max(samples), "traced_peak_mib": peak/1024**2,
            "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform == "darwin" else 1024),
            "numpy": np.__version__, "scipy": scipy.__version__}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument("--repeat", type=int, default=11)
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--case", choices=SIZES, action="append")
    p.add_argument("--worker", choices=SIZES)
    p.add_argument("--n", type=int)
    p.add_argument("--provider", choices=("python", "rust", "reference"))
    args = p.parse_args()
    if args.repeat < 2 or args.timeout <= 0:
        p.error("repeat must be at least 2; timeout must be positive")
    if args.worker:
        if not args.n or not args.provider:
            p.error("worker requires n and provider")
        print(json.dumps(measure(args.worker, args.n, args.provider, args.repeat, args.seed)))
        return
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    report = {"python": sys.version, "platform": platform.platform(), "seed": args.seed, "repeat": args.repeat, "threads": 1, "results": []}
    for case in args.case or SIZES:
        for n in SIZES[case]:
            for provider in ("python", "rust", "reference"):
                try:
                    proc = subprocess.run([sys.executable, "-X", "faulthandler", __file__, "--worker", case, "--n", str(n), "--provider", provider, "--repeat", str(args.repeat), "--seed", str(args.seed)], env=env, capture_output=True, text=True, timeout=args.timeout)
                    result = json.loads(proc.stdout) if proc.returncode == 0 else {"case": case, "n": n, "provider": provider, "error": proc.stderr[-5000:], "returncode": proc.returncode}
                except subprocess.TimeoutExpired:
                    result = {"case": case, "n": n, "provider": provider, "error": f"timeout after {args.timeout}s"}
                report["results"].append(result)
                print(case, n, provider, result.get("median_ms", result.get("error")), flush=True)
                if args.output:
                    args.output.write_text(json.dumps(report, indent=2)+"\n")
    if not args.output:
        print(json.dumps(report, indent=2))
    return int(any(r.get("error") for r in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
