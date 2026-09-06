#!/usr/bin/env python3
"""Measure the compiled backend against the pure-Python one.

Every routine is timed twice on the same input: once with the Rust extension
active and once inside ``accel.disabled()``. The comparison is therefore
between two implementations of the same algorithm on one machine, not against
some other library on some other machine.

The numbers quoted in ``README.md`` come from this script. They are not
portable -- the margin on the BLAS-bound routines in particular depends on
which BLAS NumPy was linked against -- so re-run it rather than trusting them.

Usage::

    python tools/bench_accel.py               # the standard set
    python tools/bench_accel.py --repeat 7    # more samples per measurement
    python tools/bench_accel.py --markdown    # emit the README table
    python tools/bench_accel.py --native      # compare against NumPy instead
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable

from quadrivium import numeric as np

from quadrivium import _accel, interpolate, linalg, ode, pde, special, transforms


def import_numpy():
    """NumPy, for `--native` only.

    The package no longer depends on it; this comparison does, because the
    point of it is to measure these routines against a library that has been
    binding LAPACK for thirty years.
    """
    try:
        import numpy
    except ImportError:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "--native compares against NumPy, which is not installed; "
            "`pip install numpy` or run without --native"
        ) from None
    return numpy


def warm_blas(numpy, rng) -> None:
    """Spin up NumPy's BLAS thread pool before anything is timed.

    OpenBLAS pays its pool startup on first use. Without this the cost lands on
    whichever measurement happens to run first, and NumPy reports as slower than
    it really is -- an easy way to flatter these numbers by accident.
    """
    w = rng.standard_normal((1200, 1200))
    spd = w @ w.T + 1200 * numpy.eye(1200)
    small = w[:200, :200] + w[:200, :200].T
    for _ in range(5):
        w @ w
        numpy.linalg.cholesky(spd)
        numpy.linalg.qr(w, mode="complete")
        numpy.fft.fft(w[0])
        numpy.linalg.eigh(small)


def native_cases(numpy, rng):
    """Yield ``(label, size, ours, numpy)`` for routines NumPy also provides.

    Each operand is built once in NumPy and handed to this package through the
    buffer protocol, so both sides time the same numbers in their own layout.
    """
    mm = _accel.kernel("matmul")
    ours = lambda x: np.asarray(x)
    for n in (200, 400, 800, 1600):
        a = rng.standard_normal((n, n))
        b = rng.standard_normal((n, n))
        spd = a @ a.T + n * numpy.eye(n)
        rhs = rng.standard_normal(n)
        qa, qb, qspd, qrhs = ours(a), ours(b), ours(spd), ours(rhs)
        if mm is not None:
            yield "matmul", n, (lambda x=qa, y=qb: mm(x, y)), (lambda x=a, y=b: x @ y)
        yield ("linalg.cholesky", n, lambda s=qspd: linalg.cholesky(s),
               lambda s=spd: numpy.linalg.cholesky(s))
        yield ("linalg.householder_qr", n, lambda m=qa: linalg.householder_qr(m),
               lambda m=a: numpy.linalg.qr(m, mode="complete"))
        yield ("linalg.solve", n, lambda s=qspd, v=qrhs: linalg.solve(s, v),
               lambda s=spd, v=rhs: numpy.linalg.solve(s, v))
    for n in (1024, 4096, 65536, 262144, 1000, 10000, 100000):
        v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        qv = ours(v)
        yield ("transforms.fft", n, lambda x=qv: transforms.fft(x),
               lambda x=v: numpy.fft.fft(x))
    for n in (40, 80, 160):
        m = rng.standard_normal((n, n))
        sym = m + m.T
        qsym = ours(sym)
        yield ("linalg.jacobi_eigen", n, lambda s=qsym: linalg.jacobi_eigen(s),
               lambda s=sym: numpy.linalg.eigh(s))


def measure(fn: Callable[[], object], repeat: int) -> float:
    """Best-of-`repeat` wall time in milliseconds, after one warm-up call."""
    fn()
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best * 1e3


def cases(rng: np.random.Generator):
    """Yield ``(label, size, callable)`` for each benchmarked routine."""
    for n in (100, 200, 400):
        a = rng.standard_normal((n, n))
        spd = a @ a.T + n * np.eye(n)
        b = rng.standard_normal(n)
        lower, upper = np.tril(spd), np.triu(spd)
        yield f"linalg.cholesky", n, lambda s=spd: linalg.cholesky(s)
        yield f"linalg.plu_decomposition", n, lambda m=a: linalg.plu_decomposition(m)
        yield f"linalg.householder_qr", n, lambda m=a: linalg.householder_qr(m)
        yield f"linalg.solve", n, lambda s=spd, v=b: linalg.solve(s, v)
        yield f"linalg.forward_substitution", n, lambda l=lower, v=b: linalg.forward_substitution(l, v)
        yield f"linalg.back_substitution", n, lambda u=upper, v=b: linalg.back_substitution(u, v)

    for n in (40, 80):
        m = rng.standard_normal((n, n))
        sym = m + m.T
        yield f"linalg.jacobi_eigen", n, lambda s=sym: linalg.jacobi_eigen(s)

    for n in (1024, 4096, 10000):
        v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        yield f"transforms.fft", n, lambda x=v: transforms.fft(x)
        yield f"transforms.ifft", n, lambda x=v: transforms.ifft(x)

    # The RK driver keeps calling back into Python for the right-hand side;
    # what moves to the kernel is the stage assembly and the controller.
    def decay(t, y):
        return -2.0 * y

    def van_der_pol(t, y):
        return np.array([y[1], 3.0 * (1.0 - y[0] ** 2) * y[1] - y[0]])

    def lorenz(t, y):
        return np.array([
            10.0 * (y[1] - y[0]),
            y[0] * (28.0 - y[2]) - y[1],
            y[0] * y[1] - 8.0 / 3.0 * y[2],
        ])

    yield "ode.solve_ivp (decay)", 1, lambda: ode.solve_ivp(decay, (0.0, 50.0), [1.0])
    yield "ode.solve_ivp (van der Pol)", 2, lambda: ode.solve_ivp(van_der_pol, (0.0, 20.0), [2.0, 0.0])
    yield "ode.solve_ivp (Lorenz)", 3, lambda: ode.solve_ivp(lorenz, (0.0, 25.0), [1.0, 1.0, 1.0])

    for n in (1000, 100000):
        grid = np.linspace(0.5, 20.0, n)
        wide = np.linspace(-6.0, 6.0, n)
        yield f"special.gamma", n, lambda x=grid: special.gamma(x)
        yield f"special.log_gamma", n, lambda x=grid: special.log_gamma(x)
        yield f"special.erf", n, lambda x=wide: special.erf(x)
        yield f"special.erfc", n, lambda x=wide: special.erfc(x)

    # Kernels that own a whole iteration rather than one step. The Python twin
    # of each is a per-element loop, so the gap here is the interpreter's,
    # not BLAS's.
    for n in (1000, 50000):
        diag = rng.uniform(3.0, 4.0, n)
        sub = rng.uniform(-1.0, 1.0, n - 1)
        sup = rng.uniform(-1.0, 1.0, n - 1)
        rhs = rng.standard_normal(n)
        yield ("linalg.thomas", n,
               lambda a=sub, b=diag, c=sup, d=rhs: linalg.thomas(a, b, c, d))

    for n in (20, 40):
        m = rng.standard_normal((n, n))
        yield f"linalg.qr_algorithm", n, lambda s=m + m.T: linalg.qr_algorithm(s)

    src = lambda x, y: 1.0
    for n in (20, 40):
        yield (f"pde.poisson_2d_iterative (sor)", n,
               lambda k=n: pde.poisson_2d_iterative(src, (0, 1), (0, 1), nx=k, ny=k,
                                                    method="sor"))
    yield ("pde.poisson_2d_iterative (gauss_seidel)", 30,
           lambda: pde.poisson_2d_iterative(src, (0, 1), (0, 1), nx=30, ny=30,
                                            method="gauss_seidel"))
    for n in (21, 31):
        yield f"pde.lid_driven_cavity", n, lambda k=n: pde.lid_driven_cavity(re=100.0, n=k)

    # Routines that reach the kernel through `thomas` rather than directly.
    u0 = lambda x: np.sin(np.pi * x)
    yield ("pde.heat_crank_nicolson", 800,
           lambda: pde.heat_crank_nicolson(u0, 0.1, (0, 1), (0, 0.2), nx=800, nt=400))
    yield ("interpolate.cubic_spline", 3200,
           lambda x=np.linspace(0, 1, 3200): interpolate.cubic_spline(x, np.sin(7 * x)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeat", type=int, default=5,
                        help="samples per measurement (default: 5)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--markdown", action="store_true",
                        help="print a Markdown table instead of aligned text")
    parser.add_argument("--native", action="store_true",
                        help="compare against NumPy's kernels instead of the "
                             "pure-Python fallback")
    args = parser.parse_args(argv)

    print(_accel.show_config().splitlines()[0], file=sys.stderr)
    if not _accel.available():
        print("The compiled backend is not active; there is nothing to compare "
              "against. Build it with `pip install -e .` and a Rust toolchain "
              "on PATH.", file=sys.stderr)
        return 1

    if args.native:
        numpy = import_numpy()
        rng = numpy.random.default_rng(args.seed)
        warm_blas(numpy, rng)
        print(f"{'operation':28s} {'quadrivium':>12s} {'numpy':>10s}  ratio")
        print("-" * 68)
        for label, size, ours, theirs in native_cases(numpy, rng):
            a = measure(ours, args.repeat)
            b = measure(theirs, args.repeat)
            ratio = a / b
            verdict = (f"{ratio:.2f}x slower" if ratio > 1.10
                       else f"{1 / ratio:.2f}x FASTER" if ratio < 0.91 else "~parity")
            print(f"{label + ' n=' + str(size):28s} {a:9.3f} ms {b:7.3f} ms  {verdict}")
        return 0

    rows = []
    for label, size, fn in cases(np.random.default_rng(args.seed)):
        with _accel.disabled():
            slow = measure(fn, args.repeat)
        fast = measure(fn, args.repeat)
        rows.append((label, size, slow, fast, slow / fast if fast else float("inf")))

    if args.markdown:
        print("| routine | size | python | rust | speedup |")
        print("|---|---|---|---|---|")
        for label, size, slow, fast, ratio in rows:
            print(f"| `{label}` | {size} | {slow:.2f} ms | {fast:.3f} ms | **{ratio:.1f}x** |")
    else:
        width = max(len(r[0]) for r in rows)
        print(f"{'routine':<{width}}  {'size':>7}  {'python':>11}  {'rust':>11}  {'speedup':>9}")
        for label, size, slow, fast, ratio in rows:
            print(f"{label:<{width}}  {size:>7}  {slow:>8.2f} ms  {fast:>8.3f} ms  {ratio:>8.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
