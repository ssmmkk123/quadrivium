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
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable

import numpy as np

from quadrivium import _accel, linalg, ode, special, transforms


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeat", type=int, default=5,
                        help="samples per measurement (default: 5)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--markdown", action="store_true",
                        help="print a Markdown table instead of aligned text")
    args = parser.parse_args(argv)

    print(_accel.show_config().splitlines()[0], file=sys.stderr)
    if not _accel.available():
        print("The compiled backend is not active; there is nothing to compare "
              "against. Build it with `pip install -e .` and a Rust toolchain "
              "on PATH.", file=sys.stderr)
        return 1

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
