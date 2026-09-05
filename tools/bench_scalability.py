#!/usr/bin/env python3
"""Measure time and memory in isolated processes, without optional dependencies.

Run on two source trees with the same interpreter and hardware::

    python tools/bench_scalability.py --root /path/to/before --output before.json
    python tools/bench_scalability.py --output after.json

Each case warms up, reports the median of repeated wall times, and measures
Python/NumPy-tracked allocations separately. Peak RSS includes the interpreter,
inputs, and native allocations (unlike tracemalloc), and is available on Unix.
Thread counts are fixed to one in child processes for comparable BLAS timings.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc


CASES = ("least_squares", "least_squares_python", "kernel_density", "identity_sparse",
         "csr_matvec", "fft_python", "welch_python")


def measure_case(name, root, repeat):
    sys.path.insert(0, str(root))
    import numpy as np
    from quadrivium import _accel
    from quadrivium.linalg import CSRMatrix, qr_least_squares, identity_sparse
    from quadrivium.stochastic import kernel_density
    from quadrivium.transforms import fft, welch

    rng = np.random.default_rng(20260905)
    context = _accel.disabled if name.endswith("_python") else _accel.enabled
    if name.startswith("least_squares"):
        A, b = rng.normal(size=(1500, 12)), rng.normal(size=1500)
        operation = lambda: qr_least_squares(A, b)
        size = "1500 x 12"
    elif name == "kernel_density":
        x, points = rng.normal(size=20000), np.linspace(-4, 4, 400)
        operation = lambda: kernel_density(x, points, bandwidth=0.3)
        size = "20000 samples x 400 points"
    elif name == "identity_sparse":
        operation = lambda: identity_sparse(2000)
        size = "2000 x 2000"
    elif name == "csr_matvec":
        n = 100000
        A = CSRMatrix(np.arange(n + 1), np.arange(n), np.ones(n), (n, n))
        x = rng.normal(size=n)
        operation = lambda: A.matvec(x)
        size = "100000 nonzeros"
    elif name == "fft_python":
        x = rng.normal(size=8192)
        operation = lambda: fft(x)
        size = "8192 samples"
    else:
        x = rng.normal(size=65536)
        operation = lambda: welch(x, segment=256)
        size = "65536 samples; 256 per segment"
    with context():
        operation()
        times = []
        for _ in range(repeat):
            start = time.perf_counter()
            operation()
            times.append((time.perf_counter() - start) * 1000)
        gc.collect()
        tracemalloc.start()
        operation()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        backend = _accel.backend()
    try:
        import resource
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss /= 1024**2 if sys.platform == "darwin" else 1024
    except ImportError:
        peak_rss = None
    return {"case": name, "size": size, "backend": backend,
            "median_ms": statistics.median(times), "samples_ms": times,
            "traced_peak_mib": peak / 1024**2, "process_peak_rss_mib": peak_rss,
            "numpy": np.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--case", choices=CASES, action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.worker:
        print(json.dumps(measure_case(args.worker, args.root.resolve(), args.repeat)))
        return
    results = []
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    for case in args.case or CASES:
        try:
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--root", str(args.root.resolve()),
                 "--repeat", str(args.repeat), "--worker", case],
                env=env, check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as exc:
            print(exc.stderr, file=sys.stderr)
            raise
        result = json.loads(completed.stdout)
        results.append(result)
        print(f"{case:24} {result['median_ms']:10.3f} ms  "
              f"{result['traced_peak_mib']:9.3f} MiB traced  "
              f"{result['process_peak_rss_mib']} MiB process peak RSS", file=sys.stderr)
    report = {"python": sys.version, "platform": platform.platform(),
              "root": str(args.root.resolve()), "repeat": args.repeat,
              "blas_threads": 1, "results": results}
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
