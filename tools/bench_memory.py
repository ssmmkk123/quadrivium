#!/usr/bin/env python3
"""Fresh-process native RSS measurements (optional deps: numpy, scipy, psutil).

The parent samples RSS after input creation, while the worker performs one cold
operation and holds its output. Three independent processes per case/provider
avoid warmed allocator high-water marks. Sampling can miss brief scratch peaks;
ru_maxrss is recorded separately, since its lifetime peak can include setup.
This is a memory benchmark, not a timing benchmark or a leak detector.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CASES = {"exp": 4_000_000, "qr": 512, "least_squares": 8192,
         "fft": 262144, "sparse_identity": 1_000_000, "kde": 100_000}


def worker(case, provider):
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import psutil
    from scipy import sparse
    import quadrivium as q
    from quadrivium import numeric as a

    n = CASES[case]
    lib = np if provider == "reference" else a
    if case == "exp":
        x = lib.linspace(-4, 4, n)
        op = lambda: lib.exp(x)
    elif case == "qr":
        # Deterministic, nontrivial square input, with only provider data live.
        x = lib.sin(lib.arange(n*n, dtype=float).reshape(n, n)) + lib.eye(n)
        op = (lambda: np.linalg.qr(x)) if provider == "reference" else (lambda: q.linalg.householder_qr(x))
    elif case == "least_squares":
        x = lib.cos(lib.arange(n*12, dtype=float).reshape(n, 12)**1.1)
        b = lib.sin(lib.arange(n, dtype=float))
        op = (lambda: np.linalg.lstsq(x, b, rcond=None)[0]) if provider == "reference" else (lambda: q.linalg.qr_least_squares(x, b))
    elif case == "fft":
        x = lib.sin(lib.arange(n, dtype=float)).astype(complex)
        op = (lambda: np.fft.fft(x)) if provider == "reference" else (lambda: q.transforms.fft(x))
    elif case == "sparse_identity":
        op = (lambda: sparse.eye(n, format="csr")) if provider == "reference" else (lambda: q.linalg.identity_sparse(n))
    else:
        x = lib.sin(lib.arange(n, dtype=float))
        pts = lib.linspace(-4, 4, 200)
        def reference():
            out = np.zeros(len(pts))
            for start in range(0, n, 4096):
                z = (pts[:, None] - x[None, start:start+4096])/.3
                out += np.exp(-.5*z*z).sum(axis=1)
            return out/(n*.3*np.sqrt(2*np.pi))
        op = reference if provider == "reference" else (lambda: q.stochastic.kernel_density(x, pts, bandwidth=.3))

    def payload_bytes(obj):
        if isinstance(obj, (tuple, list)):
            return sum(payload_bytes(item) for item in obj)
        if hasattr(obj, "indptr"):
            return sum(item.nbytes for item in (obj.data, obj.indices, obj.indptr))
        return getattr(obj, "nbytes", 0)

    rss_scale = 1 if sys.platform == "darwin" else 1024
    process = psutil.Process()
    with q.accel.disabled() if provider == "python" else q.accel.enabled():
        if provider == "rust" and not q.accel.available():
            raise RuntimeError("Rust acceleration unavailable")
        gc.collect()
        baseline = process.memory_info().rss
        before_hwm = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_scale
        print(json.dumps({"baseline_rss_bytes": baseline, "before_hwm_bytes": before_hwm}), flush=True)
        sys.stdin.readline()
        out = op()
        result = {"output_rss_bytes": process.memory_info().rss,
                  "after_hwm_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_scale,
                  "output_payload_bytes": payload_bytes(out)}
        # Hold the live result long enough for the parent to sample it.
        time.sleep(.02)
        del out
        gc.collect()
        result["after_release_rss_bytes"] = process.memory_info().rss
        print(json.dumps(result), flush=True)


def main():
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=CASES)
    parser.add_argument("--provider", choices=("python", "rust", "reference"))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.provider)
        return
    if args.repeat < 1 or args.output is None:
        parser.error("positive repeat and output are required")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    report = {"repeat": args.repeat, "sampling_interval_seconds": .001, "threads": 1,
              "description": __doc__, "results": []}
    for case, n in CASES.items():
        for provider in ("python", "rust", "reference"):
            samples = []
            for _ in range(args.repeat):
                proc = subprocess.Popen([sys.executable, "-X", "faulthandler", __file__,
                                         "--worker", case, "--provider", provider],
                                        env=env, text=True, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    ready = json.loads(proc.stdout.readline())
                    peak = ready["baseline_rss_bytes"]
                    child = psutil.Process(proc.pid)
                    proc.stdin.write("go\n")
                    proc.stdin.flush()
                    deadline = time.monotonic() + 120
                    while proc.poll() is None:
                        if time.monotonic() > deadline:
                            raise TimeoutError("memory worker exceeded 120s")
                        try:
                            peak = max(peak, child.memory_info().rss)
                        except psutil.NoSuchProcess:
                            break
                        time.sleep(.001)
                    stdout, stderr = proc.communicate()
                    if proc.returncode:
                        raise RuntimeError(stderr)
                    sample = {**ready, **json.loads(stdout)}
                    sample["sampled_peak_rss_bytes"] = max(peak, sample["output_rss_bytes"])
                    sample["sampled_increment_bytes"] = max(0, sample["sampled_peak_rss_bytes"] - sample["baseline_rss_bytes"])
                    samples.append(sample)
                except Exception as exc:
                    proc.kill()
                    proc.communicate()
                    samples.append({"error": str(exc)})
            result = {"case": case, "n": n, "provider": provider, "samples": samples}
            good = [s["sampled_increment_bytes"] for s in samples if "error" not in s]
            if good:
                result["median_sampled_increment_mib"] = statistics.median(good)/1024**2
            report["results"].append(result)
            args.output.write_text(json.dumps(report, indent=2)+"\n")
            print(case, provider, result.get("median_sampled_increment_mib", samples), flush=True)
    return int(any("error" in s for r in report["results"] for s in r["samples"]))


if __name__ == "__main__":
    raise SystemExit(main())
