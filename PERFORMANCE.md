# Performance and reliability improvements

Measured on 2026-09-05 using Python 3.14.4, NumPy 2.4.4, and Linux x86-64.
The baseline is the working tree at the start of this request, including its
existing local changes and compiled backend. These results measure selected
workloads, not a uniform speedup across every public function.

Each workload ran in its own process with one BLAS thread, one warm-up and
three timed samples. The table reports median wall time in milliseconds.

| Operation | Workload | Before (ms) | After (ms) | Speedup |
|---|---|---:|---:|---:|
| Least squares, Rust | 1500 x 12 | 6.512 | 0.165 | 39.5× |
| Least squares, Python | 1500 x 12 | 87.503 | 0.517 | 169.3× |
| Gaussian density estimate | 20000 samples x 400 points | 89.557 | 34.544 | 2.6× |
| Sparse identity | 2000 x 2000 | 15.665 | 0.022 | 712.1× |
| CSR matrix-vector product | 100000 nonzeros | 90.828 | 1.042 | 87.1× |
| FFT, Python | 8192 samples | 11.615 | 0.521 | 22.3× |
| Welch PSD, Python | 65536 samples; 256 per segment | 209.182 | 25.709 | 8.1× |

## Memory

- Gaussian KDE (20,000 samples, 400 points): tracked peak allocations fell
  from 183.1 MiB to 1.84 MiB. Process peak RSS fell from 225.7 MiB to 44.9 MiB.
- Python least squares (1,500 × 12): tracked peak allocations fell from
  34.61 MiB to 0.44 MiB. Both implementations now use O(m*n) workspace;
  the previous full-Q construction required O(m*m) storage.
- Sparse identity (2,000 × 2,000): tracked peak allocations fell from
  64.85 MiB to 0.05 MiB by constructing the sparse arrays directly.
- CSR matrix-vector multiplication trades temporary storage for speed:
  tracked peak allocations rose from 0.76 MiB to 3.15 MiB for 100,000
  nonzeros. The workspace scales with stored entries, not dense matrix area.
- Real FFT outputs own their retained half-spectrum; stationary wavelets
  wrap dilated taps modulo signal length instead of allocating exponentially
  long zero-filled filters.

Tracemalloc captures Python/NumPy allocations but does not capture all Rust
allocations. Native workloads must also be assessed using the recorded process
peak RSS, which includes the interpreter and inputs. Raw samples and memory
measurements are in [before](benchmarks/scalability-before.json) and
[after](benchmarks/scalability-after.json).

## Reliability and compatibility

The public numerical algorithms remain visible in Python, with optional Rust
kernels. No new runtime dependency or language toolchain is required. NumPy's
compiled array operations provide the gains in sparse storage, density
estimation, and Python transforms. The Rust build now declares the 1.83 minimum
required by PyO3/NumPy bindings and avoids a duplicate ndarray dependency.

Regression coverage includes independent residual and transform identities,
noncontiguous/read-only/unaligned inputs, sparse duplicates and empty shapes,
concurrent backend selection, backward and incomplete ODE solves, invalid
controls, and stale binaries during optional builds. CI now collects the
parametrized regressions with pytest and retains documentation doctests.

Invalid inputs fail earlier. Adaptive ODE integration no longer labels a
partial trajectory as successful or accepts a step that fails its tolerance
at the minimum step size. Mixed-radix inverse FFT and Nyquist resampling
corrections intentionally change previously incorrect numerical results.

Backend context isolation does not synchronize mutations to caller-owned
arrays; callers must still avoid writing to an array while another thread
uses it in a numerical operation. Timings vary with hardware, matrix shape,
BLAS, and callback cost. Further native-language additions should follow
profiling of the intended workload.

## Reproduce

Validation completed locally: 577 tests passed with the Rust backend; 530
passed with acceleration disabled, with 47 native-only tests skipped. The
13 Rust unit tests, formatting check, and Clippy with warnings denied passed.
Documentation examples, generated API pages, regenerated figures, and a strict
MkDocs build were checked. Native and portable wheels passed metadata and
isolated-import smoke tests; a simulated Cargo failure produced a portable
wheel without a stale extension. The editable native installation was also
rebuilt and checked.

```bash
python tools/bench_scalability.py --root /path/to/before --output before.json
python tools/bench_scalability.py --root /path/to/after --output after.json
python -m pytest -q
QUADRIVIUM_NO_ACCEL=1 python -m pytest -q
cargo fmt --manifest-path rust/Cargo.toml --check
cargo clippy --manifest-path rust/Cargo.toml --release --all-targets -- -D warnings
cargo test --manifest-path rust/Cargo.toml --release
```

Build the Rust backend in each source tree to compare compiled paths. Without
an importable extension, the benchmark records that it used Python instead.
