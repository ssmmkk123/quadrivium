# Performance and reliability improvements

Measured on Python 3.14.4 and Linux x86-64; the earlier passes used NumPy 2.4.4
as the array backend, which the last section replaces.
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

The sections below cover a second pass over the same tree. Its baseline is the
state after the work above, so its numbers compose with rather than repeat
them. It ran the whole public API through a 187-case timing sweep, then went
after what that sweep put at the top: routines whose inner loop was still a
Python `for`, and two eigenvalue iterations that were spending their entire
budget without converging.

## Array-native special functions

Forty-three of the fifty-one functions in `quadrivium.special` accepted only
scalars, so evaluating one over an array meant a Python loop at the call site.
Each series, continued fraction, recurrence and AGM iteration now advances the
whole input in one pass. The "before" column is that Python loop, since the
array call itself raised `TypeError`.

| Function | Points | Loop (ms) | Array (ms) | Speedup |
|---|---:|---:|---:|---:|
| `fresnel_c` | 5 000 | 3436.71 | 4.684 | 734× |
| `airy_ai` | 5 000 | 38.23 | 0.711 | 54× |
| `associated_legendre` | 5 000 | 2.15 | 0.053 | 40× |
| `erfcx` | 20 000 | 41.32 | 1.757 | 24× |
| `bessel_i0` | 5 000 | 4.47 | 0.184 | 24× |
| `dawson` | 20 000 | 151.67 | 6.940 | 22× |
| `bessel_j0` | 20 000 | 34.62 | 1.636 | 21× |
| `beta` | 20 000 | 47.78 | 2.518 | 19× |
| `struve_h0` | 5 000 | 5.89 | 0.290 | 20× |
| `elliptic_k` | 5 000 | 1.48 | 0.082 | 18× |
| `spherical_bessel_j` | 5 000 | 3.44 | 0.189 | 18× |
| `bessel_k0` | 20 000 | 130.45 | 8.966 | 15× |
| `hyp1f1` | 5 000 | 10.71 | 0.704 | 15× |
| `elliptic_e` | 5 000 | 4.37 | 0.310 | 14× |
| `hyp2f1` | 5 000 | 12.02 | 1.051 | 11× |
| `polygamma` | 5 000 | 12.79 | 1.120 | 11× |
| `lambert_w` | 20 000 | 13.80 | 1.396 | 10× |
| `expint_n` | 5 000 | 15.44 | 1.648 | 9× |
| `zeta` | 2 000 | 4.77 | 0.527 | 9× |
| `digamma` | 20 000 | 5.74 | 0.739 | 8× |
| `erfinv` | 20 000 | 11.46 | 2.286 | 5× |
| `sine_integral` | 5 000 | 7.60 | 1.494 | 5× |

Values are unchanged. All 51 functions were replayed over their domains against
the previous implementation and agree to 1e-12 relative or better; the two
exceptions are deliberate accuracy fixes, described below.

## Python inner loops removed

Routines whose per-element or per-row work ran in the interpreter, and now runs
in NumPy or in a compiled kernel:

| Routine | Workload | Before (ms) | After (ms) | Speedup |
|---|---|---:|---:|---:|
| `linalg.ilu0` | 200 × 200 | 690.65 | 6.216 | 111× |
| `stochastic.ks_test` | 20 000 samples | 29.28 | 0.230 | 127× |
| `linalg.givens_qr` | 120 × 120 | 52.59 | 0.842 | 62× |
| `linalg.ldl_decomposition` | 200 × 200 | 43.48 | 0.718 | 61× |
| `linalg.svd_jacobi` | 80 × 80 | 200.95 | 3.769 | 53× |
| `linalg.doolittle` | 200 × 200 | 34.02 | 0.947 | 36× |
| `linalg.crout` | 200 × 200 | 32.64 | 0.934 | 35× |
| `pde.fem_2d_triangular` | n = 32, assembly | 22.86 | 1.246 | 18× |
| `interpolate.bezier` | 40 points, 2 000 samples | 90.12 | 5.432 | 17× |
| `stochastic.halton` | 4 096 × 8 | 11.29 | 0.693 | 16× |
| `stochastic.sobol` | 4 096 × 8 | 11.16 | 0.690 | 16× |
| `pde.heat_2d_adi` | 48 × 48, 100 steps | 39.53 | 3.762 | 10.5× |
| `transforms.dwt2` | 256 × 256, db4 | 17.70 | 1.987 | 8.9× |
| `linalg.gauss_jordan` | 150 × 150 | 12.97 | 1.583 | 8.2× |
| `interpolate.newton_divided_differences` | 200 nodes | 6.19 | 0.946 | 6.5× |
| `interpolate.barycentric` | 400 nodes, 5 000 points | 28.35 | 4.552 | 6.2× |
| `linalg.svd_golub_kahan` | 80 × 80 | 28.06 | 4.522 | 6.2× |
| `linalg.modified_gram_schmidt_qr` | 200 × 200 | 38.57 | 6.949 | 5.6× |
| `linalg.jacobi_eigen` | 120 × 120 | 93.64 | 18.659 | 5.0× |
| `rootfind.durand_kerner` | degree 30 | 19.93 | 4.867 | 4.1× |
| `linalg.inverse` | 200 × 200 | 39.69 | 14.596 | 2.7× |

The discrete cosine and sine transforms were a separate case: `dct` types III
and IV, `dst` type II and `idct` were written as their defining sums, an
`O(n^2)` double loop with the outer one in Python. Each is now a single
length-`2n` inverse FFT with a twiddle on one or both ends.

| Transform | n = 512 | n = 2 048 | n = 8 192 |
|---|---:|---:|---:|
| `dct` III before / after | 4.74 / 0.03 ms | 37.16 / 0.094 ms | ~600 / 0.504 ms |
| `dct` IV before / after | 4.86 / 0.04 ms | 38.05 / 0.123 ms | ~600 / 0.641 ms |
| `dst` II before / after | 4.96 / 0.03 ms | 38.14 / 0.094 ms | ~600 / 0.522 ms |
| `idct` before / after | 4.78 / 0.03 ms | 36.39 / 0.096 ms | ~600 / 0.514 ms |

Across a 187-case sweep of the public API the aggregate is 1.60×. Routines
dominated by a user callback -- the ODE and SDE integrators, the optimizers --
are unchanged: 55 000 of the 75 000 calls in a `newton_cg` solve are the
caller's objective, so the framework around it is not what costs.

## Replacing NumPy with a C array core

`quadrivium.numeric` -- the package's own array layer, compiled from `csrc/` --
replaced NumPy. The comparison below is the same tree before and after that
change: one process per case, threads pinned to one, a warm-up and the median
of twenty-five timed samples.

| Operation | Workload | NumPy backend (ms) | C backend (ms) | | Peak RSS before | Peak RSS after |
|---|---|---:|---:|---:|---:|---:|
| Least squares, compiled kernels | 1500 x 12 | 0.148 | 0.150 | unchanged | 36 MiB | 24 MiB |
| Least squares, Python path | 1500 x 12 | 0.470 | 0.482 | unchanged | 37 MiB | 24 MiB |
| Gaussian density estimate | 20000 samples x 400 points | 32.348 | 16.623 | 1.95x faster | 38 MiB | 25 MiB |
| Sparse identity | 2000 x 2000 | 0.015 | 0.014 | 1.12x faster | 36 MiB | 23 MiB |
| CSR matrix-vector product | 100000 nonzeros | 0.974 | 1.002 | unchanged | 43 MiB | 29 MiB |
| FFT, Python path | 8192 samples | 0.443 | 0.325 | 1.36x faster | 37 MiB | 24 MiB |
| Welch PSD, Python path | 65536 samples; 256 per segment | 24.731 | 14.375 | 1.72x faster | 37 MiB | 24 MiB |

Peak resident memory falls across the board because the NumPy import no longer
happens; the tracked allocations of the workloads themselves are unchanged.

Three things carry the speed. Element-wise loops have unit-stride fast paths the
compiler vectorises, and `exp` is a vectorised Cody-Waite reduction accurate to
one ulp of libm, which is where the density estimate's gain comes from. The
matrix product is a packed kernel with an AVX2/FMA micro-kernel holding a 4x8
tile of `C` in registers, reaching about 46 GFLOPS single-threaded. And
indexing has direct paths for the two idioms this package runs hardest --
`a[mask]` and `a[indices]`, in both directions -- which is what keeps the
sparse matrix-vector product level.

Nothing is retained between operations: an array's buffer goes back to the
allocator when the array dies, which `test_temporaries_do_not_accumulate` in
`tests/test_numeric_backend.py` holds to by measuring peak RSS over four
hundred iterations of large temporaries.

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

From the second pass:

- `transforms.dft_matrix(1024)`: tracked peak fell from 50.33 MB to 25.31 MB.
  Building the exponent table with `meshgrid` materialized two full `n × n`
  index arrays before the one that was wanted; an outer product of the index
  range does not.
- `transforms.dwt2` (512 × 512, db4): 8.64 MB to 5.97 MB, by transforming all
  rows together instead of accumulating a Python list of row results and
  calling `np.array` on it.
- Working sets that scale with the *query* rather than the problem are now
  blocked, so peak allocation stops growing once the block is reached.
  `interpolate.bezier` over a 30-point control polygon: 28.6 MB at 20 000
  samples and 52.5 MB at 200 000 before blocking, 6.6 MB and 9.5 MB after --
  the remaining growth is the output itself. `interpolate.barycentric` with
  200 nodes: 25.7 MB to 6.6 MB at 20 000 points. `special.bessel_k0` evaluates
  its doubly-exponential quadrature over sorted blocks of 512 arguments, so
  the `(points × nodes)` table never exists whole.
- `interpolate.lagrange` reuses three scratch arrays across all `n^2` factors
  rather than allocating three temporaries per factor: 0.16 MB to 0.12 MB at
  5 000 points, and 1.2× faster, while staying bit-identical to the previous
  implementation on every case tested.
- `pde.fem_2d_triangular` assembly trades memory for speed in the other
  direction: forming every element matrix at once and scattering with
  `np.add.at` raised tracked peak from 9.58 MB to 10.31 MB at n = 32, for
  18× less time. The block scales with the element count, not with the dense
  system it assembles into.

Tracemalloc captures Python-side allocations but not those a compiled kernel
makes for itself. Native workloads must also be judged by the recorded process
peak RSS, which includes the interpreter and its inputs. Raw samples and memory
measurements are in [before](benchmarks/scalability-before.json) and
[after](benchmarks/scalability-after.json) for the earlier passes, and in
[NumPy backend](benchmarks/scalability-numpy-backend.json) and
[C backend](benchmarks/scalability-c-backend.json) for the array core.

## Reliability and compatibility

The public numerical algorithms remain visible in Python, with optional Rust
kernels. There is no longer any runtime dependency at all: the array layer that
provides the gains in sparse storage, density estimation and Python transforms
is `quadrivium.numeric`, compiled from this project's own C sources. The Rust
build declares the 1.83 minimum required by its PyO3 bindings.

Regression coverage includes independent residual and transform identities,
noncontiguous/read-only/unaligned inputs, sparse duplicates and empty shapes,
concurrent backend selection, backward and incomplete ODE solves, invalid
controls, and stale binaries during optional builds. CI now collects the
parametrized regressions with pytest and retains documentation doctests.

Two eigenvalue iterations could not converge on a matrix of any appreciable
size. `jacobi_eigen` and `qr_algorithm` both compared the off-diagonal mass
against `tol` without reference to the size of the matrix, and rounding holds
that mass near `eps*||A||`: at `||A||` of a few hundred the default `1e-12`
is unreachable, so both spent their whole budget and then reported failure on
an answer found in the first few sweeps. Jacobi's test is now an absolute
bound tightened to a relative one below unit norm, plus a stagnation exit --
its off-diagonal mass falls monotonically, so a sweep that fails to reduce it
has reached the floor. `qr_algorithm` adopts the relative per-subdiagonal
deflation test its siblings already used, and is now scale-invariant: the same
symmetric problem takes the same 1607 iterations to the same accuracy whether
it is scaled by `1e-4`, `1`, or `1e4`.

`barycentric` scales its node differences by the interval's logarithmic
capacity before multiplying them. The unscaled product runs like `capacity^n`
and overflowed to `inf` for 400 Chebyshev nodes on `[0, 1]` -- every weight
`inf`, every interpolated value `nan` -- while the same problem on `[-1, 1]`
worked. `lobpcg` no longer pairs eigenvalues from one iteration with
eigenvectors from the next when it runs out of iterations. `sine_integral` and
`cosine_integral` switch to the continued fraction for `E_1(ix)` past `x = 4`,
where the ascending series had lost up to ten digits to cancellation.

`gram_schmidt_qr` and `modified_gram_schmidt_qr` now project against the
already-computed block in one matrix-vector product rather than one column at a
time. The arithmetic is the same operation summed in a different order, which
is visible only where the summation order is the subject: the two
QR-conditioning figures in the documentation shift slightly. The behaviour they
illustrate is unchanged -- classical Gram-Schmidt still loses orthogonality
like `cond(A)^2`, modified like `cond(A)`, and Householder stays at machine
precision across the whole range.

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

Validation completed locally: 606 tests and 1709 subtests passed with the Rust
backend; 557 tests and 1502 subtests passed with acceleration disabled, with 49
native-only tests skipped. The 17 Rust unit tests, formatting check, and Clippy
with warnings denied passed.
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
