# The array layer

`quadrivium.numeric` is the array namespace the package computes with. It is
this project's own code — C sources under `csrc/`, compiled into
`quadrivium._qnp` — and it replaces what used to be a NumPy dependency. The
names, signatures and semantics are NumPy's, for the subset the library uses,
so code reads the same either way:

```pycon
>>> from quadrivium import numeric as np
>>> a = np.array([[4.0, 1.0], [1.0, 3.0]])
>>> np.linalg.solve(a, np.array([1.0, 2.0]))
array([0.09090909, 0.63636364])
>>> round(np.linalg.norm(a, 2), 6)
4.618034

```

You rarely need to import it. Every public function in the package accepts
lists, tuples, scalars and anything exporting a buffer, and returns arrays of
this type; the namespace is there for when you want to build or inspect one
directly.

## What it provides

Arrays are strided and N-dimensional over four dtypes — `bool`, `int64`,
`float64` and `complex128` — which is every dtype the numerical methods here
need. On top of that:

| | |
|---|---|
| Creation | `array`, `asarray`, `zeros`, `ones`, `empty`, `full`, the `_like` forms, `arange`, `linspace`, `eye`, `diag`, `tri`, `vander`, `meshgrid` |
| Shape | `reshape`, `transpose`, `ravel`, `concatenate`, `stack`, `hstack`, `vstack`, `roll`, `repeat`, `tile`, `pad`, `broadcast_to`, `diff` |
| Element-wise | arithmetic and comparison operators, `sqrt`, `exp`, `log`, the trigonometric and hyperbolic families, `sign`, `floor`, `clip`, `where`, `maximum`, `hypot` |
| Reductions | `sum`, `prod`, `mean`, `std`, `var`, `min`, `max`, `argmin`, `argmax`, `any`, `all`, `cumsum`, with `axis` and `keepdims` |
| Indexing | slices and views, `None`, `...`, boolean masks, integer arrays, `ix_`, `add.at`, and every assignment form of those |
| Linear algebra | `solve`, `inv`, `det`, `slogdet`, `cholesky`, `eig`, `eigh`, `eigvals`, `svd`, `lstsq`, `pinv`, `matrix_rank`, `qr`, `cond`, `norm` |
| Transforms | `fft`, `ifft`, `fft2`, `ifft2`, `rfft`, `irfft`, `fftfreq`, `fftshift` |
| Random | `default_rng` with `random`, `standard_normal`, `normal`, `uniform`, `integers`, `choice`, `permutation`, `shuffle`, `exponential`, `poisson` |
| Sorting | `sort`, `argsort`, `lexsort`, `searchsorted`, `unique`, `bincount`, `quantile`, `median`, `histogram` |
| Polynomials | `polyval`, `polyfit`, `poly`, `roots`, `polymul`, `polydiv`, `polyint`, and `polynomial.chebyshev` / `polynomial.legendre` |

## How closely it matches NumPy

Closely enough to be tested against it. `tests/test_numeric_backend.py` is a
differential suite of 363 tests that runs each operation on both libraries and
compares shapes, dtypes and values — exactly, wherever an exact answer exists:

- **Sums agree bit for bit.** Floating-point reductions use NumPy's pairwise
  scheme with the same block size, so `sum` over ten thousand values returns
  the identical double, not merely a close one.
- **Seeded generators produce identical streams.** `default_rng` reproduces
  NumPy's SeedSequence, its PCG64 bit generator, its ziggurat tables for the
  normal and exponential samplers, and its Lemire-bounded integers. Every
  seeded example in this documentation prints what it always did.
- **`repr` and `str` match character for character**, including column
  alignment, the switch to exponential notation, and summarised output.
- **Errors match in kind**: a broadcasting mismatch is a `ValueError`, an
  out-of-range index an `IndexError`, a singular system a `LinAlgError`.

Transcendental functions agree to within an ulp rather than exactly, since
NumPy's vectorised kernels and the system's libm already differ there by that
much.

Two differences are deliberate:

- `linalg.eigh` fixes the sign of each eigenvector so that its
  largest-magnitude entry is positive. LAPACK makes no such promise and its
  choice varies between builds; fixing it makes the decomposition reproducible
  across machines.
- `reciprocal` on an integer array promotes to float instead of performing
  integer division, which is a trap rather than a feature.

## Interoperating with NumPy

Arrays export the PEP 3118 buffer protocol, and the constructors accept
anything that does. So if you have NumPy, it passes both ways without either
library importing the other:

```python
import numpy
from quadrivium import numeric as np, linalg

data = numpy.random.default_rng(0).normal(size=(50, 4))
coefficients = linalg.qr_least_squares(data, data[:, 0])   # NumPy in
back = numpy.asarray(memoryview(coefficients))             # arrays out
```

## Printing

`set_printoptions` takes NumPy's option names — `precision`, `suppress`,
`threshold`, `edgeitems`, `linewidth` — and `printoptions` applies them for the
duration of a `with` block:

```pycon
>>> from quadrivium import numeric as np
>>> with np.printoptions(precision=3, suppress=True):
...     print(np.array([1.23456789, 1e-17]))
[1.235 0.   ]

```
