# Arrays and the numeric namespace

`quadrivium.numeric` supplies the arrays used by Quadrivium's algorithms. It is
implemented by the project's compiled C extension and Python helpers. NumPy is
not required at runtime. The familiar `np` alias in these guides refers to
**Quadrivium's numeric namespace**, unless an example explicitly imports NumPy.

```pycon
>>> from quadrivium import numeric as np
>>> A = np.array([[4.0, 1.0], [1.0, 3.0]])
>>> b = np.array([1.0, 2.0])
>>> x = np.linalg.solve(A, b)
>>> np.allclose(A @ x, b)
True

```

The namespace implements a substantial subset of NumPy-style operations.
It is not a promise that arbitrary NumPy programs, dtypes, or keyword options
work unchanged. Use this guide as the numeric overview, inspect operation docstrings for
options, and test the operations used by an external integration.

## Choose a representation deliberately

Arrays are strided and can have multiple dimensions. Supported storage types
are `bool`, `int64`, `float32`, `float64`, `complex64`, and `complex128`.
Python floating-point input normally produces double precision. Request a
dtype explicitly when memory, precision, or complex arithmetic matters.

| Dtype | Bytes per element | Typical use |
| --- | ---: | --- |
| `bool` | 1 | Masks and logical flags |
| `int64` | 8 | Indices and integer counts |
| `float32` | 4 | Compact real-valued data |
| `float64` | 8 | Default numerical calculations |
| `complex64` | 8 | Compact complex-valued data |
| `complex128` | 16 | Complex numerical calculations |

```pycon
>>> compact = np.array([1.0, 2.0, 3.0], dtype=np.float32)
>>> compact.shape, compact.ndim, compact.size
((3,), 1, 3)
>>> compact.itemsize, compact.nbytes
(4, 12)
>>> z = np.array([1+2j, 3-1j], dtype=np.complex128)
>>> np.conjugate(z).tolist()
[(1-2j), (3+1j)]

```

Storage precision and solver precision are separate decisions. Many educational
routines convert inputs to real `float64` through `core.as_vector` or
`core.as_matrix`. Several native array operations and newer factorization
interfaces preserve complex or compact input. Check the method's contract
before assuming an entire computation stays in the original dtype.

Object arrays, arbitrary-width integer types, string arrays, and structured
record dtypes are outside this array layer's supported storage model.

## Build arrays and grids

Use `array` for a new array from data, `asarray` for conversion that may avoid a
copy, and `zeros`, `ones`, or `full` for initialized storage. `empty` allocates
storage without initializing its contents: assign every element before use.

```pycon
>>> np.zeros((2, 3)).shape
(2, 3)
>>> np.full((2, 2), 7).tolist()
[[7, 7], [7, 7]]
>>> np.eye(3).tolist()
[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

```

`arange` describes a step and excludes its stop. `linspace` describes the
number of samples and includes both endpoints by default. For a periodic
signal, exclude the repeated endpoint so the first and last sample are not
two representations of the same location.

```pycon
>>> np.arange(0, 6, 2).tolist()
[0, 2, 4]
>>> np.linspace(0.0, 1.0, 5).tolist()
[0.0, 0.25, 0.5, 0.75, 1.0]
>>> periodic_grid = np.linspace(0.0, 2*np.pi, 8, endpoint=False)
>>> periodic_grid.size, bool(periodic_grid[-1] < 2*np.pi)
(8, True)

```

A grid determines which features a sampled calculation can resolve. Adding
precision to the dtype does not replace adding samples where a signal varies
rapidly. Check a calculation again on a refined grid and compare a quantity
that matters: interpolation error, an integral, a derivative, or a solution.

<figure markdown="span">
  ![Interpolation error for a sampled signal as the grid is refined](../assets/figures/numeric-grid-resolution.svg#only-light)
  ![Interpolation error for a sampled signal as the grid is refined](../assets/figures/numeric-grid-resolution-dark.svg#only-dark)
  <figcaption>Linear interpolation reconstructs sin(6πx) from grids of 9 to 257 points, measured against an independent 2,001-point grid. The second-order RMS trend shows sampling error decreasing under refinement; storing the coarse grid at higher precision would not supply its missing detail.</figcaption>
</figure>

## Shapes, axes, and broadcasting

An array's shape is part of the mathematical model. In a `(samples, variables)`
matrix, `axis=0` reduces across samples, leaving one value per variable;
`axis=1` reduces across variables, leaving one value per sample.
`keepdims=True` retains a length-one axis, making the result easy to broadcast
back over the original array.

```pycon
>>> data = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
>>> data.mean(axis=0).tolist()
[3.0, 20.0]
>>> centers = data.mean(axis=0, keepdims=True)
>>> centers.shape, (data - centers).shape
((1, 2), (3, 2))
>>> np.allclose((data - centers).mean(axis=0), [0.0, 0.0])
True

```

Broadcasting aligns trailing dimensions. Two aligned lengths are compatible
when they are equal or one of them is one. A vector with shape `(n,)` aligns
with the last axis; `v[:, None]` instead makes it a column of shape `(n, 1)`.

```pycon
>>> row = np.array([10.0, 20.0, 30.0])
>>> column = np.array([1.0, 2.0])[:, None]
>>> (column + row).tolist()
[[11.0, 21.0, 31.0], [12.0, 22.0, 32.0]]

```

Broadcasting can express a large calculation without explicitly tiling inputs,
but the result and intermediates still occupy memory. An expression forming
all pairwise distances between two large point sets can create an enormous
array even when each input is modest. Process blocks when the full result is
not needed at once.

## Views, copies, and indexing

Basic slicing commonly creates views sharing the same storage. Editing a view
can therefore change the source. Use `.copy()` when the source must remain
independent. `shares_memory` can check the relationship directly.

```pycon
>>> original = np.array([1.0, 2.0, 3.0, 4.0])
>>> view = original[1:3]
>>> np.shares_memory(original, view)
True
>>> view[0] = 9.0
>>> original.tolist()
[1.0, 9.0, 3.0, 4.0]
>>> independent = original.copy()
>>> np.shares_memory(original, independent)
False

```

Integer-array and boolean-mask indexing select entries by explicit indices or
a condition. Mask operations are elementwise: use `&`, `|`, and `~`, with
parentheses around comparisons, rather than Python's scalar `and` and `or`.

```pycon
>>> selected = original[(original >= 3) & (original <= 4)]
>>> selected.tolist()
[3.0, 4.0]
>>> original[[0, 3]].tolist()
[1.0, 4.0]

```

`reshape`, transpose, and strided slices can produce non-contiguous arrays.
Use `ascontiguousarray` when an external consumer requires contiguous storage.
Do not infer whether an operation copies only from the values it prints.

## Arithmetic and matrix operations

`*` multiplies elementwise. `@` performs matrix multiplication, with vector
and batch rules determined by the trailing dimensions. A transpose is enough
for real data; use the conjugate transpose for complex inner products.

```pycon
>>> M = np.array([[1.0, 2.0], [3.0, 4.0]])
>>> (M * M).tolist()
[[1.0, 4.0], [9.0, 16.0]]
>>> (M @ M).tolist()
[[7.0, 10.0], [15.0, 22.0]]
>>> np.allclose(np.conjugate(z) @ z, 15.0)
True

```

The `linalg` namespace includes solves, decompositions, singular values,
eigenvalues, least squares, pseudoinverses, rank, norms, and conditioning.
Batched dense operations use trailing matrix dimensions. When supplying
multiple right-hand sides, make their shape explicit, such as `(n, nrhs)`.
Use the [linear algebra guide](linalg.md) for algorithm selection and the
[workflow guide](workflows.md) for complete batch and complex examples.

## Randomness, reductions, and reproducibility

Create a generator with an explicit seed for reproducible examples. Reusing
one generator advances its state; creating another with the same seed restarts
the sequence. Supported methods include normal and uniform samples, integers,
choice, permutations, exponential samples, and Poisson samples.

```pycon
>>> rng1 = np.random.default_rng(17)
>>> rng2 = np.random.default_rng(17)
>>> np.array_equal(rng1.standard_normal(5), rng2.standard_normal(5))
True

```

Compatibility tests compare supported operations against NumPy, including
seeded streams and many reduction behaviors. This is useful evidence about
tested cases, not a guarantee about every NumPy release, platform, operation,
or floating-point evaluation order. Record the package version, seed, method,
and tolerances for a reproducible scientific calculation.

`allclose` is convenient for numerical comparisons, but its default tolerances
may be inappropriate near zero or for very small physical quantities. Set
`atol` and `rtol` from the quantity being checked. `array_equal` checks exact
entry equality instead.

## Save arrays and exchange data

`save` and `load` support numeric `.npy` files without a NumPy dependency.
Saving a path without the extension adds `.npy`; loading uses the path given.
Only supported fixed-width numeric dtypes are accepted, and object/pickle
payloads are not loaded.

```pycon
>>> import io
>>> stream = io.BytesIO()
>>> np.save(stream, data)
>>> _ = stream.seek(0)
>>> restored = np.load(stream)
>>> np.array_equal(restored, data)
True

```

`load(path, mmap_mode="r")` maps a file for read-only access; `"r+"` supports
persistent writes and `"c"` provides copy-on-write access. `open_memmap` can
create a mapped array. Mapping avoids loading the entire file eagerly, but
subsequent arithmetic can still allocate full-size arrays. Mapping requires a
path and native-endian supported data. See [workflows](workflows.md) for file
and lifetime examples.

Arrays export the PEP 3118 buffer protocol, and constructors accept compatible
buffer exporters. If NumPy is separately installed, it can exchange data
through that protocol:

```python
import numpy
from quadrivium import numeric as np

external = numpy.array([1.0, 2.0, 3.0])
internal = np.asarray(external)
back = numpy.asarray(memoryview(internal))
```

Check copying, dtype, contiguity, writability, and ownership at an external
boundary. A buffer exchange is not a promise that every external library
recognizes Quadrivium arrays directly.

## Display and further reading

Use `printoptions` to apply formatting locally. Display precision changes only
what is printed; it does not round the stored data or improve its accuracy.

```pycon
>>> with np.printoptions(precision=3, suppress=True):
...     print(np.array([1.23456789, 1e-17]))
[1.235 0.   ]

```

The namespace also provides Fourier transforms, sorting and searching,
polynomial helpers, histogram and quantile operations, and testing helpers.
Continue with [transforms](transforms.md), [approximation](approx.md), or the
[API overview](../api/index.md) for those operations. A missing
`_qnp` extension is an installation problem; follow [installation](../installation.md)
to build or install the compiled package before running these examples.
