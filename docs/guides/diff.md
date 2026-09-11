# Differentiation

A derivative can come from nearby function evaluations, differentiation of a
program's arithmetic, or differentiation of a sampled interpolant. These
approaches solve different practical problems. Choose according to the
information you have, the function's smoothness, and the accuracy you need.

Quadrivium provides finite differences, scalar and array automatic
differentiation, and Fourier/Chebyshev differentiation. The
[differentiation reference](../api/diff.md) gives the signatures; this guide
explains the assumptions and how to check results.

```pycon
>>> from quadrivium import numeric as np
>>> from quadrivium import diff as df
>>> abs(df.central_difference(np.sin, 1.0, h=1e-5) - np.cos(1.0)) < 1e-9
True

```

## Select a source of derivative information

| Available information | Method | Main assumption |
| --- | --- | --- |
| Real scalar callable | Central or one-sided finite difference | Smooth and evaluable at perturbed points |
| Complex-compatible analytic callable | `complex_step_derivative` | Imaginary perturbations survive all operations |
| Scalar arithmetic written with supported operations | `derivative`, `gradient`, `hessian` | Automatic-differentiation objects remain in the computation |
| Array-valued program using supported tensor operations | `array_gradient`, `jvp`, `vjp` | Real-valued differentiable array primitives |
| Tabulated values on known coordinates | `differentiate_data` | Data resolve the desired derivative |
| Noisy uniformly spaced samples | `savitzky_golay_derivative` | Local polynomial is a useful smoothing model |
| Smooth periodic samples on a uniform grid | `fourier_derivative` | Correct period and no duplicated endpoint |
| Smooth function on a finite nonperiodic interval | `chebyshev_derivative` | Sampling can use clustered Chebyshev points |

A derivative of noisy measurements is an inference problem. An automatic
derivative is a derivative of the executed program. A spectral derivative is
a derivative of an interpolant. None should be accepted solely because the
returned number has many decimal places.

## Finite differences and the step-size tradeoff

Forward and backward differences use points on one side; a central difference
uses symmetric points and usually has a smaller truncation error for a smooth
interior point. The basic first-derivative central formula is
`(f(x+h)-f(x-h))/(2*h)`.

A large `h` blurs curvature and increases truncation error. A tiny `h` subtracts
nearly equal values, magnifies rounding, and may even produce `x+h == x`.
There is therefore a useful intermediate step range rather than a rule that
smaller is always better.

```pycon
>>> point = 1.0
>>> reference = np.exp(point)
>>> coarse_error = abs(df.central_difference(np.exp, point, h=0.1) - reference)
>>> refined_error = abs(df.central_difference(np.exp, point, h=1e-4) - reference)
>>> refined_error < coarse_error
True

```

The public `forward_difference`, `backward_difference`, and
`central_difference` routines have fixed default steps. They do not silently
optimize a step for each problem. `optimal_step_size` provides a scale-based
heuristic, and the `core.numerical_*` helpers use their own coordinate-scaled
defaults. Different interfaces need not share the same default.

<figure markdown="span">
  ![Absolute derivative error over a range of finite-difference step sizes](../assets/figures/diff-step-selection.svg#only-light)
  ![Absolute derivative error over a range of finite-difference step sizes](../assets/figures/diff-step-selection-dark.svg#only-dark)
  <figcaption>For the derivative of exp(x) at x=1, finite differences balance truncation and roundoff. Complex-step evaluation avoids subtracting nearby real values for this analytic function. The displayed error floor marks values below the plotting threshold.</figcaption>
</figure>

Choose a step relative to the coordinate scale and function noise. Test several
nearby choices and look for a stable range of answers. At a domain boundary,
use a one-sided rule or an analytic reformulation; a central rule may evaluate
outside the domain even if the requested point itself is valid.

### Higher derivatives and extrapolation

`second_derivative`, `third_derivative`, and `fourth_derivative` supply common
stencils. Their denominators contain higher powers of `h`, making noise and
roundoff more severe. `five_point_stencil` estimates a first derivative with
a higher formal truncation order.

```pycon
>>> abs(df.second_derivative(lambda x: x*x, 1.0, h=1e-3) - 2.0) < 1e-8
True
>>> abs(df.five_point_stencil(np.sin, 1.0, h=1e-3) - np.cos(1.0)) < 1e-10
True

```

Richardson extrapolation combines estimates at a sequence of steps to cancel
leading truncation terms. `richardson_derivative` applies this to central
stencils; `richardson_extrapolation` accepts a more general step-dependent
estimate. The assumed expansion must hold, and excessive refinement eventually
amplifies floating-point noise.

```pycon
>>> extrapolated = df.richardson_derivative(np.sin, 1.0, h=0.1, levels=5)
>>> abs(extrapolated - np.cos(1.0)) < 1e-11
True

```

`fornberg_weights` creates weights on specified distinct nodes.
`finite_difference_weights` returns offsets and weights for standard stencil
choices. `differentiation_matrix` assembles a dense operator for supplied
coordinates. A dense matrix is useful for analysis and moderate grids but
requires quadratic storage even when each row uses only a short stencil.

## Complex-step differentiation

For a real-valued analytic function continued into the complex plane,
`imag(f(x+1j*h))/h` estimates the first derivative without real subtraction.
Very small `h` can then be useful because the imaginary component carries the
perturbation directly.

```pycon
>>> complex_estimate = df.complex_step_derivative(np.exp, 1.0, h=1e-20)
>>> abs(complex_estimate - np.e) < 1e-14
True

```

The callable must accept complex input and preserve its imaginary part.
`abs`, clipping, real casts, conjugation, and branch logic can invalidate the
analytic argument or erase the perturbation. A function that silently drops
imaginary parts can return a plausible but incorrect zero derivative.
Check the entire computation, including external routines it calls.

Complex step is primarily a first-derivative technique here. It is not the
same operation as differentiating a complex-valued function with respect to
complex variables, and it does not make a nonsmooth model differentiable.

## Scalar automatic differentiation

Forward-mode AD propagates a value and tangent through supported arithmetic.
`Dual` implements this mechanism, and `derivative(f, x)` seeds a unit tangent.
Use arithmetic on the supplied object and its supported elementary methods.

```pycon
>>> df.derivative(lambda x: x*x*x + 2*x, 2.0)
14.0
>>> abs(df.derivative(lambda x: x.sin(), 1.0) - np.cos(1.0)) < 1e-14
True

```

The derivative has no finite-difference truncation error, but still uses
floating-point arithmetic. The word "exact" in an AD description refers to
the chain rule on supported operations, not symbolic exactness or immunity to
rounding.

The scalar AD interfaces pass lists of scalar AD objects to multivariate
callables. Write scalar expressions using indexing and supported arithmetic;
do not convert those objects to ordinary numeric arrays or floats inside the
function, because that discards derivative information.

```pycon
>>> objective = lambda x: x[0]**2 + 3*x[1]**2 + x[0]*x[1]
>>> value, gradient = df.value_and_grad(objective, [1.0, 2.0])
>>> value, gradient.tolist()
(15.0, [4.0, 13.0])
>>> np.allclose(df.forward_gradient(objective, [1.0, 2.0]), gradient)
True

```

Reverse-mode `gradient` records operations and performs a backward sweep to
obtain all scalar-objective partial derivatives. Forward mode needs one pass
per input direction, making it attractive for few inputs or directional
questions. `jacobian` uses reverse sweeps per output; `forward_jacobian` uses
forward passes per input. For `F: R^n -> R^m`, both return shape `(m, n)`.

AD follows the branch executed at the evaluation point. A branch can be
nonsmooth at its switching surface even when AD returns a derivative on one
side. Constant-output expressions must retain a supported AD object in the
older scalar interfaces; returning a plain Python constant may not satisfy
the expected output protocol.

### Second derivatives and Hessian products

`HyperDual` propagates mixed second derivatives. `second_derivative_ad` handles
one scalar variable and `hessian` builds a dense Hessian using one function
evaluation per distinct coordinate pair.

```pycon
>>> df.second_derivative_ad(lambda x: x**4, 2.0)
48.0
>>> df.hessian(objective, [1.0, 2.0]).tolist()
[[2.0, 1.0], [1.0, 6.0]]
>>> df.hessian_vector_product(objective, [1.0, 2.0], [1.0, -1.0]).tolist()
[1.0, -5.0]

```

`hessian_vector_product` avoids storing the full Hessian but performs one
hyper-dual evaluation per coordinate in this implementation. It is useful
when only `H @ v` is needed, but it is not a constant-cost reverse-over-forward
Hessian-product implementation.

`taylor_coefficients`, despite residing alongside AD helpers, uses a finite
difference stencil. Its coefficients are ascending Taylor coefficients and
inherit the step-size sensitivity of higher numerical derivatives.

## Array differentiation and directional products

`Tensor` represents a differentiable real array. Supported operations include
broadcasting arithmetic, indexing, sums and means, reshape and transpose,
matrix multiplication, and selected smooth elementary functions.
`array_value_and_grad` returns a scalar objective value and a gradient with
the same shape as the input.

```pycon
>>> array_objective = lambda a: (a*a).sum()
>>> value, gradient = df.array_value_and_grad(array_objective,
...                                         np.array([[1.0, 2.0], [3.0, 4.0]]))
>>> value, gradient.tolist()
(30.0, [[2.0, 4.0], [6.0, 8.0]])

```

`jvp(f, x, tangent)` returns `(f(x), J @ tangent)` in a forward pass.
`vjp(f, x, cotangent)` returns `(f(x), J.T @ cotangent)` using a reverse
pullback. Neither constructs the full Jacobian. The tangent has the input
shape; the cotangent has the output shape.

```pycon
>>> value, directional = df.jvp(lambda a: a*a, [1.0, 2.0], [3.0, 4.0])
>>> value.tolist(), directional.tolist()
([1.0, 4.0], [6.0, 16.0])
>>> value, pullback = df.vjp(lambda a: a*a, [1.0, 2.0])
>>> pullback([1.0, 1.0]).tolist()
[2.0, 4.0]

```

Without a cotangent, `vjp` returns a reusable pullback callable. Each call uses
the recorded forward state; changing the original input does not update the
tape. Construct a fresh evaluation when the input changes.

Tensor data are snapshots exposed through read-only views. Mutation, `out=`
operations, complex differentiation, and conversion to `float` are deliberately
restricted. Use supported tensor operations throughout the differentiated
function. A plain external array routine is not automatically differentiable
because it accepts the same numerical values.

## Derivatives from data

`differentiate_data(x, y, order, stencil)` applies a moving finite-difference
stencil to known coordinates, including nonuniform coordinates. It returns one
value per sample. Distinct ordered finite coordinates and matching values are
necessary for meaningful weights.

```pycon
>>> coordinates = np.linspace(0.0, 1.0, 11)
>>> samples = coordinates**2
>>> derivative_samples = df.differentiate_data(coordinates, samples)
>>> np.allclose(derivative_samples, 2*coordinates, atol=1e-10)
True

```

Noise in a first difference is amplified roughly in proportion to `1/h`.
More closely spaced noisy samples therefore do not automatically improve a
pointwise derivative estimate. A Savitzky-Golay derivative fits a local
polynomial and differentiates that fit, trading spatial resolution for lower
noise sensitivity.

Use a positive uniform spacing `dx`, a window no longer than the data, and a
polynomial order high enough for the requested derivative but below the window
size. Even window lengths are increased to the next odd length. This
implementation pads with endpoint values, so estimates near either boundary
need separate scrutiny.

```pycon
>>> smoothed = df.savitzky_golay_derivative(samples, window=5, poly_order=2,
...                                       order=1, dx=0.1)
>>> np.allclose(smoothed[2:-2], 2*coordinates[2:-2], atol=1e-10)
True

```

Select the window using the feature width and noise level, not only the sample
count. Compare boundaries and interior separately and avoid interpreting a
smoothed derivative as an exact reconstruction of a sharp transition.

## Fourier and Chebyshev differentiation

`fourier_derivative` differentiates the periodic Fourier interpolant. Supply
one period of uniformly spaced samples, excluding the repeated endpoint, and
set the physical period with `L`.

```pycon
>>> grid = np.linspace(0.0, 2*np.pi, 64, endpoint=False)
>>> periodic_derivative = df.fourier_derivative(np.sin(grid), L=2*np.pi)
>>> float(np.max(np.abs(periodic_derivative - np.cos(grid)))) < 1e-12
True

```

The Fourier derivative is highly accurate for a resolved smooth periodic
function. Endpoint mismatch, jumps, insufficient sampling, and noise prevent
that behavior. The length `L` scales derivative units; an incorrect period
produces an incorrect derivative even with an otherwise accurate transform.

`chebyshev_derivative(f, n, a, b)` evaluates on `n+1` increasing Lobatto points
and returns `(coordinates, derivatives)`. `chebyshev_diff_matrix` returns
`(D, coordinates)` instead. These return orders are intentionally different.

```pycon
>>> cheb_x, cheb_d = df.chebyshev_derivative(np.exp, n=24, a=-1.0, b=1.0)
>>> cheb_x.size
25
>>> float(np.max(np.abs(cheb_d - np.exp(cheb_x)))) < 1e-10
True

```

In this submodule `chebyshev_points(n)` means **n+1 points**, unlike
`interpolate.chebyshev_nodes(n)`, which means n nodes. Preserve the returned
coordinates when using differentiation matrices. A uniform sample vector
cannot be substituted merely because it has the same length.

Validate derivatives against analytic identities, directional differences,
or a refinement study. Feed checked gradients into [optimization](optimize.md),
checked Jacobians into [root finding](rootfind.md), and appropriate spatial
operators into the [PDE methods](pde.md).
