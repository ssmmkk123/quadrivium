"""Tail accuracy and bounded workspace for vector complementary error function."""
from decimal import Decimal, localcontext
import math
import tracemalloc

import numpy as np
import pytest

from quadrivium import _accel, numeric as a


def _erfc_reference(x):
    if not math.isfinite(x) or x < 26:
        return math.erfc(x)
    # Some system libms flush the last few representable erfc values to zero.
    # For x >= 26, use the independent asymptotic expansion in 80-digit
    # arithmetic. Its remainder is bounded by the first omitted term; here
    # that is below 1e-70 relative, far below binary64 rounding uncertainty.
    with localcontext() as context:
        context.prec = 80
        value = Decimal.from_float(float(x))
        square = value * value
        pi = Decimal("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628")
        term = total = Decimal(1)
        for n in range(1, 100):
            term *= -Decimal(2 * n - 1) / (2 * square)
            total += term
            if abs(term) < Decimal("1e-70"):
                break
        else:
            raise AssertionError("erfc reference did not reach its error bound")
        return float((-square).exp() * total / (value * pi.sqrt()))


@pytest.mark.skipif(not _accel.available(), reason="C acceleration disabled")
@pytest.mark.parametrize("stride", [1, 2, -1, -3])
def test_vector_erfc_matches_reference_across_branches_and_subnormal_tails(stride):
    x = np.concatenate((np.linspace(-30., 30., 20001),
                        [np.nan, -np.inf, np.inf, 0., -0., 1.25, 2.8571414947509766,
                         np.nextafter(1.25, 0), np.nextafter(1.25, 2), 27., 27.2]))
    values = a.array(x)[::stride]
    values.flags.writeable = False
    expected = np.array([_erfc_reference(v) for v in np.asarray(values)])
    actual = np.asarray(_accel.kernel("erfc")(values))
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=5e-324, equal_nan=True)


@pytest.mark.skipif(not _accel.available(), reason="C acceleration disabled")
def test_vector_erfc_preserves_last_representable_tail_values():
    # 80-digit reference values begin 1.0189049142703155e-323,
    # 5.91118574866365e-324, 3.428694222673064e-324, 1.988364974232363e-324.
    # Their correctly rounded doubles include both final nonzero levels.
    x = np.tile([27.2, 27.21, 27.22, 27.23], 32)
    expected = np.tile([1e-323, 5e-324, 5e-324, 0.], 32)
    actual = np.asarray(_accel.kernel("erfc")(a.array(x)))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not _accel.available(), reason="C acceleration disabled")
def test_vector_erfc_allocates_only_the_result_and_fixed_scratch():
    x = a.linspace(.1, 20., 200000)
    tracemalloc.start()
    try:
        result = _accel.kernel("erfc")(x)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < result.nbytes + 65536
