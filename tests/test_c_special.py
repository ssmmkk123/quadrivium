"""Tail accuracy and bounded workspace for vector complementary error function."""
import math
import tracemalloc

import numpy as np
import pytest

from quadrivium import _accel, numeric as a


@pytest.mark.skipif(not _accel.available(), reason="C acceleration disabled")
@pytest.mark.parametrize("stride", [1, 2, -1, -3])
def test_vector_erfc_matches_libm_across_branches_and_subnormal_tails(stride):
    x = np.concatenate((np.linspace(-30., 30., 20001),
                        [np.nan, -np.inf, np.inf, 0., -0., 1.25, 2.8571414947509766,
                         np.nextafter(1.25, 0), np.nextafter(1.25, 2), 27., 27.2]))
    values = a.array(x)[::stride]
    values.flags.writeable = False
    expected = np.array([math.erfc(v) for v in np.asarray(values)])
    actual = np.asarray(_accel.kernel("erfc")(values))
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=5e-324, equal_nan=True)


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
