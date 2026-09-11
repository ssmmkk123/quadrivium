"""Exported buffers keep valid array metadata throughout native operations."""
import pytest
from quadrivium import _accel, numeric as np


def test_shape_assignment_preserves_live_buffer_metadata():
    a = np.arange(12.).reshape(3, 4)
    with memoryview(a) as view:
        with pytest.raises(BufferError, match="exported"):
            a.shape = (2, 6)
        assert view.shape == (3, 4)
        assert view.strides == (32, 8)
        assert view[2, 3] == 11.
        assert a.shape == (3, 4)
    a.shape = (2, 6)
    assert a.shape == (2, 6)
    assert a[1, 5] == 11.


def test_scalar_shape_assignment_keeps_storage():
    a = np.array(3.)
    a.shape = (1,)
    assert a[0] == 3.
    a.shape = ()
    assert float(a) == 3.


def test_shape_coercion_cannot_invalidate_a_buffer_it_exports():
    a = np.arange(6.)
    retained = []

    class Dimension:
        def __index__(self):
            retained.append(memoryview(a))
            return 2

    try:
        with pytest.raises(BufferError, match="exported"):
            a.shape = (Dimension(), 3)
        assert retained[0].shape == (6,)
        assert retained[0][5] == 5.
    finally:
        for view in retained:
            view.release()


@pytest.mark.skipif(not _accel.available(), reason="C acceleration disabled")
def test_matmul_keeps_validated_shape_during_rhs_coercion():
    left = np.eye(2)

    class Value:
        def __float__(self):
            left.shape = (4,)
            return 2.

    out = _accel.kernel("matmul")(left, [[Value()], [Value()]])
    np.testing.assert_array_equal(out, [[2.], [2.]])
