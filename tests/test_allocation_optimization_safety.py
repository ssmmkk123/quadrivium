"""Reject unrepresentable array buffers before allocating or exposing views."""

from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def _run_limited(code):
    resource = pytest.importorskip("resource")
    if not hasattr(resource, "RLIMIT_AS"):
        pytest.skip("address-space limits are unavailable")
    # Keep intentionally enormous requests isolated even if bounds validation
    # regresses. The probes never access an improperly allocated data buffer.
    preamble = """
import resource
import sys
resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
from quadrivium import numeric as np
"""
    result = subprocess.run([sys.executable, "-c", preamble + textwrap.dedent(code)],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("dtype", ["int64", "float64", "complex128"])
def test_allocation_rejects_signed_byte_overflow_and_exact_size_t_wrap(dtype):
    _run_limited(f"""
        dtype = np.dtype({dtype!r})
        limit = sys.maxsize // dtype.itemsize
        unsigned_wrap = 2 * (sys.maxsize + 1) // dtype.itemsize
        shapes = [limit + 1, unsigned_wrap, unsigned_wrap + 1,
                  (limit // 2 + 1, 2), (unsigned_wrap // 2, 2)]
        for factory in (np.empty, np.ndarray):
            for shape in shapes:
                try:
                    result = factory(shape, dtype=dtype)
                except ValueError as exc:
                    assert 'too big' in str(exc), str(exc)
                else:
                    raise AssertionError((factory.__name__, shape, result.shape))
    """)


def test_allocation_and_views_reject_element_product_overflow_for_bool():
    _run_limited("""
        huge = 1 << (sys.maxsize.bit_length() // 2 + 1)
        for make in (lambda: np.empty((huge, huge), dtype=bool),
                     lambda: np.broadcast_to(np.array(True), (huge, huge))):
            try:
                make()
            except ValueError as exc:
                assert 'too big' in str(exc), str(exc)
            else:
                raise AssertionError('overflowing bool shape was accepted')
    """)


@pytest.mark.parametrize("dtype", ["int64", "float64", "complex128"])
def test_broadcast_views_validate_logical_bytes_at_the_signed_boundary(dtype):
    _run_limited(f"""
        dtype = np.dtype({dtype!r})
        source = np.array(1, dtype=dtype)
        limit = sys.maxsize // dtype.itemsize
        valid = np.broadcast_to(source, (limit,))
        assert valid.size == limit
        assert valid.nbytes == limit * dtype.itemsize
        assert valid[-1] == 1
        assert not valid.flags.writeable
        for shape in ((limit + 1,), (limit // 2 + 1, 2)):
            try:
                np.broadcast_to(source, shape)
            except ValueError as exc:
                assert 'too big' in str(exc), str(exc)
            else:
                raise AssertionError(('overflowing logical buffer', shape))
    """)


def test_huge_empty_shapes_remain_valid_in_any_dimension_order():
    _run_limited("""
        huge = 1 << (sys.maxsize.bit_length() // 2 + 1)
        for dtype in (bool, int, float, complex):
            for shape in ((0, huge, huge), (huge, 0, huge), (huge, huge, 0)):
                for values in (np.empty(shape, dtype=dtype),
                               np.broadcast_to(np.array(1, dtype=dtype), shape)):
                    assert values.shape == shape
                    assert values.size == values.nbytes == 0
                    assert memoryview(values).nbytes == 0
                    assert values.T.size == values.T.nbytes == 0
                    assert values.T.shape == shape[::-1]
        for shape in ((0, -1), (-1, 0)):
            try:
                np.empty(shape)
            except ValueError:
                pass
            else:
                raise AssertionError('empty dimension masked a negative dimension')
    """)
