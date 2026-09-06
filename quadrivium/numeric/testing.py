"""Array assertions, mirroring ``numpy.testing`` for what the tests use.

Reports look like NumPy's: the mismatch count, the worst offender and both
arrays, which is what makes a failing numerical test readable.
"""

from __future__ import annotations

from .. import _qnp as _c

__all__ = ["assert_allclose", "assert_array_equal", "assert_array_almost_equal",
           "assert_almost_equal", "assert_equal", "assert_array_less"]


def _as_array(value):
    return _c.asarray(value)


def _report(header, got, want, extra=""):
    return (f"\n{header}\n{extra}\n"
            f" ACTUAL: {got!r}\n DESIRED: {want!r}\n")


def assert_allclose(actual, desired, rtol=1e-7, atol=0, equal_nan=True,
                    err_msg="", verbose=True):
    """Element-wise closeness, with NumPy's asymmetric tolerance rule."""
    from . import isclose, all as _all, absolute, count_nonzero, isnan, logical_not
    got = _as_array(actual)
    want = _as_array(desired)
    if got.shape != want.shape:
        # Broadcasting is allowed, exactly as NumPy allows it here.
        from . import broadcast_arrays
        got, want = broadcast_arrays(got, want)
    close = isclose(got, want, rtol=rtol, atol=atol, equal_nan=equal_nan)
    if bool(_all(close)):
        return
    bad = int(count_nonzero(logical_not(close)))
    total = got.size
    difference = absolute(got - want)
    worst = float(_c.amax(difference)) if total else 0.0
    raise AssertionError(_report(
        f"Not equal to tolerance rtol={rtol}, atol={atol}",
        got, want,
        f"{err_msg}\nMismatched elements: {bad} / {total}\n"
        f"Max absolute difference: {worst}"))


def assert_array_equal(actual, desired, err_msg="", verbose=True):
    from . import all as _all, count_nonzero, logical_not, isnan
    got = _as_array(actual)
    want = _as_array(desired)
    if got.shape != want.shape:
        from . import broadcast_arrays
        try:
            got, want = broadcast_arrays(got, want)
        except ValueError:
            raise AssertionError(_report(
                "Arrays are not equal", got, want,
                f"{err_msg}\n(shapes {got.shape}, {want.shape} mismatch)")) from None
    same = got == want
    if got.dtype == _c.float64 or got.dtype == _c.complex128:
        same = same | (isnan(got) & isnan(want))
    if bool(_all(same)):
        return
    bad = int(count_nonzero(logical_not(same)))
    raise AssertionError(_report("Arrays are not equal", got, want,
                                 f"{err_msg}\nMismatched elements: {bad} / {got.size}"))


def assert_array_almost_equal(actual, desired, decimal=6, err_msg="", verbose=True):
    assert_allclose(actual, desired, rtol=0, atol=1.5 * 10.0 ** (-decimal),
                    err_msg=err_msg)


def assert_almost_equal(actual, desired, decimal=7, err_msg=""):
    if _c.asarray(actual).ndim == 0 and _c.asarray(desired).ndim == 0:
        if abs(float(actual) - float(desired)) >= 1.5 * 10.0 ** (-decimal):
            raise AssertionError(f"{err_msg}\n{actual!r} != {desired!r} to {decimal} decimals")
        return
    assert_array_almost_equal(actual, desired, decimal, err_msg)


def assert_equal(actual, desired, err_msg=""):
    if isinstance(actual, _c.ndarray) or isinstance(desired, _c.ndarray):
        assert_array_equal(actual, desired, err_msg)
        return
    if actual != desired:
        raise AssertionError(f"{err_msg}\n{actual!r} != {desired!r}")


def assert_array_less(smaller, larger, err_msg=""):
    from . import all as _all
    if not bool(_all(_as_array(smaller) < _as_array(larger))):
        raise AssertionError(_report("Arrays are not less-ordered",
                                     _as_array(smaller), _as_array(larger), err_msg))
