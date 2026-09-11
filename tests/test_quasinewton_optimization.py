"""Numerical invariants for the quadratic-cost inverse Hessian update."""

import numpy as np
import pytest

from quadrivium import numeric as a
from quadrivium.optimize import bfgs, broyden_class
from quadrivium.optimize.quasinewton import _bfgs_update


@pytest.mark.parametrize("condition", [1.0, 1e4, 1e8, 1e12])
def test_bfgs_update_secant_symmetry_and_positive_definiteness(condition):
    rng = np.random.default_rng(17)
    q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    h = (q * np.geomspace(1, condition, 16)) @ q.T
    y = rng.normal(size=16)
    s = rng.normal(size=16)
    s += (abs(s @ y) + 1) * y / (y @ y)
    sy = float(s @ y)
    v = np.eye(16) - np.outer(s, y) / sy
    expected = v @ h @ v.T + np.outer(s, s) / sy
    ah, ass, ay = map(a.array, (h, s, y))
    for arr in (ah, ass, ay):
        arr.flags.writeable = False
    actual = np.asarray(_bfgs_update(ah, ass, ay, sy))
    scale = np.linalg.norm(expected)
    assert np.linalg.norm(actual - expected) < 2e-15 * scale
    assert np.linalg.norm(actual - actual.T) < 2e-15 * scale
    assert np.linalg.norm(actual @ y - s) < 2e-15 * scale * np.linalg.norm(y)
    assert np.linalg.eigvalsh(actual).min() > 0
    for actual_input, original in ((ah, h), (ass, s), (ay, y)):
        np.testing.assert_array_equal(np.asarray(actual_input), original)


@pytest.mark.parametrize("scale", [1e-150, 1e-100, 1.0, 1e100, 1e150])
def test_bfgs_update_extreme_scales(scale):
    s0, y0 = np.array([2., 1., 3.]), np.array([1., 2., 3.])
    sy = float(s0 @ y0)
    v = np.eye(3) - np.outer(s0, y0) / sy
    expected = v @ v.T + np.outer(s0, s0) / sy
    actual = _bfgs_update(a.eye(3) * scale * scale,
                          a.array(s0) * scale, a.array(y0) / scale, sy)
    np.testing.assert_allclose(np.asarray(actual) / scale / scale,
                               expected, rtol=2e-15, atol=2e-15)


def test_bfgs_update_preserves_nonsymmetric_initial_matrix_formula():
    h = np.array([[2., .3], [.1, 1.]])
    s, y = np.array([1., 2.]), np.array([3., 4.])
    sy = float(s @ y)
    v = np.eye(2) - np.outer(s, y) / sy
    actual = _bfgs_update(a.array(h), a.array(s), a.array(y), sy)
    np.testing.assert_allclose(np.asarray(actual),
                               v @ h @ v.T + np.outer(s, s) / sy,
                               rtol=2e-15, atol=2e-15)


def test_bfgs_update_falls_back_when_only_the_contractions_overflow():
    h = a.eye(2) * 1e200
    s, y = a.array([1e-100, 1e-100]), a.array([1e200, 0.])
    actual = np.asarray(_bfgs_update(h, s, y, float(s @ y)))
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, [[1e-300, 1e-300], [1e-300, 2e200]],
                               rtol=2e-15, atol=0.)


def test_bfgs_update_retains_correction_when_dividing_s_first_underflows():
    h = a.diag(a.array([2.**-500, 2.**20]))
    s, y = a.array([2.**-300, 2.**-160]), a.array([0., 2.**1000])
    actual = _bfgs_update(h, s, y, float(s @ y))
    np.testing.assert_array_equal(np.asarray(actual), [[2.**-260, 0.], [0., 0.]])


@pytest.mark.parametrize("method", [bfgs, broyden_class])
def test_dense_quasi_newton_converges_without_mutating_initial_hessian(method):
    weights = a.linspace(1, 50, 64)
    target = a.linspace(-2, 2, 64)
    initial = a.eye(64)
    initial.flags.writeable = False
    result = method(lambda x: float(a.sum(weights * (x - target)**2)),
                    a.ones(64) * 3,
                    grad_f=lambda x: 2 * weights * (x - target),
                    H0=initial, tol=1e-8, max_iter=300)
    assert result.converged
    assert result.fun < 1e-10
    np.testing.assert_allclose(np.asarray(result.x), np.asarray(target), atol=1e-6)
    np.testing.assert_array_equal(np.asarray(initial), np.eye(64))
