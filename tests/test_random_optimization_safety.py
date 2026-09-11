"""Invalid distribution parameters must fail before advancing the RNG."""

import math

import numpy as np
import pytest

from quadrivium import numeric as a


@pytest.mark.parametrize("lam", [-1., np.nan, np.inf, -np.inf, 1e30, float(2**63)])
@pytest.mark.parametrize("size", [None, 0, 10])
def test_poisson_rejects_invalid_rates_without_consuming_draws(lam, size):
    rng = a.random.default_rng(71)
    with pytest.raises(ValueError):
        rng.poisson(lam, size=size)
    np.testing.assert_array_equal(np.asarray(rng.random(12)),
                                  np.random.default_rng(71).random(12))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1., 1e30])
def test_poisson_validates_entire_strided_parameter_array_before_sampling(bad):
    params = a.array([2., 99., 3., 99., bad, 99.])[::2]
    rng = a.random.default_rng(72)
    with pytest.raises(ValueError):
        rng.poisson(params, size=(5, 3))
    np.testing.assert_array_equal(np.asarray(rng.random(12)),
                                  np.random.default_rng(72).random(12))


def test_poisson_int64_margin_is_checked_even_for_empty_output():
    limit = float(2**63 - 1) - 10 * math.sqrt(float(2**63 - 1))
    rng = a.random.default_rng(73)
    assert rng.poisson(limit, size=0).shape == (0,)
    with pytest.raises(ValueError):
        rng.poisson(math.nextafter(limit, math.inf), size=0)


@pytest.mark.parametrize("low,high", [
    (3., 1.), (np.nan, 1.), (0., np.nan), (-np.inf, 1.), (0., np.inf),
    (np.inf, np.inf), (-1e308, 1e308),
])
@pytest.mark.parametrize("size", [None, 0, 10])
def test_uniform_rejects_invalid_intervals_without_consuming_draws(low, high, size):
    rng = a.random.default_rng(74)
    with pytest.raises(ValueError):
        rng.uniform(low, high, size=size)
    np.testing.assert_array_equal(np.asarray(rng.random(12)),
                                  np.random.default_rng(74).random(12))


def test_uniform_checks_every_broadcast_pair_before_sampling():
    low = a.array([0., 99., 2., 99.])[::2, None]
    high = a.array([3., 99., 1., 99.])[None, ::2]
    rng = a.random.default_rng(75)
    with pytest.raises(ValueError):
        rng.uniform(low, high, size=(6, 2, 2))
    np.testing.assert_array_equal(np.asarray(rng.random(12)),
                                  np.random.default_rng(75).random(12))


@pytest.mark.parametrize("probabilities", [
    [0., 0.], [-1., 2.], [np.nan, 1.], [np.inf, 1.], [-np.inf, 1.],
    [.2, .2], [.5, .5001], [1e308, 1e308], [[.5, .5]], [.2, .3, .5],
])
@pytest.mark.parametrize("size", [None, 0, 10])
def test_choice_rejects_invalid_probabilities_without_consuming_draws(probabilities, size):
    rng = a.random.default_rng(76)
    with pytest.raises(ValueError):
        rng.choice(2, size=size, p=probabilities)
    np.testing.assert_array_equal(np.asarray(rng.random(12)),
                                  np.random.default_rng(76).random(12))


def test_valid_distributions_preserve_seeded_streams_and_broadcasting():
    actual = a.random.default_rng(77)
    expected = np.random.default_rng(77)
    low, high = np.array([-2., 0., 3.])[::-1], np.array([0., 4., 7.])[::-1]
    for result, reference in (
        (actual.uniform(a.array(low), a.array(high), size=(4, 3)),
         expected.uniform(low, high, size=(4, 3))),
        (actual.uniform(2., 2., size=7), expected.uniform(2., 2., size=7)),
        (actual.uniform(-1e300, 1e300, size=7), expected.uniform(-1e300, 1e300, size=7)),
        (actual.poisson(a.array([0., .1, 3., 10., 1e5]), size=(4, 5)),
         expected.poisson([0., .1, 3., 10., 1e5], size=(4, 5))),
        (actual.choice(4, size=25, p=[0., .2, 0., .8]),
         expected.choice(4, size=25, p=[0., .2, 0., .8])),
        (actual.random(12), expected.random(12)),
    ):
        np.testing.assert_array_equal(np.asarray(result), reference)


def test_choice_valid_strided_probabilities_remain_unchanged():
    probabilities = a.array([0., 99., .2, 99., 0., 99., .8, 99.])[::2]
    probabilities.flags.writeable = False
    result = a.random.default_rng(78).choice(4, size=40, p=probabilities)
    expected = np.random.default_rng(78).choice(4, size=40, p=[0., .2, 0., .8])
    np.testing.assert_array_equal(np.asarray(result), expected)
    np.testing.assert_array_equal(np.asarray(probabilities), [0., .2, 0., .8])


def test_choice_preserves_float32_probability_tolerance_and_stream():
    probabilities = np.full(6, 1 / 3, dtype=np.float32)[::2]
    probabilities.flags.writeable = False
    result = a.random.default_rng(80).choice(3, size=50, p=probabilities)
    expected = np.random.default_rng(80).choice(3, size=50, p=probabilities)
    np.testing.assert_array_equal(np.asarray(result), expected)


def test_uniform_and_poisson_retain_empty_parameter_broadcasts():
    rng = a.random.default_rng(79)
    assert rng.uniform(a.zeros((0, 3)), a.ones(3)).shape == (0, 3)
    assert rng.poisson(a.zeros((0, 3))).shape == (0, 3)
    np.testing.assert_array_equal(np.asarray(rng.random(12)),
                                  np.random.default_rng(79).random(12))
