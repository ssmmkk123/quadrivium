"""PCG64 limb arithmetic must preserve full-width draws and exported state."""

import numpy as np
import pytest

from quadrivium import numeric as q


@pytest.mark.parametrize("state,increment", [
    (0, 1),
    ((1 << 128) - 1, (1 << 128) - 1),
    ((1 << 64) - 1, (1 << 64) + 1),
    (1 << 127, (1 << 127) + 1),
])
def test_pcg64_full_width_arithmetic_matches_numpy(state, increment):
    actual = q.random.default_rng(0)
    expected = np.random.default_rng(0)
    mask = (1 << 64) - 1
    actual.state = {
        "bit_generator": "PCG64",
        "version": 1,
        "words": (state >> 64, state & mask, increment >> 64, increment & mask),
        "has_uint32": 0,
        "uinteger": 0,
    }
    expected.bit_generator.state = {
        "bit_generator": "PCG64",
        "state": {"state": state, "inc": increment},
        "has_uint32": 0,
        "uinteger": 0,
    }
    # The wide bound exercises 64-by-64 multiplication and Lemire rejection;
    # smaller bounds also leave a cached half-word across subsequent draws.
    for high in ((1 << 62) + 1, (1 << 63) - 1, 1000, (1 << 32) + 1):
        np.testing.assert_array_equal(
            np.asarray(actual.integers(0, high, size=257)),
            expected.integers(0, high, size=257),
        )
    np.testing.assert_array_equal(np.asarray(actual.random(257)), expected.random(257))
    got, reference = actual.state, expected.bit_generator.state
    assert (got["words"][0] << 64) | got["words"][1] == reference["state"]["state"]
    assert (got["words"][2] << 64) | got["words"][3] == reference["state"]["inc"]
    assert got["has_uint32"] == reference["has_uint32"]
    assert got["uinteger"] == reference["uinteger"]
