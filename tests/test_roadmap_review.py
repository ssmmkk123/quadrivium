"""Independent regressions for numerical edge cases found in cross-review."""
import math

import numpy as oracle
import pytest

from quadrivium import numeric as np
from quadrivium.transforms import stft, istft, PolyphaseResampler
from quadrivium.stochastic import mcmc_diagnostics, split_rhat
from quadrivium.rootfind import pseudo_arclength


@pytest.mark.parametrize("length", [1, 15, 16, 37])
def test_complex_analysis_window_preserves_full_spectrum_and_roundtrips(length):
    x = np.arange(length, dtype=float) + 0.3
    window = np.exp(1j * np.arange(8) / 3) * np.linspace(0.5, 1.0, 8)
    spectrum = stft(x, segment=8, hop=3, nfft=16, window=window)
    assert spectrum.onesided is False
    assert spectrum.coefficients.shape[0] == 16
    oracle.testing.assert_allclose(oracle.asarray(istft(spectrum)), oracle.asarray(x),
                                   rtol=1e-12, atol=1e-12)
    first = oracle.zeros(16, dtype=complex)
    first[4:4 + min(length, 4)] = oracle.asarray(x[:4])
    first[:8] *= oracle.asarray(window)
    oracle.testing.assert_allclose(oracle.asarray(spectrum.coefficients[:, 0]), oracle.fft.fft(first),
                                   rtol=1e-12, atol=1e-12)


def test_resampler_checkpoint_requires_complete_overlap_and_is_atomic():
    original = PolyphaseResampler(3, 2, taps=[0.1, 0.2, 0.4, 0.2, 0.1])
    original.process([1., 2., 3., 4.])
    state = original.state
    restored = PolyphaseResampler(3, 2, taps=[0.1, 0.2, 0.4, 0.2, 0.1])
    restored.state = state
    oracle.testing.assert_allclose(oracle.asarray(original.process([5., 6., 7.])),
                                   oracle.asarray(restored.process([5., 6., 7.])), atol=1e-14)
    good = restored.state
    invalid = dict(good, tail_real=good["tail_real"][:-1], tail_imag=good["tail_imag"][:-1])
    with pytest.raises(ValueError, match="overlap"):
        restored.state = invalid
    assert restored.state == good
    invalid = dict(good, tail_real=[[1., 2.]], tail_imag=[[0., 0.]])
    with pytest.raises(ValueError, match="vectors"):
        restored.state = invalid
    assert restored.state == good
    with pytest.raises(ValueError, match="finite"):
        PolyphaseResampler(2, 3, taps=[1., math.nan])


def test_degenerate_constant_chain_diagnostics_are_undefined():
    result = mcmc_diagnostics(np.ones((4, 100)))
    assert all(math.isnan(value) for value in result.values())
    # Constants disagreeing across chains indicate failure, not indeterminacy.
    disagree = np.array([[0.] * 100, [1.] * 100])
    assert math.isinf(split_rhat(disagree))
    # Detect constant parameters independently without hiding well-mixing ones.
    rng = oracle.random.default_rng(912)
    chains = oracle.stack((rng.normal(size=(4, 500)), oracle.ones((4, 500))), axis=-1)
    values = mcmc_diagnostics(chains)
    assert float(values["rhat"][0]) < 1.05
    assert math.isnan(float(values["rhat"][1]))
    assert math.isnan(float(values["mcse_mean"][1]))


def test_final_allowed_newton_correction_can_accept_continuation_point():
    result = pseudo_arclength(lambda x, parameter: x * x - parameter, [1.], 1.,
                             jac=lambda x, parameter: [[2 * x[0]]],
                             parameter_derivative=lambda x, parameter: [-1.],
                             ds=0.2, min_step=0.2, max_step=0.2,
                             max_steps=1, max_newton=1, tol=1e-3)
    assert result.converged, result.message
    assert result.x.shape == (2, 1)
    residual = oracle.asarray(result.x[:, 0]) ** 2 - oracle.asarray(result.parameters)
    assert oracle.max(oracle.abs(residual)) <= 1e-3
