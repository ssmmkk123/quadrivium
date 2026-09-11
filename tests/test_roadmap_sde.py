"""Pathwise, seeded, jump-sampling, and bounded-memory SDE output checks."""
import gc
import tracemalloc

import numpy as np
import pytest

from quadrivium import numeric as q
from quadrivium.stochastic import sde

SCHEMES = ["euler_maruyama", "milstein", "implicit_milstein", "stochastic_heun",
           "stochastic_rk", "srk_strong_1_5", "tamed_euler"]


@pytest.mark.parametrize("name", SCHEMES)
def test_sde_final_stride_and_seed_stream_match_default(name):
    function = getattr(sde, name)
    options = dict(a=lambda x, t: .2*x, b=lambda x, t: .3*q.ones_like(x),
                   t_span=(2., 3.), x0=[1., 2.], n=71)
    generators = [q.random.default_rng(91) for _ in range(3)]
    full = function(**options, rng=generators[0])
    final = function(**options, rng=generators[1], final_only=True)
    stride = function(**options, rng=generators[2], save_every=7)
    assert final.y.shape == (1, 2)
    np.testing.assert_array_equal(np.asarray(final.y_final), np.asarray(full.y[-1]))
    expected = list(range(0, 72, 7))
    if expected[-1] != 71:
        expected.append(71)
    np.testing.assert_array_equal(np.asarray(stride.y), np.asarray(full.y)[expected])
    assert generators[0].state == generators[1].state == generators[2].state
    assert full.n_steps == final.n_steps == stride.n_steps == 71


@pytest.mark.parametrize("name", ["euler_maruyama", "milstein", "stochastic_heun", "stochastic_rk"])
def test_provided_increments_match_independent_additive_solution(name):
    increments = np.array([.1, -.2, .3, -.4, .1, .2])
    x0, drift, diffusion, dt = 2., .5, .3, 1/6
    result = getattr(sde, name)(lambda x,t: drift, lambda x,t: diffusion,
                               (0, 1), [x0], n=6, dW=increments, save_every=2)
    exact = x0 + drift*dt*np.arange(7) + diffusion*np.r_[0, np.cumsum(increments)]
    np.testing.assert_allclose(np.asarray(result.y)[:, 0], exact[[0, 2, 4, 6]], atol=1e-14)


def test_save_at_interpolation_and_actual_endpoint():
    increments = q.array([.2, -.1, .3, -.4])
    options = dict(a=lambda x,t: 0, b=lambda x,t: 1, t_span=(0,1), x0=[0], n=4, dW=increments)
    result = sde.euler_maruyama(**options, save_at=[.125, .625])
    np.testing.assert_allclose(np.asarray(result.y)[:, 0], [.1, .25], atol=1e-14)
    assert abs(float(result.y_final[0])) < 1e-14
    assert result.t[-1] == .625


@pytest.mark.parametrize("name", SCHEMES)
def test_sde_callback_stops_and_cannot_mutate_internal_state(name):
    observed = []
    def callback(t, state):
        observed.append((t, state.copy()))
        state[:] = 999
        return len(observed) == 4
    options = dict(a=lambda x,t: .1*x, b=lambda x,t: .2*q.ones_like(x),
                   t_span=(0,1), x0=[1], n=40, rng=12)
    full = getattr(sde,name)(**options)
    stopped = getattr(sde,name)(**options, callback=callback, final_only=True)
    assert not stopped.success
    assert stopped.n_steps == 3
    assert stopped.message == "callback stopped"
    np.testing.assert_array_equal(np.asarray(stopped.y_final), np.asarray(full.y[3]))
    assert len(observed) == 4


@pytest.mark.parametrize("name,parameters", [
    ("geometric_brownian_motion", dict(x0=2.,mu=.3,sigma=.4)),
    ("ornstein_uhlenbeck", dict(x0=[1.,2.],theta=1.5,mu=.2,sigma=.4)),
    ("cox_ingersoll_ross", dict(x0=[.05],theta=.5,mu=.2,sigma=2.)),
])
def test_named_process_output_preserves_seeded_path(name, parameters):
    function = getattr(sde,name)
    generators = [q.random.default_rng(7), q.random.default_rng(7)]
    full = function(**parameters, t_span=(0,2), n=67, rng=generators[0])
    final = function(**parameters, t_span=(0,2), n=67, rng=generators[1], final_only=True)
    np.testing.assert_allclose(np.asarray(final.y_final), np.asarray(full.y[-1]), rtol=5e-15, atol=0)
    assert generators[0].state == generators[1].state
    assert final.y.shape[0] == 1


def test_exact_ou_final_distribution_moments():
    # Independent closed-form transition moments across vectorized independent paths.
    result = sde.ornstein_uhlenbeck(q.full(16000, 1.), 1.2, .3, .4, (0,1),
                                  n=4, rng=234, final_only=True)
    expected_mean = .3 + .7*np.exp(-1.2)
    expected_variance = .4**2*(-np.expm1(-2.4))/2.4
    values = np.asarray(result.y_final)
    assert abs(values.mean()-expected_mean) < 5*np.sqrt(expected_variance/len(values))
    assert abs(values.var()-expected_variance) < .05*expected_variance


def test_gillespie_save_at_is_right_continuous_and_never_fractional():
    options = dict(propensities=lambda x,t: [3.], stoichiometry=[[1]], x0=[0],
                   t_span=(0,2), rng=19)
    times, states = sde.gillespie_ssa(**options)
    samples = np.linspace(0, 2, 61)
    selected_t, selected_y = sde.gillespie_ssa(**options, save_at=samples)
    idx = np.searchsorted(np.asarray(times), samples, side="right")-1
    np.testing.assert_array_equal(np.asarray(selected_y), np.asarray(states)[idx])
    np.testing.assert_array_equal(np.asarray(selected_t), samples)
    final_t, final_y = sde.gillespie_ssa(**options, final_only=True)
    assert final_t[-1] == 2
    np.testing.assert_array_equal(np.asarray(final_y[-1]), np.asarray(states[-1]))


def test_jump_callback_and_absorbing_endpoint():
    seen = []
    def callback(t, state):
        seen.append(t)
        state[:] = -100
        return len(seen) == 3
    t, x = sde.gillespie_ssa(lambda x,t:[20.], [[1]], [0], (0,1), rng=7,
                            callback=callback, final_only=True)
    assert len(seen) == 3
    assert x[-1, 0] == 2
    assert t[-1] < 1
    t, x = sde.gillespie_ssa(lambda x,t:[0.], [[1]], [5], (0,1), final_only=True)
    assert t[-1] == 1 and x[-1, 0] == 5


def test_tau_leaping_output_draw_order_and_step_sampling():
    options = dict(propensities=lambda x,t:[4.], stoichiometry=[[1]], x0=[0],
                   t_span=(0,1),tau=.125)
    generators = [q.random.default_rng(10),q.random.default_rng(10)]
    times, states = sde.tau_leaping(**options, rng=generators[0])
    selected_t, selected = sde.tau_leaping(**options, rng=generators[1], save_at=[.1,.2,.6,1])
    idx = np.searchsorted(np.asarray(times), np.asarray(selected_t), side="right")-1
    np.testing.assert_array_equal(np.asarray(selected), np.asarray(states)[idx])
    assert generators[0].state == generators[1].state


@pytest.mark.parametrize("name", ["euler_maruyama", "srk_strong_1_5"])
def test_final_only_memory_is_bounded_by_state_not_step_count(name, monkeypatch):
    class GuardedGenerator:
        def __init__(self):
            self.generator = q.random.default_rng(47)
            self.largest = 0
        def standard_normal(self, size):
            count = int(np.prod(size)) if isinstance(size, tuple) else size
            self.largest = max(self.largest, count)
            assert count <= 4096
            return self.generator.standard_normal(size)
    generator = GuardedGenerator()
    # Inject only the RNG adapter, keeping the native sampler and solver real.
    monkeypatch.setattr(sde, "_rng", lambda rng: generator)
    def peak(n):
        gc.collect()
        tracemalloc.start()
        result = getattr(sde,name)(lambda x,t: 0*x, lambda x,t: q.ones_like(x),
                                  (0,1), [0], n=n, final_only=True)
        _, memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert result.y.shape == (1,1)
        return memory
    small, large = peak(100), peak(7000)
    assert large < small + 200000
    assert generator.largest <= 4096


@pytest.mark.parametrize("options", [{"save_every":0}, {"save_at":[]},
                                      {"save_at":[2]}, {"save_at":[.5],"final_only":True}])
def test_invalid_sde_output_controls(options):
    with pytest.raises(ValueError):
        sde.euler_maruyama(lambda x,t:x, lambda x,t:1, (0,1), [1], **options)
