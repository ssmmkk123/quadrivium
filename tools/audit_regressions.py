"""Minimal reproducers discovered by the September 2026 audit.

These deliberately FAIL on the audited revision. Run explicitly with:
    python -m pytest -q tools/audit_regressions.py
Kept outside the default suite so an audit does not silently change CI policy.
They can be promoted to the regular suite as the underlying defects are fixed.
"""
import math
import subprocess
import sys

import pytest

from quadrivium import numeric as a, special, transforms


@pytest.mark.parametrize("operator", ["//", "%"])
def test_integer_minimum_divided_by_minus_one_does_not_crash(operator):
    code = (
        "import resource;resource.setrlimit(resource.RLIMIT_CORE,(0,0));"
        "from quadrivium import numeric as a;"
        f"print(a.array([-2**63],dtype=int){operator}a.array([-1],dtype=int))"
    )
    proc = subprocess.run([sys.executable, "-X", "faulthandler", "-c", code],
                          capture_output=True, text=True, timeout=5)
    assert proc.returncode == 0, proc.stderr


def test_excessive_new_axes_raise_instead_of_overwriting_stack():
    code = """import resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
from quadrivium import numeric as a
try:
    a.ones(1)[(None,)*20]
except (IndexError, ValueError):
    print('rejected')
"""
    proc = subprocess.run([sys.executable, "-X", "faulthandler", "-c", code],
                          capture_output=True, text=True, timeout=5)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "rejected"


@pytest.mark.parametrize("dtype", [float, int, complex])
def test_reverse_assignment_preserves_source(dtype):
    x = a.arange(6, dtype=dtype)
    x[:] = x[::-1]
    assert x.tolist() == [5, 4, 3, 2, 1, 0]


def test_overlapping_inplace_add_uses_original_operands():
    x = a.arange(6, dtype=float)
    x[1:] += x[:-1]
    assert x.tolist() == [0, 1, 3, 5, 7, 9]


def test_empty_mean_is_not_a_valid_zero():
    assert a.isnan(a.mean(a.array([], dtype=float)))


def test_fractional_advanced_indices_are_rejected():
    with pytest.raises((IndexError, TypeError)):
        a.arange(3)[a.array([0.5])]


def test_duplicate_reduction_axes_are_rejected():
    with pytest.raises(ValueError):
        a.sum(a.ones((2, 2)), axis=(0, 0))


def test_negative_normal_scale_is_rejected():
    with pytest.raises(ValueError):
        a.random.default_rng(1).normal(scale=-1, size=2)


def test_airy_ai_at_eight_retains_eight_relative_digits():
    # Independently evaluated using mpmath at 80 decimal digits.
    assert special.airy_ai(8.) == pytest.approx(4.6922076160992316e-8, rel=1e-8, abs=0.)


def test_struve_h0_at_fifty():
    assert special.struve_h0(50.) == pytest.approx(-0.085337674826119, rel=1e-8, abs=1e-10)


def test_bessel_y0_near_a_zero():
    assert special.bessel_y0(7.0745594228198785) == pytest.approx(
        -0.003451336416626002, rel=1e-8, abs=1e-10)


@pytest.mark.parametrize("function", [a.fft.irfft, transforms.irfft])
def test_irfft_zero_padding_preserves_hermitian_spectrum(function):
    # DC=1, frequency 1=2, other frequencies zero, including Nyquist.
    expected = [(1 + 4*math.cos(2*math.pi*k/8))/8 for k in range(8)]
    assert function([1., 2.], n=8).tolist() == pytest.approx(expected, abs=1e-12)
