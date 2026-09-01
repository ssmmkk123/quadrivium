"""Spectral methods for PDEs.

Global basis functions give exponential accuracy for smooth solutions, so a
handful of modes can match what a finite difference grid needs thousands of
points to achieve.
"""

from __future__ import annotations

import numpy as np

from ..core.types import PDESolution
from ..diff.spectral import chebyshev_diff_matrix, fourier_derivative

__all__ = [
    "fourier_heat",
    "fourier_advection",
    "chebyshev_poisson_1d",
    "chebyshev_bvp",
    "spectral_burgers",
    "kuramoto_sivashinsky",
    "pseudospectral_step",
]


def fourier_heat(u0, alpha: float, L: float = 2 * np.pi, n: int = 128,
                 t_span=(0.0, 1.0), nt: int = 100):
    """Heat equation with periodic boundaries, solved exactly in Fourier space.

    Each mode decays as ``exp(-alpha k^2 t)``, so the time stepping is exact and
    unconditionally stable whatever the step size.
    """
    x = np.linspace(0.0, L, n, endpoint=False)
    t = np.linspace(float(t_span[0]), float(t_span[1]), nt + 1)
    u = np.array([u0(xi) for xi in x]) if callable(u0) else np.asarray(u0, dtype=float)
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    U_hat = np.fft.fft(u)
    out = np.empty((nt + 1, n))
    for i, ti in enumerate(t):
        out[i] = np.real(np.fft.ifft(U_hat * np.exp(-alpha * k**2 * (ti - t[0]))))
    return PDESolution(out, (x,), t, "fourier_heat")


def fourier_advection(u0, c: float, L: float = 2 * np.pi, n: int = 128,
                      t_span=(0.0, 1.0), nt: int = 100):
    """Linear advection with periodic boundaries, exact in Fourier space.

    Each mode is translated by ``exp(-i c k t)``: no dispersion, no dissipation.
    """
    x = np.linspace(0.0, L, n, endpoint=False)
    t = np.linspace(float(t_span[0]), float(t_span[1]), nt + 1)
    u = np.array([u0(xi) for xi in x]) if callable(u0) else np.asarray(u0, dtype=float)
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    U_hat = np.fft.fft(u)
    out = np.empty((nt + 1, n))
    for i, ti in enumerate(t):
        out[i] = np.real(np.fft.ifft(U_hat * np.exp(-1j * c * k * (ti - t[0]))))
    return PDESolution(out, (x,), t, "fourier_advection")


def chebyshev_poisson_1d(f, x_span=(-1.0, 1.0), bc=(0.0, 0.0), n: int = 32):
    """Solve ``u'' = f`` on an interval by Chebyshev collocation."""
    D, x = chebyshev_diff_matrix(n, x_span[0], x_span[1])
    D2 = D @ D
    A = D2.copy()
    b = np.array([f(xi) for xi in x])
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = bc[0]
    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    b[-1] = bc[1]
    u = np.linalg.solve(A, b)
    return PDESolution(u, (x,), None, "chebyshev_poisson")


def chebyshev_bvp(p, q, r, x_span=(-1.0, 1.0), bc=(0.0, 0.0), n: int = 32):
    """Chebyshev collocation for ``u'' + p(x) u' + q(x) u = r(x)``."""
    D, x = chebyshev_diff_matrix(n, x_span[0], x_span[1])
    D2 = D @ D
    P = np.diag([p(xi) for xi in x])
    Q = np.diag([q(xi) for xi in x])
    A = D2 + P @ D + Q
    b = np.array([r(xi) for xi in x])
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = bc[0]
    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    b[-1] = bc[1]
    return PDESolution(np.linalg.solve(A, b), (x,), None, "chebyshev_bvp")


def pseudospectral_step(u, nonlinear, k, dt: float, linear_symbol):
    """One IMEX step: linear part exactly in spectral space, nonlinear explicitly.

    The nonlinear term is evaluated in physical space and transformed back,
    which is what makes the method "pseudospectral".
    """
    U = np.fft.fft(u)
    N = np.fft.fft(nonlinear(u))
    E = np.exp(linear_symbol * dt)
    return np.real(np.fft.ifft(E * U + dt * E * N))


def spectral_burgers(u0, nu: float = 0.01, L: float = 2 * np.pi, n: int = 256,
                     t_span=(0.0, 1.0), nt: int = 2000, dealias: bool = True):
    """Viscous Burgers by a Fourier pseudospectral method with ETD-RK2 in time.

    The 2/3 dealiasing rule removes the aliasing error that the quadratic
    nonlinearity would otherwise fold back onto the resolved modes.
    """
    x = np.linspace(0.0, L, n, endpoint=False)
    t = np.linspace(float(t_span[0]), float(t_span[1]), nt + 1)
    dt = t[1] - t[0]
    u = np.array([u0(xi) for xi in x]) if callable(u0) else np.asarray(u0, dtype=float)
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    mask = np.ones(n) if not dealias else (np.abs(k) < (2.0 / 3.0) * np.max(np.abs(k)))
    Lsym = -nu * k**2
    E = np.exp(Lsym * dt)
    E2 = np.exp(Lsym * dt / 2)
    out = np.empty((nt + 1, n))
    out[0] = u

    def nonlinear_hat(v_hat):
        v = np.real(np.fft.ifft(v_hat))
        return -0.5 * 1j * k * np.fft.fft(v * v) * mask

    U = np.fft.fft(u)
    for i in range(nt):
        N1 = nonlinear_hat(U)
        a = E * U + dt * E * N1                     # predictor
        N2 = nonlinear_hat(a)
        U = E * U + dt * (E * N1 + N2) / 2.0        # trapezoidal correction
        out[i + 1] = np.real(np.fft.ifft(U))
    return PDESolution(out, (x,), t, "spectral_burgers")


def kuramoto_sivashinsky(u0, L: float = 32 * np.pi, n: int = 256,
                         t_span=(0.0, 50.0), nt: int = 5000):
    """Kuramoto-Sivashinsky equation ``u_t + u u_x + u_xx + u_xxxx = 0``.

    A canonical chaotic PDE; the fourth-derivative term makes it very stiff, so
    the linear part is integrated exactly by an exponential (ETDRK2) step.
    """
    x = np.linspace(0.0, L, n, endpoint=False)
    t = np.linspace(float(t_span[0]), float(t_span[1]), nt + 1)
    dt = t[1] - t[0]
    u = np.array([u0(xi) for xi in x]) if callable(u0) else np.asarray(u0, dtype=float)
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    Lsym = k**2 - k**4                     # from -u_xx - u_xxxx
    E = np.exp(Lsym * dt)
    mask = np.abs(k) < (2.0 / 3.0) * np.max(np.abs(k))
    out = np.empty((nt + 1, n))
    out[0] = u
    U = np.fft.fft(u)

    def nl(v_hat):
        v = np.real(np.fft.ifft(v_hat))
        return -0.5 * 1j * k * np.fft.fft(v * v) * mask

    for i in range(nt):
        N1 = nl(U)
        a = E * U + dt * E * N1
        N2 = nl(a)
        U = E * U + dt * (E * N1 + N2) / 2.0
        out[i + 1] = np.real(np.fft.ifft(U))
    return PDESolution(out, (x,), t, "kuramoto_sivashinsky")
