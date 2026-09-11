"""High-resolution schemes and incompressible flow.

WENO reconstruction with SSP time stepping for conservation laws, and a
projection method for the incompressible Navier-Stokes equations.
"""

from __future__ import annotations

from .. import numeric as np

from .. import _accel

from ..core.storage import (pde_solution as PDESolution, TimeGrid,
                            Trajectory, output_control, OutputRecorder)

__all__ = [
    "weno5_reconstruct",
    "weno3_reconstruct",
    "ssp_rk3",
    "ssp_rk2",
    "weno_conservation_law",
    "weno_burgers",
    "navier_stokes_2d",
    "lid_driven_cavity",
    "vorticity_streamfunction",
    "poisson_periodic_fft",
]


def weno3_reconstruct(u, eps: float = 1e-6):
    """Third-order WENO reconstruction of the left state at each interface."""
    um1 = np.roll(u, 1)
    up1 = np.roll(u, -1)
    p0 = -0.5 * um1 + 1.5 * u
    p1 = 0.5 * u + 0.5 * up1
    b0 = (u - um1) ** 2
    b1 = (up1 - u) ** 2
    a0 = (1.0 / 3.0) / (eps + b0) ** 2
    a1 = (2.0 / 3.0) / (eps + b1) ** 2
    return (a0 * p0 + a1 * p1) / (a0 + a1)


def weno5_reconstruct(u, eps: float = 1e-6):
    """Fifth-order WENO reconstruction of the left interface state ``u_{i+1/2}^-``.

    Three candidate third-order stencils are blended with weights driven by
    *smoothness indicators*.  Where the solution is smooth the weights approach
    the linear ones and the combination is fifth-order accurate; near a
    discontinuity the indicator on the offending stencil blows up, its weight
    collapses, and the scheme quietly falls back to the smooth stencils.  That
    automatic, continuous switching is why WENO captures shocks without the
    oscillations of a fixed high-order stencil and without the smearing of a
    limiter that clips to first order.

    Periodic boundaries; ``u`` is the cell-average array.
    """
    um2, um1 = np.roll(u, 2), np.roll(u, 1)
    up1, up2 = np.roll(u, -1), np.roll(u, -2)
    p0 = (2 * um2 - 7 * um1 + 11 * u) / 6.0
    p1 = (-um1 + 5 * u + 2 * up1) / 6.0
    p2 = (2 * u + 5 * up1 - up2) / 6.0
    b0 = (13 / 12) * (um2 - 2 * um1 + u) ** 2 + 0.25 * (um2 - 4 * um1 + 3 * u) ** 2
    b1 = (13 / 12) * (um1 - 2 * u + up1) ** 2 + 0.25 * (um1 - up1) ** 2
    b2 = (13 / 12) * (u - 2 * up1 + up2) ** 2 + 0.25 * (3 * u - 4 * up1 + up2) ** 2
    a0 = 0.1 / (eps + b0) ** 2
    a1 = 0.6 / (eps + b1) ** 2
    a2 = 0.3 / (eps + b2) ** 2
    s = a0 + a1 + a2
    return (a0 * p0 + a1 * p1 + a2 * p2) / s


def ssp_rk2(rhs, u, dt):
    """Strong-stability-preserving RK2 (Heun) -- one step."""
    u1 = u + dt * rhs(u)
    return 0.5 * u + 0.5 * (u1 + dt * rhs(u1))


def ssp_rk3(rhs, u, dt):
    """Strong-stability-preserving RK3 (Shu-Osher) -- one step.

    Every stage is a convex combination of forward-Euler steps, so any bound
    that forward Euler respects -- positivity, a maximum principle, total
    variation -- survives the whole step.  A general third-order RK method has
    negative coefficients somewhere and gives no such guarantee, which is why
    a high-resolution spatial scheme is paired with an SSP integrator and not
    just any RK3.
    """
    u1 = u + dt * rhs(u)
    u2 = 0.75 * u + 0.25 * (u1 + dt * rhs(u1))
    return (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * rhs(u2))


def weno_conservation_law(u0, flux, wave_speed, x_span, t_span, nx: int = 200,
                          cfl: float = 0.4, order: int = 5):
    """Solve ``u_t + f(u)_x = 0`` with WENO reconstruction and SSP-RK3.

    Uses a Lax-Friedrichs flux split so that each half carries information in a
    single direction; WENO is then applied to each half with the appropriate
    upwind bias.  Periodic boundaries.
    """
    x = np.linspace(float(x_span[0]), float(x_span[1]), nx, endpoint=False)
    dx = x[1] - x[0]
    u = np.array([u0(xi) for xi in x], dtype=float) if callable(u0) else np.asarray(u0, float)
    recon = weno5_reconstruct if order == 5 else weno3_reconstruct

    def rhs(v):
        alpha = float(np.max(np.abs(wave_speed(v))))
        # Global Lax-Friedrichs splitting: f = f+ + f- with f+' >= 0 >= f-'.
        fp = 0.5 * (flux(v) + alpha * v)
        fm = 0.5 * (flux(v) - alpha * v)
        f_left = recon(fp)                              # biased from the left
        f_right = np.roll(recon(fm[::-1])[::-1], -1)    # mirrored for the right
        f_hat = f_left + f_right
        return -(f_hat - np.roll(f_hat, 1)) / dx

    t0, tf = float(t_span[0]), float(t_span[1])
    t = t0
    recorder = OutputRecorder.current((t0, tf))
    recorder.append(t, u)
    while t < tf and not recorder.stopped:
        if recorder.stopped:
            break
        alpha = max(float(np.max(np.abs(wave_speed(u)))), 1e-12)
        dt = min(cfl * dx / alpha, tf - t)
        u = ssp_rk3(rhs, u, dt)
        t += dt
        recorder.append(t, u)
    times, states, _ = recorder.finish()
    result = PDESolution(states, (x,), times, f"weno{order}_ssprk3")
    result.checkpoint = recorder.checkpoint(result.method)
    result._final_state = result.checkpoint.y
    return result


def weno_burgers(u0, x_span, t_span, nx: int = 200, cfl: float = 0.4):
    """Inviscid Burgers with WENO5 and SSP-RK3."""
    return weno_conservation_law(u0, lambda v: 0.5 * v * v, lambda v: v,
                                 x_span, t_span, nx, cfl)


def poisson_periodic_fft(f, dx: float, dy: float, stencil: str = "wide"):
    """Doubly periodic Poisson solve ``lap p = f`` by FFT.

    The mean of ``f`` is removed first: with periodic boundaries on all sides
    the Laplacian annihilates constants, so a source with nonzero mean has no
    solution at all.  The pressure is likewise pinned to zero mean.

    ``stencil`` selects which *discrete* symbol is inverted, and it must match
    the difference operators the caller uses, not the continuous ``-k^2``:

    - ``"compact"`` inverts the 5-point Laplacian, the composition of forward
      and backward differences.
    - ``"wide"`` (default) inverts the composition of two *central*
      differences, which is what a collocated projection method needs.  Its
      symbol vanishes at the Nyquist modes -- the classic odd-even
      decoupling -- so those modes are simply pinned to zero rather than
      divided by something near zero.
    """
    ny, nx = f.shape
    fh = np.fft.fft2(f - f.mean())
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    if stencil == "compact":
        lam = ((2 * np.cos(kx * dx) - 2) / dx**2)[None, :] + \
              ((2 * np.cos(ky * dy) - 2) / dy**2)[:, None]
    elif stencil == "wide":
        lam = (-(np.sin(kx * dx) / dx) ** 2)[None, :] + \
              (-(np.sin(ky * dy) / dy) ** 2)[:, None]
    else:
        raise ValueError("stencil must be 'compact' or 'wide'")
    scale = max(1.0 / dx**2, 1.0 / dy**2)
    null = np.abs(lam) < 1e-12 * scale        # constant and checkerboard modes
    lam = np.where(null, 1.0, lam)
    ph = fh / lam
    ph[null] = 0.0
    return np.real(np.fft.ifft2(ph))


def navier_stokes_2d(u0, v0, nu: float, t_span, nx: int = 64, ny: int = 64,
                     L: float = 2 * np.pi, nt: int = 200, forcing=None):
    """Incompressible Navier-Stokes in 2-D by Chorin's projection method.

    Each step advances velocity ignoring pressure, then *projects* the result
    onto the divergence-free subspace by solving a Poisson equation for the
    pressure correction.  The projection is what enforces incompressibility:
    pressure in an incompressible flow is not a thermodynamic variable at all
    but the Lagrange multiplier of the constraint ``div u = 0``, which is why
    it is solved for rather than stepped.

    Doubly periodic.  Returns a :class:`PDESolution` whose ``u`` is stacked
    ``(u, v)`` fields over time.
    """
    dx = dy = L / nx
    x = np.linspace(0.0, L, nx, endpoint=False)
    y = np.linspace(0.0, L, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="xy")
    u = u0(X, Y) if callable(u0) else np.asarray(u0, float)
    v = v0(X, Y) if callable(v0) else np.asarray(v0, float)
    t0, tf = float(t_span[0]), float(t_span[1])
    dt = (tf - t0) / nt
    times = TimeGrid(t0, tf, nt + 1)
    out = Trajectory(times)
    out[0] = np.stack([u, v])

    def ddx(a):
        return (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1)) / (2 * dx)

    def ddy(a):
        return (np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0)) / (2 * dy)

    def lap(a):
        return ((np.roll(a, -1, axis=1) - 2 * a + np.roll(a, 1, axis=1)) / dx**2
                + (np.roll(a, -1, axis=0) - 2 * a + np.roll(a, 1, axis=0)) / dy**2)



    for k in range(nt):
        if out.recorder.stopped:
            break
        fx, fy = (forcing(X, Y, times[k]) if forcing is not None
                  else (np.zeros_like(u), np.zeros_like(v)))
        # Predictor: momentum without the pressure gradient.
        us = u + dt * (-u * ddx(u) - v * ddy(u) + nu * lap(u) + fx)
        vs = v + dt * (-u * ddx(v) - v * ddy(v) + nu * lap(v) + fy)
        # Pressure Poisson equation from the divergence of the corrector.
        # Divergence and gradient both use central differences, and the FFT
        # inverts the symbol of *their* composition.  Any mismatch here leaves
        # a residual divergence of order dx^2, so the projection would only
        # approximately enforce the constraint it exists to enforce.
        div = ddx(us) + ddy(vs)
        p = poisson_periodic_fft(div / dt, dx, dy, stencil="wide")
        u = us - dt * ddx(p)
        v = vs - dt * ddy(p)
        out[k + 1] = np.stack([u, v])
    return PDESolution(out, (x, y), times, "navier_stokes_projection")


def vorticity_streamfunction(omega0, nu: float, t_span, nx: int = 64, ny: int = 64,
                             L: float = 2 * np.pi, nt: int = 200):
    """2-D Navier-Stokes in vorticity-streamfunction form, pseudo-spectral.

    Eliminating pressure and velocity in favour of the scalar vorticity turns
    the system into one transport equation plus one Poisson solve.  In two
    dimensions this is exact -- there is no vortex stretching -- so nothing is
    lost, and incompressibility is satisfied identically because the velocity
    is defined as the curl of a stream function.
    """
    x = np.linspace(0.0, L, nx, endpoint=False)
    y = np.linspace(0.0, L, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="xy")
    w = omega0(X, Y) if callable(omega0) else np.asarray(omega0, float)
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=L / nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=L / ny)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    K2 = KX**2 + KY**2
    K2inv = np.where(K2 == 0, 1.0, K2)
    # 2/3 dealiasing: the quadratic nonlinearity aliases the top third of the
    # modes back onto the resolved ones, which is a classic instability.
    mask = (np.abs(KX) < (2 / 3) * np.max(np.abs(kx))) & \
           (np.abs(KY) < (2 / 3) * np.max(np.abs(ky)))
    t0, tf = float(t_span[0]), float(t_span[1])
    dt = (tf - t0) / nt
    times = TimeGrid(t0, tf, nt + 1)
    out = Trajectory(times)
    out[0] = w

    def rhs(wh):
        psih = wh / K2inv
        psih[0, 0] = 0.0
        u = np.real(np.fft.ifft2(1j * KY * psih))
        v = np.real(np.fft.ifft2(-1j * KX * psih))
        wx = np.real(np.fft.ifft2(1j * KX * wh))
        wy = np.real(np.fft.ifft2(1j * KY * wh))
        return -np.fft.fft2(u * wx + v * wy) * mask - nu * K2 * wh

    wh = np.fft.fft2(w)
    for k in range(nt):
        if out.recorder.stopped:
            break
        k1 = rhs(wh)
        k2 = rhs(wh + 0.5 * dt * k1)
        k3 = rhs(wh + 0.5 * dt * k2)
        k4 = rhs(wh + dt * k3)
        wh = wh + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        out[k + 1] = np.real(np.fft.ifft2(wh))
    return PDESolution(out, (x, y), times, "vorticity_streamfunction")


def lid_driven_cavity(re: float = 100.0, n: int = 41, tol: float = 1e-6,
                      max_iter: int = 50000, dt=None):
    """Steady lid-driven cavity flow, vorticity-streamfunction on a unit square.

    The canonical incompressible benchmark.  The wall vorticity comes from
    Thom's formula, which encodes the no-slip condition -- getting that
    boundary treatment right, rather than the interior scheme, is what
    determines whether the solution is correct.

    Returns ``(psi, omega, x)``.
    """
    h = 1.0 / (n - 1)
    x = np.linspace(0.0, 1.0, n)
    psi = np.zeros((n, n))
    w = np.zeros((n, n))
    nu = 1.0 / re
    dt = 0.5 * min(0.25 * h * h / nu, h) if dt is None else dt

    fast = _accel.kernel("lid_driven_cavity")
    if fast is not None and n >= 3:
        fast(psi, w, h, nu, dt, tol, max_iter)
        return psi, w, x

    for it in range(1, max_iter + 1):
        # Stream function from vorticity: lap psi = -omega (SOR sweep).
        for _ in range(30):
            psi_old = psi.copy()
            psi[1:-1, 1:-1] = 0.25 * (psi[2:, 1:-1] + psi[:-2, 1:-1]
                                      + psi[1:-1, 2:] + psi[1:-1, :-2]
                                      + h * h * w[1:-1, 1:-1])
            if np.max(np.abs(psi - psi_old)) < 1e-10:
                break
        # Thom's wall vorticity: a one-sided expansion of lap psi = -omega
        # combined with psi = 0 and the prescribed wall velocity.
        w_new = w.copy()
        w_new[0, :] = 2.0 * (psi[0, :] - psi[1, :]) / h**2
        w_new[-1, :] = 2.0 * (psi[-1, :] - psi[-2, :]) / h**2 - 2.0 / h   # moving lid
        w_new[:, 0] = 2.0 * (psi[:, 0] - psi[:, 1]) / h**2
        w_new[:, -1] = 2.0 * (psi[:, -1] - psi[:, -2]) / h**2
        u = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * h)
        v = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * h)
        wx = (w[1:-1, 2:] - w[1:-1, :-2]) / (2 * h)
        wy = (w[2:, 1:-1] - w[:-2, 1:-1]) / (2 * h)
        lapw = (w[2:, 1:-1] + w[:-2, 1:-1] + w[1:-1, 2:] + w[1:-1, :-2]
                - 4 * w[1:-1, 1:-1]) / h**2
        w_new[1:-1, 1:-1] = w[1:-1, 1:-1] + dt * (-u * wx - v * wy + nu * lapw)
        change = float(np.max(np.abs(w_new - w)))
        w = w_new
        if change < tol:
            break
    return psi, w, x


# Share output policy through nested method-of-lines and wrapper calls.
for _name in __all__:
    if "t_span" in __import__("inspect").signature(globals()[_name]).parameters:
        globals()[_name] = output_control(globals()[_name])
del _name
