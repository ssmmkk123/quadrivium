"""Optional compiled backend.

The package ships a Rust extension holding compiled versions of the kernels
that dominate runtime -- factorizations, eigensolvers, transforms. It is
strictly optional: when the extension is missing (a source install without a
Rust toolchain, an unsupported platform, or ``QUADRIVIUM_NO_ACCEL=1`` in the
environment) every routine falls back to the pure-Python implementation and
computes exactly the same thing, only slower.

Nothing in the public API changes based on which backend is active. Use
:func:`available` to report what is in use, and :func:`disabled` in tests or
benchmarks to force the Python path::

    from quadrivium import _accel
    with _accel.disabled():
        ...  # runs the readable reference implementation
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Optional

__all__ = ["available", "backend", "version", "disabled", "enabled", "kernel",
           "kernels", "show_config", "translate_error"]


def _env_off() -> bool:
    return os.environ.get("QUADRIVIUM_NO_ACCEL", "").strip().lower() not in ("", "0", "false", "no")


try:  # pragma: no cover - depends on how the package was installed
    if _env_off():
        raise ImportError("disabled by QUADRIVIUM_NO_ACCEL")
    from .. import _quadrivium_rs as _rs  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001 - any import failure means "no fast path"
    _rs = None

# A context-local switch keeps nested calls, threads and asyncio tasks from
# changing one another's backend or restoring a stale process-wide value.
_active = ContextVar("quadrivium_accel_active", default=_rs is not None)


def available() -> bool:
    """True when the compiled backend is importable *and* currently enabled."""
    return _active.get()


def backend() -> str:
    """``'rust'`` or ``'python'`` -- whichever will actually run."""
    return "rust" if available() else "python"


def version() -> Optional[str]:
    """Version string of the compiled extension, or ``None`` if absent."""
    return getattr(_rs, "__version__", None) if _rs is not None else None


@contextmanager
def disabled():
    """Force Python in this context, independently of other threads/tasks."""
    token = _active.set(False)
    try:
        yield
    finally:
        _active.reset(token)


@contextmanager
def enabled():
    """Enable the compiled path in this context, if it is importable."""
    token = _active.set(_rs is not None)
    try:
        yield
    finally:
        _active.reset(token)


def kernel(name: str) -> Optional[Callable]:
    """Return the compiled kernel called ``name``, or ``None``.

    Callers use the ``None`` result to select their own Python implementation,
    which keeps the fallback visible at the call site rather than hidden behind
    a dispatcher.
    """
    if not _active.get() or _rs is None:
        return None
    return getattr(_rs, name, None)


def translate_error(exc: Exception, singular, not_positive_definite):
    """Re-raise a kernel error as the package's own exception type.

    The Rust side tags failures with a short prefix rather than carrying the
    Python exception hierarchy across the boundary.
    """
    msg = str(exc)
    if msg.startswith("singular:"):
        return singular(msg[len("singular:"):])
    if msg.startswith("notpd:"):
        return not_positive_definite(msg[len("notpd:"):])
    return None


def kernels() -> "list[str]":
    """Names of the compiled kernels this build provides, sorted."""
    if _rs is None:
        return []
    return sorted(n for n in dir(_rs) if not n.startswith("_"))


def show_config() -> str:
    """One-glance summary of which backend is active and what it covers.

    >>> import quadrivium as qd
    >>> print(qd.accel.show_config())          # doctest: +SKIP
    quadrivium acceleration: rust (extension 1.2.0)
      compiled kernels (14): back_substitution, cholesky, erf, ...
    """
    if not available():
        reason = (
            "QUADRIVIUM_NO_ACCEL is set"
            if _env_off()
            else "extension not built for this interpreter"
            if _rs is None
            else "disabled at runtime"
        )
        return f"quadrivium acceleration: python ({reason})"
    names = kernels()
    listed = ", ".join(n for n in names if n != "__version__")
    return (
        f"quadrivium acceleration: rust (extension {version()})\n"
        f"  compiled kernels ({len(names)}): {listed}"
    )
