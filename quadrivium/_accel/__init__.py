"""C acceleration built into the required array core.

All numerical kernels share the array core's storage and require only a C
compiler to build. The readable Python implementations remain selectable via
``disabled()`` or ``QUADRIVIUM_NO_ACCEL=1`` for comparison and debugging.
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
    from .._qnp import _accel as _native
except Exception:  # noqa: BLE001 - any import failure means "no fast path"
    _native = None

# A context-local switch keeps nested calls, threads and asyncio tasks from
# changing one another's backend or restoring a stale process-wide value.
_active = ContextVar("quadrivium_accel_active", default=_native is not None)


def available() -> bool:
    """True when the compiled backend is importable *and* currently enabled."""
    return _active.get()


def backend() -> str:
    """``'c'`` or ``'python'`` -- whichever will actually run."""
    return "c" if available() else "python"


def version() -> Optional[str]:
    """Version string of the compiled extension, or ``None`` if absent."""
    return getattr(_native, "__version__", None) if _native is not None else None


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
    token = _active.set(_native is not None)
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
    if not _active.get() or _native is None:
        return None
    return getattr(_native, name, None)


def translate_error(exc: Exception, singular, not_positive_definite):
    """Re-raise a kernel error as the package's own exception type.

    The C side tags failures with a short prefix rather than carrying the
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
    if _native is None:
        return []
    return sorted(n for n in dir(_native) if not n.startswith("_"))


def show_config() -> str:
    """One-glance summary of which backend is active and what it covers.

    >>> import quadrivium as qd
    >>> print(qd.accel.show_config())          # doctest: +SKIP
    quadrivium acceleration: c (extension 1.2.0)
      compiled kernels (14): back_substitution, cholesky, erf, ...
    """
    if not available():
        reason = (
            "QUADRIVIUM_NO_ACCEL is set"
            if _env_off()
            else "extension not built for this interpreter"
            if _native is None
            else "disabled at runtime"
        )
        return f"quadrivium acceleration: python ({reason})"
    names = kernels()
    listed = ", ".join(n for n in names if n != "__version__")
    return (
        f"quadrivium acceleration: c (extension {version()})\n"
        f"  compiled kernels ({len(names)}): {listed}"
    )
