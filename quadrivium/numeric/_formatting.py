"""Rendering arrays the way NumPy renders them.

Printing is not on any hot path, so it lives in Python where the alignment
rules are easy to read.  The rules are NumPy's: every element gets the shortest
representation that round-trips (capped at eight decimals), the integer and
fractional parts are padded to a common width so the decimal points line up,
the whole array switches to exponential notation once the magnitudes spread too
far, and arrays past a thousand elements are summarised with an ellipsis.
"""

from __future__ import annotations

import math

#: Print options, with NumPy's defaults and names. `set_printoptions` in the
#: package namespace edits this in place.
OPTIONS = {
    "threshold": 1000,     # summarise arrays larger than this
    "edgeitems": 3,        # items shown at each end when summarising
    "linewidth": 75,
    "precision": 8,
    "suppress": False,     # never fall back to exponential for small values
}


def _split_fixed(value):
    """Shortest round-tripping decimal for `value`, split at the point."""
    precision = OPTIONS["precision"]
    text = repr(float(value))
    if "e" in text or "E" in text:
        text = f"{value:.{precision}f}"
    if "." in text:
        left, right = text.split(".")
        if len(right) > precision:
            left, right = f"{value:.{precision}f}".split(".")
        right = right.rstrip("0")
    else:
        left, right = text, ""
    return left, right, ""


def _split_exponential(value):
    text = f"{value:.{OPTIONS['precision']}e}"
    mantissa, exponent = text.split("e")
    left, right = mantissa.split(".")
    return left, right.rstrip("0"), "e" + exponent


class _FloatFormat:
    """One shared layout for every element of a float array."""

    def __init__(self, values):
        finite = [v for v in values if math.isfinite(v)]
        self.specials = {}
        nonzero = [abs(v) for v in finite if v != 0.0]
        self.exponential = False
        if nonzero:
            hi, lo = max(nonzero), min(nonzero)
            if OPTIONS["suppress"]:
                # Fixed point unless the magnitudes are outright too large.
                self.exponential = hi >= 1e8
            else:
                self.exponential = hi >= 1e8 or lo < 1e-4 or hi / lo > 1000.0
        split = _split_exponential if self.exponential else _split_fixed
        self.pieces = {}
        left_width = right_width = 0
        for v in finite:
            left, right, suffix = split(v)
            self.pieces[v] = (left, right, suffix)
            left_width = max(left_width, len(left))
            right_width = max(right_width, len(right))
        self.left_width = left_width
        self.right_width = right_width
        self.special_width = 0
        for v in values:
            if not math.isfinite(v):
                text = "nan" if v != v else ("inf" if v > 0 else "-inf")
                self.specials[repr(v)] = text
                self.special_width = max(self.special_width, len(text))
        suffix_width = len(next(iter(self.pieces.values()))[2]) if self.pieces else 0
        self.width = max(left_width + 1 + right_width + suffix_width, self.special_width)

    def __call__(self, value):
        value = float(value)
        if not math.isfinite(value):
            text = "nan" if value != value else ("inf" if value > 0 else "-inf")
            return text.rjust(self.width)
        left, right, suffix = self.pieces.get(value, _split_fixed(value))
        # Exponential notation pads the fraction with zeros; fixed-point pads
        # with spaces so the columns line up under the decimal point.
        filler = "0" if self.exponential else " "
        text = (left.rjust(self.left_width) + "." +
                right.ljust(self.right_width, filler) + suffix)
        return text.rjust(self.width)


class _IntFormat:
    def __init__(self, values):
        self.width = max((len(str(v)) for v in values), default=1)

    def __call__(self, value):
        return str(value).rjust(self.width)


class _BoolFormat:
    def __init__(self, values):
        self.width = 5 if any(not v for v in values) else 4

    def __call__(self, value):
        return ("True" if value else "False").rjust(self.width)


class _ComplexFormat:
    def __init__(self, values):
        self.real = _FloatFormat([v.real for v in values])
        self.imag = _FloatFormat([abs(v.imag) for v in values])

    def __call__(self, value):
        sign = "-" if math.copysign(1.0, value.imag) < 0 else "+"
        imaginary = sign + self.imag(abs(value.imag))
        # NumPy puts the `j` before the alignment padding, not after it.
        body = imaginary.rstrip()
        padding = " " * (len(imaginary) - len(body))
        return f"{self.real(value.real)}{body}j{padding}"

    @property
    def width(self):
        return self.real.width + self.imag.width + 2


def _sample(array):
    """Every element, or just the ones a summarised view will show."""
    if array.ndim == 0:
        return [array.item()]
    if array.size <= OPTIONS["threshold"]:
        return array.reshape(-1).tolist()
    edge = OPTIONS["edgeitems"] * 2
    flat = array.reshape(-1)
    return flat[:edge].tolist() + flat[-edge:].tolist()


def _formatter(array):
    values = _sample(array)
    kind = array.dtype.name
    if not values:
        return _IntFormat([])
    if kind == "bool":
        return _BoolFormat(values)
    if kind == "int64":
        return _IntFormat(values)
    if kind == "complex128":
        return _ComplexFormat(values)
    return _FloatFormat([float(v) for v in values])


def _render(array, fmt, indent, summarise, comma=True):
    edgeitems = OPTIONS["edgeitems"]
    linewidth = OPTIONS["linewidth"]
    if array.ndim == 1:
        n = array.shape[0]
        if summarise and n > 2 * edgeitems:
            pieces = ([fmt(array[i]) for i in range(edgeitems)] + ["..."] +
                      [fmt(array[n - edgeitems + i]) for i in range(edgeitems)])
        else:
            pieces = [fmt(array[i]) for i in range(n)]
        joiner = ", " if comma else " "
        line = "[" + joiner.join(pieces) + "]"
        if len(line) + indent <= linewidth or not pieces:
            return line
        # Wrap, counting the characters already printed to the left of the
        # opening bracket only on the first line; continuation lines carry
        # their own leading spaces inside `current`.
        out, current, margin = [], "[", indent
        for i, piece in enumerate(pieces):
            last = i == len(pieces) - 1
            addition = piece + ("" if last or not comma else ",")
            if current != "[" and margin + len(current) + len(addition) > linewidth:
                out.append(current.rstrip())
                current = " " * (indent + 1)
                margin = 0
            current += addition + ("" if last else " ")
        # The final element keeps its alignment padding, as NumPy's does; only
        # the space after a trailing comma is trimmed from wrapped lines.
        out.append(current + "]")
        return "\n".join(out)
    n = array.shape[0]
    if summarise and n > 2 * edgeitems:
        indices = list(range(edgeitems)) + [None] + list(range(n - edgeitems, n))
    else:
        indices = list(range(n))
    rows = []
    for i in indices:
        if i is None:
            rows.append(" " * (indent + 1) + "...")
            continue
        text = _render(array[i], fmt, indent + 1, summarise, comma)
        prefix = "" if i == indices[0] else " " * (indent + 1)
        rows.append(prefix + text)
    separator = ("," if comma else "") + "\n" + "\n" * (array.ndim - 2)
    return "[" + separator.join(rows) + "]"


def format_array(array, is_repr):
    """Render `array`; `is_repr` selects the ``array(...)`` wrapper."""
    if array.ndim == 0:
        fmt = _formatter(array)
        body = fmt(array.item()).strip()
        return f"array({body})" if is_repr else body
    if array.size == 0:
        body = "[" * array.ndim + "]" * array.ndim
        if is_repr:
            return f"array({body}, dtype={array.dtype.name})"
        return body
    fmt = _formatter(array)
    indent = 6 if is_repr else 0
    summarise = array.size > OPTIONS["threshold"]
    # `str` separates with spaces and `repr` with commas, as NumPy does.
    body = _render(array, fmt, indent, summarise, comma=is_repr)
    if not is_repr:
        return body
    # A summarised repr no longer shows the full extent, so NumPy states it.
    suffix = f", shape={array.shape}" if summarise else ""
    return f"array({body}{suffix})"
