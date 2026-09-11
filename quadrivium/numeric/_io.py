"""Dependency-free NumPy .npy persistence and memory mapping.

Only fixed-width numeric arrays are accepted; loading never unpickles objects.
Headers are bounded and parsed with literal_eval. Native-endian mapped arrays
hold a buffer export, so a live view cannot outlast its mapping.
"""
from __future__ import annotations

import ast
import math
import mmap
import os
import struct
import sys

from .. import _qnp as _c

_CODES = {"bool": "b1", "int64": "i8", "float32": "f4", "float64": "f8",
          "complex64": "c8", "complex128": "c16"}
_DTYPES = {value: key for key, value in _CODES.items()}
_MAGIC = b"\x93NUMPY"
_NATIVE = "<" if sys.byteorder == "little" else ">"
_MAX_HEADER = 65536


def _path(file):
    path = os.fspath(file)
    suffix = b".npy" if isinstance(path, bytes) else ".npy"
    return path if path.endswith(suffix) else path + suffix


def _shape(shape):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    if len(shape) > 16 or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("shape must contain at most 16 nonnegative integer dimensions")
    return shape


def _write_header(stream, shape, dtype, fortran=False):
    code = _CODES[dtype.name]
    descr = ("|" if dtype.itemsize == 1 else _NATIVE) + code
    data = repr({"descr": descr, "fortran_order": bool(fortran), "shape": shape}).encode("latin1")
    # The data start is aligned to 64 bytes (also safe for native C scalars).
    data += b" " * ((-(10 + len(data) + 1)) % 64) + b"\n"
    if len(data) >= 65536:
        raise ValueError("array header is too large")
    stream.write(_MAGIC + bytes((1, 0)) + struct.pack("<H", len(data)) + data)
    return 10 + len(data)


def _read_exact(stream, count):
    data = stream.read(count)
    if len(data) != count:
        raise ValueError("truncated .npy file")
    return data


def _read_header(stream):
    if _read_exact(stream, 6) != _MAGIC:
        raise ValueError("expected a .npy numeric array")
    version = tuple(_read_exact(stream, 2))
    if version == (1, 0):
        size = struct.unpack("<H", _read_exact(stream, 2))[0]
    elif version in ((2, 0), (3, 0)):
        size = struct.unpack("<I", _read_exact(stream, 4))[0]
    else:
        raise ValueError("unsupported .npy version")
    if size > _MAX_HEADER:
        raise ValueError("array header exceeds the 64 KiB limit")
    try:
        header = ast.literal_eval(_read_exact(stream, size).decode("utf8" if version == (3, 0) else "latin1"))
    except (SyntaxError, UnicodeError, MemoryError, RecursionError) as exc:
        raise ValueError("invalid .npy header") from exc
    if not isinstance(header, dict) or set(header) != {"descr", "shape", "fortran_order"}:
        raise ValueError("invalid .npy header fields")
    descr = header["descr"]
    if not isinstance(descr, str) or len(descr) < 2 or descr[0] not in "<>=|" or descr[1:] not in _DTYPES:
        raise ValueError("only fixed-width bool/int64/float/complex arrays are supported; object arrays are forbidden")
    if type(header["fortran_order"]) is not bool:
        raise ValueError("invalid array storage order")
    shape = _shape(header["shape"])
    dt = _c.dtype(_DTYPES[descr[1:]])
    count = math.prod(shape)
    if count > sys.maxsize // dt.itemsize:
        raise ValueError("array size overflows the address space")
    native = descr[0] in ("=", "|", _NATIVE) or dt.itemsize == 1
    return shape, dt, header["fortran_order"], native, stream.tell()


def _reshape(flat, shape, fortran):
    if fortran and len(shape) > 1:
        return flat.reshape(shape[::-1]).transpose(tuple(range(len(shape) - 1, -1, -1)))
    return flat.reshape(shape)


def save(file, array, allow_pickle=False):
    """Write a numeric .npy file, in chunks for strided inputs.

    Path names acquire ``.npy`` when absent. File objects stay open. Object
    serialization is always disabled, regardless of ``allow_pickle``.
    """
    a = _c.asarray(array)
    close = not hasattr(file, "write")
    stream = open(_path(file), "wb") if close else file
    try:
        _write_header(stream, a.shape, a.dtype)
        if a.size == 0:
            return
        if a.flags.c_contiguous:
            stream.write(memoryview(a).cast("B"))
        else:
            # Fixed-size pieces avoid a contiguous copy of the whole input.
            step = max(1, 65536 // a.itemsize)
            block = _c.empty((min(step, a.size),), dtype=a.dtype)
            used = 0
            for value in a.flat:
                block[used] = value
                used += 1
                if used == block.size:
                    stream.write(memoryview(block).cast("B"))
                    used = 0
            if used:
                stream.write(memoryview(block[:used]).cast("B"))
    finally:
        if close:
            stream.close()


def load(file, mmap_mode=None, allow_pickle=False):
    """Read a numeric .npy array, optionally mapping in ``r``, ``r+``, or ``c`` mode.

    Ordinary loading returns writable owned storage. Object/pickle payloads
    are rejected. Non-native-endian data is converted while loading and may
    not be memory mapped.
    """
    if mmap_mode is not None:
        if hasattr(file, "read"):
            raise TypeError("memory mapping requires a file path")
        return open_memmap(file, mode=mmap_mode)
    close = not hasattr(file, "read")
    stream = open(file, "rb") if close else file
    try:
        shape, dt, fortran, native, _ = _read_header(stream)
        count = math.prod(shape)
        flat = _c.empty((count,), dtype=dt)
        if count == 0:
            return _reshape(flat, shape, fortran)
        view = memoryview(flat).cast("B")
        # readinto avoids a second full-size bytes allocation.
        if hasattr(stream, "readinto"):
            pos = 0
            while pos < len(view):
                n = stream.readinto(view[pos:])
                if not n:
                    raise ValueError("truncated .npy payload")
                pos += n
        else:
            for pos in range(0, len(view), 65536):
                part = _read_exact(stream, min(65536, len(view) - pos))
                view[pos:pos+len(part)] = part
        if not native:
            width = dt.itemsize // (2 if dt.kind == "c" else 1)
            for pos in range(0, len(view), width):
                view[pos:pos+width] = bytes(view[pos:pos+width])[::-1]
        return _reshape(flat, shape, fortran)
    finally:
        if close:
            stream.close()


def open_memmap(filename, mode="r+", dtype=None, shape=None, fortran_order=False):
    """Open/create a .npy array backed by the OS page cache.

    ``w+`` creates a file and requires ``shape``; ``r`` is read-only, ``r+``
    persists writes, and ``c`` is copy-on-write. Call :func:`flush` to persist
    dirty pages. Views retain the mapping automatically. Truncating a mapped
    file externally while it is in use is unsupported.
    """
    if mode not in ("r", "r+", "w+", "c"):
        raise ValueError("mode must be 'r', 'r+', 'w+', or 'c'")
    if mode == "w+":
        if shape is None:
            raise ValueError("shape is required when creating a mapping")
        shape = _shape(shape)
        dt = _c.dtype(_c.float64 if dtype is None else dtype)
        count = math.prod(shape)
        if count > (sys.maxsize - _MAX_HEADER) // dt.itemsize:
            raise ValueError("mapped array size overflows the address space")
    with open(filename, "w+b" if mode == "w+" else "rb" if mode in ("r", "c") else "r+b") as stream:
        if mode == "w+":
            offset = _write_header(stream, shape, dt, fortran_order)
            count = math.prod(shape)
            if count > (sys.maxsize - offset) // dt.itemsize:
                raise ValueError("mapped array size overflows the address space")
            stream.truncate(offset + count * dt.itemsize)
            stream.flush()
            fortran = bool(fortran_order)
        else:
            shape, dt, fortran, native, offset = _read_header(stream)
            if not native:
                raise ValueError("memory mapping requires native-endian numeric data")
            count = math.prod(shape)
            if os.fstat(stream.fileno()).st_size < offset + count * dt.itemsize:
                raise ValueError("truncated .npy payload")
        access = mmap.ACCESS_READ if mode == "r" else mmap.ACCESS_COPY if mode == "c" else mmap.ACCESS_WRITE
        mapping = mmap.mmap(stream.fileno(), 0, access=access)
        try:
            flat = _c.frombuffer(mapping, dtype=dt, count=count, offset=offset)
        except Exception:
            mapping.close()
            raise
    return _reshape(flat, shape, fortran)


def flush(array):
    """Flush a memory-mapped array (or any view of one) to its backing file."""
    owner = array
    seen = set()
    while id(owner) not in seen:
        seen.add(id(owner))
        if isinstance(owner, mmap.mmap):
            owner.flush()
            return
        owner = owner.obj if isinstance(owner, memoryview) else getattr(owner, "base", None)
        if owner is None:
            break
    raise ValueError("array is not backed by a memory map")
