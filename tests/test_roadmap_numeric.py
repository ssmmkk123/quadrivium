"""Independent references and ownership checks for compact arrays and factors."""
import gc
import io
import json
import mmap
import struct
import weakref

import numpy as np
import pytest

from quadrivium import numeric as q


@pytest.mark.parametrize("name", ["bool", "int64", "float32", "float64", "complex64", "complex128"])
def test_storage_buffer_cast_creation(name):
    dt = q.dtype(name)
    raw = np.array([[0, 2, 3], [4, 5, 6]], dtype=name)
    if dt.kind == "c":
        raw += 1j * raw
    a = q.asarray(raw)
    assert a.dtype == dt
    assert a.nbytes == raw.nbytes
    assert memoryview(a).itemsize == raw.itemsize
    np.testing.assert_array_equal(np.asarray(a), raw)
    np.testing.assert_array_equal(np.asarray(a.T[::-1]), raw.T[::-1])
    for other in (q.float32, q.complex64, q.float64, q.complex128):
        with np.errstate(all="ignore"):
            expected = raw.astype(str(other))
        np.testing.assert_array_equal(np.asarray(a.astype(other)), expected)
    for creator in (q.zeros, q.ones, q.empty):
        assert creator((2, 3), dtype=dt).nbytes == 6 * dt.itemsize
    np.testing.assert_array_equal(np.asarray(q.full((4,), 2, dtype=dt)), np.full(4, 2, dtype=name))
    np.testing.assert_array_equal(np.asarray(q.eye(3, dtype=dt)), np.eye(3, dtype=name))
    np.testing.assert_array_equal(np.asarray(q.arange(4, dtype=dt)), np.arange(4).astype(name))
    np.testing.assert_array_equal(np.asarray(q.linspace(0, 2, 4, dtype=dt)), np.linspace(0, 2, 4).astype(name))


@pytest.mark.parametrize("dtype", [q.float32, q.complex64])
@pytest.mark.parametrize("name", ["negative", "absolute", "sqrt", "exp", "log", "log2", "log10", "log1p", "expm1", "sin", "cos", "tan", "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan", "arcsinh", "arccosh", "arctanh", "square", "reciprocal", "sign", "conjugate", "real", "imag", "angle", "isfinite", "isnan", "isinf", "logical_not"])
def test_compact_unary_strided(dtype, name):
    raw = np.linspace(.15, .75, 12).reshape(3, 4).astype(str(dtype))
    if dtype.kind == "c":
        raw += .1j
    a = q.asarray(raw).T[::-1]
    with np.errstate(all="ignore"):
        expected = getattr(np, name)(raw.T[::-1])
        result = getattr(q, name)(a)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=3e-6, atol=2e-7, equal_nan=True)
    assert result.dtype.name == expected.dtype.name


@pytest.mark.parametrize("dtype", [q.float32, q.complex64])
@pytest.mark.parametrize("name", ["add", "subtract", "multiply", "divide", "power", "maximum", "minimum", "equal", "not_equal", "less", "greater"])
def test_compact_binary_broadcast(dtype, name):
    raw = np.array([[1, 2], [3, 4]], dtype=str(dtype))
    b = np.array([.5, 2], dtype=str(dtype))
    if dtype.kind == "c":
        raw += .2j
    result = getattr(q, name)(q.asarray(raw), q.asarray(b))
    np.testing.assert_allclose(np.asarray(result), getattr(np, name)(raw, b), rtol=2e-6, atol=1e-6)
    if result.dtype.kind != "b":
        assert result.dtype == dtype
    a = q.asarray(raw)
    a[1:] += a[:-1]
    raw[1:] += raw[:-1]
    np.testing.assert_allclose(np.asarray(a), raw, rtol=2e-6)


def test_dtype_inference_promotion_and_weak_scalars():
    assert q.array([1, 2]).dtype == q.int64
    assert q.array([True, False]).dtype == q.bool_
    assert q.array([1., 2.]).dtype == q.float64
    assert q.result_type(q.float32, q.int64) == q.float64
    assert q.result_type(q.float32, q.complex64) == q.complex64
    assert q.result_type(q.float64, q.complex64) == q.complex128
    assert (q.ones(4, dtype=q.float32) + 1.).dtype == q.float32
    assert (q.ones(4, dtype=q.complex64) * (1+2j)).dtype == q.complex64
    assert q.finfo(q.float32).eps == np.finfo(np.float32).eps
    assert q.finfo(q.complex64).tiny == np.finfo(np.complex64).tiny


@pytest.mark.parametrize("dtype", [q.float32, q.complex64])
@pytest.mark.parametrize("name", ["sum", "prod", "mean", "max", "min", "argmax", "argmin", "cumsum", "cumprod", "sort", "argsort"])
def test_compact_reductions(dtype, name):
    raw = np.array([[1, 3, 2], [2, 0, 4]], dtype=str(dtype))
    if dtype.kind == "c":
        raw += .125j
    a = q.asarray(raw).T
    result = getattr(q, name)(a, axis=0)
    expected = getattr(np, name)(raw.T, axis=0)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=3e-6, atol=1e-6)


def test_compact_real_imag_views_and_scatter():
    a = q.array([1+2j, 3+4j], dtype=q.complex64)
    assert a.real.dtype == q.float32
    a.real += 2
    a.imag[:] = -1
    q.add.at(a, q.array([0, 0], dtype=q.int64), 1+2j)
    np.testing.assert_array_equal(np.asarray(a), [5+3j, 5-1j])


@pytest.mark.parametrize("shapes", [((2, 1, 3, 4), (5, 4, 2)), ((3,), (2, 3, 4)), ((2, 4, 3), (3,)), ((0, 3, 4), (4, 2))])
@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128", "int64"])
def test_batched_matmul(shapes, dtype):
    rng = np.random.default_rng(91)
    left = rng.normal(size=shapes[0]).astype(dtype)
    right = rng.normal(size=shapes[1]).astype(dtype)
    if "complex" in dtype:
        left += .2j
        right += .3j
    result = q.asarray(left) @ q.asarray(right)
    np.testing.assert_allclose(np.asarray(result), left @ right, rtol=4e-6, atol=1e-6)
    assert result.dtype.name == dtype


def test_batched_solve_broadcast_vectors_and_empty():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(2, 1, 4, 4)) + 5 * np.eye(4)
    b = rng.normal(size=(3, 4, 2))
    np.testing.assert_allclose(np.asarray(q.linalg.solve(a, b)), np.linalg.solve(a, b), rtol=1e-12)
    bv = rng.normal(size=(3, 4))
    expected = np.linalg.solve(a, bv[..., None])[..., 0]
    np.testing.assert_allclose(np.asarray(q.linalg.solve(a, bv, vector=True)), expected, rtol=1e-12)
    assert q.linalg.solve(q.empty((0, 3, 3)), q.ones(3)).shape == (0, 3)


@pytest.mark.parametrize("shape", [(6, 3), (3, 6), (4, 4), (0, 4), (4, 0)])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_complex_svd_and_qr(shape, full, dtype):
    rng = np.random.default_rng(21)
    a = (rng.normal(size=shape) + 1j*rng.normal(size=shape)).astype(dtype)
    tol = 2e-5 if dtype == "complex64" else 2e-12
    u, s, vh = map(np.asarray, q.linalg.svd(a, full_matrices=full))
    k = min(shape)
    np.testing.assert_allclose((u[:, :k]*s) @ vh[:k], a, atol=tol)
    np.testing.assert_allclose(u.conj().T @ u, np.eye(u.shape[1]), atol=tol)
    np.testing.assert_allclose(vh @ vh.conj().T, np.eye(vh.shape[0]), atol=tol)
    np.testing.assert_allclose(s, np.linalg.svd(a, compute_uv=False), atol=tol)
    Q, R = map(np.asarray, q.linalg.qr(a, mode="complete" if full else "reduced"))
    np.testing.assert_allclose(Q @ R, a, atol=tol)
    np.testing.assert_allclose(Q.conj().T @ Q, np.eye(Q.shape[1]), atol=tol)


def test_complex_eigen_repeated_values_and_factor_reuse():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(2, 5, 5)) + 1j*rng.normal(size=(2, 5, 5))
    a = x @ x.conj().swapaxes(-1, -2) + np.eye(5)
    w, v = map(np.asarray, q.linalg.eigh(a))
    np.testing.assert_allclose(w, np.linalg.eigvalsh(a), rtol=1e-12)
    np.testing.assert_allclose(a @ v, v * w[..., None, :], atol=3e-12)
    L = np.asarray(q.linalg.cholesky(a))
    np.testing.assert_allclose(L @ L.conj().swapaxes(-1, -2), a, atol=1e-12)
    factor = q.linalg.lu_factor(a[0])
    rhs = rng.normal(size=(5, 3)) + 1j*rng.normal(size=(5, 3))
    np.testing.assert_allclose(np.asarray(q.linalg.lu_solve(factor, rhs)), np.linalg.solve(a[0], rhs), rtol=1e-12)
    w, v = map(np.asarray, q.linalg.eigh(q.eye(5, dtype=q.complex128)))
    np.testing.assert_array_equal(w, np.ones(5))
    np.testing.assert_allclose(v.conj().T @ v, np.eye(5), atol=1e-14)


def test_frombuffer_lifetime_readonly_alignment_and_resize():
    owner = bytearray(struct.pack("4f", 1, 2, 3, 4))
    a = q.frombuffer(owner, dtype=q.float32)
    a[0] = 7
    assert struct.unpack("4f", owner)[0] == 7
    with pytest.raises(BufferError):
        owner.extend(b"0000")
    view = a[::2]
    del a
    gc.collect()
    assert view[1] == 3
    del view
    gc.collect()
    owner.extend(b"0000")
    read = q.frombuffer(bytes(owner), dtype=q.float32)
    assert not read.flags.writeable
    with pytest.raises(ValueError):
        read[0] = 1
    with pytest.raises(ValueError):
        q.frombuffer(owner, dtype=q.float64, count=1, offset=1)
    with pytest.raises(ValueError):
        q.frombuffer(owner, dtype=q.float64, count=100)


@pytest.mark.parametrize("dtype", ["bool", "int64", "float32", "float64", "complex64", "complex128"])
def test_npy_roundtrip_strides_and_numpy(tmp_path, dtype):
    a = np.arange(24).reshape(4, 6).astype(dtype)
    if "complex" in dtype:
        a += .5j
    path = tmp_path / "array.npy"
    q.save(path, q.asarray(a).T[:, ::-1])
    np.testing.assert_array_equal(np.load(path), a.T[:, ::-1])
    np.testing.assert_array_equal(np.asarray(q.load(path)), a.T[:, ::-1])
    np.save(path, np.asfortranarray(a))
    result = q.load(path)
    np.testing.assert_array_equal(np.asarray(result), a)
    assert result.flags.f_contiguous
    mapped = q.load(path, mmap_mode="r")
    np.testing.assert_array_equal(np.asarray(mapped), a)
    assert not mapped.flags.writeable


def test_npy_endian_header_rejection_and_mapping(tmp_path):
    path = tmp_path / "map.npy"
    a = q.open_memmap(path, mode="w+", dtype=q.float32, shape=(3, 4))
    a[:] = q.arange(12).reshape(3, 4)
    view = a[1:]
    del a
    gc.collect()
    view[0, 0] = 99
    q.flush(view)
    assert q.load(path)[1, 0] == 99
    copy = q.open_memmap(path, mode="c")
    copy[1, 0] = -10
    assert q.load(path)[1, 0] == 99
    owner = view.base
    while not isinstance(owner, mmap.mmap):
        owner = owner.obj if isinstance(owner, memoryview) else owner.base
    with pytest.raises(BufferError):
        owner.close()
    path = tmp_path / "endian.npy"
    np.save(path, np.array([1+2j, 3-4j], dtype=">c8"))
    np.testing.assert_array_equal(np.asarray(q.load(path)), [1+2j, 3-4j])
    with pytest.raises(ValueError):
        q.open_memmap(path, mode="r")
    payload = io.BytesIO()
    np.save(payload, np.array([{}], dtype=object))
    payload.seek(0)
    with pytest.raises(ValueError, match="forbidden"):
        q.load(payload, allow_pickle=True)
    with pytest.raises(ValueError):
        q.load(io.BytesIO(b"\x93NUMPY\x02\x00" + struct.pack("<I", 1000000)))


def test_generator_state_atomic_json_roundtrip():
    g = q.random.default_rng(39)
    g.integers(0, 50)
    state = json.loads(json.dumps(g.state))
    expected = g.integers(0, 100, size=20)
    g.state = state
    np.testing.assert_array_equal(np.asarray(g.integers(0, 100, size=20)), np.asarray(expected))
    before = g.state
    bad = dict(before, words=[0, 0, 0, 2])
    with pytest.raises(ValueError):
        g.state = bad
    assert g.state == before


def test_public_array_dispatch_hook():
    class Custom:
        def __quadrivium_function__(self, name, *args, **kwargs):
            return (name, kwargs)
    obj = Custom()
    assert q.exp(obj) == ("exp", {})
    assert q.sum(obj, axis=1) == ("sum", {"axis": 1})
    assert q.matmul(q.eye(2), obj) == ("matmul", {})


@pytest.mark.parametrize("operation", ["fill", "real", "imag", "scatter", "outer", "shuffle", "reduce"])
def test_readonly_exports_reject_every_mutation(operation):
    # Immutable bytes make any accidental native write observable as corruption.
    original = struct.pack("8f", *range(8))
    a = q.frombuffer(original, dtype=q.complex64)
    with pytest.raises(ValueError):
        if operation == "fill":
            a.fill(2)
        elif operation == "real":
            a.real = 2
        elif operation == "imag":
            a.imag = 2
        elif operation == "scatter":
            q.add.at(a, [0, 0], 1)
        elif operation == "outer":
            q.outer(q.ones(2), q.ones(2), out=a.reshape(2, 2))
        elif operation == "shuffle":
            q.random.default_rng(0).shuffle(a)
        else:
            q.sum(q.ones((2, 4)), axis=0, out=a)
    assert original == struct.pack("8f", *range(8))


def test_empty_persistence_and_ieee_compact_reductions(tmp_path):
    for shape in ((0,), (0, 3), (3, 0), ()):
        a = q.zeros(shape, dtype=q.complex64)
        path = tmp_path / "empty.npy"
        q.save(path, a)
        assert q.load(path).shape == shape
        assert q.load(path, mmap_mode="r").shape == shape
    a = q.array([q.inf, 1], dtype=q.float32)
    assert q.sum(a) == q.inf
    assert q.prod(a) == q.inf
    assert q.cumprod(a)[-1] == q.inf


def test_invalid_mapping_creation_preserves_existing_file(tmp_path):
    path = tmp_path / "existing.npy"
    path.write_bytes(b"keep me")
    with pytest.raises(ValueError):
        q.open_memmap(path, mode="w+")
    assert path.read_bytes() == b"keep me"


def test_solve_result_does_not_leak():
    result = q.linalg.solve(q.eye(4), q.ones(4))
    reference = weakref.ref(result)
    del result
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("source", ["bytes", "readonly_memoryview", "readonly_mmap"])
def test_imported_readonly_flags_cannot_escalate(tmp_path, source):
    payload = struct.pack("4f", 1, 2, 3, 4)
    if source == "bytes":
        owner = payload
    elif source == "readonly_memoryview":
        owner = memoryview(bytearray(payload)).toreadonly()
    else:
        path = tmp_path / "readonly.raw"
        path.write_bytes(payload)
        with open(path, "rb") as stream:
            owner = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    array = q.frombuffer(owner, dtype=q.float32)
    for view in (array, array[::2], array.reshape(2, 2).T, array.real):
        assert not view.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            view.flags.writeable = True
        assert not view.flags.writeable
    np.testing.assert_array_equal(np.asarray(array), [1, 2, 3, 4])


def test_imported_buffer_owner_cannot_release_live_storage():
    owner = bytearray(struct.pack("4f", 1, 2, 3, 4))
    external_view = memoryview(owner)
    array = q.frombuffer(external_view, dtype=q.float32)
    assert array.base is external_view
    with pytest.raises(BufferError):
        external_view.release()
    view = array[::2]
    del array
    gc.collect()
    with pytest.raises(BufferError):
        external_view.release()
    del view
    gc.collect()
    external_view.release()
    owner.extend(b"more")


def test_imported_writable_buffer_can_restore_its_flag():
    owner = bytearray(struct.pack("4f", 1, 2, 3, 4))
    array = q.frombuffer(owner, dtype=q.float32)
    array.flags.writeable = False
    array.flags.writeable = True
    array[0] = 7
    assert struct.unpack("4f", owner)[0] == 7
