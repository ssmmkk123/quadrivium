"""Differential tests: the C array backend against NumPy itself.

`quadrivium.numeric` replaced NumPy inside this package, so the thing worth
testing is that it agrees with NumPy -- not approximately, but on the same
inputs producing the same shapes, dtypes and values.  NumPy is a test-only
dependency for exactly this reason; the package itself no longer imports it.

Where a result is a floating-point computation the comparison is exact where it
can be (sums use NumPy's pairwise scheme, the generators reproduce NumPy's
streams bit for bit) and tolerance-based where an algorithm legitimately
differs, such as a factorisation reaching the same subspace by another route.
"""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")

from quadrivium import numeric as qp


def to_numpy(value):
    """Bring a `quadrivium.numeric` result into NumPy for comparison."""
    if isinstance(value, qp.ndarray):
        return np.asarray(memoryview(value))
    if isinstance(value, (list, tuple)):
        return type(value)(to_numpy(v) for v in value)
    return value


def assert_same(mine, theirs, *, exact=True, tol=1e-12, name=""):
    if isinstance(theirs, tuple):
        assert len(mine) == len(theirs), f"{name}: tuple length"
        for a, b in zip(mine, theirs):
            assert_same(a, b, exact=exact, tol=tol, name=name)
        return
    got = to_numpy(mine)
    if isinstance(theirs, np.ndarray):
        assert got.shape == theirs.shape, f"{name}: shape {got.shape} != {theirs.shape}"
        assert got.dtype == theirs.dtype, f"{name}: dtype {got.dtype} != {theirs.dtype}"
        if exact:
            assert np.array_equal(got, theirs, equal_nan=got.dtype.kind == "f"), \
                f"{name}: values differ\n{got}\n{theirs}"
        else:
            assert np.allclose(got, theirs, rtol=tol, atol=tol, equal_nan=True), \
                f"{name}: values differ\n{got}\n{theirs}"
        return
    if isinstance(theirs, (bool, np.bool_)):
        assert bool(got) == bool(theirs), f"{name}: {got} != {theirs}"
        return
    if isinstance(theirs, (int, np.integer)):
        assert int(got) == int(theirs), f"{name}: {got} != {theirs}"
        return
    if isinstance(theirs, (float, np.floating, complex, np.complexfloating)):
        if exact:
            assert got == theirs or (got != got and theirs != theirs), \
                f"{name}: {got!r} != {theirs!r}"
        else:
            assert abs(got - theirs) <= tol * builtins_max(1.0, abs(theirs)), \
                f"{name}: {got!r} != {theirs!r}"
        return
    assert got == theirs, f"{name}: {got!r} != {theirs!r}"


def builtins_max(a, b):
    return a if a > b else b


@pytest.fixture(scope="module")
def data():
    """A fixed set of operands, in both libraries."""
    rng = np.random.default_rng(7)
    raw = {
        "v": rng.normal(size=9),
        "w": rng.normal(size=9),
        "m": rng.normal(size=(5, 4)),
        "n": rng.normal(size=(4, 6)),
        "sq": rng.normal(size=(6, 6)),
        "spd": None,
        "i": np.arange(-4, 5),
        "b": np.array([True, False, True, True, False, False, True, False, True]),
        "c": rng.normal(size=9) + 1j * rng.normal(size=9),
        "cm": rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)),
        "t": rng.normal(size=(3, 4, 5)),
    }
    a = raw["sq"]
    raw["spd"] = a @ a.T + 6 * np.eye(6)
    mine = {k: qp.array(v.tolist()) if v.dtype != bool else qp.array(v.tolist(), dtype=bool)
            for k, v in raw.items()}
    for k, v in raw.items():
        if v.dtype.kind == "i":
            mine[k] = qp.array(v.tolist(), dtype=int)
    return raw, mine


#: Functions whose result is exactly reproducible: correctly rounded in IEEE
#: terms, or pure data movement.
EXACT_UNARY = ["sqrt", "sign", "floor", "ceil", "trunc", "square", "conj",
               "real", "imag", "isfinite", "isnan", "isinf", "absolute",
               "negative"]
#: Transcendentals, where NumPy's vectorised kernels and libm may differ in the
#: last place.  The library's own tolerances are far wider than this.
APPROX_UNARY = ["exp", "log", "log2", "log10", "log1p", "expm1", "sin", "cos",
                "tan", "arctan", "sinh", "cosh", "tanh", "arcsinh", "angle",
                "reciprocal"]
UNARY = EXACT_UNARY + APPROX_UNARY


@pytest.mark.parametrize("fn", UNARY)
@pytest.mark.parametrize("key", ["v", "m", "i", "c"])
def test_unary_ufuncs(data, fn, key):
    raw, mine = data
    if fn == "reciprocal" and key == "i":
        # NumPy's integer reciprocal is integer division, which is a trap
        # rather than a feature; this backend promotes to float instead.
        pytest.skip("deliberate difference from NumPy's integer reciprocal")
    with np.errstate(all="ignore"):
        try:
            expected = getattr(np, fn)(raw[key])
        except TypeError:
            pytest.skip(f"{fn} rejects this dtype in NumPy too")
    got = getattr(qp, fn)(mine[key])
    exact = fn in EXACT_UNARY and not (key == "c" and fn in ("sqrt", "absolute", "square"))
    assert_same(got, expected, exact=exact, tol=1e-13, name=f"{fn}({key})")


BINARY = ["add", "subtract", "multiply", "divide", "power", "maximum",
          "minimum", "hypot", "arctan2", "copysign", "equal", "not_equal",
          "less", "less_equal", "greater", "greater_equal", "logical_and",
          "logical_or", "floor_divide", "remainder"]


APPROX_BINARY = {"power", "hypot", "arctan2"}


@pytest.mark.parametrize("fn", BINARY)
def test_binary_ufuncs(data, fn):
    raw, mine = data
    with np.errstate(all="ignore"):
        expected = getattr(np, fn)(raw["v"], raw["w"])
    got = getattr(qp, fn)(mine["v"], mine["w"])
    assert_same(got, expected, exact=fn not in APPROX_BINARY, tol=1e-13, name=fn)


def test_binary_broadcasting(data):
    raw, mine = data
    cases = [
        (raw["m"], raw["m"][0]),
        (raw["m"], raw["m"][:, :1]),
        (raw["t"], raw["t"][0]),
        (raw["v"], 2.0),
        (3, raw["i"]),
        (raw["m"][:, None, :], raw["m"][None, :, :]),
    ]
    qm = {id(raw[k]): mine[k] for k in mine}
    for left, right in cases:
        lq = qp.array(left.tolist()) if isinstance(left, np.ndarray) else left
        rq = qp.array(right.tolist()) if isinstance(right, np.ndarray) else right
        assert_same(lq + rq, left + right, name="broadcast add")
        assert_same(lq * rq, left * right, name="broadcast mul")
    assert qm is not None


def test_operators_and_inplace(data):
    raw, mine = data
    a, qa = raw["m"].copy(), mine["m"].copy()
    a += 1.5
    qa += 1.5
    assert_same(qa, a, name="+=")
    a *= raw["m"]
    qa *= mine["m"]
    assert_same(qa, a, name="*=")
    a -= 0.25
    qa -= 0.25
    assert_same(qa, a, name="-=")
    a /= 3.0
    qa /= 3.0
    assert_same(qa, a, name="/=")
    assert_same(-qa, -a, name="neg")
    assert_same(abs(qa), abs(a), name="abs")
    assert_same(qa ** 2, a ** 2, name="**2")
    assert_same(qa ** 0.5, a ** 0.5, exact=False, name="**0.5")


REDUCTIONS = ["sum", "prod", "mean", "std", "var", "max", "min", "argmax",
              "argmin", "any", "all", "cumsum", "count_nonzero"]


@pytest.mark.parametrize("fn", REDUCTIONS)
@pytest.mark.parametrize("key", ["v", "m", "t", "i", "b"])
def test_reductions(data, fn, key):
    raw, mine = data
    expected = getattr(np, fn)(raw[key])
    got = getattr(qp, fn)(mine[key])
    assert_same(got, expected, name=f"{fn}({key})")


@pytest.mark.parametrize("fn", ["sum", "mean", "max", "min", "argmax", "argmin",
                                "any", "all", "std", "var", "cumsum", "prod"])
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_reductions_with_axis(data, fn, axis):
    raw, mine = data
    expected = getattr(np, fn)(raw["m"], axis=axis)
    got = getattr(qp, fn)(mine["m"], axis=axis)
    assert_same(got, expected, name=f"{fn}(m, axis={axis})")


def test_reduction_over_three_dimensions(data):
    raw, mine = data
    for axis in (0, 1, 2, -1):
        assert_same(qp.sum(mine["t"], axis=axis), np.sum(raw["t"], axis=axis),
                    name=f"sum(t, axis={axis})")
        assert_same(qp.mean(mine["t"], axis=axis), np.mean(raw["t"], axis=axis),
                    name=f"mean(t, axis={axis})")
    # Reducing several axes at once repacks first, so the summation order --
    # and with it the last bit -- can differ from NumPy's nested reduction.
    assert_same(qp.sum(mine["t"], axis=(0, 2)), np.sum(raw["t"], axis=(0, 2)),
                exact=False, tol=1e-14, name="sum(t, axis=(0,2))")


def test_sum_matches_numpy_pairwise_exactly():
    """Long float sums must agree bit for bit, not just to a tolerance."""
    rng = np.random.default_rng(11)
    for n in (7, 8, 129, 1000, 4096, 10007):
        values = rng.normal(size=n) * 1e6
        assert float(qp.sum(qp.array(values.tolist()))) == float(np.sum(values))


INDEXING = [
    "a[0]", "a[-1]", "a[1:4]", "a[::2]", "a[::-1]", "a[1:4:2]",
    "a[0, 1]", "a[:, 1]", "a[1, :]", "a[:, ::2]", "a[..., 0]", "a[None]",
    "a[:, None]", "a[None, :, 0]", "a[a > 0]", "a[[0, 2, 1]]",
    "a[[0, 2], [1, 3]]", "a[:, [0, 2]]", "a[[0, 2], :]", "a[1:, 1:]",
    "a[a[:, 0] > 0]", "a[:, a[0] > 0]",
]


@pytest.mark.parametrize("expr", INDEXING)
def test_indexing(data, expr):
    raw, mine = data
    a = raw["m"]
    expected = eval(expr)
    a = mine["m"]
    got = eval(expr)
    assert_same(got, expected, name=expr)


ASSIGNMENT = [
    ("a[0] = 5.0", None), ("a[1:3] = 0.0", None), ("a[:, 1] = 7.0", None),
    ("a[a > 0] = -1.0", None), ("a[[0, 2]] = 2.0", None),
    ("a[[0, 2], [1, 3]] = 9.0", None), ("a[:, ::2] = 3.0", None),
    ("a[0, :] = b", "row"), ("a[:, 0] = c", "col"),
]


@pytest.mark.parametrize("expr,extra", ASSIGNMENT)
def test_assignment(data, expr, extra):
    raw, mine = data
    b = np.arange(4, dtype=float)
    c = np.arange(5, dtype=float)
    a = raw["m"].copy()
    exec(expr)
    expected = a
    b = qp.arange(4, dtype=float)
    c = qp.arange(5, dtype=float)
    a = mine["m"].copy()
    exec(expr)
    assert_same(a, expected, name=expr)


def test_add_at(data):
    raw, mine = data
    target = np.zeros(6)
    np.add.at(target, np.array([0, 2, 2, 5]), np.array([1.0, 2.0, 3.0, 4.0]))
    got = qp.zeros(6)
    qp.add.at(got, qp.array([0, 2, 2, 5], dtype=int), qp.array([1.0, 2.0, 3.0, 4.0]))
    assert_same(got, target, name="add.at")


SHAPE_OPS = [
    ("reshape", lambda lib, m: lib.reshape(m, (4, 5))),
    ("transpose", lambda lib, m: lib.transpose(m)),
    ("ravel", lambda lib, m: m.ravel()),
    ("concatenate0", lambda lib, m: lib.concatenate([m, m], axis=0)),
    ("concatenate1", lambda lib, m: lib.concatenate([m, m], axis=1)),
    ("hstack", lambda lib, m: lib.hstack([m, m])),
    ("vstack", lambda lib, m: lib.vstack([m, m])),
    ("column_stack", lambda lib, m: lib.column_stack([m[:, 0], m[:, 1]])),
    ("stack", lambda lib, m: lib.stack([m, m], axis=1)),
    ("roll", lambda lib, m: lib.roll(m, 2)),
    ("roll_axis", lambda lib, m: lib.roll(m, 1, axis=1)),
    ("repeat", lambda lib, m: lib.repeat(m, 2, axis=0)),
    ("diff", lambda lib, m: lib.diff(m, axis=0)),
    ("diff2", lambda lib, m: lib.diff(m, 2, axis=1)),
    ("atleast_2d", lambda lib, m: lib.atleast_2d(m[0])),
    ("broadcast_to", lambda lib, m: lib.broadcast_to(m[0], (3, 4))),
    ("tile", lambda lib, m: lib.tile(m, (2, 1))),
    ("flip", lambda lib, m: lib.flip(m, 0)),
    ("triu", lambda lib, m: lib.triu(m[:4, :4])),
    ("tril", lambda lib, m: lib.tril(m[:4, :4], -1)),
    ("diag_extract", lambda lib, m: lib.diag(m[:4, :4])),
    ("diag_build", lambda lib, m: lib.diag(m[0])),
    ("outer", lambda lib, m: lib.outer(m[0], m[1])),
    ("kron", lambda lib, m: lib.kron(m[:2, :2], m[2:4, 2:4])),
    ("vander", lambda lib, m: lib.vander(m[0], 3)),
    ("append", lambda lib, m: lib.append(m[0], m[1])),
    ("delete", lambda lib, m: lib.delete(m, 1, axis=0)),
    ("pad", lambda lib, m: lib.pad(m, 1, mode="constant")),
    ("pad_edge", lambda lib, m: lib.pad(m, 2, mode="edge")),
    ("cumsum_axis", lambda lib, m: lib.cumsum(m, axis=1)),
    ("clip", lambda lib, m: lib.clip(m, -0.5, 0.5)),
    ("where", lambda lib, m: lib.where(m > 0, m, -m)),
    ("sort", lambda lib, m: lib.sort(m, axis=1)),
    ("argsort", lambda lib, m: lib.argsort(m, axis=1)),
    ("nonzero", lambda lib, m: lib.nonzero(m > 0)),
    ("flatnonzero", lambda lib, m: lib.flatnonzero(m > 0)),
    ("unique", lambda lib, m: lib.unique(lib.round(m, 1))),
    ("meshgrid", lambda lib, m: lib.meshgrid(m[0], m[1])),
    ("trace", lambda lib, m: lib.trace(m[:4, :4])),
]


@pytest.mark.parametrize("name,op", SHAPE_OPS, ids=[n for n, _ in SHAPE_OPS])
def test_shape_operations(data, name, op):
    raw, mine = data
    expected = op(np, raw["m"])
    got = op(qp, mine["m"])
    if isinstance(expected, list):
        expected = tuple(expected)
        got = tuple(got)
    assert_same(got, expected, name=name)


def test_ix_and_fancy_combinations(data):
    raw, mine = data
    rows, cols = [0, 2, 4], [1, 3]
    assert_same(mine["m"][qp.ix_(rows, cols)], raw["m"][np.ix_(rows, cols)], name="ix_")
    assert_same(mine["m"][qp.array(rows, dtype=int)][:, qp.array(cols, dtype=int)],
                raw["m"][np.array(rows)][:, np.array(cols)], name="two-step fancy")


def test_searchsorted_and_bincount():
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    probes = np.array([-1.0, 0.5, 2.0, 9.0])
    assert_same(qp.searchsorted(qp.array(values.tolist()), qp.array(probes.tolist())),
                np.searchsorted(values, probes), name="searchsorted")
    assert_same(qp.searchsorted(qp.array(values.tolist()), qp.array(probes.tolist()),
                                side="right"),
                np.searchsorted(values, probes, side="right"), name="searchsorted right")
    counts = np.array([0, 1, 1, 3, 3, 3, 7])
    assert_same(qp.bincount(qp.array(counts.tolist(), dtype=int)),
                np.bincount(counts), name="bincount")
    weights = np.linspace(0.5, 3.5, counts.size)
    assert_same(qp.bincount(qp.array(counts.tolist(), dtype=int),
                            weights=qp.array(weights.tolist())),
                np.bincount(counts, weights=weights), name="bincount weighted")


def test_matmul_and_dot(data):
    raw, mine = data
    assert_same(mine["m"] @ mine["n"], raw["m"] @ raw["n"], exact=False, name="m@n")
    assert_same(mine["m"][0] @ mine["m"][1], float(raw["m"][0] @ raw["m"][1]),
                exact=False, name="v@v")
    assert_same(mine["m"] @ mine["n"][:, 0], raw["m"] @ raw["n"][:, 0],
                exact=False, name="m@v")
    assert_same(mine["m"][0] @ mine["n"][:4], raw["m"][0] @ raw["n"][:4],
                exact=False, name="v@m")
    assert_same(mine["cm"] @ mine["cm"], raw["cm"] @ raw["cm"], exact=False,
                name="complex matmul")


def test_matmul_larger_sizes():
    rng = np.random.default_rng(3)
    for m, k, n in [(64, 64, 64), (129, 77, 53), (200, 200, 200), (17, 300, 9)]:
        a, b = rng.normal(size=(m, k)), rng.normal(size=(k, n))
        got = qp.array(a.tolist()) @ qp.array(b.tolist())
        assert np.allclose(to_numpy(got), a @ b, rtol=1e-12, atol=1e-12)


def test_linalg_solve_and_inverse(data):
    raw, mine = data
    rhs = raw["sq"][:, 0]
    assert_same(qp.linalg.solve(mine["sq"], mine["sq"][:, 0]),
                np.linalg.solve(raw["sq"], rhs), exact=False, name="solve")
    assert_same(qp.linalg.solve(mine["sq"], mine["sq"][:, :2]),
                np.linalg.solve(raw["sq"], raw["sq"][:, :2]), exact=False, name="solve 2 rhs")
    assert_same(qp.linalg.inv(mine["sq"]), np.linalg.inv(raw["sq"]),
                exact=False, tol=1e-10, name="inv")
    assert_same(qp.linalg.det(mine["sq"]), float(np.linalg.det(raw["sq"])),
                exact=False, tol=1e-10, name="det")
    sign, logdet = np.linalg.slogdet(raw["spd"])
    got_sign, got_log = qp.linalg.slogdet(mine["spd"])
    assert got_sign == sign
    assert abs(got_log - logdet) < 1e-10


def test_linalg_cholesky_and_eigh(data):
    raw, mine = data
    assert_same(qp.linalg.cholesky(mine["spd"]), np.linalg.cholesky(raw["spd"]),
                exact=False, tol=1e-10, name="cholesky")
    w_mine, v_mine = qp.linalg.eigh(mine["spd"])
    w_theirs, v_theirs = np.linalg.eigh(raw["spd"])
    assert_same(w_mine, w_theirs, exact=False, tol=1e-9, name="eigh values")
    # Eigenvectors are only defined up to sign, so compare the reconstruction.
    reconstructed = to_numpy(v_mine) @ np.diag(to_numpy(w_mine)) @ to_numpy(v_mine).T
    assert np.allclose(reconstructed, raw["spd"], atol=1e-9)
    assert_same(qp.linalg.eigvalsh(mine["spd"]), np.linalg.eigvalsh(raw["spd"]),
                exact=False, tol=1e-9, name="eigvalsh")


def test_linalg_eigvals_and_eig(data):
    raw, mine = data
    got = np.sort_complex(np.asarray(to_numpy(qp.linalg.eigvals(mine["sq"])),
                                     dtype=complex))
    expected = np.sort_complex(np.linalg.eigvals(raw["sq"]).astype(complex))
    assert np.allclose(got, expected, atol=1e-8)
    values, vectors = qp.linalg.eig(mine["sq"])
    A = raw["sq"]
    V = np.asarray(to_numpy(vectors), dtype=complex)
    w = np.asarray(to_numpy(values), dtype=complex)
    assert np.allclose(A @ V, V * w, atol=1e-7)


def test_linalg_svd_and_lstsq(data):
    raw, mine = data
    for full in (True, False):
        U, s, Vt = qp.linalg.svd(mine["m"], full_matrices=full)
        eU, es, eVt = np.linalg.svd(raw["m"], full_matrices=full)
        assert_same(s, es, exact=False, tol=1e-10, name="svd singular values")
        Un, Vtn = to_numpy(U), to_numpy(Vt)
        assert Un.shape == eU.shape and Vtn.shape == eVt.shape
        k = es.size
        assert np.allclose(Un[:, :k] * to_numpy(s) @ Vtn[:k], raw["m"], atol=1e-10)
        assert np.allclose(Un.T @ Un, np.eye(Un.shape[1]), atol=1e-10)
    assert_same(qp.linalg.svd(mine["m"], compute_uv=False),
                np.linalg.svd(raw["m"], compute_uv=False), exact=False, tol=1e-10,
                name="svd values only")
    x, res, rank, sv = qp.linalg.lstsq(mine["m"], mine["m"][:, 0], rcond=None)
    ex, eres, erank, esv = np.linalg.lstsq(raw["m"], raw["m"][:, 0], rcond=None)
    assert_same(x, ex, exact=False, tol=1e-9, name="lstsq x")
    assert rank == erank
    assert_same(sv, esv, exact=False, tol=1e-10, name="lstsq singular values")
    assert_same(res, eres, exact=False, tol=1e-8, name="lstsq residual")


def test_linalg_pinv_and_rank(data):
    raw, mine = data
    assert_same(qp.linalg.pinv(mine["m"]), np.linalg.pinv(raw["m"]),
                exact=False, tol=1e-9, name="pinv")
    assert qp.linalg.matrix_rank(mine["m"]) == np.linalg.matrix_rank(raw["m"])


NORMS = [None, 1, 2, np.inf, -np.inf]


@pytest.mark.parametrize("ord_value", NORMS)
def test_vector_norms(data, ord_value):
    raw, mine = data
    assert_same(qp.linalg.norm(mine["v"], ord_value),
                float(np.linalg.norm(raw["v"], ord_value)), exact=False, name="vector norm")


@pytest.mark.parametrize("ord_value", [None, 1, 2, np.inf, "fro", "nuc"])
def test_matrix_norms(data, ord_value):
    raw, mine = data
    assert_same(qp.linalg.norm(mine["m"], ord_value),
                float(np.linalg.norm(raw["m"], ord_value)), exact=False, tol=1e-10,
                name="matrix norm")


@pytest.mark.parametrize("axis", [0, 1])
def test_norm_with_axis(data, axis):
    raw, mine = data
    assert_same(qp.linalg.norm(mine["m"], axis=axis),
                np.linalg.norm(raw["m"], axis=axis), exact=False, name="norm axis")


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 12, 15, 97, 128, 256, 360, 1024])
def test_fft_round_trip_and_values(n):
    rng = np.random.default_rng(n)
    values = rng.normal(size=n)
    expected = np.fft.fft(values)
    got = to_numpy(qp.fft.fft(qp.array(values.tolist())))
    assert np.allclose(got, expected, atol=1e-9 * max(1.0, n / 16))
    back = to_numpy(qp.fft.ifft(qp.fft.fft(qp.array(values.tolist()))))
    assert np.allclose(back.real, values, atol=1e-10)


def test_fft_two_dimensional_and_real():
    rng = np.random.default_rng(5)
    values = rng.normal(size=(8, 12))
    assert np.allclose(to_numpy(qp.fft.fft2(qp.array(values.tolist()))),
                       np.fft.fft2(values), atol=1e-9)
    assert np.allclose(to_numpy(qp.fft.ifft2(qp.array(values.tolist()))),
                       np.fft.ifft2(values), atol=1e-9)
    line = rng.normal(size=64)
    assert np.allclose(to_numpy(qp.fft.rfft(qp.array(line.tolist()))),
                       np.fft.rfft(line), atol=1e-10)
    assert np.allclose(to_numpy(qp.fft.irfft(qp.fft.rfft(qp.array(line.tolist())))),
                       line, atol=1e-10)
    assert_same(qp.fft.fftfreq(10), np.fft.fftfreq(10), name="fftfreq")
    assert_same(qp.fft.fftfreq(9, 0.25), np.fft.fftfreq(9, 0.25), name="fftfreq d")


def test_fft_with_explicit_length_and_axis():
    rng = np.random.default_rng(6)
    values = rng.normal(size=(4, 10))
    assert np.allclose(to_numpy(qp.fft.fft(qp.array(values.tolist()), 16)),
                       np.fft.fft(values, 16), atol=1e-10)
    assert np.allclose(to_numpy(qp.fft.fft(qp.array(values.tolist()), None, 0)),
                       np.fft.fft(values, None, 0), atol=1e-10)


def test_empty_transform_axis_is_rejected_like_numpy():
    """There is no spectrum of nothing, and this layer mirrors ``numpy.fft``.

    Returning an empty array here used to hide the degenerate input instead of
    reporting it; the policy is deliberate, so it is pinned rather than left
    to whatever the transform happens to do with a zero-length axis.
    """
    empty = qp.array([])
    for name in ("fft", "ifft", "rfft"):
        with pytest.raises(ValueError, match="number of data points"):
            getattr(qp.fft, name)(empty)
        with pytest.raises(ValueError):
            getattr(np.fft, name)(np.array([]))
    # An explicit non-positive length is refused on non-empty input too.
    for bad in (0, -1):
        with pytest.raises(ValueError, match="number of data points"):
            qp.fft.fft(qp.array([1.0, 2.0]), bad)
    # A zero-length axis is only rejected when it is the transform axis.
    block = qp.zeros((0, 4))
    assert to_numpy(qp.fft.fft(block, None, 1)).shape == (0, 4)
    with pytest.raises(ValueError, match="number of data points"):
        qp.fft.fft(block, None, 0)


@pytest.mark.parametrize("seed", [0, 1, 42, 20260905])
def test_random_streams_match_numpy_exactly(seed):
    mine = qp.random.default_rng(seed)
    theirs = np.random.default_rng(seed)
    assert list(to_numpy(mine.random(50))) == list(theirs.random(50))
    assert list(to_numpy(mine.standard_normal(50))) == list(theirs.standard_normal(50))
    assert list(to_numpy(mine.uniform(-2.0, 3.0, 50))) == list(theirs.uniform(-2.0, 3.0, 50))
    assert list(to_numpy(mine.normal(1.5, 0.25, 50))) == list(theirs.normal(1.5, 0.25, 50))
    assert list(to_numpy(mine.standard_exponential(50))) == \
        list(theirs.standard_exponential(50))
    assert list(to_numpy(mine.integers(0, 1000, 50))) == list(theirs.integers(0, 1000, 50))
    assert list(to_numpy(mine.permutation(20))) == list(theirs.permutation(20))


def test_random_shapes_and_scalars():
    mine = qp.random.default_rng(3)
    theirs = np.random.default_rng(3)
    assert to_numpy(mine.random((3, 4))).shape == theirs.random((3, 4)).shape
    assert isinstance(mine.random(), float)
    assert isinstance(qp.random.default_rng(1).integers(0, 5), int)
    assert to_numpy(mine.choice(10, size=5)).shape == (5,)


def test_statistics_helpers(data):
    raw, mine = data
    assert_same(qp.median(mine["v"]), float(np.median(raw["v"])), exact=False, name="median")
    assert_same(qp.quantile(mine["v"], 0.25), float(np.quantile(raw["v"], 0.25)),
                exact=False, name="quantile")
    assert_same(qp.cov(mine["m"]), np.cov(raw["m"]), exact=False, tol=1e-12, name="cov")
    counts, edges = qp.histogram(mine["v"], bins=5)
    ecounts, eedges = np.histogram(raw["v"], bins=5)
    assert_same(counts, ecounts, name="histogram counts")
    assert_same(edges, eedges, exact=False, name="histogram edges")
    assert_same(qp.gradient(mine["v"]), np.gradient(raw["v"]), exact=False, name="gradient")
    assert_same(qp.interp(qp.array([0.5, 1.5]), qp.array([0.0, 1.0, 2.0]),
                          qp.array([0.0, 10.0, 20.0])),
                np.interp(np.array([0.5, 1.5]), np.array([0.0, 1.0, 2.0]),
                          np.array([0.0, 10.0, 20.0])), name="interp")


def test_polynomial_helpers():
    coeffs = np.array([2.0, -3.0, 0.5, 1.0])
    x = np.linspace(-2, 2, 7)
    assert_same(qp.polyval(qp.array(coeffs.tolist()), qp.array(x.tolist())),
                np.polyval(coeffs, x), exact=False, name="polyval")
    assert_same(qp.polymul(qp.array([1.0, 2.0]), qp.array([1.0, -1.0, 3.0])),
                np.polymul(np.array([1.0, 2.0]), np.array([1.0, -1.0, 3.0])),
                exact=False, name="polymul")
    q, r = qp.polydiv(qp.array([1.0, -3.0, 2.0]), qp.array([1.0, -1.0]))
    eq, er = np.polydiv(np.array([1.0, -3.0, 2.0]), np.array([1.0, -1.0]))
    assert_same(q, eq, exact=False, name="polydiv q")
    assert_same(qp.polyint(qp.array([3.0, 2.0])), np.polyint(np.array([3.0, 2.0])),
                exact=False, name="polyint")
    roots = np.sort(np.real(np.asarray(to_numpy(qp.roots(qp.array([1.0, -3.0, 2.0]))))))
    assert np.allclose(roots, [1.0, 2.0], atol=1e-10)
    assert_same(qp.poly(qp.array([1.0, 2.0])), np.poly(np.array([1.0, 2.0])),
                exact=False, name="poly")
    cheb = np.array([1.0, 2.0, 3.0])
    assert_same(qp.polynomial.chebyshev.chebval(qp.array(x.tolist()),
                                                qp.array(cheb.tolist())),
                np.polynomial.chebyshev.chebval(x, cheb), exact=False, name="chebval")


def test_dtype_promotion_rules():
    cases = [
        (qp.array([1, 2], dtype=int), np.array([1, 2]), "int"),
        (qp.array([True, False]), np.array([True, False]), "bool"),
    ]
    for mine, theirs, name in cases:
        assert_same(mine + 1, theirs + 1, name=f"{name} + int")
        assert_same(mine + 1.5, theirs + 1.5, name=f"{name} + float")
        assert_same(mine * 2, theirs * 2, name=f"{name} * int")
        assert_same(mine / 2, theirs / 2, name=f"{name} / int")
    ints = qp.array([1, 2, 3], dtype=int)
    floats = qp.array([1.0, 2.0, 3.0])
    assert (ints + floats).dtype == qp.float64
    assert (ints + ints).dtype == qp.int64
    assert (ints / ints).dtype == qp.float64
    assert (floats > 1).dtype == qp.bool_
    assert (floats + 1j).dtype == qp.complex128


def test_views_share_memory(data):
    raw, mine = data
    a = mine["m"].copy()
    b = raw["m"].copy()
    a[1:3, 1:3] += 10.0
    b[1:3, 1:3] += 10.0
    assert_same(a, b, name="view write-through")
    row = a[0]
    row[0] = 99.0
    assert float(a[0, 0]) == 99.0
    assert a.T.base is a or a.T.base is not None


def test_buffer_protocol_round_trip(data):
    raw, mine = data
    view = np.asarray(memoryview(mine["m"]))
    assert np.array_equal(view, raw["m"])
    back = qp.asarray(raw["m"])
    assert_same(back, raw["m"], name="from numpy buffer")


def test_predicates_and_conversions(data):
    raw, mine = data
    assert qp.allclose(mine["v"], mine["v"]) == np.allclose(raw["v"], raw["v"])
    assert not qp.allclose(mine["v"], mine["w"])
    assert_same(qp.isclose(mine["v"], mine["v"]), np.isclose(raw["v"], raw["v"]),
                name="isclose")
    assert qp.array_equal(mine["v"], mine["v"])
    assert qp.iscomplexobj(mine["c"]) == np.iscomplexobj(raw["c"])
    assert qp.ndim(mine["m"]) == np.ndim(raw["m"])
    assert qp.shape(mine["m"]) == np.shape(raw["m"])
    assert mine["m"].tolist() == raw["m"].tolist()
    assert float(mine["v"][0]) == float(raw["v"][0])


def test_repr_matches_numpy(data):
    raw, mine = data
    for key in ("v", "m", "i", "b", "c"):
        assert repr(mine[key]) == repr(raw[key]), key
        assert str(mine[key]) == str(raw[key]), key


def _peak_rss_kib():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


@pytest.mark.skipif(not hasattr(__import__("resource", fromlist=["getrusage"]), "getrusage"),
                    reason="peak RSS is only available on Unix")
def test_temporaries_do_not_accumulate():
    """Churning large temporaries must not grow the process.

    Array data is freed as soon as the array dies, with nothing retained
    between operations. A backend that held onto buffers -- a recycling cache,
    say -- would show up here as steady growth.
    """
    size = 200_000
    a = qp.ones(size)
    b = qp.arange(size, dtype=qp.int64)
    for _ in range(20):                      # settle the allocator first
        _ = a * a[b]
    before = _peak_rss_kib()
    for _ in range(400):
        _ = a * a[b]
        _ = qp.exp(a) + qp.sqrt(a)
        _ = a[b][b]
    growth_mib = (_peak_rss_kib() - before) / 1024
    assert growth_mib < 8, f"peak RSS grew by {growth_mib:.1f} MiB over 400 iterations"


def test_freed_arrays_release_their_memory():
    """A large array's buffer goes back to the allocator when it dies."""
    before = _peak_rss_kib()
    for _ in range(50):
        block = qp.zeros(2_000_000)          # 16 MiB each
        block[0] = 1.0
        del block
    growth_mib = (_peak_rss_kib() - before) / 1024
    assert growth_mib < 32, f"peak RSS grew by {growth_mib:.1f} MiB over 50 allocations"


def test_errors_are_raised_like_numpy():
    with pytest.raises(ValueError):
        qp.array([1.0, 2.0]) + qp.array([1.0, 2.0, 3.0])
    with pytest.raises(IndexError):
        qp.zeros(3)[5]
    with pytest.raises(qp.linalg.LinAlgError):
        qp.linalg.solve(qp.zeros((2, 2)), qp.ones(2))
    with pytest.raises(qp.linalg.LinAlgError):
        qp.linalg.cholesky(-qp.eye(2))
    with pytest.raises(ValueError):
        bool(qp.array([True, False]))
    assert math.isnan(float(qp.array([float("nan")])[0]))
