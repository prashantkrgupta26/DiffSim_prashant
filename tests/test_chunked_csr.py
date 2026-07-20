"""Task #38 ChunkedCSR gates: block-row chunked nnz-space storage past
warp's 2^31-ELEMENT array ceiling (types.py check_array_shape rejects any
array dimension >= 2^31 — the wall the #34 GH200 probe measured at
265x265x75 / 2.29B nnz with HBM half empty).

Gates covered here:
  G1 (invariance): chunking FORCED at small size (multi-chunk via a tiny
     chunk_cap) matches the unchunked path bit-for-bit on CPU / few-ULP
     on CUDA (_assert_scatter_equal contract) across the node-graph,
     masked and COO scatter paths, strong rows, diag, operator SpMV and
     the film blockch_dev solve; the knob resolution keeps chunking OFF
     below ~90% of 2^31 (no new kernels, no new arrays).
  G2 (the synthetic): the chunked scatter kernels driven through a chunk
     table whose base offset sits PAST 2^31 — the int64 slot arithmetic
     and the chunk-located write are exercised with a tiny values buffer
     (no 17 GB allocation).
"""
import numpy as np
import pytest
import warp as wp

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import (DeviceMesh, CSROperator,
                                        make_csr_spmv_chunked)
from diffsim.assembly.device_assembly import (
    DeviceNSAssembler, ChunkTable, ChunkedArray, _resolve_chunking,
    _scatter_kernel_chunked, _scatter_node_kernel_chunked,
    CHUNK_NNZ_THRESHOLD, CHUNK_CAP_DEFAULT)
from diffsim.physics.poisson import gauss_points
from diffsim.errors import BackendError, ConfigError

pytestmark = pytest.mark.tier2


def _setup(dim, level, device, ndof_fields=True):
    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), device)
    if not ndof_fields:
        return dm, None, None, None
    xq = gauss_points(mesh, dm.tables_by_p)
    rng = np.random.default_rng(3)
    pv = list(dm.bins)[0]
    ngp = len(xq[pv])
    aq = {pv: rng.standard_normal((ngp, dim)) * 0.5}
    dq = {pv: rng.standard_normal(ngp) * 0.1}
    fq = {pv: rng.standard_normal((ngp, dim))}
    return dm, aq, dq, fq


def _assert_scatter_equal(x, y, device):
    """Chunked vs flat scatter agreement: bit-equal on CPU (serial
    deterministic scatter — the index/layout is the ONLY difference);
    few-ULP on CUDA (separately compiled kernels order their atomics
    differently — same band as the #33 wide-vs-narrow gate)."""
    if str(device).startswith("cpu"):
        assert np.array_equal(x, y), np.abs(x - y).max()
    else:
        np.testing.assert_allclose(x, y, rtol=1e-13, atol=1e-14)


# ---------------------------------------------------------------------
def test_resolve_chunking():
    """Knob + auto-selection: OFF below ~90% of 2^31, ON at/above it;
    'force' always ON; 'off' past 2^31 REFUSED loudly (warp cannot
    construct the buffers at all); bogus knob rejected."""
    assert _resolve_chunking("auto", 1000) is False
    assert _resolve_chunking("auto", None) is False
    assert _resolve_chunking("auto", CHUNK_NNZ_THRESHOLD) is True
    assert _resolve_chunking("auto", 2 ** 31) is True
    assert _resolve_chunking("force", 1) is True
    assert _resolve_chunking("off", 1000) is False
    with pytest.raises(BackendError):
        _resolve_chunking("off", 2 ** 31)
    with pytest.raises(ConfigError):
        _resolve_chunking("bogus", 1000)
    # capacity: a full f64 chunk row's byte stride stays under 2^31
    assert CHUNK_CAP_DEFAULT * 8 < 2 ** 31


def test_chunk_table_partition(device):
    """ChunkTable invariants on an irregular synthetic indptr: chunk
    boundaries sit at ROW starts (no row straddles a chunk), every chunk
    holds <= cap entries, the chunks tile [0, nnz) exactly, and a single
    row larger than cap is refused."""
    rng = np.random.default_rng(7)
    rows = rng.integers(1, 40, size=200)
    indptr = np.concatenate(([0], np.cumsum(rows))).astype(np.int64)
    nnz = int(indptr[-1])
    t = ChunkTable(indptr, cap=97, device=device)
    assert t.nnz == nnz and t.nchunks >= 2
    assert t.bases[0] == 0 and t.bases[-1] == nnz
    assert (np.diff(t.bases) > 0).all()
    assert (t.counts() <= 97).all()
    # row alignment: every base is some indptr value at its row_start
    assert np.array_equal(t.bases, indptr[t.row_start])
    assert t.row_start[0] == 0 and t.row_start[-1] == len(indptr) - 1
    assert np.array_equal(t.bases_d.numpy(), t.bases)
    with pytest.raises(BackendError):
        ChunkTable(np.array([0, 500], np.int64), cap=97, device=device)


def test_chunked_array_roundtrip(device):
    """ChunkedArray upload/numpy round-trip across several chunks (the
    padded 2-D storage is invisible at the API)."""
    rng = np.random.default_rng(11)
    rows = rng.integers(1, 20, size=64)
    indptr = np.concatenate(([0], np.cumsum(rows))).astype(np.int64)
    t = ChunkTable(indptr, cap=53, device=device)
    a = ChunkedArray(t, wp.float64, device)
    host = rng.standard_normal(t.nnz)
    a.upload(host)
    assert np.array_equal(a.numpy(), host)
    assert len(a) == t.nnz and a.size == t.nnz
    a.zero_()
    assert np.array_equal(a.numpy(), np.zeros(t.nnz))


def test_chunking_narrow_conflict(device):
    """chunking='force' with index_width='narrow' is a config conflict
    (chunked kernels carry int64 global slots) — refused loudly."""
    dm, aq, dq, fq = _setup(2, 3, device)
    with pytest.raises(ConfigError):
        DeviceNSAssembler(dm, index_width="narrow", chunking="force")


def test_chunked_off_below_threshold(device):
    """G1 invariance guard: default knobs below the ceiling build the
    IDENTICAL flat arrays (no chunk table, plain 1-D vals_d)."""
    dm, aq, dq, fq = _setup(2, 3, device)
    a = DeviceNSAssembler(dm)
    assert a._chunked is False and a._ctab is None
    assert isinstance(a.vals_d, wp.array) and a.vals_d.ndim == 1


# ---------------------------------------------------------------------
# G1 parity: chunking FORCED at small size (multi-chunk) vs unchunked.
# ---------------------------------------------------------------------
def _mk_pair(dm, cap, **kw):
    a_f = DeviceNSAssembler(dm, **kw)                       # flat
    a_c = DeviceNSAssembler(dm, chunking="force", chunk_cap=cap, **kw)
    assert a_c._chunked and a_c._idx_dtype is wp.int64
    assert a_c._ctab.nchunks >= 3, a_c._ctab.nchunks       # multi-chunk
    assert a_f.nnz == a_c.nnz
    assert np.array_equal(np.asarray(a_f.indptr, np.int64),
                          np.asarray(a_c.indptr, np.int64))
    # the chunked dof-indices build must reproduce the flat columns
    assert np.array_equal(np.asarray(a_f.indices, np.int64),
                          np.asarray(a_c.indices, np.int64))
    return a_f, a_c


def test_chunked_parity_node_pattern(device):
    """Node-graph pattern: chunked-forced (>=3 chunks) matches the flat
    path — pattern exact, scatter/rhs/add_matrix_values/strong-rows/
    diag/operator-SpMV parity."""
    dm, aq, dq, fq = _setup(3, 3, device)
    probe = DeviceNSAssembler(dm, ndof=4, node_pattern=True)
    cap = probe.nnz // 5 + 7
    a_f, a_c = _mk_pair(dm, cap, ndof=4, node_pattern=True)
    pv, b, ne, nbf, _ = a_f._bins[0]
    nl = 4 * nbf
    rng = np.random.default_rng(11)
    Ae = wp.array(rng.standard_normal((ne, nl, nl)), dtype=wp.float64,
                  device=device)
    be = wp.array(rng.standard_normal((ne, nl)), dtype=wp.float64,
                  device=device)
    # extra device values at CSR slots (the film's top-flux idiom)
    diag_rows = np.arange(0, a_f.Nfull, 7, dtype=np.int64)
    slots = a_f.csr_slots(diag_rows, diag_rows)
    extra = wp.array(rng.standard_normal(len(slots)), dtype=wp.float64,
                     device=device)
    strong = np.arange(0, a_f.Nfull, 13, dtype=np.int64)
    bvals = rng.standard_normal(len(strong))
    for asm in (a_f, a_c):
        asm.set_strong_rows(strong)
        asm.zero_fill()
        asm.scatter_bin(0, Ae, be)
        asm.add_matrix_values(
            wp.array(slots.astype(asm._idx_np), dtype=asm._idx_dtype,
                     device=device), extra)
        asm.apply_strong_rows(bvals)
    _assert_scatter_equal(a_f.vals_d.numpy(), a_c.vals_d.numpy(), device)
    _assert_scatter_equal(a_f.F_d.numpy(), a_c.F_d.numpy(), device)
    _assert_scatter_equal(a_f.diag_host(), a_c.diag_host(), device)
    x = rng.standard_normal(a_f.Nfull)
    x_d = wp.array(x, dtype=wp.float64, device=device)
    ys = []
    for asm in (a_f, a_c):
        y_d = wp.zeros(asm.Nfull, dtype=wp.float64, device=device)
        asm.device_operator().matvec(x_d, y_d)
        ys.append(y_d.numpy())
    # SpMV is a deterministic per-row reduction in both kernels ->
    # same order, same values (the operands were gated few-ULP above)
    np.testing.assert_allclose(ys[0], ys[1], rtol=1e-13, atol=1e-14)


def test_chunked_parity_masked(device):
    """Block-masked node pattern (kron(G, mask)): chunked dof-indices +
    chunked masked scatter match the flat path."""
    dm, aq, dq, fq = _setup(3, 3, device)
    bm = np.array([[1, 1, 0, 0], [1, 1, 0, 0],
                   [0, 0, 1, 1], [0, 0, 1, 1]], bool)
    probe = DeviceNSAssembler(dm, ndof=4, node_pattern=True, blockmask=bm)
    cap = probe.nnz // 4 + 3
    a_f, a_c = _mk_pair(dm, cap, ndof=4, node_pattern=True, blockmask=bm)
    pv, b, ne, nbf, _ = a_f._bins[0]
    nl = 4 * nbf
    rng = np.random.default_rng(2)
    Ae = wp.array(rng.standard_normal((ne, nl, nl)), dtype=wp.float64,
                  device=device)
    be = wp.array(rng.standard_normal((ne, nl)), dtype=wp.float64,
                  device=device)
    for asm in (a_f, a_c):
        asm.zero_fill()
        asm.scatter_bin(0, Ae, be)
    _assert_scatter_equal(a_f.vals_d.numpy(), a_c.vals_d.numpy(), device)
    _assert_scatter_equal(a_f.F_d.numpy(), a_c.F_d.numpy(), device)


def test_chunked_parity_coo_assemble(device):
    """COO (slot-map) pattern: a full assemble() in chunked-forced mode
    matches the flat assembly (structure exact, values per contract)."""
    dm, aq, dq, fq = _setup(2, 5, device)
    nu, sigma = 0.05, 20.0
    probe = DeviceNSAssembler(dm)
    cap = probe.nnz // 4 + 5
    a_f = DeviceNSAssembler(dm)
    a_c = DeviceNSAssembler(dm, chunking="force", chunk_cap=cap)
    assert a_c._chunked and a_c._ctab.nchunks >= 3
    Af, bf = a_f.assemble(aq, dq, fq, nu, sigma)
    Ac, bc = a_c.assemble(aq, dq, fq, nu, sigma)
    assert np.array_equal(Af.indptr, Ac.indptr)
    assert np.array_equal(Af.indices, Ac.indices)
    _assert_scatter_equal(Af.data, Ac.data, device)
    _assert_scatter_equal(bf, bc, device)


def test_chunked_coo_unsupported_modes(device):
    """Chunking scope: colored / constraint-aware COO scatters are
    refused loudly (identity-T non-colored only)."""
    dm, aq, dq, fq = _setup(2, 3, device)
    with pytest.raises(BackendError):
        DeviceNSAssembler(dm, coloring=True, chunking="force",
                          node_pattern=False)


def test_chunked_spmv_operator(device):
    """CSROperator.from_device_arrays over chunked (indices, data)
    matches scipy on a random matrix split across >=4 chunks."""
    import scipy.sparse as sp
    rng = np.random.default_rng(1)
    n = 400
    A = (sp.random(n, n, density=0.03, random_state=1, format="csr")
         + sp.identity(n, format="csr")).tocsr()
    t = ChunkTable(A.indptr, cap=int(A.nnz) // 5 + 3, device=device)
    assert t.nchunks >= 4
    ind_ch = ChunkedArray(t, wp.int32, device)
    ind_ch.upload(A.indices.astype(np.int32))
    val_ch = ChunkedArray(t, wp.float64, device)
    val_ch.upload(A.data.astype(np.float64))
    ptr_d = wp.array(A.indptr.astype(np.int64), dtype=wp.int64,
                     device=device)
    op = CSROperator.from_device_arrays(ptr_d, ind_ch, val_ch, n, device)
    x = rng.standard_normal(n)
    assert np.abs(op.matvec_numpy(x) - A @ x).max() < 1e-12


def test_chunked_device_csr_sync(device):
    """cuDSS-path contract (#38): device_csr() in chunked mode
    concatenates the chunk rows into ONE contiguous torch values tensor
    (torch sizes are int64 — no ceiling); the tensor is a COPY, so
    after an in-place refill it is STALE until sync_csr_values() — the
    documented zero-copy loss on the chunked cuDSS path."""
    torch = pytest.importorskip("torch")
    dm, aq, dq, fq = _setup(3, 3, device)
    probe = DeviceNSAssembler(dm, ndof=4, node_pattern=True)
    cap = probe.nnz // 4 + 1
    a_c = DeviceNSAssembler(dm, ndof=4, node_pattern=True,
                            chunking="force", chunk_cap=cap)
    assert a_c._ctab.nchunks >= 3
    pv, b, ne, nbf, _ = a_c._bins[0]
    nl = 4 * nbf
    rng = np.random.default_rng(8)
    be = wp.array(rng.standard_normal((ne, nl)), dtype=wp.float64,
                  device=device)
    Ae1 = wp.array(rng.standard_normal((ne, nl, nl)), dtype=wp.float64,
                   device=device)
    a_c.zero_fill()
    a_c.scatter_bin(0, Ae1, be)
    A_t, F_t = a_c.device_csr()
    v1 = a_c.vals_d.numpy()
    assert np.array_equal(A_t.values().cpu().numpy(), v1)
    assert np.array_equal(A_t.col_indices().cpu().numpy(), a_c.indices)
    # refill in place -> the cached tensor is stale until the sync
    Ae2 = wp.array(rng.standard_normal((ne, nl, nl)), dtype=wp.float64,
                   device=device)
    a_c.zero_fill()
    a_c.scatter_bin(0, Ae2, be)
    v2 = a_c.vals_d.numpy()
    assert not np.array_equal(A_t.values().cpu().numpy(), v2)
    a_c.sync_csr_values()
    assert np.array_equal(A_t.values().cpu().numpy(), v2)


def test_chunked_film_blockch_parity(device):
    """End-to-end G1: the film blockch_dev solve path (chunked pair-fill
    kernel, chunked outer SpMV, chunked column-index upload in
    blockch_pairs_device) with chunking FORCED multi-chunk matches the
    unchunked device-assembly trajectory over a short march.  Both runs
    pin node_pattern + wide so the chunk LAYOUT is the only difference
    (wide-vs-narrow is #33's gate, not this one)."""
    from diffsim.octree.build import Octree
    from diffsim.physics.wodo_film import WodoFilmStepper

    def run(force_chunked):
        tree0 = build_uniform(4, dim=2)
        keep = tree0.centers()[:, 0] < 4 / 16
        tree = Octree(tree0.keys[keep], tree0.levels[keep], dim=2,
                      periodic=tree0.periodic)
        mesh = build_mesh(tree, p=1)
        cons = build_constraints(mesh)
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2),
                                  device)
        st = WodoFilmStepper(dm, chi=(1.0, 0.3, 0.3), N=(5.0, 5.0, 1.0),
                             M=(0.225, 0.0, 0.225), kappa=(2e-4, 2e-4),
                             k_e=1.0, dt=1e-3, linsolver="blockch",
                             use_device_assembly=True)
        st._node_pattern = True
        st._index_width = "wide"
        if force_chunked:
            st._chunking = "force"
            st._chunk_cap = 2500        # multi-chunk at this toy size
        rng = np.random.default_rng(3)
        st.set_initial(
            lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)),
            lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)))
        rec = []
        st.march(h_min=0.8, phis_stop=0.05, max_steps=4,
                 callback=lambda s, K, dt, it:
                 rec.append((dt, s.x.copy())))
        if force_chunked:
            assert st._asm._chunked and st._asm._ctab.nchunks >= 3
        return rec

    rec_f = run(False)
    rec_c = run(True)
    assert len(rec_f) == len(rec_c) and len(rec_f) >= 1
    for (dt_f, x_f), (dt_c, x_c) in zip(rec_f, rec_c):
        assert dt_f == dt_c
        denom = max(np.abs(x_f).max(), 1e-30)
        assert np.abs(x_f - x_c).max() / denom < 1e-9

