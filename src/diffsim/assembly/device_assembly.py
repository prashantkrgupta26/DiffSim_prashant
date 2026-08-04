"""M1d D1-item-1: device-side CSR scatter for the NS volume assembly.

Measured motivation (profile_stages.json, 2026-07-06): the Ae/be element
blocks are ALREADY device-computed; the host cost is the .numpy() pull +
COO->CSR (2.4 s at 3-D L5, 0.54 s at 2-D L8). This module keeps values
on device: a host-once-per-epoch SLOT MAP sends each (element, local row,
local col) straight to its CSR value index; per-step numeric fill is one
scatter kernel. Two scatter variants per the M1d spec (both gated at
1e-12 vs the host assembler, both benchmarked):

  - slot-map + atomicAdd (shared slots collide across elements)
  - element COLORING: per-color launches, no atomics (bit-deterministic)

Scope: meshes WITHOUT hanging constraints (T == identity — uniform-tree
carves; both gate meshes qualify). Constraint-aware scatter (cuFEM-style
master-DOF elimination) is D1 item 3.
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import warp as wp

from ..errors import BackendError, ConfigError

from .operators import _kernel_cache, _chunk_of

# ---------------------------------------------------------------------------
# W2b: opt-in per-phase profiling.  Activated by DIFFSIM_ASM_PROFILE=1.
# Completely inert when the env var is absent or "0" — no timing objects
# are created, no wp.synchronize() calls are injected.
# ---------------------------------------------------------------------------
_ASM_PROFILE = os.environ.get("DIFFSIM_ASM_PROFILE", "0").strip() not in ("", "0")

# ---------------------------------------------------------------------------
# W2c: opt-in device-resident CSR handoff.  Activated by SADDLE_DEVICE_CSR=1.
# When set, the NS driver's device-assembly path swaps assemble() for
# assemble_handoff(), which keeps the 19 GB assembled values DEVICE-RESIDENT
# (no ChunkedArray.numpy() pull) and hands the solver a DeviceSaddleCSR that
# exposes a Warp SpMV operator + a device-gathered diagonal (73 MB pull only).
# Defaults byte-identical: absent/"0" => the classic host-CSR pull path.
# ---------------------------------------------------------------------------
_DEVICE_CSR_HANDOFF = os.environ.get(
    "SADDLE_DEVICE_CSR", "0").strip() not in ("", "0")


class DeviceSaddleCSR:
    """Lightweight device-resident handoff for the monolithic saddle solve.

    Carries the assembler whose ``vals_d`` holds the just-filled CSR values
    (still on device) plus the STATIC sparsity (indptr/indices, uploaded once
    per mesh epoch).  Exposes just enough scipy-CSR surface (``.shape``,
    ``.nnz``, ``.dtype``, ``.tocsr()``, ``.tocsc()``) that the existing driver
    and the splu fallback keep working, but the fgmres_bdiag / fused_bdiag
    branches of ``solve_linear`` recognize it and consume the device buffers
    directly — no 19 GB device->host pull, no CSROperator re-upload.

    ``.tocsr()`` / ``.tocsc()`` deliberately fall back to the full host pull
    (the classic path) so any code that genuinely needs host values still
    works; the device-fast path in solve_linear never calls them.
    """

    __slots__ = ("asm", "shape", "nnz", "dtype")

    def __init__(self, asm):
        self.asm = asm
        self.shape = (asm.Nfull, asm.Nfull)
        self.nnz = asm.nnz
        self.dtype = np.float64

    # Device-resident consumption surface (used by solve_linear fast paths).
    def device_operator(self):
        """Warp SpMV operator over the resident CSR values (no host copy)."""
        return self.asm.device_operator()

    def diagonal_device(self):
        """The current matrix diagonal as a device wp.array (device gather;
        no host copy).  Slots are precomputed once per epoch (see
        DeviceNSAssembler.diagonal_device)."""
        return self.asm.diagonal_device()

    # scipy-CSR compatibility surface (host fallbacks).
    def tocsr(self):
        """Full host CSR — the classic 19 GB pull.  Only reached by callers
        that did not take the device fast path (e.g. splu)."""
        return sp.csr_matrix((self.asm.vals_d.numpy(), self.asm.indices,
                              self.asm.indptr), shape=self.shape)

    def tocsc(self):
        return self.tocsr().tocsc()

    def diagonal(self):
        """Host diagonal via the device gather + small (73 MB at L7r9) pull."""
        return self.asm.diag_host()


def _asm_prof_sync(label, t0_ref):
    """Synchronize device then return (elapsed_ms, new_t0).  ONLY called
    when _ASM_PROFILE is True — never in the default hot path."""
    wp.synchronize()
    now = time.perf_counter()
    return (now - t0_ref) * 1e3, now


# ---------------------------------------------------------------------
# Mixed-width CSR index templating (Horizon P0-2, idx-widening).
#
# The int32 nnz ceiling: row pointers, slot arrays and the in-kernel
# slot ARITHMETIC index into nnz-space, which climbs past 2^31 at the
# campaign scales (1.7B and rising). Column indices only ever address
# DOFS (<< 2^31 at all our scales), so the fix is a MIXED-width CSR:
# int64 offsets/slots, int32 columns.  Mechanism = the kernel-factory
# templating already in use — the index dtype joins the cache key, so
# each warp module is compiled per (idx_dtype, ...) lazily.
#
# Narrow mode (idx_dtype == wp.int32) stays the DEFAULT and remains
# bit-for-bit the pre-P0-2 path: same kernels, same cache keys tail,
# same int32 arrays.  Wide mode allocates offset/slot arrays as int64
# and specializes the slot-computing kernels so no arithmetic wraps.
# ---------------------------------------------------------------------
IDX_WIDE_THRESHOLD = int(0.9 * 2 ** 31)     # >~90% of 2^31 -> wide path


# ---------------------------------------------------------------------
# Task #36: mixed-precision value storage (8j activation).  The CSR
# VALUES buffer vals_d ALWAYS accumulates in fp64 (the scatter atomic-adds
# sum across elements sharing a slot — accumulation must be fp64; "never
# accumulate fp32").  val_dtype="fp32" opts a SEPARATE round-on-store
# snapshot buffer (_vals_fp32) into existence; it is refreshed from vals_d
# per fill and fed ONLY to the cuDSS factorization.  The FP64 iterative
# refinement residual rides the fp64 device SpMV against vals_d, never the
# fp32 snapshot.  Default "fp64": no snapshot, no kernels, bit-for-bit the
# pre-#36 path.  The value dtype joins the kernel-factory cache keys the
# same way #33 added the index width.
# ---------------------------------------------------------------------
def _resolve_val_dtype(val_dtype):
    """Map the config knob to a warp float dtype for the fp32 snapshot.

    val_dtype: "fp64" (default) | "fp32".  "fp64" -> wp.float64 (no
    snapshot — the accumulation buffer IS the storage); "fp32" ->
    wp.float32 (the round-on-store snapshot's element type).  Bogus input
    is REFUSED loudly."""
    if val_dtype not in ("fp64", "fp32"):
        raise ConfigError(
            f"val_dtype must be 'fp64'|'fp32', got {val_dtype!r}")
    return wp.float64 if val_dtype == "fp64" else wp.float32


def _resolve_idx_width(index_width, nnz_estimate):
    """Map the config knob + an nnz estimate to a warp index dtype.

    index_width: "auto" (default) | "narrow" | "wide".  "auto" picks
    wide once the estimate crosses ~90% of 2^31 (the slot arithmetic
    headroom); "narrow"/"wide" force the choice.  Returns wp.int32 or
    wp.int64.  A forced-narrow request that would overflow is REFUSED
    loudly here (no silent wrap)."""
    if index_width not in ("auto", "narrow", "wide"):
        raise ConfigError(
            f"index_width must be 'auto'|'narrow'|'wide', got "
            f"{index_width!r}")
    if index_width == "wide":
        return wp.int64
    if index_width == "narrow":
        if nnz_estimate is not None and nnz_estimate >= 2 ** 31:
            raise BackendError(
                f"index_width='narrow' forced but nnz {nnz_estimate} "
                f">= 2^31: the int32 slot arithmetic would wrap — use "
                f"index_width='auto' or 'wide'")
        return wp.int32
    # auto
    if nnz_estimate is not None and nnz_estimate >= IDX_WIDE_THRESHOLD:
        return wp.int64
    return wp.int32


# ---------------------------------------------------------------------
# Task #38: block-row ChunkedCSR — past warp's 2^31-ELEMENT array
# ceiling.  The #34 probe measured that warp 1.15 rejects ANY array
# dimension >= 2^31 at construction (types.py check_array_shape; the
# array_t ABI carries int32 shapes AND int32 BYTE-strides), so the #33
# wide slot arithmetic is correct but the nnz-length buffers it indexes
# (vals_d f64, column indices i32) cannot exist past 2.15B nnz —
# measured wall: 265x265x75 film, 2.29B nnz, FAILED on a 96 GB GH200
# with HBM half empty (gh200-capacity-findings.md §2).
#
# Fix: every nnz-space buffer becomes ONE 2-D [nchunks, cap] array whose
# per-chunk byte stride stays < 2^31; chunk boundaries sit at CSR ROW
# starts (no row straddles a chunk), carried by a device-agnostic
# ChunkTable (int64 bases) that doubles as the later multi-GPU row
# decomposition (Horizon gb-large — do NOT map chunks to devices here).
# Kernels keep the int64 global-slot arithmetic (#33) and only at the
# memory access locate the chunk (binary search over bases, <= 6 steps)
# and address [c, int32(slot - bases[c])].  Chunking OFF (the default
# below ~90% of 2^31 nnz) leaves the #33 paths bit-for-bit untouched.
# ---------------------------------------------------------------------
CHUNK_NNZ_THRESHOLD = IDX_WIDE_THRESHOLD    # auto-chunk at >= 90% of 2^31
# Elements per chunk: the largest power-of-nothing round count whose f64
# byte stride keeps 10% margin under 2^31 (int64 buffers share it; int32
# buffers are even safer).  241,591,910 elements = 1.93 GB f64 rows.
CHUNK_CAP_DEFAULT = int(0.9 * 2 ** 31) // 8


def _resolve_chunking(chunking, nnz_estimate):
    """Map the chunking knob + an nnz estimate to a bool.

    chunking: "auto" (default) | "off" | "force".  "auto" activates
    block-row chunking once the estimate crosses ~90% of 2^31 (the same
    threshold as the wide index auto-switch, so wide+chunked fire
    together); "off"/"force" force the choice.  An "off" past 2^31 is
    REFUSED loudly — warp cannot construct the buffers at all."""
    if chunking not in ("auto", "off", "force"):
        raise ConfigError(
            f"chunking must be 'auto'|'off'|'force', got {chunking!r}")
    if chunking == "force":
        return True
    if chunking == "off":
        if nnz_estimate is not None and nnz_estimate >= 2 ** 31:
            raise BackendError(
                f"chunking='off' forced but nnz {nnz_estimate} >= 2^31: "
                f"warp's array_t ABI (int32 shapes/byte-strides) cannot "
                f"construct any array of >= 2^31 elements — use "
                f"chunking='auto'")
        return False
    return (nnz_estimate is not None
            and nnz_estimate >= CHUNK_NNZ_THRESHOLD)


class ChunkTable:
    """Block-row partition of a CSR's nnz space.

    Chunk c covers rows [row_start[c], row_start[c+1]) and nnz-space
    positions [bases[c], bases[c+1]); bases[c] = indptr[row_start[c]]
    (int64), so rows NEVER straddle chunks and every chunk holds at most
    `cap` entries.  The table is deliberately storage-agnostic — it is
    the row decomposition a later multi-GPU task maps to devices; today
    ChunkedArray realizes it as one 2-D device array."""

    def __init__(self, indptr, cap=None, device=None):
        cap = CHUNK_CAP_DEFAULT if cap is None else int(cap)
        if cap * 8 >= 2 ** 31:          # f64/int64 row byte-stride bound
            raise ConfigError(
                f"chunk_cap {cap} overflows warp's int32 BYTE-stride "
                f"ABI for 8-byte dtypes (cap*8 must stay < 2^31)")
        indptr = np.asarray(indptr, np.int64)
        nnz = int(indptr[-1])
        nrow = len(indptr) - 1
        bases = [0]
        row_start = [0]
        while bases[-1] < nnz:
            # last row whose START lies within cap of the current base
            r = int(np.searchsorted(indptr, bases[-1] + cap,
                                    side="right") - 1)
            r = min(r, nrow)
            if r <= row_start[-1]:
                raise BackendError(
                    f"ChunkTable: row {row_start[-1]} alone carries "
                    f"{int(indptr[row_start[-1] + 1] - bases[-1])} nnz "
                    f"> chunk capacity {cap}")
            row_start.append(r)
            bases.append(int(indptr[r]))
        if len(bases) == 1:                     # degenerate empty CSR
            row_start.append(nrow)
            bases.append(0)
        self.nnz = nnz
        self.cap = cap
        self.bases = np.asarray(bases, np.int64)
        self.row_start = np.asarray(row_start, np.int64)
        self.nchunks = len(bases) - 1
        self.device = device
        self.bases_d = wp.array(self.bases, dtype=wp.int64, device=device)

    def counts(self):
        """Per-chunk entry counts (host int64)."""
        return np.diff(self.bases)


class ChunkedArray:
    """One nnz-space device buffer in block-row chunked storage: a
    single 2-D warp array [nchunks, cap] (each dimension and each byte
    stride < 2^31 — warp's array_t ABI is respected per dimension while
    the TOTAL element count passes 2^31 freely).  Chunk c's valid span
    is [0, bases[c+1]-bases[c]); the tail of each row is zeroed padding,
    never addressed by any kernel."""

    def __init__(self, table, dtype, device):
        self.table = table
        self.dtype = dtype
        self.device = device
        self.data = wp.zeros((table.nchunks, table.cap), dtype=dtype,
                             device=device)

    @property
    def size(self):
        return self.table.nnz

    def __len__(self):
        return self.table.nnz

    def zero_(self):
        self.data.zero_()

    def numpy(self):
        """Valid spans concatenated to ONE host 1-D array (per-chunk
        pulls — the padded whole-array copy never exists on host)."""
        t = self.table
        cnt = t.counts()
        if t.nnz == 0:
            return np.empty(0, dtype=self.data.numpy().dtype)
        return np.concatenate(
            [self.data[c].numpy()[:cnt[c]] for c in range(t.nchunks)])

    def upload(self, host_1d):
        """Host 1-D nnz-length array -> per-chunk device copies (no
        padded host mirror is ever built)."""
        t = self.table
        host_1d = np.ascontiguousarray(host_1d)
        for c in range(t.nchunks):
            b0, b1 = int(t.bases[c]), int(t.bases[c + 1])
            if b1 == b0:
                continue
            src = wp.array(host_1d[b0:b1], dtype=self.dtype,
                           device="cpu", copy=False)
            wp.copy(self.data[c], src, count=b1 - b0)


# Auto-switch threshold for the node-graph pattern build (G5 rung c):
# above this many dof-pair entries (sum over bins of ne*(nbf*ndof)^2)
# the old host COO/slot build becomes the memory binder (measured
# exit-137 ladder at 62 GB host: 4.12M dofs = 4.2G entries dies; 3.26M
# = 805M entries works, 37 s), so identity-T assemblers flip to the
# node-graph path, which is exactness-gated against the old one.
NODE_PATTERN_AUTO_ENTRIES = 2 * 10 ** 8


# ---------------------------------------------------------------------
# W1: per-step element-block (Ae/be) intermediate bound.  assemble()
# used to allocate the WHOLE bin's element matrices in one array,
# Ae = wp.zeros((ne, nl, nl)) with nl = nbf*ndof — sized to the ENTIRE
# element set.  Measured wall: a single 137,367,584,768-byte (137.4 GB)
# device allocation at 3d-L8 (ne = 16,768,504 active elements, nl = 32
# -> ne * 1024 * 8 B) hard-OOM'd on a 95 GiB GH200 whose PERSISTENT
# state was only 62.7 GiB (campaign doc 2026-07-28 §10.10).  It was an
# unbounded intermediate, not a capacity wall.
#
# Fix: compute Ae/be in element BATCHES whose byte footprint stays
# <= AE_BATCH_BYTES and scatter each batch (reusing the scatter_batch /
# batched-scatter machinery already in this module).  The element-block
# math is per-element and independent, so the assembled CSR is
# BIT-IDENTICAL to the whole-bin path (parity-gated).  2 GiB is the
# bound: it caps the transient at the same order as the per-step
# scatter-batch launch note (~2 GB at full res) while leaving ample
# headroom under 95 GiB HBM for the persistent CSR/solver state; the
# derived batch element count is AE_BATCH_BYTES // (npair * 8) —
# 262,144 elements at npair = 1024 (3-D NS hex).  SINGLE-BATCH scales
# (ne <= that count) keep the launch/scatter sequence byte-for-byte the
# pre-W1 path; larger meshes (e.g. 3d-L6/L7 node_mode) now run multiple
# batches — functionally identical (bit-identical CSR, parity-gated),
# just more launches.
# ---------------------------------------------------------------------
AE_BATCH_BYTES = 2 * 2 ** 30            # ~2 GiB per Ae element-block batch

# W2: constraint-expansion scatter chunk cap (ENTRIES per element-aligned
# chunk of the per-bin slots/src/w arrays).  Each entry costs 8 B (int64
# slot) + 8 B (f64 weight) + 4 B (int32 src) = 20 B; capping at
# AE_BATCH_BYTES // 8 keeps the largest single array (slots int64/w f64)
# <= 2 GiB AND well under 2^31 entries (268M << 2.1B).  Because every
# element expands to >= (nbf*ndof)^2 entries (>= 1 master per dof), this
# same cap also bounds the per-chunk Ae (nl^2 entries/element <= expansion
# entries/element), so ONE cap sizes both the scatter arrays and the Ae
# element-block transient.  The MEASURED expansion factor per bin
# (exp_eptr[-1]/ne) sets how many elements land in each chunk.
EXP_CHUNK_ENTRIES = AE_BATCH_BYTES // 8


class DeviceNSAssembler:
    """Per-epoch object: symbolic pattern + slot maps once; numeric fill
    per step on device."""

    def __init__(self, dm, sigma_like=1.0, coloring=False, ndof=None,
                 node_pattern=None, blockmask=None, matvec_only=False,
                 index_width="auto", chunking="auto", chunk_cap=None,
                 val_dtype="fp64"):
        # ndof: dofs per node (default dim+1 = the NS layout; 4 for the
        # ternary CH film system, 2 for binary CH — M4 device-bound)
        # blockmask (B5): bool [ndof, ndof] compile-time dof-pair block
        # sparsity — pattern = kron(G, blockmask) instead of
        # kron(G, ones): the structural-zero couplings (the recorded
        # cuDSS 3-D fill penalty, Sec 6 of the device-assembly note)
        # never exist.  The element kernel MUST write only inside live
        # blocks (masked-out scatter entries are skipped; exactness is
        # gated masked-vs-unmasked).  Node-pattern mode only; the
        # diagonal is forced live (strong rows need it).
        self.dm = dm
        self.coloring = coloring
        # index-width config (P0-2): resolved to a concrete wp dtype
        # once the pattern nnz is known (in the pattern builders); the
        # default "auto" keeps narrow (int32) below ~90% of 2^31 so the
        # pre-P0-2 path is bit-for-bit unchanged.
        self._index_width = index_width
        self._idx_dtype = wp.int32          # provisional; finalized below
        self._idx_np = np.int32
        # Task #38 block-row chunking config: resolved with the index
        # width once nnz is known; OFF below ~90% of 2^31 keeps every
        # buffer/kernel bit-for-bit the pre-#38 path.  chunk_cap
        # overrides the per-chunk element capacity (tests force small
        # caps so multi-chunk paths run at toy sizes).
        self._chunking = chunking
        self._chunk_cap = chunk_cap
        self._chunked = False               # provisional; finalized below
        self._ctab = None
        # W1: per-step Ae/be element-block batch size (elements).  None
        # -> derived per bin from AE_BATCH_BYTES (assemble() bounds the
        # element-matrix transient).  Tests set a small value to force
        # the multi-batch path at toy sizes; a whole-bin value (>= ne)
        # reproduces the pre-W1 single-launch path byte-for-byte.
        self._ae_batch = None
        # Task #36 mixed-precision: fp32 round-on-store snapshot config.
        # vals_d ALWAYS stays fp64 (accumulation contract); _vals_fp32 is
        # allocated only when val_dtype='fp32' and refreshed from vals_d
        # per fill (refresh_fp32_snapshot).  Default fp64: _vals_fp32 stays
        # None and the assembler is bit-for-bit the pre-#36 path.
        self._val_dtype_cfg = val_dtype
        self._val_dtype = _resolve_val_dtype(val_dtype)
        self._vals_fp32 = None              # allocated below iff fp32
        ndof = (dm.dim + 1) if ndof is None else int(ndof)
        self.ndof = ndof
        if blockmask is not None:
            blockmask = np.asarray(blockmask, bool).reshape(ndof, ndof) \
                | np.eye(ndof, dtype=bool)
        self._blockmask = blockmask
        T = dm.constraints.T.tocsr()
        n_free = T.shape[1]
        self.n_free = n_free
        identity_T = (T.shape[0] == T.shape[1]) and (
            T != sp.identity(T.shape[0], format="csr")).nnz == 0
        self._identity_T = identity_T
        self.node_mode = False
        # ---- node-graph pattern path (G5 rung c) -----------------------
        # node_pattern: True forces it, False forbids it, None = auto by
        # entry count. Identity-T, non-colored meshes only.
        tot_entries = sum(
            len(dm.mesh.conn_of[pv]) * (dm.mesh.conn_of[pv].shape[1]
                                        * ndof) ** 2
            for pv in dm.bins)
        if blockmask is not None:
            if node_pattern is False:
                raise ConfigError(
                    "blockmask requires the node-graph pattern")
            node_pattern = True
        if node_pattern is None:
            node_pattern = (identity_T and not coloring
                            and tot_entries > NODE_PATTERN_AUTO_ENTRIES)
        if node_pattern:
            if not (identity_T and not coloring):
                raise BackendError(
                    "node-graph pattern requires identity constraints and "
                    "no coloring")
            self.Nfull = dm.n_nodes * ndof
            if matvec_only:
                # MATRIX-FREE ONLY (M3): build ONLY what apply_batch_matvec
                # needs (bins + per-bin device conn); SKIP the CSR pattern
                # entirely — no indices (nnz int32, ~91 GB host at 256^3),
                # no vals_d (~148 GB), and NO int32-nnz ceiling.  This is
                # the true footprint of the matrix-free outer solve.
                self.node_mode = True
                self._blockmask = blockmask
                self._bins, self._conn_d = [], []
                for pv, b in dm.bins.items():
                    ne, nbf = dm.mesh.conn_of[pv].shape
                    self._bins.append((pv, b, ne, nbf, None))
                    self._conn_d.append(b["conn"])
                self.nnz = None
                return
            self._init_node_pattern(dm, ndof)
            return
        # CONSTRAINT-AWARE (D1 item 3, cuFEM design): element entries are
        # expanded THROUGH the constraint weights host-once — full dof
        # (node n, comp c) -> masters (m_i, c) with weights w_i; each
        # element pair contributes w_r*w_c at (master_r, master_c) in the
        # CONSTRAINED pattern. The device scatter is the same kernel with
        # a weights array + a source-index array.
        self.Nfull = (dm.n_nodes if identity_T else n_free) * ndof
        # masters per NODE from T (row n: free masters + weights)
        Tind, Tptr, Tdat = T.indices, T.indptr, T.data
        # ---- symbolic pattern + slot maps (host, once) -----------------
        rows_all, cols_all = [], []
        exp_bins = []            # per bin: (src_idx, weights, rows, cols)
        self._bins = []
        for pv, b in dm.bins.items():
            conn = dm.mesh.conn_of[pv].astype(np.int64)
            ne, nbf = conn.shape
            gdof = (conn[:, :, None] * ndof
                    + np.arange(ndof)[None, None, :]).reshape(ne,
                                                              nbf * ndof)
            self._bins.append((pv, b, ne, nbf, gdof))
            if identity_T:
                r = np.repeat(gdof, nbf * ndof, axis=1).ravel()
                c = np.tile(gdof, (1, nbf * ndof)).ravel()
                rows_all.append(r)
                cols_all.append(c)
                exp_bins.append(None)
                continue
            # expand each local dof through its node's masters
            nodes = np.repeat(conn, ndof, axis=1)          # [ne, nbf*ndof]
            comps = np.tile(np.tile(np.arange(ndof), nbf), (ne, 1))
            cnt = (Tptr[nodes.ravel() + 1]
                   - Tptr[nodes.ravel()])                  # masters/dof
            # flatten per-dof master lists
            m_idx, m_w, dof_of = [], [], []
            for j, n_ in enumerate(nodes.ravel()):
                s_, e_ = Tptr[n_], Tptr[n_ + 1]
                m_idx.append(Tind[s_:e_])
                m_w.append(Tdat[s_:e_])
                dof_of.append(np.full(e_ - s_, j))
            m_idx = np.concatenate(m_idx)
            m_w = np.concatenate(m_w)
            dof_of = np.concatenate(dof_of)
            free_dof = m_idx * ndof + comps.ravel()[dof_of]
            # per element: cross the row-expansions with col-expansions
            nl = nbf * ndof
            e_of = dof_of // nl
            l_of = dof_of % nl
            src_r, src_c, w_rc, rows_e, cols_e = [], [], [], [], []
            # per-element expansion-entry count -> offsets (W2: chunking
            # the constraint-expansion scatter slices these arrays by
            # element, so the boundaries must be element-aligned).
            exp_cnt = np.empty(ne, np.int64)
            # group by element via sorted order (dof_of already grouped)
            # build per-element index lists
            order = np.argsort(e_of, kind="stable")
            eb = np.searchsorted(e_of[order], np.arange(ne + 1))
            for e_ in range(ne):
                sl = order[eb[e_]:eb[e_ + 1]]
                li = l_of[sl]
                fd = free_dof[sl]
                ww = m_w[sl]
                # cross product of expansions within the element
                A_, B_ = np.meshgrid(np.arange(len(sl)),
                                     np.arange(len(sl)), indexing="ij")
                a_, b2 = A_.ravel(), B_.ravel()
                exp_cnt[e_] = len(a_)
                src_r.append(e_ * nl * nl + li[a_] * nl + li[b2])
                rows_e.append(fd[a_])
                cols_e.append(fd[b2])
                w_rc.append(ww[a_] * ww[b2])
            exp_eptr = np.concatenate(([0], np.cumsum(exp_cnt)))
            exp_bins.append((np.concatenate(src_r),
                             np.concatenate(w_rc),
                             np.concatenate(rows_e),
                             np.concatenate(cols_e),
                             free_dof, m_w, dof_of, exp_eptr))
            rows_all.append(np.concatenate(rows_e))
            cols_all.append(np.concatenate(cols_e))
        self._exp_bins = exp_bins
        r = np.concatenate(rows_all)
        c = np.concatenate(cols_all)
        K = sp.coo_array((np.ones(len(r)), (r, c)),
                         shape=(self.Nfull, self.Nfull)).tocsr()
        K.sort_indices()
        self.indptr = K.indptr.copy()
        self.indices = K.indices.copy()
        self.nnz = K.nnz
        # W2: the constraint-expansion scatter overflows warp's 2^31
        # per-dimension array ceiling in the PER-BIN slots/src/w arrays,
        # whose length is the EXPANSION-ENTRY count (ne*(nbf*ndof)^2*
        # masters^2) — a DIFFERENT, larger quantity than the CSR nnz.
        # At 3d-L7r9 it is 2.37B while nnz is well under 2^31, so the nnz-
        # gated chunking auto-switch would NOT fire and the flat upload
        # would crash.  Trigger chunking on max(nnz, max expansion entries)
        # so the exp-chunk path engages whenever EITHER quantity crosses
        # the ceiling.  chunking='off' past the ceiling still refuses
        # loudly inside _resolve_chunking.
        max_exp = 0
        if not identity_T:
            max_exp = max((len(eb[2]) for eb in exp_bins if eb is not None),
                          default=0)
        chunk_trigger = max(self.nnz, max_exp)
        # P0-2: resolve the CSR index width from the realized nnz.
        idx_dt = self._finalize_idx_width(self.nnz, chunk_trigger)
        inp = self._idx_np
        # Task #38 + W2: block-row chunk table over the realized pattern.
        # COO-path chunking supports the identity-T non-colored scatter
        # AND (W2) the constraint-expansion weighted scatter (the adaptive
        # AMR path — hanging nodes make identity_T=False; the expansion
        # entries reach billions at 9M+ DOF, past warp's 2^31 array
        # ceiling in BOTH vals_d and the per-bin slot/src/w arrays).  The
        # colored scatter is still narrow-only by scope.
        if self._chunked:
            if coloring:
                raise BackendError(
                    "chunking supports the identity-constraint and "
                    "constraint-expansion (non-colored) scatter paths "
                    "only (COO or node-graph pattern)")
            self._ctab = ChunkTable(self.indptr, cap=self._chunk_cap,
                                    device=dm.device)
        # slot index per (element-pair entry): position in the CSR
        # values array. FULLY VECTORIZED via sparse fancy indexing: give
        # the pattern matrix data = arange(nnz), then K2[rr, cc] returns
        # each entry's slot directly (replaced a per-entry Python loop
        # measured at 56 s for 3-D L5; now milliseconds).
        K2 = K.copy()
        K2.data = np.arange(self.nnz, dtype=np.float64)
        slot_bins = []
        self._weight_bins = []
        self._src_bins = []
        for k_bin, (pv, b, ne, nbf, gdof) in enumerate(self._bins):
            if identity_T:
                rr = np.repeat(gdof, nbf * ndof, axis=1).ravel()
                cc = np.tile(gdof, (1, nbf * ndof)).ravel()
                slots = np.asarray(K2[rr, cc]).ravel().astype(np.int64)
                slot_bins.append(slots)
                self._weight_bins.append(None)
                self._src_bins.append(None)
            else:
                src, w, rr, cc, fd, mw, dof_of, _eptr = self._exp_bins[k_bin]
                slots = np.asarray(K2[rr, cc]).ravel().astype(np.int64)
                slot_bins.append(slots)
                self._weight_bins.append(w)
                self._src_bins.append(src)
        self._slot_bins = slot_bins
        # W2: when chunked AND constraint-expanded, the per-bin
        # slots/src/w arrays themselves pass 2^31 entries (measured 2.37B
        # at 3d-L7r9) — warp cannot construct the flat 1-D uploads at all.
        # Chunk them ELEMENT-ALIGNED (a chunk covers elements [e0, e1),
        # its slice of the element-contiguous expansion arrays) so each
        # per-chunk array stays under the same byte/element bound as the
        # Ae batch and the src values rebase cleanly to a batch-local Ae.
        # Built per bin as _exp_chunks[k_bin] = list of
        # (e0, e1, slots_d int64, src_d int32 (batch-local), w_d f64).
        self._exp_chunks = None
        if self._chunked and not identity_T:
            self._build_exp_chunks(slot_bins, dm.device)
            self._slots_d = [None] * len(slot_bins)
            self._w_d = [None] * len(slot_bins)
            self._src_d = [None] * len(slot_bins)
        else:
            # device uploads.  Slot arrays index nnz-space -> widen with
            # the CSR (inp/idx_dt); the source (src_d) and dof (gdof_d)
            # arrays are element-block-local / dof-space and stay int32.
            self._slots_d = [wp.array(np.ascontiguousarray(
                s.astype(inp).ravel()), dtype=idx_dt, device=dm.device)
                for s in slot_bins]
            self._w_d = [None if w is None else wp.array(
                np.ascontiguousarray(w), dtype=wp.float64,
                device=dm.device) for w in self._weight_bins]
            self._src_d = [None if s2 is None else wp.array(
                np.ascontiguousarray(s2.astype(np.int32)), dtype=wp.int32,
                device=dm.device) for s2 in self._src_bins]
        self.vals_d = (ChunkedArray(self._ctab, wp.float64, dm.device)
                       if self._chunked
                       else wp.zeros(self.nnz, dtype=wp.float64,
                                     device=dm.device))
        self._alloc_fp32_snapshot()
        self.F_d = wp.zeros(self.Nfull, dtype=wp.float64,
                            device=dm.device)
        gdof_bins = []
        self._bw_d = []
        self._bsrc_d = []
        for k_bin, (pv, b, ne, nbf, g) in enumerate(self._bins):
            if identity_T:
                gdof_bins.append(wp.array(g.astype(np.int32).ravel(),
                                          dtype=wp.int32,
                                          device=dm.device))
                self._bw_d.append(None)
                self._bsrc_d.append(None)
            else:
                src, w, rr, cc, fd, mw, dof_of, _eptr = self._exp_bins[k_bin]
                gdof_bins.append(wp.array(fd.astype(np.int32),
                                          dtype=wp.int32,
                                          device=dm.device))
                self._bw_d.append(wp.array(np.ascontiguousarray(mw),
                                           dtype=wp.float64,
                                           device=dm.device))
                self._bsrc_d.append(wp.array(
                    dof_of.astype(np.int32), dtype=wp.int32,
                    device=dm.device))
        self._gdof_d = gdof_bins
        # coloring (variant b): greedy on node sharing
        if coloring:
            self._colors = []
            for pv, b, ne, nbf, gdof in self._bins:
                conn = dm.mesh.conn_of[pv]
                color = -np.ones(ne, np.int32)
                node_color_used = {}
                for e in range(ne):
                    used = set()
                    for a in conn[e]:
                        used |= node_color_used.get(int(a), set())
                    col = 0
                    while col in used:
                        col += 1
                    color[e] = col
                    for a in conn[e]:
                        node_color_used.setdefault(int(a), set()).add(col)
                order = np.argsort(color, kind="stable")
                bounds = np.searchsorted(color[order],
                                         np.arange(color.max() + 2))
                self._colors.append((order.astype(np.int32), bounds))

    def _finalize_idx_width(self, nnz, chunk_estimate=None):
        """Resolve the concrete index dtype from the config knob + the
        realized pattern nnz.  Sets self._idx_dtype (wp) / self._idx_np
        (numpy).  Also asserts dofs (Nfull) fit int32 — always true at
        our scales but checked so a wide-mode column-index assumption
        can never silently break.  Returns the chosen wp dtype.

        Task #38: also resolves the chunking knob (block-row ChunkedCSR
        past warp's 2^31-element array ceiling).  Chunked storage
        carries int64 global slots by construction, so chunking forces
        the WIDE index dtype; an explicit index_width='narrow' conflicts
        and is refused.

        W2: `chunk_estimate` (default = nnz) drives the CHUNKING switch
        separately from the index width — the constraint-expansion path
        overflows in its per-bin arrays (expansion-entry count) while nnz
        itself stays under the ceiling, so chunking must trigger on the
        larger of the two.  The index dtype still keys off nnz (column
        indices are dof-space; only chunked storage forces int64)."""
        self._idx_dtype = _resolve_idx_width(self._index_width, nnz)
        if chunk_estimate is None:
            chunk_estimate = nnz
        self._chunked = _resolve_chunking(self._chunking, chunk_estimate)
        if self._chunked:
            if self._index_width == "narrow":
                raise ConfigError(
                    "chunking is active but index_width='narrow' forced: "
                    "chunked kernels carry int64 global slots — use "
                    "index_width='auto' or 'wide'")
            self._idx_dtype = wp.int64
        self._idx_np = (np.int64 if self._idx_dtype is wp.int64
                        else np.int32)
        if self.Nfull >= 2 ** 31:
            raise BackendError(
                f"dof count Nfull={self.Nfull} >= 2^31: column indices "
                f"stay int32 in mixed-width CSR — this exceeds that "
                f"assumption (dof-space overflow, not nnz-space)")
        self.index_wide = self._idx_dtype is wp.int64
        return self._idx_dtype

    # ------------------------------------------------------------------
    # G5 rung c: node-graph symbolic pattern. Build the NODE adjacency
    # graph G (nbf^2 pairs/element — ndof^2-fold fewer host COO entries
    # than the dof pattern), then the dof-level CSR structurally
    # = kron(G, ones(ndof, ndof)) with indptr/indices in CLOSED FORM
    # (no dof-level COO ever exists; indices built by a device kernel).
    # Element slot maps are computed IN-KERNEL from the per-element
    # node-pair position in G:
    #   dof row (ndof*na + ca) starts at
    #       indptr[ndof*na + ca] = ndof^2*Gptr[na] + ca*ndof*deg(na),
    #   and the entry for G-neighbor at offset q = s - Gptr[na] (s the
    #   node-pair slot) and dof col comp cb sits at + ndof*q + cb.
    # This removes the three measured full-res binders (15.15M dofs):
    # the host dof COO (~61 GB), the K2[rr, cc] slot fancy-indexing
    # (exit-137 at 4.12M dofs / 62 GB host), and the ne x (nbf*ndof)^2
    # device slot map (15.2 GB); the per-element G-slot map is
    # ne x nbf^2 int32 (~0.95 GB at full res).
    # ------------------------------------------------------------------
    def _init_node_pattern(self, dm, ndof):
        self.node_mode = True
        d = dm.device
        n = dm.n_nodes
        rows_all, cols_all = [], []
        self._bins = []
        self._conn_d = []
        for pv, b in dm.bins.items():
            conn = dm.mesh.conn_of[pv].astype(np.int32)
            ne, nbf = conn.shape
            self._bins.append((pv, b, ne, nbf, None))
            self._conn_d.append(b["conn"])
            # a varies slowly, b fast — matches gslot index a*nbf + b
            rows_all.append(np.repeat(conn, nbf, axis=1).ravel())
            cols_all.append(np.tile(conn, (1, nbf)).ravel())
        r = np.concatenate(rows_all) if len(rows_all) > 1 else rows_all[0]
        c = np.concatenate(cols_all) if len(cols_all) > 1 else cols_all[0]
        # duplicate count per node pair <= elements sharing the pair
        # (<= 2^dim on conforming hexes) — int8 cannot overflow
        G = sp.coo_array((np.ones(len(r), np.int8), (r, c)),
                         shape=(n, n)).tocsr()
        G.sort_indices()
        del r, c
        Gptr = G.indptr.astype(np.int64)
        Gind = G.indices.astype(np.int32)
        del G
        gnnz = int(Gptr[-1])
        mask = self._blockmask
        deg = np.diff(Gptr)
        if mask is None:
            self.nnz = gnnz * ndof * ndof
        else:
            # B5 block-masked pattern = kron(G, mask): per dof row
            # (na, ca) the live cols are deg(na) * rowcnt[ca]
            rowcnt = mask.sum(axis=1).astype(np.int64)
            rowoff = np.concatenate(([0], np.cumsum(rowcnt)))
            lcols = np.concatenate(
                [np.where(mask[ca])[0] for ca in range(ndof)])
            colpos = np.full(ndof * ndof, -1, np.int64)
            for ca in range(ndof):
                colpos[ca * ndof + np.where(mask[ca])[0]] = \
                    np.arange(rowcnt[ca])
            self.nnz = gnnz * int(rowcnt.sum())
        # P0-2: pick the CSR index width from the realized nnz (auto:
        # wide once past ~90% of 2^31).  Replaces the old hard raise —
        # wide mode carries int64 offsets/slots so the slot arithmetic
        # no longer wraps.  A forced-narrow past the ceiling still FAILS
        # loudly inside _finalize_idx_width (no silent wrap).
        idx_dt = self._finalize_idx_width(self.nnz)
        inp = self._idx_np
        # dof-level indptr in closed form (int64: values reach nnz)
        if mask is None:
            self.indptr = np.concatenate(
                ([0], np.cumsum(np.repeat(deg * ndof,
                                          ndof)))).astype(np.int64)
        else:
            self.indptr = np.concatenate(
                ([0], np.cumsum((deg[:, None]
                                 * rowcnt[None, :]).ravel()))
            ).astype(np.int64)
        assert self.indptr[-1] == self.nnz
        # Task #38: block-row chunk table over the closed-form indptr —
        # built BEFORE any nnz-length allocation (the whole point: warp
        # cannot construct a 1-D >2^31-element buffer at all).
        if self._chunked:
            self._ctab = ChunkTable(self.indptr, cap=self._chunk_cap,
                                    device=d)
            # in-kernel int64 reads of Gind assume node-pair nnz < 2^31
            # (nnz/ndof^2 — holds to ~34B dof-level nnz)
            assert gnnz < 2 ** 31
        # dof-level indices by device kernel (one thread per dof row);
        # int32 mirror kept on host (blockch symbolic setup, csr_slots).
        # Gptr indexes into nnz-space (via g0) -> uploaded at the CSR
        # index width; the COLUMN indices (ind_d) stay int32 and go
        # CHUNKED past the ceiling (a dof row never straddles chunks —
        # ChunkTable is row-aligned — so the write kernel locates its
        # chunk once).
        self._Gptr_d = wp.array(Gptr.astype(inp), dtype=idx_dt, device=d)
        Gind_d = wp.array(Gind, dtype=wp.int32, device=d)
        ind_d = (ChunkedArray(self._ctab, wp.int32, d) if self._chunked
                 else wp.zeros(self.nnz, dtype=wp.int32, device=d))
        if mask is None:
            self._mask_d = None
            if self._chunked:
                wp.launch(_dof_indices_kernel_chunked(), dim=self.Nfull,
                          inputs=[self._Gptr_d, Gind_d, wp.int32(ndof),
                                  ind_d.data, self._ctab.bases_d,
                                  wp.int32(self._ctab.nchunks)], device=d)
            else:
                wp.launch(_dof_indices_kernel(idx_dt), dim=self.Nfull,
                          inputs=[self._Gptr_d, Gind_d, wp.int32(ndof),
                                  ind_d], device=d)
        else:
            i32 = lambda a_: wp.array(a_.astype(np.int32),
                                      dtype=wp.int32, device=d)
            self._mask_d = dict(rowoff=i32(rowoff), lcols=i32(lcols),
                                colpos=i32(colpos),
                                blocknnz=int(rowcnt.sum()))
            if self._chunked:
                wp.launch(_dof_indices_masked_kernel_chunked(),
                          dim=self.Nfull,
                          inputs=[self._Gptr_d, Gind_d, wp.int32(ndof),
                                  self._mask_d["rowoff"],
                                  self._mask_d["lcols"], ind_d.data,
                                  self._ctab.bases_d,
                                  wp.int32(self._ctab.nchunks)], device=d)
            else:
                wp.launch(_dof_indices_masked_kernel(idx_dt),
                          dim=self.Nfull,
                          inputs=[self._Gptr_d, Gind_d, wp.int32(ndof),
                                  self._mask_d["rowoff"],
                                  self._mask_d["lcols"], ind_d], device=d)
        self.indices = ind_d.numpy()
        del ind_d, Gind_d
        # per-element node-pair slot in G via keyed searchsorted
        # (Gkey strictly increasing: CSR row-major + sorted columns)
        Gkey = np.repeat(np.arange(n, dtype=np.int64), deg) * n + Gind
        self._gslot_d = []
        for (pv, b, ne, nbf, _), rr, cc in zip(self._bins, rows_all,
                                               cols_all):
            key = rr.astype(np.int64) * n + cc
            slot = np.searchsorted(Gkey, key)
            if not (slot.max() < gnnz and (Gkey[slot] == key).all()):
                raise BackendError("node-pair slot lookup failed")
            self._gslot_d.append(wp.array(slot.astype(np.int32),
                                          dtype=wp.int32, device=d))
            del key, slot
        del Gkey, Gind, rows_all, cols_all
        self.vals_d = (ChunkedArray(self._ctab, wp.float64, d)
                       if self._chunked
                       else wp.zeros(self.nnz, dtype=wp.float64,
                                     device=d))
        self._alloc_fp32_snapshot()
        self.F_d = wp.zeros(self.Nfull, dtype=wp.float64, device=d)

    # ------------------------------------------------------------------
    # Task #36: fp32 round-on-store snapshot of vals_d for cuDSS fp32
    # factorization.  vals_d stays fp64 (accumulation contract); this is a
    # SEPARATE buffer rounded from it per fill.  No-op / None for fp64.
    # ------------------------------------------------------------------
    def _alloc_fp32_snapshot(self):
        """Allocate _vals_fp32 mirroring vals_d's storage (flat or #38
        chunked) when val_dtype='fp32'; leave it None for fp64."""
        if self._val_dtype is wp.float64:
            self._vals_fp32 = None
            return
        if self._chunked:
            self._vals_fp32 = ChunkedArray(self._ctab, wp.float32,
                                           self.dm.device)
        else:
            self._vals_fp32 = wp.zeros(self.nnz, dtype=wp.float32,
                                       device=self.dm.device)

    def refresh_fp32_snapshot(self):
        """Round-on-store vals_d (fp64) -> _vals_fp32 (fp32).  One cast
        kernel per nnz; no-op when val_dtype='fp64' (no snapshot).  Call
        after a fill and before the cuDSS fp32 factorization consumes it
        (device_csr_fp32)."""
        if self._vals_fp32 is None:
            return
        d = self.dm.device
        if self._chunked:
            # per-chunk cast: both buffers share the same 2-D layout, so
            # a straight element-wise round over each chunk's valid span.
            wp.launch(_round_store_kernel_chunked(),
                      dim=(self._ctab.nchunks, self._ctab.cap),
                      inputs=[self.vals_d.data, self._vals_fp32.data],
                      device=d)
        else:
            wp.launch(_round_store_kernel(), dim=self.nnz,
                      inputs=[self.vals_d, self._vals_fp32], device=d)

    def set_strong_rows(self, rows, diag_vals=None):
        """D1 item 2: per-epoch strong-row plan. rows: global dof ids
        whose equations become identity rows. Precomputes slot spans and
        diagonal slots; assemble() applies them ON DEVICE after scatter
        (replaces the host LIL surgery — measured 0.8-1.4 s/step)."""
        rows = np.asarray(rows, np.int64)
        starts = self.indptr[rows]
        ends = self.indptr[rows + 1]
        # diagonal slot per row (column == row)
        diag = np.empty(len(rows), np.int64)
        for i, (r_, s_, e_) in enumerate(zip(rows, starts, ends)):
            diag[i] = s_ + np.searchsorted(self.indices[s_:e_], r_)
        # flat list of ALL slots in the strong rows (to zero)
        spans = np.concatenate([np.arange(s_, e_)
                                for s_, e_ in zip(starts, ends)])
        # spans/diag index nnz-space (positions in the values array) ->
        # widen with the CSR; rows are dof-space (stay int32).
        idx_dt, inp = self._idx_dtype, self._idx_np
        self._strong = dict(
            rows_d=wp.array(rows.astype(np.int32), dtype=wp.int32,
                            device=self.dm.device),
            spans_d=wp.array(spans.astype(inp), dtype=idx_dt,
                             device=self.dm.device),
            diag_d=wp.array(diag.astype(inp), dtype=idx_dt,
                            device=self.dm.device),
            n_spans=len(spans), n_rows=len(rows))

    def _exp_chunk_entries(self):
        """W2: entries per element-aligned constraint-expansion chunk.
        Honors the `_ae_batch`-style test hook `_exp_chunk_cap` (tests
        force a tiny cap to run the multi-chunk path at toy sizes);
        otherwise the ~2 GiB-derived EXP_CHUNK_ENTRIES bound."""
        cap = getattr(self, "_exp_chunk_cap", None)
        if cap is not None:
            return max(1, int(cap))
        return EXP_CHUNK_ENTRIES

    def _build_exp_chunks(self, slot_bins, device):
        """W2: partition each bin's element-contiguous constraint-
        expansion arrays (slots/src/w) into ELEMENT-ALIGNED chunks whose
        per-chunk arrays stay < 2^31 entries and <= the ~2 GiB bound.

        A chunk covers elements [e0, e1); its slice of the (already
        element-contiguous) expansion arrays is uploaded as warp arrays.
        `src` values are element-local offsets into the WHOLE-bin Ae
        (e*nl*nl + ...); they are REBASED by e0*nl*nl so the scatter reads
        the batch-local Ae [e1-e0, nl, nl].  slots stay int64 (nnz-space,
        chunk-located in-kernel against vals_d's chunk table).  The
        per-chunk element count is derived from the MEASURED expansion
        factor: elements are packed greedily until the running entry count
        would exceed the cap."""
        cap = self._exp_chunk_entries()
        ndof = self.ndof
        self._exp_chunks = []
        for k_bin, (pv, b, ne, nbf, gdof) in enumerate(self._bins):
            src, w, rr, cc, fd, mw, dof_of, eptr = self._exp_bins[k_bin]
            slots = slot_bins[k_bin]
            nl = nbf * ndof
            npair = nl * nl
            chunks = []
            e0 = 0
            while e0 < ne:
                # largest e1 with (eptr[e1]-eptr[e0]) <= cap; >= e0+1 (a
                # single element's expansion never exceeds cap at any real
                # scale — nl^2 * masters^2).
                budget = eptr[e0] + cap
                e1 = int(np.searchsorted(eptr, budget, side="right") - 1)
                e1 = min(max(e1, e0 + 1), ne)
                a, z = int(eptr[e0]), int(eptr[e1])
                slots_c = np.ascontiguousarray(slots[a:z].astype(np.int64))
                src_c = np.ascontiguousarray(
                    (src[a:z] - e0 * npair).astype(np.int32))
                w_c = np.ascontiguousarray(w[a:z])
                chunks.append((
                    e0, e1,
                    wp.array(slots_c, dtype=wp.int64, device=device),
                    wp.array(src_c, dtype=wp.int32, device=device),
                    wp.array(w_c, dtype=wp.float64, device=device)))
                e0 = e1
            self._exp_chunks.append(chunks)

    def _ae_batch_for(self, npair):
        """Elements per Ae/be batch for a bin whose element matrix has
        `npair` = (nbf*ndof)^2 fp64 entries.  Honors the `_ae_batch`
        override (tests force small batches); otherwise derives the
        largest count whose Ae footprint (nb*npair*8 B) stays within
        AE_BATCH_BYTES (>= 1 always)."""
        if self._ae_batch is not None:
            return max(1, int(self._ae_batch))
        return max(1, AE_BATCH_BYTES // (npair * 8))

    # ------------------------------------------------------------------
    def _scatter_be_weighted(self, k_bin, be, d):
        """Scatter element RHS `be` into F_d using the weighted (constraint-
        expansion) kernel.  Factored from two call sites in assemble() that
        have bit-identical scatter logic; floating-point summation order is
        preserved — the launch dim and input arrays are unchanged."""
        scat_bw = _scatter_vec_weighted_kernel()
        wp.launch(scat_bw, dim=len(self._exp_bins[k_bin][4]),
                  inputs=[be.reshape((-1,)), self._bsrc_d[k_bin],
                          self._bw_d[k_bin], self._gdof_d[k_bin],
                          self.F_d], device=d)

    # ------------------------------------------------------------------
    def assemble(self, aq_by_bin, div_aq_by_bin, fq_by_bin, nu, sigma,
                 sig2tau=None, tau_scale=1.0, s_skew=0.5, strong_b_vals=None,
                 extra_matrix=None, extra_rhs=None):
        """Numeric fill on device; returns (csr, F) with HOST copies for
        now (the solver interface); vals stay resident in self.vals_d.

        M1d closure: the GP-field inputs (aq/div/fq per bin) may be
        DEVICE wp.arrays (assembly/gp_field.py products) — consumed
        directly, no host round-trip; numpy inputs upload as before.
        extra_matrix=(slots_d, vals_d) / extra_rhs=(dofs_d, vals_d):
        additional device-resident contributions (e.g. a cached SBM face
        system on the fixed pattern) atomically added AFTER the volume
        scatter and BEFORE the strong rows — the composition order the
        host path realizes as A_vol + Af then row surgery."""
        from ..api.ns_bricks import make_linear_ns_Ae, make_linear_ns_be
        dm = self.dm
        d = dm.device

        def _dev(x):
            return x if isinstance(x, wp.array) else wp.array(
                np.ascontiguousarray(x), dtype=wp.float64, device=d)

        ndof = self.ndof
        if sig2tau is None:
            sig2tau = (2.0 * sigma) ** 2

        # W2b: per-phase profiling accumulators (only allocated when
        # DIFFSIM_ASM_PROFILE=1; otherwise these lines are never reached
        # and _ASM_PROFILE stays False — zero overhead in the default path).
        if _ASM_PROFILE:
            _t_upload = 0.0    # input GP-field host->device uploads (ms)
            _t_ae = 0.0        # Ae/be compute kernels (ms)
            _t_scat = 0.0      # scatter kernels into vals_d/F_d (ms)
            _t_extra = 0.0     # add_matrix_values / add_rhs_values (ms)
            _t_strong = 0.0    # apply_strong_rows (ms)
            _t_pull = 0.0      # vals_d.numpy() + F_d.numpy() host pull (ms)
            _nchunks_total = 0 # total constraint-expansion chunks fired
            _t0 = time.perf_counter()

        self.vals_d.zero_()
        self.F_d.zero_()
        if not hasattr(self, "_gaq_d"):
            self._gaq_d = {}
        for k_bin, (pv, b, ne, nbf, gdof) in enumerate(self._bins):
            nqp = b["nqp"]
            if _ASM_PROFILE:
                _t_up0 = time.perf_counter()
            aq = _dev(aq_by_bin[pv])
            dq = _dev(div_aq_by_bin[pv])
            # frozen-a linearization: gaq only read at newton=1 (never
            # here) — a cached zero buffer, not a per-step upload
            gaq = self._gaq_d.get(pv)
            if gaq is None:
                gaq = wp.zeros((ne * nqp, dm.dim * dm.dim),
                               dtype=wp.float64, device=d)
                self._gaq_d[pv] = gaq
            fq = _dev(fq_by_bin[pv])
            if _ASM_PROFILE:
                wp.synchronize()
                _t_upload += (time.perf_counter() - _t_up0) * 1e3
            kA = make_linear_ns_Ae(nbf, nqp, dm.dim)
            kb = make_linear_ns_be(nbf, nqp, dm.dim)
            npair = (nbf * ndof) ** 2
            nl = nbf * ndof
            # W1: the identity-T non-colored and node-graph scatters read
            # Ae/be as element-CONTIGUOUS block ranges (scatter_batch
            # slices its slot maps / computes slots in-kernel by element
            # offset), so the whole-bin Ae — the 137 GB OOM at 3d-L8 — is
            # replaced by element BATCHES bounded to AE_BATCH_BYTES.  The
            # element-block math is per-element independent, so batching
            # is bit-identical to the whole-bin launch.  The weighted
            # (constraint-aware) and colored scatters index Ae through
            # non-contiguous whole-bin slot maps and are scoped to
            # small/mid meshes (never near the OOM scale), so they keep
            # the single whole-bin allocation unchanged.
            if self.node_mode or (self._identity_T and not self.coloring):
                step = self._ae_batch_for(npair)
                for e0 in range(0, ne, step):
                    nb = min(step, ne - e0)
                    Ae = wp.zeros((nb, nl, nl), dtype=wp.float64, device=d)
                    be = wp.zeros((nb, nl), dtype=wp.float64, device=d)
                    conn_v = b["conn"][e0:e0 + nb]
                    h_v = b["h"][e0:e0 + nb]
                    aq_v = aq[e0 * nqp:(e0 + nb) * nqp]
                    dq_v = dq[e0 * nqp:(e0 + nb) * nqp]
                    gaq_v = gaq[e0 * nqp:(e0 + nb) * nqp]
                    fq_v = fq[e0 * nqp:(e0 + nb) * nqp]
                    if _ASM_PROFILE:
                        _t_ae0 = time.perf_counter()
                    wp.launch(kA, dim=nb,
                              inputs=[conn_v, h_v, b["N"], b["dN"],
                                      b["lapN"], b["w"],
                                      aq_v, dq_v, gaq_v, wp.float64(nu),
                                      wp.float64(sigma),
                                      wp.float64(sig2tau),
                                      wp.float64(tau_scale),
                                      wp.float64(s_skew), wp.int32(0), Ae],
                              device=d)
                    wp.launch(kb, dim=nb,
                              inputs=[conn_v, h_v, b["N"], b["dN"], b["w"],
                                      aq_v, fq_v, wp.float64(nu),
                                      wp.float64(sig2tau),
                                      wp.float64(tau_scale), be], device=d)
                    if _ASM_PROFILE:
                        wp.synchronize()
                        _t_ae += (time.perf_counter() - _t_ae0) * 1e3
                        _t_sc0 = time.perf_counter()
                    self.scatter_batch(k_bin, e0, Ae, be, nb)
                    if _ASM_PROFILE:
                        wp.synchronize()
                        _t_scat += (time.perf_counter() - _t_sc0) * 1e3
                continue
            # W2: chunked constraint-expansion path.  The per-bin
            # slots/src/w arrays and vals_d all pass 2^31 at adaptive
            # scale, so the matrix scatter runs per ELEMENT-ALIGNED chunk:
            # each chunk builds ONLY its elements' Ae ([e1-e0, nl, nl],
            # bounded ~2 GiB) and scatters through the chunked weighted
            # kernel (int64 slot chunk-located against vals_d's table).
            # The rhs stays whole-bin (its expansion arrays are linear in
            # masters, far under 2^31), so be/its weighted-vec scatter are
            # unchanged.  Bit-identical to the unchunked weighted scatter:
            # same per-entry w*Ae contributions, just partitioned.
            if self._chunked and not self._identity_T:
                scwc = _scatter_weighted_kernel_chunked()
                for (e0, e1, slots_c, src_c, w_c) in \
                        self._exp_chunks[k_bin]:
                    nb = e1 - e0
                    if _ASM_PROFILE:
                        _nchunks_total += 1
                        _t_ae0 = time.perf_counter()
                    Ae = wp.zeros((nb, nl, nl), dtype=wp.float64, device=d)
                    conn_v = b["conn"][e0:e1]
                    h_v = b["h"][e0:e1]
                    aq_v = aq[e0 * nqp:e1 * nqp]
                    dq_v = dq[e0 * nqp:e1 * nqp]
                    gaq_v = gaq[e0 * nqp:e1 * nqp]
                    wp.launch(kA, dim=nb,
                              inputs=[conn_v, h_v, b["N"], b["dN"],
                                      b["lapN"], b["w"], aq_v, dq_v, gaq_v,
                                      wp.float64(nu), wp.float64(sigma),
                                      wp.float64(sig2tau),
                                      wp.float64(tau_scale),
                                      wp.float64(s_skew), wp.int32(0), Ae],
                              device=d)
                    if _ASM_PROFILE:
                        wp.synchronize()
                        _t_ae += (time.perf_counter() - _t_ae0) * 1e3
                        _t_sc0 = time.perf_counter()
                    wp.launch(scwc, dim=len(w_c),
                              inputs=[Ae.reshape((-1,)), src_c, w_c,
                                      slots_c, self.vals_d.data,
                                      self._ctab.bases_d,
                                      wp.int32(self._ctab.nchunks)],
                              device=d)
                    if _ASM_PROFILE:
                        wp.synchronize()
                        _t_scat += (time.perf_counter() - _t_sc0) * 1e3
                be = wp.zeros((ne, nl), dtype=wp.float64, device=d)
                wp.launch(kb, dim=ne,
                          inputs=[b["conn"], b["h"], b["N"], b["dN"],
                                  b["w"], aq, fq, wp.float64(nu),
                                  wp.float64(sig2tau),
                                  wp.float64(tau_scale), be], device=d)
                self._scatter_be_weighted(k_bin, be, d)
                continue
            # constraint-aware / colored paths: whole-bin (unchanged).
            if _ASM_PROFILE:
                _t_ae0 = time.perf_counter()
            Ae = wp.zeros((ne, nl, nl), dtype=wp.float64, device=d)
            be = wp.zeros((ne, nl), dtype=wp.float64, device=d)
            wp.launch(kA, dim=ne,
                      inputs=[b["conn"], b["h"], b["N"], b["dN"],
                              b["lapN"],   # G4: complete SUPG/PSPG resu
                              b["w"],
                              aq, dq, gaq, wp.float64(nu),
                              wp.float64(sigma), wp.float64(sig2tau),
                              wp.float64(tau_scale),
                              wp.float64(s_skew), wp.int32(0), Ae],
                      device=d)
            wp.launch(kb, dim=ne,
                      inputs=[b["conn"], b["h"], b["N"], b["dN"], b["w"],
                              aq, fq, wp.float64(nu), wp.float64(sig2tau),
                              wp.float64(tau_scale), be], device=d)
            if _ASM_PROFILE:
                wp.synchronize()
                _t_ae += (time.perf_counter() - _t_ae0) * 1e3
                _t_sc0 = time.perf_counter()
            if not self._identity_T:
                scw = _scatter_weighted_kernel(self._idx_dtype)
                wp.launch(scw, dim=len(self._slot_bins[k_bin]),
                          inputs=[Ae.reshape((-1,)), self._src_d[k_bin],
                                  self._w_d[k_bin], self._slots_d[k_bin],
                                  self.vals_d], device=d)
            else:
                order, bounds = self._colors[k_bin]
                order_d = wp.array(order, dtype=wp.int32, device=d)
                scat_c = _scatter_colored_kernel(self._idx_dtype)
                for ci in range(len(bounds) - 1):
                    lo, hi = int(bounds[ci]), int(bounds[ci + 1])
                    if hi > lo:
                        wp.launch(scat_c, dim=(hi - lo) * npair,
                                  inputs=[Ae.reshape((-1,)),
                                          self._slots_d[k_bin], order_d,
                                          wp.int32(lo), wp.int32(npair),
                                          self.vals_d], device=d)
            if self._identity_T:
                scat_b = _scatter_vec_kernel()
                wp.launch(scat_b, dim=ne * nbf * ndof,
                          inputs=[be.reshape((-1,)), self._gdof_d[k_bin],
                                  self.F_d], device=d)
            else:
                self._scatter_be_weighted(k_bin, be, d)
            if _ASM_PROFILE:
                wp.synchronize()
                _t_scat += (time.perf_counter() - _t_sc0) * 1e3
        if _ASM_PROFILE:
            _t_ex0 = time.perf_counter()
        if extra_matrix is not None:
            self.add_matrix_values(*extra_matrix)
        if extra_rhs is not None:
            self.add_rhs_values(*extra_rhs)
        if _ASM_PROFILE:
            wp.synchronize()
            _t_extra = (time.perf_counter() - _t_ex0) * 1e3
            _t_str0 = time.perf_counter()
        if getattr(self, "_strong", None) is not None:
            self.apply_strong_rows(strong_b_vals)
        if _ASM_PROFILE:
            wp.synchronize()
            _t_strong = (time.perf_counter() - _t_str0) * 1e3
        if getattr(self, "_return_device", False) == "raw":
            if _ASM_PROFILE:
                _t_total = (time.perf_counter() - _t0) * 1e3
                self._last_profile = dict(
                    upload_ms=_t_upload, ae_ms=_t_ae, scatter_ms=_t_scat,
                    extra_ms=_t_extra, strong_ms=_t_strong, pull_ms=0.0,
                    chunks=_nchunks_total, total_ms=_t_total)
            return None                    # fill-only (assemble_fill)
        if getattr(self, "_return_device", False) == "handoff":
            # W2c: keep values device-resident.  Pull only the small rhs
            # (Nfull*8 B) to host — the driver's post-processing (traction,
            # reshape) consumes a host x; the 19 GB values array stays on
            # device for the SpMV / preconditioner.
            if _ASM_PROFILE:
                _t_total = (time.perf_counter() - _t0) * 1e3
                self._last_profile = dict(
                    upload_ms=_t_upload, ae_ms=_t_ae, scatter_ms=_t_scat,
                    extra_ms=_t_extra, strong_ms=_t_strong, pull_ms=0.0,
                    chunks=_nchunks_total, total_ms=_t_total)
            return DeviceSaddleCSR(self), self.F_d.numpy()
        if getattr(self, "_return_device", False):
            if _ASM_PROFILE:
                _t_total = (time.perf_counter() - _t0) * 1e3
                self._last_profile = dict(
                    upload_ms=_t_upload, ae_ms=_t_ae, scatter_ms=_t_scat,
                    extra_ms=_t_extra, strong_ms=_t_strong, pull_ms=0.0,
                    chunks=_nchunks_total, total_ms=_t_total)
            return self.device_csr()
        if _ASM_PROFILE:
            _t_pull0 = time.perf_counter()
        A = sp.csr_matrix((self.vals_d.numpy(), self.indices,
                           self.indptr), shape=(self.Nfull, self.Nfull))
        F_host = self.F_d.numpy()
        if _ASM_PROFILE:
            wp.synchronize()
            _t_pull = (time.perf_counter() - _t_pull0) * 1e3
            _t_total = (time.perf_counter() - _t0) * 1e3
            self._last_profile = dict(
                upload_ms=_t_upload, ae_ms=_t_ae, scatter_ms=_t_scat,
                extra_ms=_t_extra, strong_ms=_t_strong, pull_ms=_t_pull,
                chunks=_nchunks_total, total_ms=_t_total)
            return A, F_host
        return A, self.F_d.numpy()

    def assemble_fill(self, *a, **k):
        """Numeric fill ONLY (vals_d/F_d updated in place, nothing
        returned/pulled) — for solvers consuming the device buffers
        directly (device_operator() + F_d; the fused-Krylov step path)."""
        self._return_device = "raw"
        try:
            self.assemble(*a, **k)
        finally:
            self._return_device = False

    def assemble_handoff(self, *a, **k):
        """W2c: like assemble() but returns ``(DeviceSaddleCSR, F_host)``.

        The assembled values stay DEVICE-RESIDENT (no ChunkedArray.numpy()
        pull) — the killer 19 GB device->host round-trip W2b profiled.  Only
        the small rhs vector is pulled.  The fgmres_bdiag / fused_bdiag paths
        in solve_linear recognize DeviceSaddleCSR and consume vals_d via a
        Warp SpMV + a device-gathered diagonal.  splu (or any host caller)
        transparently falls back to the full pull via .tocsr()."""
        self._return_device = "handoff"
        try:
            return self.assemble(*a, **k)
        finally:
            self._return_device = False

    def assemble_device(self, *a, **k):
        """M1d D3: like assemble() but returns a DEVICE-RESIDENT torch
        CSR (dlpack zero-copy over vals_d) + device rhs — feed directly
        to nvmath DirectSolver; no host round-trip. Measured: agreement
        1e-15, faster than the host path (0.29 vs 0.40 s at 2D L7)."""
        self._return_device = True
        try:
            return self.assemble(*a, **k)
        finally:
            self._return_device = False

    # ------------------------------------------------------------------
    # M4 generic fill API (physics-agnostic): a stepper whose element
    # blocks come from its OWN kernel (e.g. the Wodo film's 4-dof CH
    # Newton) reuses the slot maps directly — zero_fill() once per
    # iterate, scatter_bin() per bin, optional add_matrix_values()/
    # add_rhs_values() for small host-side extras (boundary fluxes),
    # then device_csr() for the zero-copy torch CSR -> cuDSS.
    # ------------------------------------------------------------------
    def zero_fill(self):
        """Begin a numeric fill: zero device CSR values + rhs."""
        self.vals_d.zero_()
        self.F_d.zero_()

    def _scatter_A(self, vals_e, slots_d, n):
        """Atomic-add vals_e into the device CSR values at nnz-space
        slots — the ONE dispatch point between the flat (pre-#38,
        bit-for-bit) and chunked scatter kernels."""
        if self._chunked:
            wp.launch(_scatter_kernel_chunked(), dim=n,
                      inputs=[vals_e, slots_d, self.vals_d.data,
                              self._ctab.bases_d,
                              wp.int32(self._ctab.nchunks)],
                      device=self.dm.device)
        else:
            wp.launch(_scatter_kernel(self._idx_dtype), dim=n,
                      inputs=[vals_e, slots_d, self.vals_d],
                      device=self.dm.device)

    def scatter_batch(self, k_bin, e0, Ae_d, be_d, nb):
        """Scatter BATCH-LOCAL device element blocks for the nb elements
        [e0, e0 + nb) of bin k_bin (Ae_d flat or [>=nb, nl, nl], be_d
        flat or [>=nb, nl]) into vals_d / F_d. This is the launch unit
        that keeps (a) the Ae transient bounded (~2 GB at full res vs
        30 GB whole-mesh) and (b) kernel dims < 2^31. Works in both
        pattern modes; node mode computes dof slots in-kernel."""
        d = self.dm.device
        pv, b, ne, nbf, gdof = self._bins[k_bin]
        ndof = self.ndof
        nl = nbf * ndof
        npair = nl * nl
        if self.node_mode:
            if self._blockmask is not None:
                if self._chunked:
                    wp.launch(_scatter_node_masked_kernel_chunked(),
                              dim=nb * npair,
                              inputs=[Ae_d.reshape((-1,)),
                                      self._gslot_d[k_bin],
                                      self._conn_d[k_bin], self._Gptr_d,
                                      wp.int32(e0), wp.int32(nbf),
                                      wp.int32(ndof),
                                      self._mask_d["rowoff"],
                                      self._mask_d["colpos"],
                                      self.vals_d.data,
                                      self._ctab.bases_d,
                                      wp.int32(self._ctab.nchunks)],
                              device=d)
                else:
                    wp.launch(_scatter_node_masked_kernel(
                                  self._idx_dtype),
                              dim=nb * npair,
                              inputs=[Ae_d.reshape((-1,)),
                                      self._gslot_d[k_bin],
                                      self._conn_d[k_bin], self._Gptr_d,
                                      wp.int32(e0), wp.int32(nbf),
                                      wp.int32(ndof),
                                      self._mask_d["rowoff"],
                                      self._mask_d["colpos"],
                                      self.vals_d], device=d)
            else:
                if self._chunked:
                    wp.launch(_scatter_node_kernel_chunked(),
                              dim=nb * npair,
                              inputs=[Ae_d.reshape((-1,)),
                                      self._gslot_d[k_bin],
                                      self._conn_d[k_bin], self._Gptr_d,
                                      wp.int32(e0), wp.int32(nbf),
                                      wp.int32(ndof), self.vals_d.data,
                                      self._ctab.bases_d,
                                      wp.int32(self._ctab.nchunks)],
                              device=d)
                else:
                    wp.launch(_scatter_node_kernel(self._idx_dtype),
                              dim=nb * npair,
                              inputs=[Ae_d.reshape((-1,)),
                                      self._gslot_d[k_bin],
                                      self._conn_d[k_bin], self._Gptr_d,
                                      wp.int32(e0), wp.int32(nbf),
                                      wp.int32(ndof), self.vals_d],
                              device=d)
            wp.launch(_scatter_node_vec_kernel(), dim=nb * nl,
                      inputs=[be_d.reshape((-1,)), self._conn_d[k_bin],
                              wp.int32(e0), wp.int32(nbf),
                              wp.int32(ndof), self.F_d], device=d)
            return
        if not self._identity_T:
            raise BackendError(
                "scatter_batch: constraint-aware path is whole-bin only")
        slots_v = self._slots_d[k_bin][e0 * npair:(e0 + nb) * npair]
        self._scatter_A(Ae_d.reshape((-1,)), slots_v, nb * npair)
        gdof_v = self._gdof_d[k_bin][e0 * nl:(e0 + nb) * nl]
        wp.launch(_scatter_vec_kernel(), dim=nb * nl,
                  inputs=[be_d.reshape((-1,)), gdof_v, self.F_d],
                  device=d)

    def apply_batch_matvec(self, k_bin, e0, Ae_d, nb, v_d, y_d):
        """Matrix-FREE element apply: for the nb elements [e0, e0+nb) of
        bin k_bin, compute y_e = Ae @ v_e (element-local gather of v via
        conn, dense nl x nl multiply) and scatter-ADD into y_d — the
        matvec analogue of scatter_batch (which writes the SAME Ae into
        the global CSR).  Never touches vals_d: this is the launch unit
        of the matrix-free OUTER operator (no monolithic CSR stored).
        Node-graph pattern only (identity constraints); Ae_d flat or
        [>=nb, nl, nl] batch-local (the make_mpf_newton buffer)."""
        assert self.node_mode, (
            "apply_batch_matvec: node-graph pattern only")
        d = self.dm.device
        pv, b, ne, nbf, gdof = self._bins[k_bin]
        ndof = self.ndof
        nl = nbf * ndof
        wp.launch(_matvec_node_kernel(), dim=nb * nl,
                  inputs=[Ae_d.reshape((-1,)), self._conn_d[k_bin],
                          wp.int32(e0), wp.int32(nbf), wp.int32(ndof),
                          v_d, y_d], device=d)

    def scatter_bin(self, k_bin, Ae_d, be_d):
        """Scatter one bin's device element blocks (Ae [ne, nl, nl],
        be [ne, nl]; nl = nbf*ndof, dof-major layout node*ndof + comp)
        into vals_d / F_d via the precomputed slot maps."""
        d = self.dm.device
        pv, b, ne, nbf, gdof = self._bins[k_bin]
        npair = (nbf * self.ndof) ** 2
        if self.node_mode:
            # chunk launches: dim = ne*npair can exceed 2^31 at full res
            nl = nbf * self.ndof
            Ae_f = Ae_d.reshape((-1,))
            be_f = be_d.reshape((-1,))
            step = max(1, (2 ** 31 - 1) // npair)
            for e0 in range(0, ne, step):
                nb = min(step, ne - e0)
                self.scatter_batch(
                    k_bin, e0, Ae_f[e0 * npair:(e0 + nb) * npair],
                    be_f[e0 * nl:(e0 + nb) * nl], nb)
            return
        if not self._identity_T:
            wp.launch(_scatter_weighted_kernel(self._idx_dtype),
                      dim=len(self._slot_bins[k_bin]),
                      inputs=[Ae_d.reshape((-1,)), self._src_d[k_bin],
                              self._w_d[k_bin], self._slots_d[k_bin],
                              self.vals_d], device=d)
            wp.launch(_scatter_vec_weighted_kernel(),
                      dim=len(self._exp_bins[k_bin][4]),
                      inputs=[be_d.reshape((-1,)), self._bsrc_d[k_bin],
                              self._bw_d[k_bin], self._gdof_d[k_bin],
                              self.F_d], device=d)
        else:
            self._scatter_A(Ae_d.reshape((-1,)), self._slots_d[k_bin],
                            ne * npair)
            wp.launch(_scatter_vec_kernel(), dim=ne * nbf * self.ndof,
                      inputs=[be_d.reshape((-1,)), self._gdof_d[k_bin],
                              self.F_d], device=d)

    def apply_strong_rows(self, b_vals=None):
        """Apply the set_strong_rows plan to the CURRENT fill (device):
        zero the strong-row slot spans, write unit diagonals, and set
        F[row] = b_vals[i] (replacement, not add — the row equation
        becomes x_row = b_val exactly).  Part of the generic fill API
        (zero_fill / scatter_bin / add_* / apply_strong_rows /
        device_csr) so physics steppers with their OWN element kernels
        (the M5 multiphase Newton) realize the same strong-row
        semantics as assemble()."""
        st = self._strong
        bv = wp.array(np.ascontiguousarray(
            b_vals if b_vals is not None
            else np.zeros(st["n_rows"])), dtype=wp.float64,
            device=self.dm.device)
        if self._chunked:
            ct = self._ctab
            wp.launch(_zero_slots_kernel_chunked(), dim=st["n_spans"],
                      inputs=[st["spans_d"], self.vals_d.data,
                              ct.bases_d, wp.int32(ct.nchunks)],
                      device=self.dm.device)
            wp.launch(_diag_one_kernel_chunked(), dim=st["n_rows"],
                      inputs=[st["diag_d"], st["rows_d"], bv,
                              self.vals_d.data, ct.bases_d,
                              wp.int32(ct.nchunks), self.F_d],
                      device=self.dm.device)
            return
        wp.launch(_zero_slots_kernel(self._idx_dtype), dim=st["n_spans"],
                  inputs=[st["spans_d"], self.vals_d],
                  device=self.dm.device)
        wp.launch(_diag_one_kernel(self._idx_dtype), dim=st["n_rows"],
                  inputs=[st["diag_d"], st["rows_d"], bv,
                          self.vals_d, self.F_d],
                  device=self.dm.device)

    def csr_slots(self, rows, cols):
        """CSR value index per (row, col) pair. Entries MUST exist in
        the symbolic pattern (dof pairs sharing an element — true for
        boundary-face pairs). Identity-T dof numbering."""
        rows = np.asarray(rows, np.int64)
        cols = np.asarray(cols, np.int64)
        slots = np.empty(len(rows), np.int64)
        for i in range(len(rows)):
            s_, e_ = self.indptr[rows[i]], self.indptr[rows[i] + 1]
            k = s_ + np.searchsorted(self.indices[s_:e_], cols[i])
            if not (k < e_ and self.indices[k] == cols[i]):
                raise BackendError(
                    f"CSR slot lookup failed for entry "
                    f"(row={rows[i]}, col={cols[i]})")
            slots[i] = k
        return slots

    def add_matrix_values(self, slots_d, vals_d):
        """Atomic-add values (device array) at CSR slots (device).
        slots_d must carry the assembler's CSR index dtype
        (self._idx_dtype: int32 narrow, int64 wide/chunked) —
        csr_slots() returns host int64 slots the caller casts to that
        width."""
        self._scatter_A(vals_d, slots_d, len(vals_d))

    def add_rhs_values(self, dofs_d, vals_d):
        """Atomic-add values (device array) into F_d at dof rows."""
        wp.launch(_scatter_vec_kernel(), dim=len(vals_d),
                  inputs=[vals_d, dofs_d, self.F_d],
                  device=self.dm.device)

    def device_operator(self):
        """Operator-protocol view over the DEVICE-RESIDENT CSR (vals_d
        zero-copy; indptr/indices uploaded once per epoch): .matvec(x_wp,
        y_wp) for the fused Krylov — the direct-solver fallback when the
        factorization exceeds HBM (measured: cuDSS ALLOC_FAILED at 3-D
        L6, 1.1M dofs, on a 48 GB card — default AND hybrid memory mode;
        the GH200 capacity question made concrete)."""
        if not hasattr(self, "_op_idx"):
            from .operators import make_csr_spmv, make_csr_spmv_chunked
            # indptr (row offsets) indexes nnz-space -> CSR index width;
            # indices are dof-space (int32).  The spmv kernel specializes
            # on the offset dtype so the inner loop bound never wraps.
            # Chunked (#38): the column indices go up as a ChunkedArray
            # sharing vals_d's chunk table (a >2^31-element 1-D upload
            # cannot exist) and the chunked SpMV walks within-chunk.
            if self._chunked:
                ind_ch = ChunkedArray(self._ctab, wp.int32,
                                      self.dm.device)
                ind_ch.upload(np.asarray(self.indices, np.int32))
                self._op_idx = (
                    wp.array(self.indptr.astype(np.int64),
                             dtype=wp.int64, device=self.dm.device),
                    ind_ch)
                self._op_spmv = make_csr_spmv_chunked()
            else:
                self._op_idx = (
                    wp.array(self.indptr.astype(self._idx_np),
                             dtype=self._idx_dtype,
                             device=self.dm.device),
                    wp.array(self.indices.astype(np.int32),
                             dtype=wp.int32, device=self.dm.device))
                self._op_spmv = make_csr_spmv(self._idx_dtype)
        asm = self

        class _Op:
            device = asm.dm.device
            n_free = asm.Nfull

            def matvec(self, x, y):
                if asm._chunked:
                    wp.launch(asm._op_spmv, dim=asm.Nfull,
                              inputs=[asm._op_idx[0],
                                      asm._op_idx[1].data,
                                      asm.vals_d.data,
                                      asm._ctab.bases_d,
                                      wp.int32(asm._ctab.nchunks),
                                      x, y],
                              device=asm.dm.device)
                else:
                    wp.launch(asm._op_spmv, dim=asm.Nfull,
                              inputs=[asm._op_idx[0], asm._op_idx[1],
                                      asm.vals_d, x, y],
                              device=asm.dm.device)

        return _Op()

    def diagonal_device(self):
        """Current matrix diagonal as a DEVICE wp.array (no host pull).

        Slot ids (nnz-space positions of the diagonal entries) are static per
        mesh epoch — computed once host-side from the CSR pattern — then each
        call is a pure device gather into a reused ``_diag_d`` buffer.  This is
        the W2c device-resident preconditioner-diagonal source; ``diag_host``
        is the same gather followed by one small (Nfull*8 B) download."""
        if not hasattr(self, "_diag_slots_d"):
            probe = sp.csr_matrix(
                (np.arange(self.nnz, dtype=np.float64), self.indices,
                 self.indptr), shape=(self.Nfull, self.Nfull))
            slots = probe.diagonal().astype(np.int64)
            self._diag_slots_d = wp.array(slots.astype(self._idx_np),
                                          dtype=self._idx_dtype,
                                          device=self.dm.device)
            self._diag_d = wp.zeros(self.Nfull, dtype=wp.float64,
                                    device=self.dm.device)
        if self._chunked:
            wp.launch(_gather_kernel_chunked(), dim=self.Nfull,
                      inputs=[self.vals_d.data, self._diag_slots_d,
                              self._ctab.bases_d,
                              wp.int32(self._ctab.nchunks),
                              self._diag_d], device=self.dm.device)
        else:
            wp.launch(_gather_kernel(self._idx_dtype), dim=self.Nfull,
                      inputs=[self.vals_d, self._diag_slots_d,
                              self._diag_d], device=self.dm.device)
        return self._diag_d

    def diag_host(self):
        """Current matrix diagonal (host, for the Krylov Jacobi
        preconditioner): slot ids once per epoch, then a device gather +
        one small download per step."""
        return self.diagonal_device().numpy()

    def device_csr(self):
        """Zero-copy torch CSR over vals_d + device rhs (dlpack).

        Task #37 fix: the indptr/indices tensors were hardcoded to
        torch device "cuda" (= cuda:0) while the dlpack'd values ride
        the assembler's ACTUAL warp device — on cuda:1 the mixed-device
        CSR made every cuDSS solve fail (swallowed as a numerical NaN
        -> instant dt-underflow ladder; measured on the film front-end,
        runs/t37-g3-dev).  The torch device now follows dm.device.

        Task #38 chunked mode: cuDSS needs ONE contiguous values
        tensor, so the chunk rows are CONCATENATED into a torch f64
        [nnz] tensor (torch sizes are int64 — no 2^31 ceiling).  This
        COSTS the zero-copy contract: +nnz*8 bytes resident and a
        device-to-device copy per refresh — callers holding the tensor
        across fills must call sync_csr_values() before each factorize
        (the blockch path never materializes this)."""
        import torch
        if not hasattr(self, "_indptr_t"):
            tdev = str(self.dm.device)
            self._indptr_t = torch.tensor(self.indptr, dtype=torch.int64,
                                          device=tdev)
            self._indices_t = torch.tensor(self.indices,
                                           dtype=torch.int64,
                                           device=tdev)
        if self._chunked:
            if not hasattr(self, "_vals_t"):
                self._vals_t = torch.empty(
                    self.nnz, dtype=torch.float64,
                    device=str(self.dm.device))
            self.sync_csr_values()
            vals_t = self._vals_t
        else:
            vals_t = torch.from_dlpack(self.vals_d.__dlpack__())
        A_t = torch.sparse_csr_tensor(
            self._indptr_t, self._indices_t, vals_t,
            size=(self.Nfull, self.Nfull))
        return A_t, torch.from_dlpack(self.F_d.__dlpack__())

    def device_csr_fp32(self):
        """Task #36: torch fp32 CSR over the round-on-store snapshot, for
        the cuDSS FP32 factorization.  Refreshes _vals_fp32 from vals_d
        (a cast — vals_d itself stays fp64), then wraps it as a torch
        float32 sparse CSR sharing the (int64) indptr/indices tensors.

        The values tensor is STABLE across fills (refreshed in place), so
        the caller plans the fp32 DirectSolver ONCE and only refactorizes
        per iterate — the #37 stable-operand contract, in fp32.  The
        FP64 iterative-refinement residual rides device_operator() (fp64,
        against vals_d), NOT this fp32 CSR (brief scope 2).  Requires
        val_dtype='fp32' (else there is no snapshot)."""
        if self._vals_fp32 is None:
            raise BackendError(
                "device_csr_fp32() requires val_dtype='fp32' (no fp32 "
                "snapshot exists for a fp64 assembler)")
        import torch
        self.refresh_fp32_snapshot()
        tdev = str(self.dm.device)
        if not hasattr(self, "_indptr_t"):
            self._indptr_t = torch.tensor(self.indptr, dtype=torch.int64,
                                          device=tdev)
            self._indices_t = torch.tensor(self.indices,
                                           dtype=torch.int64, device=tdev)
        if self._chunked:
            # per-chunk concat into a STABLE contiguous fp32 tensor (the
            # #38 fp64 path's fp32 sibling); cuDSS needs one 1-D buffer.
            if not hasattr(self, "_vals_t32"):
                self._vals_t32 = torch.empty(self.nnz, dtype=torch.float32,
                                             device=tdev)
            ct = self._ctab
            for c in range(ct.nchunks):
                b0, b1 = int(ct.bases[c]), int(ct.bases[c + 1])
                if b1 == b0:
                    continue
                row = torch.from_dlpack(
                    self._vals_fp32.data[c].__dlpack__())
                self._vals_t32[b0:b1].copy_(row[:b1 - b0])
            vals_t = self._vals_t32
        else:
            vals_t = torch.from_dlpack(self._vals_fp32.__dlpack__())
        A_t = torch.sparse_csr_tensor(
            self._indptr_t, self._indices_t, vals_t,
            size=(self.Nfull, self.Nfull))
        return A_t, torch.from_dlpack(self.F_d.__dlpack__())

    def sync_csr_values(self):
        """Chunked mode (#38): refresh the contiguous torch values
        tensor from the chunked vals_d (device-to-device, per chunk).
        No-op when unchunked (values are a zero-copy dlpack view) or
        before device_csr() ever ran."""
        if not self._chunked or not hasattr(self, "_vals_t"):
            return
        import torch
        ct = self._ctab
        for c in range(ct.nchunks):
            b0, b1 = int(ct.bases[c]), int(ct.bases[c + 1])
            if b1 == b0:
                continue
            row = torch.from_dlpack(self.vals_d.data[c].__dlpack__())
            self._vals_t[b0:b1].copy_(row[:b1 - b0])


# ---------------------------------------------------------------------
# Task #36: fp32 round-on-store snapshot kernels.  vals_d (fp64) is cast
# element-wise into _vals_fp32 (fp32) — the ONLY place fp32 values are
# produced, and it is a pure cast of already-accumulated fp64 sums (never
# an fp32 accumulation).  Compiled only when val_dtype='fp32'.
# ---------------------------------------------------------------------
def _round_store_kernel():
    key = ("dev_round_store",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique", enable_backward=False)
    def rs(vals64: wp.array(dtype=wp.float64),
           out32: wp.array(dtype=wp.float32)):
        i = wp.tid()
        out32[i] = wp.float32(vals64[i])

    _kernel_cache[key] = rs
    return rs


def _round_store_kernel_chunked():
    """Chunked (#38) round-store: both buffers share the [nchunks, cap]
    2-D layout, so a straight element-wise cast over every slot (padding
    tails are cast 0.0->0.0, never addressed by any consumer)."""
    key = ("dev_round_store_ch",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique", enable_backward=False)
    def rsc(vals64: wp.array2d(dtype=wp.float64),
            out32: wp.array2d(dtype=wp.float32)):
        c, j = wp.tid()
        out32[c, j] = wp.float32(vals64[c, j])

    _kernel_cache[key] = rsc
    return rsc


def _gather_kernel(idx_dtype=wp.int32):
    # slots[] holds nnz-space positions -> its element type widens with
    # the CSR; narrow (int32) keeps the original cache key untouched.
    key = ("dev_gather",) if idx_dtype is wp.int32 else ("dev_gather64",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def gat(vals: wp.array(dtype=wp.float64),
            slots: wp.array(dtype=idx_dtype),
            out: wp.array(dtype=wp.float64)):
        i = wp.tid()
        out[i] = vals[slots[i]]

    _kernel_cache[key] = gat
    return gat


def _scatter_kernel(idx_dtype=wp.int32):
    key = ("dev_scatter",) if idx_dtype is wp.int32 else ("dev_scatter64",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def scat(vals_e: wp.array(dtype=wp.float64),
             slots: wp.array(dtype=idx_dtype),
             out: wp.array(dtype=wp.float64)):
        i = wp.tid()
        wp.atomic_add(out, slots[i], vals_e[i])

    _kernel_cache[key] = scat
    return scat


# ---------------------------------------------------------------------
# Task #38 chunked kernel variants: identical arithmetic to the wide
# (int64) kernels above; ONLY the final memory access changes — the
# int64 global slot is located in the block-row chunk table (binary
# search, _chunk_of) and addressed [c, int32(slot - bases[c])] in the
# 2-D [nchunks, cap] buffer.  Compiled only when chunking is active
# (own cache keys/modules); the flat narrow/wide kernels are untouched.
# ---------------------------------------------------------------------
def _scatter_kernel_chunked():
    key = ("dev_scatter_ch",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def scat(vals_e: wp.array(dtype=wp.float64),
             slots: wp.array(dtype=wp.int64),
             out: wp.array2d(dtype=wp.float64),
             bases: wp.array(dtype=wp.int64),
             nc: wp.int32):
        i = wp.tid()
        s = slots[i]
        c = _chunk_of(bases, nc, s)
        wp.atomic_add(out, c, wp.int32(s - bases[c]), vals_e[i])

    _kernel_cache[key] = scat
    return scat


def _gather_kernel_chunked():
    key = ("dev_gather_ch",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def gat(vals: wp.array2d(dtype=wp.float64),
            slots: wp.array(dtype=wp.int64),
            bases: wp.array(dtype=wp.int64),
            nc: wp.int32,
            out: wp.array(dtype=wp.float64)):
        i = wp.tid()
        s = slots[i]
        c = _chunk_of(bases, nc, s)
        out[i] = vals[c, wp.int32(s - bases[c])]

    _kernel_cache[key] = gat
    return gat


def _zero_slots_kernel_chunked():
    key = ("dev_zero_slots_ch",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def zk(slots: wp.array(dtype=wp.int64),
           vals: wp.array2d(dtype=wp.float64),
           bases: wp.array(dtype=wp.int64),
           nc: wp.int32):
        i = wp.tid()
        s = slots[i]
        c = _chunk_of(bases, nc, s)
        vals[c, wp.int32(s - bases[c])] = wp.float64(0.0)

    _kernel_cache[key] = zk
    return zk


def _diag_one_kernel_chunked():
    key = ("dev_diag_one_ch",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def dk(diag: wp.array(dtype=wp.int64),
           rows: wp.array(dtype=wp.int32),
           bvals: wp.array(dtype=wp.float64),
           vals: wp.array2d(dtype=wp.float64),
           bases: wp.array(dtype=wp.int64),
           nc: wp.int32,
           F: wp.array(dtype=wp.float64)):
        i = wp.tid()
        s = diag[i]
        c = _chunk_of(bases, nc, s)
        vals[c, wp.int32(s - bases[c])] = wp.float64(1.0)
        F[rows[i]] = bvals[i]

    _kernel_cache[key] = dk
    return dk


def _scatter_node_kernel_chunked():
    """Chunked variant of _scatter_node_kernel (wide arithmetic): the
    int64 closed-form slot is chunk-located before the atomic — the
    ONLY difference from the int64 flat kernel."""
    key = ("dev_scatter_node_ch",)
    if key in _kernel_cache:
        return _kernel_cache[key]
    IX = wp.int64

    @wp.kernel(module="unique")
    def scn(vals_e: wp.array(dtype=wp.float64),
            gslot: wp.array(dtype=wp.int32),
            conn: wp.array2d(dtype=wp.int32),
            Gptr: wp.array(dtype=IX),
            e0: wp.int32, nbf: wp.int32, ndof: wp.int32,
            out: wp.array2d(dtype=wp.float64),
            bases: wp.array(dtype=wp.int64),
            nc: wp.int32):
        i = wp.tid()
        nl = nbf * ndof
        npair = nl * nl
        el = i / npair
        rem = i % npair
        rl = rem / nl
        cl = rem % nl
        a = rl / ndof
        ca = rl % ndof
        bb = cl / ndof
        cb = cl % ndof
        e = e0 + el
        s = gslot[e * nbf * nbf + a * nbf + bb]
        na = conn[e, a]
        g0 = Gptr[na]                        # nnz-space (int64)
        dnb = Gptr[na + 1] - g0
        slot = IX(ndof) * IX(ndof) * g0 + IX(ca) * IX(ndof) * dnb \
            + IX(ndof) * (IX(s) - g0) + IX(cb)
        c = _chunk_of(bases, nc, slot)
        wp.atomic_add(out, c, wp.int32(slot - bases[c]), vals_e[i])

    _kernel_cache[key] = scn
    return scn


def _scatter_node_masked_kernel_chunked():
    """Chunked variant of _scatter_node_masked_kernel — same masked
    slot arithmetic (int64), chunk-located atomic."""
    key = ("dev_scatter_node_masked_ch",)
    if key in _kernel_cache:
        return _kernel_cache[key]
    IX = wp.int64

    @wp.kernel(module="unique")
    def scnm(vals_e: wp.array(dtype=wp.float64),
             gslot: wp.array(dtype=wp.int32),
             conn: wp.array2d(dtype=wp.int32),
             Gptr: wp.array(dtype=IX),
             e0: wp.int32, nbf: wp.int32, ndof: wp.int32,
             rowoff: wp.array(dtype=wp.int32),
             colpos: wp.array(dtype=wp.int32),
             out: wp.array2d(dtype=wp.float64),
             bases: wp.array(dtype=wp.int64),
             nc: wp.int32):
        i = wp.tid()
        nl = nbf * ndof
        npair = nl * nl
        el = i / npair
        rem = i % npair
        rl = rem / nl
        cl = rem % nl
        a = rl / ndof
        ca = rl % ndof
        bb = cl / ndof
        cb = cl % ndof
        j = colpos[ca * ndof + cb]
        if j >= 0:
            e = e0 + el
            s = gslot[e * nbf * nbf + a * nbf + bb]
            na = conn[e, a]
            g0 = Gptr[na]                    # nnz-space (int64)
            dnb = Gptr[na + 1] - g0
            ro = rowoff[ca]
            rc = rowoff[ca + 1] - ro
            slot = IX(rowoff[ndof]) * g0 + IX(ro) * dnb \
                + IX(rc) * (IX(s) - g0) + IX(j)
            c = _chunk_of(bases, nc, slot)
            wp.atomic_add(out, c, wp.int32(slot - bases[c]), vals_e[i])

    _kernel_cache[key] = scnm
    return scnm


def _dof_indices_kernel_chunked():
    """Chunked variant of _dof_indices_kernel: one thread per dof row;
    the row's whole span lives in ONE chunk (ChunkTable is row-aligned)
    so the chunk is located once and the writes are int32 within-chunk.
    Column VALUES stay int32."""
    key = ("dev_dof_indices_ch",)
    if key in _kernel_cache:
        return _kernel_cache[key]
    IX = wp.int64

    @wp.kernel(module="unique")
    def dik(Gptr: wp.array(dtype=IX),
            Gind: wp.array(dtype=wp.int32),
            ndof: wp.int32,
            out: wp.array2d(dtype=wp.int32),
            bases: wp.array(dtype=wp.int64),
            nc: wp.int32):
        r = wp.tid()
        na = r / ndof
        ca = r % ndof
        g0 = Gptr[na]                       # nnz-space (int64)
        dnb = wp.int32(Gptr[na + 1] - g0)
        base = IX(ndof) * IX(ndof) * g0 + IX(ca) * IX(ndof) * IX(dnb)
        c = _chunk_of(bases, nc, base)
        lb = wp.int32(base - bases[c])
        for q in range(dnb):
            col = ndof * Gind[g0 + IX(q)]
            for cb in range(ndof):
                out[c, lb + ndof * q + cb] = col + cb

    _kernel_cache[key] = dik
    return dik


def _dof_indices_masked_kernel_chunked():
    """Chunked variant of _dof_indices_masked_kernel (row-aligned
    chunks: one lookup per dof row, int32 within-chunk writes)."""
    key = ("dev_dof_indices_masked_ch",)
    if key in _kernel_cache:
        return _kernel_cache[key]
    IX = wp.int64

    @wp.kernel(module="unique")
    def dikm(Gptr: wp.array(dtype=IX),
             Gind: wp.array(dtype=wp.int32),
             ndof: wp.int32,
             rowoff: wp.array(dtype=wp.int32),
             lcols: wp.array(dtype=wp.int32),
             out: wp.array2d(dtype=wp.int32),
             bases: wp.array(dtype=wp.int64),
             nc: wp.int32):
        r = wp.tid()
        na = r / ndof
        ca = r % ndof
        g0 = Gptr[na]                       # nnz-space (int64)
        dnb = wp.int32(Gptr[na + 1] - g0)   # degree fits int32
        ro = rowoff[ca]
        rc = rowoff[ca + 1] - ro
        base = IX(rowoff[ndof]) * g0 + IX(ro) * IX(dnb)
        c = _chunk_of(bases, nc, base)
        lb = wp.int32(base - bases[c])
        for q in range(dnb):
            col = ndof * Gind[g0 + IX(q)]
            for j in range(rc):
                out[c, lb + rc * q + j] = col + lcols[ro + j]

    _kernel_cache[key] = dikm
    return dikm


def _scatter_colored_kernel(idx_dtype=wp.int32):
    # slots[] index nnz-space -> widen; idx (element-local read) stays
    # int32-representable (bounded by ne*npair per bin, chunked).
    key = ("dev_scatter_col",) if idx_dtype is wp.int32 \
        else ("dev_scatter_col64",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def scatc(vals_e: wp.array(dtype=wp.float64),
              slots: wp.array(dtype=idx_dtype),
              order: wp.array(dtype=wp.int32),
              lo: wp.int32, npair: wp.int32,
              out: wp.array(dtype=wp.float64)):
        i = wp.tid()
        e = order[lo + i / npair]
        j = i % npair
        idx = e * npair + j
        out[slots[idx]] = out[slots[idx]] + vals_e[idx]   # atomic-free

    _kernel_cache[key] = scatc
    return scatc


def _scatter_vec_kernel():
    key = ("dev_scatter_vec",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def scatv(vals_e: wp.array(dtype=wp.float64),
              gdof: wp.array(dtype=wp.int32),
              out: wp.array(dtype=wp.float64)):
        i = wp.tid()
        wp.atomic_add(out, gdof[i], vals_e[i])

    _kernel_cache[key] = scatv
    return scatv


def _zero_slots_kernel(idx_dtype=wp.int32):
    # spans[] index nnz-space -> widen with the CSR.
    key = ("dev_zero_slots",) if idx_dtype is wp.int32 \
        else ("dev_zero_slots64",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def zk(slots: wp.array(dtype=idx_dtype),
           vals: wp.array(dtype=wp.float64)):
        i = wp.tid()
        vals[slots[i]] = wp.float64(0.0)

    _kernel_cache[key] = zk
    return zk


def _diag_one_kernel(idx_dtype=wp.int32):
    # diag[] index nnz-space (rows[] is dof-space, stays int32).
    key = ("dev_diag_one",) if idx_dtype is wp.int32 \
        else ("dev_diag_one64",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def dk(diag: wp.array(dtype=idx_dtype),
           rows: wp.array(dtype=wp.int32),
           bvals: wp.array(dtype=wp.float64),
           vals: wp.array(dtype=wp.float64),
           F: wp.array(dtype=wp.float64)):
        i = wp.tid()
        vals[diag[i]] = wp.float64(1.0)
        F[rows[i]] = bvals[i]

    _kernel_cache[key] = dk
    return dk


def _scatter_weighted_kernel(idx_dtype=wp.int32):
    # slots[] index nnz-space; src[] is element-block-local (int32).
    key = ("dev_scatter_w",) if idx_dtype is wp.int32 \
        else ("dev_scatter_w64",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def scw(vals_e: wp.array(dtype=wp.float64),
            src: wp.array(dtype=wp.int32),
            w: wp.array(dtype=wp.float64),
            slots: wp.array(dtype=idx_dtype),
            out: wp.array(dtype=wp.float64)):
        i = wp.tid()
        wp.atomic_add(out, slots[i], w[i] * vals_e[src[i]])

    _kernel_cache[key] = scw
    return scw


def _scatter_weighted_kernel_chunked():
    """W2: chunked variant of _scatter_weighted_kernel (the constraint-
    expansion matrix scatter).  Same arithmetic as the wide (int64) flat
    kernel — the constraint weight w[i] times the element-block value
    vals_e[src[i]] — with the int64 global slot chunk-located before the
    atomic (mirrors _scatter_kernel_chunked).  src[] is BATCH-LOCAL into
    vals_e (the per-chunk element-block slice), slots[] index nnz-space."""
    key = ("dev_scatter_w_ch",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def scwc(vals_e: wp.array(dtype=wp.float64),
             src: wp.array(dtype=wp.int32),
             w: wp.array(dtype=wp.float64),
             slots: wp.array(dtype=wp.int64),
             out: wp.array2d(dtype=wp.float64),
             bases: wp.array(dtype=wp.int64),
             nc: wp.int32):
        i = wp.tid()
        s = slots[i]
        c = _chunk_of(bases, nc, s)
        wp.atomic_add(out, c, wp.int32(s - bases[c]), w[i] * vals_e[src[i]])

    _kernel_cache[key] = scwc
    return scwc


def _dof_indices_kernel(idx_dtype=wp.int32):
    """dof-level CSR indices from the node graph, one thread per dof
    row r = ndof*na + ca: entries (q, cb) get column ndof*Gind[g0+q]+cb
    at indptr[r] + ndof*q + cb, indptr[r] = ndof^2*Gptr[na]
    + ca*ndof*deg(na).  The WRITE POSITION (base + ndof*q + cb) is in
    nnz-space and can exceed 2^31 — in wide mode it is computed in
    int64 (Gptr is uploaded as int64 too, so g0 is already wide) so no
    arithmetic wraps.  Column VALUES stay int32 (out.dtype)."""
    key = ("dev_dof_indices",) if idx_dtype is wp.int32 \
        else ("dev_dof_indices64",)
    if key in _kernel_cache:
        return _kernel_cache[key]
    IX = idx_dtype

    @wp.kernel(module="unique")
    def dik(Gptr: wp.array(dtype=IX),
            Gind: wp.array(dtype=wp.int32),
            ndof: wp.int32,
            out: wp.array(dtype=wp.int32)):
        r = wp.tid()
        na = r / ndof
        ca = r % ndof
        g0 = Gptr[na]                       # nnz-space (IX)
        # degree fits int32 (used as the range bound); offsets stay IX.
        dnb = wp.int32(Gptr[na + 1] - g0)
        base = IX(ndof) * IX(ndof) * g0 + IX(ca) * IX(ndof) * IX(dnb)
        for q in range(dnb):
            col = ndof * Gind[g0 + IX(q)]
            for cb in range(ndof):
                out[base + IX(ndof) * IX(q) + IX(cb)] = col + cb

    _kernel_cache[key] = dik
    return dik


def _dof_indices_masked_kernel(idx_dtype=wp.int32):
    """Masked variant of _dof_indices_kernel (pattern = kron(G, mask)):
    dof row r = ndof*na + ca has deg(na) * rowcnt[ca] entries; the
    entry for neighbor q and the j-th live col of block-row ca sits at
    indptr[r] + rowcnt[ca]*q + j, indptr[r] = blocknnz*Gptr[na]
    + rowoff[ca]*deg(na); blocknnz = rowoff[ndof].  Write positions are
    nnz-space (int64 in wide mode); column values stay int32."""
    key = ("dev_dof_indices_masked",) if idx_dtype is wp.int32 \
        else ("dev_dof_indices_masked64",)
    if key in _kernel_cache:
        return _kernel_cache[key]
    IX = idx_dtype

    @wp.kernel(module="unique")
    def dikm(Gptr: wp.array(dtype=IX),
             Gind: wp.array(dtype=wp.int32),
             ndof: wp.int32,
             rowoff: wp.array(dtype=wp.int32),
             lcols: wp.array(dtype=wp.int32),
             out: wp.array(dtype=wp.int32)):
        r = wp.tid()
        na = r / ndof
        ca = r % ndof
        g0 = Gptr[na]                       # nnz-space (IX)
        dnb = wp.int32(Gptr[na + 1] - g0)   # degree fits int32
        ro = rowoff[ca]
        rc = rowoff[ca + 1] - ro
        base = IX(rowoff[ndof]) * g0 + IX(ro) * IX(dnb)
        for q in range(dnb):
            col = ndof * Gind[g0 + IX(q)]
            for j in range(rc):
                out[base + IX(rc) * IX(q) + IX(j)] = col + lcols[ro + j]

    _kernel_cache[key] = dikm
    return dikm


def _scatter_node_masked_kernel(idx_dtype=wp.int32):
    """Masked variant of _scatter_node_kernel: local pairs whose
    (ca, cb) block is masked out are SKIPPED (the element kernel is
    contractually zero there — exactness gated masked-vs-unmasked);
    live pairs land at blocknnz*Gptr[na] + rowoff[ca]*deg(na)
    + rowcnt[ca]*(s - g0) + colpos[ca, cb].  In wide mode Gptr is
    int64 (so g0 is nnz-space) and the slot is computed in int64 so
    the atomic_add index never wraps."""
    key = ("dev_scatter_node_masked",) if idx_dtype is wp.int32 \
        else ("dev_scatter_node_masked64",)
    if key in _kernel_cache:
        return _kernel_cache[key]
    IX = idx_dtype

    @wp.kernel(module="unique")
    def scnm(vals_e: wp.array(dtype=wp.float64),
             gslot: wp.array(dtype=wp.int32),
             conn: wp.array2d(dtype=wp.int32),
             Gptr: wp.array(dtype=IX),
             e0: wp.int32, nbf: wp.int32, ndof: wp.int32,
             rowoff: wp.array(dtype=wp.int32),
             colpos: wp.array(dtype=wp.int32),
             out: wp.array(dtype=wp.float64)):
        i = wp.tid()
        nl = nbf * ndof
        npair = nl * nl
        el = i / npair
        rem = i % npair
        rl = rem / nl
        cl = rem % nl
        a = rl / ndof
        ca = rl % ndof
        bb = cl / ndof
        cb = cl % ndof
        j = colpos[ca * ndof + cb]
        if j >= 0:
            e = e0 + el
            s = gslot[e * nbf * nbf + a * nbf + bb]
            na = conn[e, a]
            g0 = Gptr[na]                    # nnz-space (IX)
            dnb = Gptr[na + 1] - g0
            ro = rowoff[ca]
            rc = rowoff[ca + 1] - ro
            slot = IX(rowoff[ndof]) * g0 + IX(ro) * dnb \
                + IX(rc) * (IX(s) - g0) + IX(j)
            wp.atomic_add(out, slot, vals_e[i])

    _kernel_cache[key] = scnm
    return scnm


def _scatter_node_kernel(idx_dtype=wp.int32):
    """Closed-form slot scatter (node-graph pattern): the dof slot is
    derived in-kernel from the element node-pair's position in G —
    no ne x (nbf*ndof)^2 slot map exists. Batch-local vals_e for the
    nb elements at global offset e0.  In NARROW mode all slot
    arithmetic stays below 2^31; in WIDE mode Gptr is int64 (g0 is
    nnz-space) and the slot is int64 so nothing wraps past 2^31."""
    key = ("dev_scatter_node",) if idx_dtype is wp.int32 \
        else ("dev_scatter_node64",)
    if key in _kernel_cache:
        return _kernel_cache[key]
    IX = idx_dtype

    @wp.kernel(module="unique")
    def scn(vals_e: wp.array(dtype=wp.float64),
            gslot: wp.array(dtype=wp.int32),
            conn: wp.array2d(dtype=wp.int32),
            Gptr: wp.array(dtype=IX),
            e0: wp.int32, nbf: wp.int32, ndof: wp.int32,
            out: wp.array(dtype=wp.float64)):
        i = wp.tid()
        nl = nbf * ndof
        npair = nl * nl
        el = i / npair
        rem = i % npair
        rl = rem / nl
        cl = rem % nl
        a = rl / ndof
        ca = rl % ndof
        bb = cl / ndof
        cb = cl % ndof
        e = e0 + el
        s = gslot[e * nbf * nbf + a * nbf + bb]
        na = conn[e, a]
        g0 = Gptr[na]                        # nnz-space (IX)
        dnb = Gptr[na + 1] - g0
        slot = IX(ndof) * IX(ndof) * g0 + IX(ca) * IX(ndof) * dnb \
            + IX(ndof) * (IX(s) - g0) + IX(cb)
        wp.atomic_add(out, slot, vals_e[i])

    _kernel_cache[key] = scn
    return scn


def _scatter_node_vec_kernel():
    key = ("dev_scatter_node_vec",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def scnv(vals_e: wp.array(dtype=wp.float64),
             conn: wp.array2d(dtype=wp.int32),
             e0: wp.int32, nbf: wp.int32, ndof: wp.int32,
             out: wp.array(dtype=wp.float64)):
        i = wp.tid()
        nl = nbf * ndof
        el = i / nl
        loc = i % nl
        a = loc / ndof
        c = loc % ndof
        wp.atomic_add(out, conn[e0 + el, a] * ndof + c, vals_e[i])

    _kernel_cache[key] = scnv
    return scnv


def _matvec_node_kernel():
    """Element-local matrix-free matvec (node-graph pattern): one thread
    per output row entry (el, rl).  rl = a*ndof + ca is the local dof;
    its global row is conn[e, a]*ndof + ca — the SAME local->global map
    _scatter_node_kernel uses to place Ae into the CSR.  The thread
    reads the nl-long Ae row and gathers v at the matching global column
    dofs (conn[e, ac]*ndof + cc), accumulates in float64, and atomic-adds
    into y.  Result == (global CSR) @ v to summation-order tolerance."""
    key = ("dev_matvec_node",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def mvn(Ae: wp.array(dtype=wp.float64),
            conn: wp.array2d(dtype=wp.int32),
            e0: wp.int32, nbf: wp.int32, ndof: wp.int32,
            v: wp.array(dtype=wp.float64),
            y: wp.array(dtype=wp.float64)):
        i = wp.tid()
        nl = nbf * ndof
        el = i / nl
        rl = i % nl
        a = rl / ndof
        ca = rl % ndof
        e = e0 + el
        base = el * nl * nl + rl * nl
        acc = wp.float64(0.0)
        for cl in range(nl):
            ac = cl / ndof
            cc = cl % ndof
            acc += Ae[base + cl] * v[conn[e, ac] * ndof + cc]
        wp.atomic_add(y, conn[e, a] * ndof + ca, acc)

    _kernel_cache[key] = mvn
    return mvn


def _scatter_vec_weighted_kernel():
    key = ("dev_scatter_vw",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def scvw(vals_e: wp.array(dtype=wp.float64),
             src: wp.array(dtype=wp.int32),
             w: wp.array(dtype=wp.float64),
             gdof: wp.array(dtype=wp.int32),
             out: wp.array(dtype=wp.float64)):
        i = wp.tid()
        wp.atomic_add(out, gdof[i], w[i] * vals_e[src[i]])

    _kernel_cache[key] = scvw
    return scvw
