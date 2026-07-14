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
import numpy as np
import scipy.sparse as sp
import warp as wp

from .operators import _kernel_cache


# Auto-switch threshold for the node-graph pattern build (G5 rung c):
# above this many dof-pair entries (sum over bins of ne*(nbf*ndof)^2)
# the old host COO/slot build becomes the memory binder (measured
# exit-137 ladder at 62 GB host: 4.12M dofs = 4.2G entries dies; 3.26M
# = 805M entries works, 37 s), so identity-T assemblers flip to the
# node-graph path, which is exactness-gated against the old one.
NODE_PATTERN_AUTO_ENTRIES = 2 * 10 ** 8


class DeviceNSAssembler:
    """Per-epoch object: symbolic pattern + slot maps once; numeric fill
    per step on device."""

    def __init__(self, dm, sigma_like=1.0, coloring=False, ndof=None,
                 node_pattern=None, blockmask=None):
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
            assert node_pattern is not False, (
                "blockmask requires the node-graph pattern")
            node_pattern = True
        if node_pattern is None:
            node_pattern = (identity_T and not coloring
                            and tot_entries > NODE_PATTERN_AUTO_ENTRIES)
        if node_pattern:
            assert identity_T and not coloring, (
                "node-graph pattern requires identity constraints and "
                "no coloring")
            self.Nfull = dm.n_nodes * ndof
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
                src_r.append(e_ * nl * nl + li[a_] * nl + li[b2])
                rows_e.append(fd[a_])
                cols_e.append(fd[b2])
                w_rc.append(ww[a_] * ww[b2])
            exp_bins.append((np.concatenate(src_r),
                             np.concatenate(w_rc),
                             np.concatenate(rows_e),
                             np.concatenate(cols_e),
                             free_dof, m_w, dof_of))
            rows_all.append(np.concatenate(rows_e))
            cols_all.append(np.concatenate(cols_e))
        self._exp_bins = exp_bins
        r = np.concatenate(rows_all)
        c = np.concatenate(cols_all)
        K = sp.coo_matrix((np.ones(len(r)), (r, c)),
                          shape=(self.Nfull, self.Nfull)).tocsr()
        K.sort_indices()
        self.indptr = K.indptr.copy()
        self.indices = K.indices.copy()
        self.nnz = K.nnz
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
                src, w, rr, cc, fd, mw, dof_of = self._exp_bins[k_bin]
                slots = np.asarray(K2[rr, cc]).ravel().astype(np.int64)
                slot_bins.append(slots)
                self._weight_bins.append(w)
                self._src_bins.append(src)
        self._slot_bins = slot_bins
        # device uploads
        self._slots_d = [wp.array(np.ascontiguousarray(
            s.astype(np.int32).ravel()), dtype=wp.int32, device=dm.device)
            for s in slot_bins]
        self._w_d = [None if w is None else wp.array(
            np.ascontiguousarray(w), dtype=wp.float64, device=dm.device)
            for w in self._weight_bins]
        self._src_d = [None if s2 is None else wp.array(
            np.ascontiguousarray(s2.astype(np.int32)), dtype=wp.int32,
            device=dm.device) for s2 in self._src_bins]
        self.vals_d = wp.zeros(self.nnz, dtype=wp.float64,
                               device=dm.device)
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
                src, w, rr, cc, fd, mw, dof_of = self._exp_bins[k_bin]
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
        G = sp.coo_matrix((np.ones(len(r), np.int8), (r, c)),
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
        assert self.nnz < 2 ** 31, (
            f"node-pattern dof nnz {self.nnz} >= 2^31: the int32 device "
            f"slot arithmetic overflows — needs an int64 kernel variant")
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
        # dof-level indices by device kernel (one thread per dof row);
        # int32 mirror kept on host (blockch symbolic setup, csr_slots)
        self._Gptr_d = wp.array(Gptr.astype(np.int32), dtype=wp.int32,
                                device=d)
        Gind_d = wp.array(Gind, dtype=wp.int32, device=d)
        ind_d = wp.zeros(self.nnz, dtype=wp.int32, device=d)
        if mask is None:
            self._mask_d = None
            wp.launch(_dof_indices_kernel(), dim=self.Nfull,
                      inputs=[self._Gptr_d, Gind_d, wp.int32(ndof),
                              ind_d], device=d)
        else:
            i32 = lambda a_: wp.array(a_.astype(np.int32),
                                      dtype=wp.int32, device=d)
            self._mask_d = dict(rowoff=i32(rowoff), lcols=i32(lcols),
                                colpos=i32(colpos),
                                blocknnz=int(rowcnt.sum()))
            wp.launch(_dof_indices_masked_kernel(), dim=self.Nfull,
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
            assert slot.max() < gnnz and (Gkey[slot] == key).all(), \
                "node-pair slot lookup failed"
            self._gslot_d.append(wp.array(slot.astype(np.int32),
                                          dtype=wp.int32, device=d))
            del key, slot
        del Gkey, Gind, rows_all, cols_all
        self.vals_d = wp.zeros(self.nnz, dtype=wp.float64, device=d)
        self.F_d = wp.zeros(self.Nfull, dtype=wp.float64, device=d)

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
        self._strong = dict(
            rows_d=wp.array(rows.astype(np.int32), dtype=wp.int32,
                            device=self.dm.device),
            spans_d=wp.array(spans.astype(np.int32), dtype=wp.int32,
                             device=self.dm.device),
            diag_d=wp.array(diag.astype(np.int32), dtype=wp.int32,
                            device=self.dm.device),
            n_spans=len(spans), n_rows=len(rows))

    # ------------------------------------------------------------------
    def assemble(self, aq_by_bin, div_aq_by_bin, fq_by_bin, nu, sigma,
                 sig2tau=None, s_skew=0.5, strong_b_vals=None,
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
        self.vals_d.zero_()
        self.F_d.zero_()
        if not hasattr(self, "_gaq_d"):
            self._gaq_d = {}
        for k_bin, (pv, b, ne, nbf, gdof) in enumerate(self._bins):
            nqp = b["nqp"]
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
            Ae = wp.zeros((ne, nbf * ndof, nbf * ndof), dtype=wp.float64,
                          device=d)
            be = wp.zeros((ne, nbf * ndof), dtype=wp.float64, device=d)
            kA = make_linear_ns_Ae(nbf, nqp, dm.dim)
            kb = make_linear_ns_be(nbf, nqp, dm.dim)
            wp.launch(kA, dim=ne,
                      inputs=[b["conn"], b["h"], b["N"], b["dN"],
                              b["lapN"],   # G4: complete SUPG/PSPG resu
                              b["w"],
                              aq, dq, gaq, wp.float64(nu),
                              wp.float64(sigma), wp.float64(sig2tau),
                              wp.float64(s_skew), wp.int32(0), Ae],
                      device=d)
            wp.launch(kb, dim=ne,
                      inputs=[b["conn"], b["h"], b["N"], b["dN"], b["w"],
                              aq, fq, wp.float64(nu), wp.float64(sig2tau),
                              be], device=d)
            npair = (nbf * ndof) ** 2
            scat = _scatter_kernel()
            if self.node_mode:
                self.scatter_bin(k_bin, Ae, be)   # A and b together
                continue
            if not self._identity_T:
                scw = _scatter_weighted_kernel()
                wp.launch(scw, dim=len(self._slot_bins[k_bin]),
                          inputs=[Ae.reshape((-1,)), self._src_d[k_bin],
                                  self._w_d[k_bin], self._slots_d[k_bin],
                                  self.vals_d], device=d)
            elif not self.coloring:
                wp.launch(scat, dim=ne * npair,
                          inputs=[Ae.reshape((-1,)), self._slots_d[k_bin],
                                  self.vals_d], device=d)
            else:
                order, bounds = self._colors[k_bin]
                order_d = wp.array(order, dtype=wp.int32, device=d)
                scat_c = _scatter_colored_kernel()
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
                scat_bw = _scatter_vec_weighted_kernel()
                wp.launch(scat_bw, dim=len(self._exp_bins[k_bin][4]),
                          inputs=[be.reshape((-1,)), self._bsrc_d[k_bin],
                                  self._bw_d[k_bin], self._gdof_d[k_bin],
                                  self.F_d], device=d)
        if extra_matrix is not None:
            self.add_matrix_values(*extra_matrix)
        if extra_rhs is not None:
            self.add_rhs_values(*extra_rhs)
        if getattr(self, "_strong", None) is not None:
            self.apply_strong_rows(strong_b_vals)
        if getattr(self, "_return_device", False) == "raw":
            return None                    # fill-only (assemble_fill)
        if getattr(self, "_return_device", False):
            return self.device_csr()
        A = sp.csr_matrix((self.vals_d.numpy(), self.indices,
                           self.indptr), shape=(self.Nfull, self.Nfull))
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
                wp.launch(_scatter_node_masked_kernel(),
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
                wp.launch(_scatter_node_kernel(), dim=nb * npair,
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
        assert self._identity_T, (
            "scatter_batch: constraint-aware path is whole-bin only")
        slots_v = self._slots_d[k_bin][e0 * npair:(e0 + nb) * npair]
        wp.launch(_scatter_kernel(), dim=nb * npair,
                  inputs=[Ae_d.reshape((-1,)), slots_v, self.vals_d],
                  device=d)
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
            wp.launch(_scatter_weighted_kernel(),
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
            wp.launch(_scatter_kernel(), dim=ne * npair,
                      inputs=[Ae_d.reshape((-1,)), self._slots_d[k_bin],
                              self.vals_d], device=d)
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
        wp.launch(_zero_slots_kernel(), dim=st["n_spans"],
                  inputs=[st["spans_d"], self.vals_d],
                  device=self.dm.device)
        bv = wp.array(np.ascontiguousarray(
            b_vals if b_vals is not None
            else np.zeros(st["n_rows"])), dtype=wp.float64,
            device=self.dm.device)
        wp.launch(_diag_one_kernel(), dim=st["n_rows"],
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
            assert k < e_ and self.indices[k] == cols[i], \
                (rows[i], cols[i])
            slots[i] = k
        return slots

    def add_matrix_values(self, slots_d, vals_d):
        """Atomic-add values (device array) at CSR slots (device)."""
        wp.launch(_scatter_kernel(), dim=len(vals_d),
                  inputs=[vals_d, slots_d, self.vals_d],
                  device=self.dm.device)

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
            from .operators import csr_spmv
            self._op_idx = (
                wp.array(self.indptr.astype(np.int32), dtype=wp.int32,
                         device=self.dm.device),
                wp.array(self.indices.astype(np.int32), dtype=wp.int32,
                         device=self.dm.device))
            self._op_spmv = csr_spmv
        asm = self

        class _Op:
            device = asm.dm.device
            n_free = asm.Nfull

            def matvec(self, x, y):
                wp.launch(asm._op_spmv, dim=asm.Nfull,
                          inputs=[asm._op_idx[0], asm._op_idx[1],
                                  asm.vals_d, x, y],
                          device=asm.dm.device)

        return _Op()

    def diag_host(self):
        """Current matrix diagonal (host, for the Krylov Jacobi
        preconditioner): slot ids once per epoch, then a device gather +
        one small download per step."""
        if not hasattr(self, "_diag_slots_d"):
            probe = sp.csr_matrix(
                (np.arange(self.nnz, dtype=np.float64), self.indices,
                 self.indptr), shape=(self.Nfull, self.Nfull))
            slots = probe.diagonal().astype(np.int64)
            self._diag_slots_d = wp.array(slots.astype(np.int32),
                                          dtype=wp.int32,
                                          device=self.dm.device)
            self._diag_d = wp.zeros(self.Nfull, dtype=wp.float64,
                                    device=self.dm.device)
        wp.launch(_gather_kernel(), dim=self.Nfull,
                  inputs=[self.vals_d, self._diag_slots_d, self._diag_d],
                  device=self.dm.device)
        return self._diag_d.numpy()

    def device_csr(self):
        """Zero-copy torch CSR over vals_d + device rhs (dlpack)."""
        import torch
        vals_t = torch.from_dlpack(self.vals_d.__dlpack__())
        if not hasattr(self, "_indptr_t"):
            self._indptr_t = torch.tensor(self.indptr, dtype=torch.int64,
                                          device="cuda")
            self._indices_t = torch.tensor(self.indices,
                                           dtype=torch.int64,
                                           device="cuda")
        A_t = torch.sparse_csr_tensor(
            self._indptr_t, self._indices_t, vals_t,
            size=(self.Nfull, self.Nfull))
        return A_t, torch.from_dlpack(self.F_d.__dlpack__())


def _gather_kernel():
    key = ("dev_gather",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def gat(vals: wp.array(dtype=wp.float64),
            slots: wp.array(dtype=wp.int32),
            out: wp.array(dtype=wp.float64)):
        i = wp.tid()
        out[i] = vals[slots[i]]

    _kernel_cache[key] = gat
    return gat


def _scatter_kernel():
    key = ("dev_scatter",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def scat(vals_e: wp.array(dtype=wp.float64),
             slots: wp.array(dtype=wp.int32),
             out: wp.array(dtype=wp.float64)):
        i = wp.tid()
        wp.atomic_add(out, slots[i], vals_e[i])

    _kernel_cache[key] = scat
    return scat


def _scatter_colored_kernel():
    key = ("dev_scatter_col",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def scatc(vals_e: wp.array(dtype=wp.float64),
              slots: wp.array(dtype=wp.int32),
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


def _zero_slots_kernel():
    key = ("dev_zero_slots",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def zk(slots: wp.array(dtype=wp.int32),
           vals: wp.array(dtype=wp.float64)):
        i = wp.tid()
        vals[slots[i]] = wp.float64(0.0)

    _kernel_cache[key] = zk
    return zk


def _diag_one_kernel():
    key = ("dev_diag_one",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def dk(diag: wp.array(dtype=wp.int32),
           rows: wp.array(dtype=wp.int32),
           bvals: wp.array(dtype=wp.float64),
           vals: wp.array(dtype=wp.float64),
           F: wp.array(dtype=wp.float64)):
        i = wp.tid()
        vals[diag[i]] = wp.float64(1.0)
        F[rows[i]] = bvals[i]

    _kernel_cache[key] = dk
    return dk


def _scatter_weighted_kernel():
    key = ("dev_scatter_w",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def scw(vals_e: wp.array(dtype=wp.float64),
            src: wp.array(dtype=wp.int32),
            w: wp.array(dtype=wp.float64),
            slots: wp.array(dtype=wp.int32),
            out: wp.array(dtype=wp.float64)):
        i = wp.tid()
        wp.atomic_add(out, slots[i], w[i] * vals_e[src[i]])

    _kernel_cache[key] = scw
    return scw


def _dof_indices_kernel():
    """dof-level CSR indices from the node graph, one thread per dof
    row r = ndof*na + ca: entries (q, cb) get column ndof*Gind[g0+q]+cb
    at indptr[r] + ndof*q + cb, indptr[r] = ndof^2*Gptr[na]
    + ca*ndof*deg(na). int32-safe: nnz < 2^31 asserted at build."""
    key = ("dev_dof_indices",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def dik(Gptr: wp.array(dtype=wp.int32),
            Gind: wp.array(dtype=wp.int32),
            ndof: wp.int32,
            out: wp.array(dtype=wp.int32)):
        r = wp.tid()
        na = r / ndof
        ca = r % ndof
        g0 = Gptr[na]
        dnb = Gptr[na + 1] - g0
        base = ndof * ndof * g0 + ca * ndof * dnb
        for q in range(dnb):
            col = ndof * Gind[g0 + q]
            for cb in range(ndof):
                out[base + ndof * q + cb] = col + cb

    _kernel_cache[key] = dik
    return dik


def _dof_indices_masked_kernel():
    """Masked variant of _dof_indices_kernel (pattern = kron(G, mask)):
    dof row r = ndof*na + ca has deg(na) * rowcnt[ca] entries; the
    entry for neighbor q and the j-th live col of block-row ca sits at
    indptr[r] + rowcnt[ca]*q + j, indptr[r] = blocknnz*Gptr[na]
    + rowoff[ca]*deg(na); blocknnz = rowoff[ndof]."""
    key = ("dev_dof_indices_masked",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def dikm(Gptr: wp.array(dtype=wp.int32),
             Gind: wp.array(dtype=wp.int32),
             ndof: wp.int32,
             rowoff: wp.array(dtype=wp.int32),
             lcols: wp.array(dtype=wp.int32),
             out: wp.array(dtype=wp.int32)):
        r = wp.tid()
        na = r / ndof
        ca = r % ndof
        g0 = Gptr[na]
        dnb = Gptr[na + 1] - g0
        ro = rowoff[ca]
        rc = rowoff[ca + 1] - ro
        base = rowoff[ndof] * g0 + ro * dnb
        for q in range(dnb):
            col = ndof * Gind[g0 + q]
            for j in range(rc):
                out[base + rc * q + j] = col + lcols[ro + j]

    _kernel_cache[key] = dikm
    return dikm


def _scatter_node_masked_kernel():
    """Masked variant of _scatter_node_kernel: local pairs whose
    (ca, cb) block is masked out are SKIPPED (the element kernel is
    contractually zero there — exactness gated masked-vs-unmasked);
    live pairs land at blocknnz*Gptr[na] + rowoff[ca]*deg(na)
    + rowcnt[ca]*(s - g0) + colpos[ca, cb]."""
    key = ("dev_scatter_node_masked",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def scnm(vals_e: wp.array(dtype=wp.float64),
             gslot: wp.array(dtype=wp.int32),
             conn: wp.array2d(dtype=wp.int32),
             Gptr: wp.array(dtype=wp.int32),
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
            g0 = Gptr[na]
            dnb = Gptr[na + 1] - g0
            ro = rowoff[ca]
            rc = rowoff[ca + 1] - ro
            slot = rowoff[ndof] * g0 + ro * dnb + rc * (s - g0) + j
            wp.atomic_add(out, slot, vals_e[i])

    _kernel_cache[key] = scnm
    return scnm


def _scatter_node_kernel():
    """Closed-form slot scatter (node-graph pattern): the dof slot is
    derived in-kernel from the element node-pair's position in G —
    no ne x (nbf*ndof)^2 slot map exists. Batch-local vals_e for the
    nb elements at global offset e0. All index arithmetic stays below
    nnz < 2^31 (asserted at build), so int32 is exact."""
    key = ("dev_scatter_node",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def scn(vals_e: wp.array(dtype=wp.float64),
            gslot: wp.array(dtype=wp.int32),
            conn: wp.array2d(dtype=wp.int32),
            Gptr: wp.array(dtype=wp.int32),
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
        g0 = Gptr[na]
        dnb = Gptr[na + 1] - g0
        slot = ndof * ndof * g0 + ca * ndof * dnb + ndof * (s - g0) + cb
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
