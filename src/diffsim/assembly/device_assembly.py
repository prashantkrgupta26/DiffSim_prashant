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


class DeviceNSAssembler:
    """Per-epoch object: symbolic pattern + slot maps once; numeric fill
    per step on device."""

    def __init__(self, dm, sigma_like=1.0, coloring=False):
        self.dm = dm
        self.coloring = coloring
        ndof = dm.dim + 1
        self.ndof = ndof
        T = dm.constraints.T.tocsr()
        n_free = T.shape[1]
        self.n_free = n_free
        identity_T = (T.shape[0] == T.shape[1]) and (
            T != sp.identity(T.shape[0], format="csr")).nnz == 0
        self._identity_T = identity_T
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
                 sig2tau=None, s_skew=0.5, strong_b_vals=None):
        """Numeric fill on device; returns (csr, F) with HOST copies for
        now (the solver interface); vals stay resident in self.vals_d."""
        from ..api.ns_bricks import make_linear_ns_Ae, make_linear_ns_be
        dm = self.dm
        d = dm.device
        ndof = self.ndof
        if sig2tau is None:
            sig2tau = (2.0 * sigma) ** 2
        self.vals_d.zero_()
        self.F_d.zero_()
        for k_bin, (pv, b, ne, nbf, gdof) in enumerate(self._bins):
            nqp = b["nqp"]
            aq = wp.array(np.ascontiguousarray(aq_by_bin[pv]),
                          dtype=wp.float64, device=d)
            dq = wp.array(np.ascontiguousarray(div_aq_by_bin[pv]),
                          dtype=wp.float64, device=d)
            ga = np.zeros((len(aq_by_bin[pv]), dm.dim * dm.dim))
            gaq = wp.array(ga, dtype=wp.float64, device=d)
            fq = wp.array(np.ascontiguousarray(fq_by_bin[pv]),
                          dtype=wp.float64, device=d)
            Ae = wp.zeros((ne, nbf * ndof, nbf * ndof), dtype=wp.float64,
                          device=d)
            be = wp.zeros((ne, nbf * ndof), dtype=wp.float64, device=d)
            kA = make_linear_ns_Ae(nbf, nqp, dm.dim)
            kb = make_linear_ns_be(nbf, nqp, dm.dim)
            wp.launch(kA, dim=ne,
                      inputs=[b["conn"], b["h"], b["N"], b["dN"], b["w"],
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
        if getattr(self, "_strong", None) is not None:
            st = self._strong
            zk = _zero_slots_kernel()
            wp.launch(zk, dim=st["n_spans"],
                      inputs=[st["spans_d"], self.vals_d],
                      device=self.dm.device)
            dk = _diag_one_kernel()
            bv = wp.array(np.ascontiguousarray(
                strong_b_vals if strong_b_vals is not None
                else np.zeros(st["n_rows"])), dtype=wp.float64,
                device=self.dm.device)
            wp.launch(dk, dim=st["n_rows"],
                      inputs=[st["diag_d"], st["rows_d"], bv,
                              self.vals_d, self.F_d],
                      device=self.dm.device)
        if getattr(self, "_return_device", False):
            import torch
            vals_t = torch.from_dlpack(self.vals_d.__dlpack__())
            if not hasattr(self, "_indptr_t"):
                self._indptr_t = torch.tensor(self.indptr,
                                              dtype=torch.int64,
                                              device="cuda")
                self._indices_t = torch.tensor(self.indices,
                                               dtype=torch.int64,
                                               device="cuda")
            A_t = torch.sparse_csr_tensor(
                self._indptr_t, self._indices_t, vals_t,
                size=(self.Nfull, self.Nfull))
            return A_t, torch.from_dlpack(self.F_d.__dlpack__())
        A = sp.csr_matrix((self.vals_d.numpy(), self.indices,
                           self.indptr), shape=(self.Nfull, self.Nfull))
        return A, self.F_d.numpy()

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
