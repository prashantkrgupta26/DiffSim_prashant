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
        T = dm.constraints.T
        n_free = T.shape[1]
        if (T.shape[0] != T.shape[1]) or (T != sp.identity(
                T.shape[0], format="csr")).nnz != 0:
            raise NotImplementedError(
                "DeviceNSAssembler: hanging-constrained meshes are D1 "
                "item 3 (constraint-aware scatter); this epoch has a "
                "non-identity T")
        self.Nfull = dm.n_nodes * ndof
        # ---- symbolic pattern + slot maps (host, once) -----------------
        rows_all, cols_all = [], []
        self._bins = []
        for pv, b in dm.bins.items():
            conn = dm.mesh.conn_of[pv].astype(np.int64)
            ne, nbf = conn.shape
            gdof = (conn[:, :, None] * ndof
                    + np.arange(ndof)[None, None, :]).reshape(ne,
                                                              nbf * ndof)
            r = np.repeat(gdof, nbf * ndof, axis=1).ravel()
            c = np.tile(gdof, (1, nbf * ndof)).ravel()
            rows_all.append(r)
            cols_all.append(c)
            self._bins.append((pv, b, ne, nbf, gdof))
        r = np.concatenate(rows_all)
        c = np.concatenate(cols_all)
        K = sp.coo_matrix((np.ones(len(r)), (r, c)),
                          shape=(self.Nfull, self.Nfull)).tocsr()
        K.sort_indices()
        self.indptr = K.indptr.copy()
        self.indices = K.indices.copy()
        self.nnz = K.nnz
        # slot index per (element-pair entry): position in the CSR values
        slot_bins = []
        for pv, b, ne, nbf, gdof in self._bins:
            rr = np.repeat(gdof, nbf * ndof, axis=1).ravel()
            cc = np.tile(gdof, (1, nbf * ndof)).ravel()
            starts = self.indptr[rr]
            ends = self.indptr[rr + 1]
            # searchsorted within each row's index range
            slots = np.empty(len(rr), np.int64)
            # vectorized per-row search: indices is globally sorted per row
            for chunk in range(0, len(rr), 2_000_000):
                sl = slice(chunk, min(chunk + 2_000_000, len(rr)))
                s0 = starts[sl]
                e0 = ends[sl]
                tgt = cc[sl]
                # positions via searchsorted on the concatenated row spans
                pos = np.array([np.searchsorted(
                    self.indices[s:e], t) + s
                    for s, e, t in zip(s0, e0, tgt)], np.int64)
                slots[sl] = pos
            slot_bins.append(slots.reshape(ne, (nbf * ndof) ** 2))
        self._slot_bins = slot_bins
        # device uploads
        self._slots_d = [wp.array(s.astype(np.int32).ravel(),
                                  dtype=wp.int32, device=dm.device)
                         for s in slot_bins]
        self.vals_d = wp.zeros(self.nnz, dtype=wp.float64,
                               device=dm.device)
        self.F_d = wp.zeros(self.Nfull, dtype=wp.float64,
                            device=dm.device)
        self._gdof_d = [wp.array(g.astype(np.int32).ravel(),
                                 dtype=wp.int32, device=dm.device)
                        for _, _, _, _, g in self._bins]
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
    def assemble(self, aq_by_bin, div_aq_by_bin, fq_by_bin, nu, sigma,
                 sig2tau=None, s_skew=0.5):
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
            if not self.coloring:
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
            scat_b = _scatter_vec_kernel()
            wp.launch(scat_b, dim=ne * nbf * ndof,
                      inputs=[be.reshape((-1,)), self._gdof_d[k_bin],
                              self.F_d], device=d)
        A = sp.csr_matrix((self.vals_d.numpy(), self.indices,
                           self.indptr), shape=(self.Nfull, self.Nfull))
        return A, self.F_d.numpy()


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
