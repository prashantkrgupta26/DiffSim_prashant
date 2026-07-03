"""Host-side field evaluation + trace-conformity checking (spec S13.3).

The trace check is THE hanging-node acceptance criterion: nodal agreement is
insufficient (the group's Delaunay counterexample) — the interpolated TRACE
along every shared face must match from both sides.

Care-point fix (S13.3): when face_neighbors returns a finer neighbor
(h[j] < h[e]), we skip from the coarse side — the fine element covers the
interface when it processes its own face. This avoids evaluating the single
representative fine neighbor at sample points that may lie outside its box.
"""
import numpy as np
from diffsim.octree import morton
from diffsim.octree.lookup import face_neighbors, face_offsets
from diffsim.mesh.basis import lagrange_1d
from diffsim.mesh.nodes import _local_offsets


class FieldEvaluator:
    def __init__(self, mesh, u_all):
        self.mesh, self.u = mesh, np.asarray(u_all, np.float64)
        self.dim = mesh.dim
        L = morton.lmax(self.dim)
        self.lo = mesh.tree.anchors() / (1 << L)   # physical lower corner [Ne, dim]
        self.h = mesh.tree.h()                      # physical element size  [Ne]
        self.row_of = {}
        for pv, eids in mesh.bins.items():
            for r, e in enumerate(eids):
                self.row_of[int(e)] = (pv, r)

    def eval(self, e, x_phys):
        """Evaluate the Lagrange interpolant of element e at physical points x_phys [M, dim]."""
        pv, r = self.row_of[int(e)]
        conn = self.mesh.conn_of[pv][r]
        xi = 2.0 * (np.atleast_2d(x_phys) - self.lo[e]) / self.h[e] - 1.0
        offs = _local_offsets(pv, self.dim)
        out = np.zeros(len(xi))
        # N1[d] has shape [M, pv+1]: 1-D basis values at each sample point along axis d
        N1 = [np.array([lagrange_1d(pv, x)[0] for x in xi[:, d]])
              for d in range(self.dim)]
        for a in range(len(offs)):
            w = np.ones(len(xi))
            for d in range(self.dim):
                w *= N1[d][:, offs[a, d]]
            out += w * self.u[conn[a]]
        return out


def trace_conformity_max_jump(mesh, u_all, n_samples: int = 4) -> float:
    """Return max |u_left - u_right| sampled on an n_samples^(dim-1) lattice
    strictly inside every shared face (spec S13.3 C0 criterion).

    Same-size pairs are visited once (from the higher-index element).
    Coarse-fine pairs are visited from the FINE side only: when h[j] < h[e]
    (j is finer), the coarse side e skips — the fine element j will cover the
    interface when it processes its own opposite face.  This guarantees the
    sample points always lie inside BOTH elements' domains.
    """
    dim = mesh.dim
    ev = FieldEvaluator(mesh, u_all)
    lo, h = ev.lo, ev.h
    offs = face_offsets(dim)
    nbrs = face_neighbors(mesh.tree)
    t = np.linspace(0.15, 0.85, n_samples)          # strictly inside the face
    max_jump = 0.0
    for f, off in enumerate(offs):
        ax = f // 2
        for e in range(len(mesh.tree)):
            j = nbrs[f][e]
            # skip: no neighbor, or same-size pair already counted from higher index
            if j < 0 or (j <= e and h[j] == h[e]):
                continue
            # skip: j is finer (h[j] < h[e]) — the fine side covers this interface
            if h[j] < h[e]:
                continue
            # skip periodic-wrapped pairs (physical coords don't coincide)
            if abs((lo[j] - lo[e])[ax]) > h[e] + h[j]:
                continue
            # sample lattice on e's face f (e is the fine side of a coarse-fine
            # pair or the higher-index element of a same-size pair)
            grids = np.meshgrid(*([t] * (dim - 1)), indexing="ij")
            pts = np.empty((n_samples ** (dim - 1), dim))
            k = 0
            for d in range(dim):
                if d == ax:
                    pts[:, d] = lo[e, d] + (h[e] if off[ax] > 0 else 0.0)
                else:
                    pts[:, d] = lo[e, d] + grids[k].ravel() * h[e]
                    k += 1
            jump = np.abs(ev.eval(e, pts) - ev.eval(j, pts)).max()
            max_jump = max(max_jump, jump)
    return max_jump
