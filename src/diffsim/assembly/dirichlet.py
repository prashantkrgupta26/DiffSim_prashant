import numpy as np
import warp as wp

# Mask kernels are never taped (see operators.py note); skip backward codegen.
wp.set_module_options({"enable_backward": False})

from .operators import ConstrainedOperator, csr_spmv
from ..physics.poisson import make_load_kernel, gauss_points
from ..solvers.krylov import cg


@wp.kernel
def _mask_kernel(v: wp.array(dtype=wp.float64), m: wp.array(dtype=wp.float64)):
    """Zero entries where m[i] == 1.0 (Dirichlet mask); keep others."""
    i = wp.tid()
    v[i] = v[i] * (wp.float64(1.0) - m[i])


@wp.kernel
def _add_masked_kernel(y: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.float64),
                       m: wp.array(dtype=wp.float64)):
    """y[i] += m[i] * x[i]  — restores Dirichlet entries from x."""
    i = wp.tid()
    y[i] = y[i] + m[i] * x[i]


def _apply_mask(v, m, device):
    """Zero entries at Dirichlet (masked) positions in-place."""
    wp.launch(_mask_kernel, dim=len(v), inputs=[v, m], device=device)


def _add_masked(y, x, m, device):
    """y[C] += x[C] for Dirichlet positions C."""
    wp.launch(_add_masked_kernel, dim=len(y), inputs=[y, x, m], device=device)


class _BCOperator:
    """Constrained operator with identity rows on Dirichlet free-node set.

    Implements  y = (I-M) A ((I-M) x) + M x,
    which is SPD on the whole free-node space: identity on Dirichlet DOFs,
    and the original Poisson operator on the interior DOFs.
    """

    def __init__(self, base: ConstrainedOperator, dir_mask_free: np.ndarray):
        self.base = base
        self.dm = base.dm
        self.device = base.dm.device
        self.n_free = base.dm.n_free
        # 1.0 on Dirichlet free nodes, 0.0 elsewhere
        self.mask = wp.array(dir_mask_free.astype(np.float64), dtype=wp.float64,
                             device=self.device)

    def matvec(self, x, y):
        """y = (I-M) A ((I-M) x) + M x"""
        d = self.device
        xm = wp.clone(x)
        _apply_mask(xm, self.mask, d)       # zero Dirichlet entries of xm
        self.base.matvec(xm, y)             # y = A (I-M) x
        _apply_mask(y, self.mask, d)        # zero Dirichlet rows of y
        _add_masked(y, x, self.mask, d)     # y[C] = x[C]


class DirichletPoisson:
    """Strong Dirichlet Poisson solver on the unit-cube boundary.

    Enforces g on all free nodes that lie on the mesh boundary, then
    solves via CG on the BC-constrained operator.
    """

    def __init__(self, dm):
        self.dm = dm
        self.op = ConstrainedOperator(dm)
        c = dm.constraints
        # Boolean mask [n_free]: which free nodes are Dirichlet (on the boundary)
        self.dir_free = dm.mesh.boundary_nodes[c.free_nodes]   # bool [n_free]

    def solve(self, g_fn, f_fn, tol=1e-12):
        dm, c, d = self.dm, self.dm.constraints, self.dm.device
        n_free = dm.n_free

        # --- Assemble load vector F (full nodes) projected to free nodes ---
        # Loop over per-degree bins; each bin contributes to the same F_full
        # via atomic_add inside make_load_kernel.
        xq_by_bin = gauss_points(dm.mesh, dm.tables_by_p)
        F_full = wp.zeros(dm.n_nodes, dtype=wp.float64, device=d)
        for pv, b in dm.bins.items():
            xq = xq_by_bin[pv]
            fq = wp.array(f_fn(xq), dtype=wp.float64, device=d)
            lk = make_load_kernel(b["nbf"], b["nqp"], dm.dim)
            wp.launch(lk, dim=len(b["eids"]),
                      inputs=[b["conn"], b["h"], b["N"], b["w"], fq, F_full], device=d)
        F_free = wp.zeros(n_free, dtype=wp.float64, device=d)
        wp.launch(csr_spmv, dim=n_free, inputs=[*dm.Tt_dev, F_full, F_free], device=d)

        # --- Lift: g is non-zero only on Dirichlet free nodes ---
        g = np.zeros(n_free)
        g[self.dir_free] = g_fn(dm.mesh.node_coords[c.free_nodes][self.dir_free])
        gd = wp.array(g, dtype=wp.float64, device=d)
        Ag = wp.zeros(n_free, dtype=wp.float64, device=d)
        self.op.matvec(gd, Ag)

        # --- RHS: b = F - A g, with Dirichlet entries zeroed ---
        b = F_free.numpy() - Ag.numpy()
        b[self.dir_free] = 0.0

        # --- Solve with BC operator (identity on Dirichlet DOFs) ---
        bc_op = _BCOperator(self.op, self.dir_free)
        u0, info = cg(bc_op, b, tol=tol, maxiter=5000)
        assert info["converged"], info

        # --- Reconstruct: add lift, then expand to all nodes ---
        u_free = u0 + g
        return c.T @ u_free      # host scipy SpMV: (Nn, n_free) @ (n_free,) -> (Nn,)
