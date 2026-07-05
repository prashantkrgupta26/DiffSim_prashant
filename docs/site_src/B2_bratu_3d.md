# B2 — Bratu in 3-D + continuation toward the fold

LEARNING OUTCOME. You can (i) carry a nonlinear solver to 3-D unchanged,
and (ii) use CONTINUATION — marching a parameter and warm-starting Newton —
to walk a solution branch that a cold Newton cannot reach. You watch the
Newton step count grow as the turning point (fold) approaches: the
Jacobian is going singular, and the solver is telling you.

BACKGROUND. Same equations as B1 with dim=3 (the physical problem now:
f = 0, g = 0, so the trivial branch bends at a critical lam*). For the
unit cube lam* ~ 9.9. We march lam = 1, 2, ..., warm-starting from the
previous solution, and record ||u||_inf and Newton iterations.

EXPECTED RESULTS.
    3-D MMS check first: order ~2.0 at p1 (levels 2->3->4).
    Continuation: ||u||_inf grows with lam; Newton counts creep up
    (typically 3 -> 4 -> 5+) as lam approaches ~9; past the fold Newton
    fails — that failure is the measurement.

Run:  python tutorials/B_nonlinear/B2_bratu_3d.py

??? example "Full script — `tutorials/B_nonlinear/B2_bratu_3d.py` (run it!)"

    ```python
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu
    
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh, assemble_csr
    from diffsim.physics.poisson import gauss_points, l2_error_masked
    
    DEVICE = "cuda:0"
    
    
    class Bratu3D:
        def __init__(self, level, p=1):
            tree = build_uniform(level, dim=3)
            self.mesh = build_mesh(tree, p=p)
            self.cons = build_constraints(self.mesh)
            self.dm = DeviceMesh.from_mesh(self.mesh, self.cons,
                                           basis_tables(p, dim=3), DEVICE)
            self.K = assemble_csr(self.dm)
            self.T = self.cons.T.tocsr()
            self.xq = gauss_points(self.mesh, self.dm.tables_by_p)
            self.bdry = np.where(
                self.mesh.boundary_nodes[self.cons.free_nodes])[0]
            self.n_free = self.T.shape[1]
    
        def gp_values(self, u_free):
            full = np.asarray(self.T @ u_free)
            return {pv: np.einsum("qa,ea->eq", self.dm.tables_by_p[pv].N,
                                  full[self.mesh.conn_of[pv]])
                    for pv in self.dm.bins}
    
        def weighted_integrals(self, w_gp):
            dm, mesh = self.dm, self.mesh
            Fv = np.zeros(dm.n_nodes)
            rows, cols, vals = [], [], []
            for pv in dm.bins:
                tb = dm.tables_by_p[pv]
                h = mesh.tree.h()[mesh.bins[pv]]
                jac = (h / 2.0) ** dm.dim
                conn = mesh.conn_of[pv].astype(np.int64)
                be = np.einsum("qa,eq,q,e->ea", tb.N, w_gp[pv], tb.w, jac)
                np.add.at(Fv, conn.ravel(), be.ravel())
                Me = np.einsum("qa,qb,eq,q,e->eab", tb.N, tb.N, w_gp[pv],
                               tb.w, jac)
                nbf = conn.shape[1]
                rows.append(np.repeat(conn, nbf, axis=1).ravel())
                cols.append(np.tile(conn, (1, nbf)).ravel())
                vals.append(Me.ravel())
            M = sp.coo_matrix((np.concatenate(vals),
                               (np.concatenate(rows), np.concatenate(cols))),
                              shape=(dm.n_nodes,) * 2).tocsr()
            return np.asarray(self.T.T @ Fv), (self.T.T @ M @ self.T).tocsr()
    
        def newton(self, lam, u0, f_fn=None, g=0.0, tol=1e-11, maxit=25):
            u = u0.copy()
            u[self.bdry] = g
            for it in range(maxit):
                uq = self.gp_values(u)
                w_exp = {pv: lam * np.exp(uq[pv]) for pv in uq}
                if f_fn is None:
                    w_rhs = w_exp
                else:
                    w_rhs = {pv: w_exp[pv]
                             + f_fn(self.xq[pv]).reshape(uq[pv].shape)
                             for pv in uq}
                b_nl, _ = self.weighted_integrals(w_rhs)
                _, M_jac = self.weighted_integrals(w_exp)
                R = self.K @ u - b_nl
                R[self.bdry] = 0.0
                if np.linalg.norm(R, np.inf) < tol:
                    return u, it
                J = (self.K - M_jac).tolil()
                for i in self.bdry:
                    J.rows[i] = [int(i)]
                    J.data[i] = [1.0]
                u = u - splu(J.tocsr().tocsc()).solve(R)
            return None, maxit                       # Newton failed
    
    
    if __name__ == "__main__":
        # 1) MMS sanity in 3-D (lam=2)
        LAM = 2.0
        us = lambda x: (np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
                        * np.sin(np.pi * x[:, 2]))
        fs = lambda x: 3 * np.pi ** 2 * us(x) - LAM * np.exp(us(x))
        errs = []
        for lv in (2, 3, 4):
            ws = Bratu3D(lv)
            gv = us(ws.mesh.node_coords[ws.cons.free_nodes][ws.bdry])
            u = np.zeros(ws.n_free)
            u[ws.bdry] = gv
    
            # small wrapper: Dirichlet g = u* on the box for the MMS check
            def newton_mms():
                uu = u.copy()
                for _ in range(15):
                    uq = ws.gp_values(uu)
                    w_exp = {pv: LAM * np.exp(uq[pv]) for pv in uq}
                    w_rhs = {pv: w_exp[pv] + fs(ws.xq[pv]).reshape(
                        uq[pv].shape) for pv in uq}
                    b_nl, _ = ws.weighted_integrals(w_rhs)
                    _, M_jac = ws.weighted_integrals(w_exp)
                    R = ws.K @ uu - b_nl
                    R[ws.bdry] = uu[ws.bdry] - gv
                    if np.linalg.norm(R, np.inf) < 1e-11:
                        break
                    J = (ws.K - M_jac).tolil()
                    for i in ws.bdry:
                        J.rows[i] = [int(i)]
                        J.data[i] = [1.0]
                    uu = uu - splu(J.tocsr().tocsc()).solve(R)
                return uu
            uu = newton_mms()
            errs.append(l2_error_masked(ws.dm, np.asarray(ws.T @ uu), us,
                                        lambda x: np.ones(len(x), bool)))
        print("3-D MMS p1: errors " + "  ".join(f"{e:.3e}" for e in errs)
              + "   orders "
              + "  ".join(f"{np.log2(errs[i] / errs[i + 1]):.2f}"
                          for i in range(2)))
    
        # 2) continuation on the physical problem (f = 0, g = 0), level 3
        ws = Bratu3D(3)
        u = np.zeros(ws.n_free)
        print("\ncontinuation (physical Bratu, unit cube, level 3):")
        print(f"{'lam':>5} {'|u|_inf':>10} {'newton its':>11}")
        for lam in (1.0, 3.0, 5.0, 7.0, 8.0, 9.0, 9.5, 10.0, 11.0):
            u_new, its = ws.newton(lam, u)
            if u_new is None:
                print(f"{lam:>5.1f} {'—':>10} {'FAILED':>11}   <- past the fold?")
                break
            u = u_new
            print(f"{lam:>5.1f} {np.abs(u).max():>10.4f} {its:>11}")
        print("""
    EXPLORE
      (a) Refine the fold: bisect lam between the last success and first
          failure. How does lam* change from level 2 to level 3 to level 4?
          (Literature for the cube: ~9.9.)
      (b) Newton fails AT the fold because J is singular there. Pseudo-arclength
          continuation fixes this by treating (u, lam) jointly with an arclength
          constraint. Write the augmented system on paper; implementing it is a
          week-project.
      (c) PERFORMANCE CORNER: the weighted-integrals host assembly is now the
          per-iteration bottleneck (measure it!). Estimate the flops of the
          einsum "qa,qb,eq,q,e->eab" at level 4 (count indices!) and compare
          with the measured time x your machine's GFLOP/s — how far from peak
          are we, and why (memory-bound einsum)?
    """)
    ```
