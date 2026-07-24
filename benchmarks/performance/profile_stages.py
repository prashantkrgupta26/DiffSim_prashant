"""M1d profiling pass: per-stage wall times for the forward + adjoint
pipelines across sizes — the measured migration priority table (spec D1).

Run: python benchmarks/profile_stages.py            # writes JSON + table
"""

import os as _bos, sys as _bsys  # noqa: E402  (benchmark import bootstrap)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import _bench_bootstrap  # noqa: E402,F401
import json
import time

import numpy as np

RESULTS = {}


def t(label, fn, store, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    dt = time.perf_counter() - t0
    store[label] = round(dt, 4)
    return out


def profile_case(dim, level, device=None):
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu
    from diffsim import default_device
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    from diffsim.api.ns_bricks import assemble_linear_ns
    from diffsim.physics.poisson import gauss_points
    from diffsim.sbm.ns_adjoint import (ns_volume_cotangents,
                                        ns_load_cotangents)

    device = default_device() if device is None else device
    S = {}
    tree = t("mesh:build_uniform", build_uniform, S, level, dim=dim)
    mesh = t("mesh:build_mesh", build_mesh, S, tree, p=1)
    cons = t("mesh:constraints", build_constraints, S, mesh)
    dm = t("mesh:device_mesh", DeviceMesh.from_mesh, S, mesh, cons,
           basis_tables(1, dim=dim), device)
    ndof = dim + 1
    T = cons.T.tocsr()
    T_vec = t("host:kron_Tvec", lambda: sp.kron(
        T, sp.identity(ndof, format="csr"), format="csr"), S)
    nfree = T.shape[1]
    xq = gauss_points(mesh, dm.tables_by_p)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(nfree * ndof)

    # GP field eval (stepper-style host einsums)
    def gp_field():
        aq, dq = {}, {}
        for pv in dm.bins:
            tb = dm.tables_by_p[pv]
            conn = mesh.conn_of[pv]
            vals = np.stack([np.asarray(
                T @ x.reshape(nfree, ndof)[:, c])[conn]
                for c in range(dim)], axis=-1)
            aq[pv] = np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
            h = mesh.tree.h()[mesh.bins[pv]]
            dq[pv] = (np.einsum("qad,ead->eq", tb.dN, vals)
                      * (2.0 / h)[:, None]).reshape(-1)
        return aq, dq
    aq, dq = t("step:gp_field", gp_field, S)
    fq = {pv: np.zeros_like(aq[pv]) for pv in xq}

    A, b = t("step:assemble_linear_ns", assemble_linear_ns, S,
             dm, aq, dq, fq, 0.01, sigma=20.0)

    def strong_rows():
        Al = A.tolil()
        nb = min(1000, nfree)
        for i in range(0, nb, 7):
            for c in range(dim):
                r = i * ndof + c
                Al.rows[r] = [int(r)]
                Al.data[r] = [1.0]
        return Al.tocsr()
    A2 = t("step:strong_rows_lil", strong_rows, S)

    lu = t("solve:splu_factor", lambda: splu(A2.tocsc()), S)
    t("solve:splu_solve", lambda: lu.solve(b), S)
    try:
        from diffsim.solvers.linsolve import solve_linear
        t("solve:cudss_oneshot", lambda: solve_linear(
            A2, b, solver="cudss"), S)
    except Exception:
        S["solve:cudss_oneshot"] = None

    # adjoint sweeps
    T_full = np.asarray(T_vec @ x)
    lam_full = rng.standard_normal(len(T_full))
    t("adj:volume_cotangents", lambda: ns_volume_cotangents(
        dm, aq, dq, 0.01, 20.0, 0.5, T_full, lam_full), S)
    t("adj:load_cotangents", lambda: ns_load_cotangents(
        dm, aq, fq, 0.01, 20.0, lam_full), S)
    t("adj:Tvec_project", lambda: np.asarray(T_vec.T @ lam_full), S)

    S["_meta"] = {"dim": dim, "level": level, "nfree": int(nfree),
                  "nnz": int(A.nnz)}
    return S


def main():
    cases = [(2, 6), (2, 7), (2, 8), (3, 4), (3, 5)]
    for dim, lv in cases:
        key = f"{dim}d_L{lv}"
        print(f"== {key} ==", flush=True)
        try:
            RESULTS[key] = profile_case(dim, lv)
            for k, v in sorted(RESULTS[key].items(),
                               key=lambda kv: -(kv[1] or 0)
                               if kv[0] != "_meta" else 0):
                if k != "_meta" and v is not None:
                    print(f"  {k:28s} {v:9.3f} s")
        except Exception as e:
            RESULTS[key] = {"error": str(e)[:200]}
            print(f"  ERROR: {str(e)[:120]}")
        with open("profile_stages.json", "w") as fh:
            json.dump(RESULTS, fh, indent=2)
    print("\nwritten: profile_stages.json")


if __name__ == "__main__":
    main()
