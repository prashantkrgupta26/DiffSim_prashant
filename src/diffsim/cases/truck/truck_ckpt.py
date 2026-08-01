"""TRUCK case checkpoint helpers: mesh-sequenced restart interpolation.

Moved verbatim from tests/truck_flow.py.  Functions: interpolate_checkpoint.
Note: checkpoint save/load is inline in run_truck (truck_march.py), so only
interpolate_checkpoint lives here.
"""
import numpy as np


def interpolate_checkpoint(ckpt_path, fx_old, fx_new, out_dir):
    """Mesh-sequenced restart (Baskar / plan T-A1): transfer a march
    checkpoint from mesh A to mesh B by FE interpolation, so a run can be
    resumed on a REFINED (or otherwise rebuilt) mesh.

    The state is p=1 trilinear fields: reconstruct FULL nodal fields on A
    (constraint expansion T), locate every free node of B in A's octree
    (LeafLookup on the integer lattice), evaluate the trilinear interpolant
    from the containing cell's corner nodes, and write a mesh-B checkpoint
    (same npz schema; run_truck(resume=True) picks it up).  A->A transfer is
    EXACT (nodes coincide with corners => weights select nodal values)."""
    import pathlib as _pl
    from diffsim.octree import morton as _morton
    from diffsim.octree.lookup import LeafLookup as _LeafLookup

    # Read every needed member into memory before touching the output directory.
    # When out_dir == the source directory, the unlink loop below would otherwise
    # delete the source file while we still hold a lazy npz handle — safe today
    # via the POSIX open-file guarantee but fragile across OS / zip backends.
    with np.load(ckpt_path) as _ck_f:
        ck = {k: np.array(_ck_f[k]) for k in _ck_f.files}
    mesh_o, cons_o, tree_o = fx_old["mesh"], fx_old["cons"], fx_old["tree"]
    mesh_n, cons_n = fx_new["mesh"], fx_new["cons"]
    ndof = 4
    T_o = cons_o.T.tocsr()                      # scalar constraint expansion

    def _full(free_vec, ncomp):
        v = np.asarray(free_vec, np.float64).reshape(len(cons_o.free_nodes),
                                                     ncomp)
        return np.asarray(T_o @ v)              # [n_nodes_old, ncomp]

    full_x = _full(ck["x_cur"], ndof)
    full_u1 = _full(ck["u_pre1"], 3)
    full_u2 = _full(ck["u_pre2"], 3)

    # target points: mesh B free-node coords
    pts = mesh_n.node_coords[cons_n.free_nodes]
    L = _morton.lmax(3)
    G = 1 << L
    lk = _LeafLookup(tree_o)
    ip = np.clip((pts * G).astype(np.int64), 0, G - 1)
    cell = lk.find(ip)
    # nodes on cell/carve boundaries can lattice-land in an excluded or
    # neighboring cell: search all 8 corner-adjacent offsets until found
    for dx in (0, -1):
        for dy in (0, -1):
            for dz in (0, -1):
                miss = np.where(cell < 0)[0]
                if len(miss) == 0:
                    break
                probe = np.clip(ip[miss] + np.array([dx, dy, dz]), 0, G - 1)
                got = lk.find(probe)
                ok = got >= 0
                cell[miss[ok]] = got[ok]
    assert (cell >= 0).all(), \
        f"{int((cell < 0).sum())} target nodes outside the old mesh"
    sc = 2.0 ** -L
    anc = tree_o.anchors()[cell] * sc
    hh = tree_o.h()[cell]
    xi = np.clip((pts - anc) / hh[:, None], 0.0, 1.0)   # [n,3] in [0,1]
    conn_o = np.asarray(mesh_o.conn_of[1], np.int64)[cell]   # [n,8]
    # trilinear weights in build_mesh corner ordering (z-major bit pattern:
    # corner k has offsets ((k>>0)&1, (k>>1)&1, (k>>2)&1) -- matches the
    # octree child/anchor convention used by build_mesh(p=1))
    w = np.empty((len(pts), 8))
    for k in range(8):
        ox, oy, oz = (k >> 0) & 1, (k >> 1) & 1, (k >> 2) & 1
        w[:, k] = (np.where(ox, xi[:, 0], 1 - xi[:, 0])
                   * np.where(oy, xi[:, 1], 1 - xi[:, 1])
                   * np.where(oz, xi[:, 2], 1 - xi[:, 2]))

    def _interp(full):
        return np.einsum("nk,nkc->nc", w, full[conn_o])

    x_new = _interp(full_x).ravel()
    u1_new = _interp(full_u1)
    u2_new = _interp(full_u2)

    out = _pl.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("march_ckpt_*.npz"):
        old.unlink()
    dst = out / "march_ckpt_0.npz"
    nst = len(ck["cd"])
    np.savez(dst, step=int(ck["step"]), t_cur=float(ck["t_cur"]),
             dt_prev=ck["dt_prev"], x_cur=x_new, u_pre1=u1_new,
             u_pre2=u2_new, cd=ck["cd"], cd_surr=ck["cd_surr"],
             cl_y=ck["cl_y"], cl_z=ck["cl_z"],
             cl_y_surr=ck["cl_y_surr"], cl_z_surr=ck["cl_z_surr"])
    print(f"[truck] checkpoint interpolated: {len(pts)} target nodes, "
          f"A cells hit, -> {dst}", flush=True)
    return dst
