"""FN2 diagnostic: compare raw traction vs consistent-flux drag on the SAME
rung-B mesh, for BOTH the projection and monolithic steady fields. Dumps the
per-component wall numbers (pressure force, viscous force, penalty reaction)
so we can see WHY the same surrogate_traction reads right on the monolithic
(+2.02) but wrong on the projection (-1.07), and whether the consistent flux
recovers Cd for BOTH."""
import numpy as np

from diffsim.sbm.vector import surrogate_traction, sbm_consistent_flux
from ladder_rungB_square_nitsche import (
    march_projection, march_monolithic, build_square_channel_2d, ALPHA,
    qref)


def _pieces(dm, sf, geo, x_full, nu, ndof, alpha):
    """Per-piece wall force breakdown (pressure / viscous / penalty) + wall
    field stats, in the surrogate_traction orientation (n = geo.n)."""
    dim = dm.dim
    mesh = dm.mesh
    from diffsim.mesh.faces import face_tables
    pv = int(np.unique(np.asarray(mesh.p_elem)[sf.elem])[0])
    ftab = face_tables(pv, dim)
    nqf = ftab.nqf
    conn = mesh.conn_of[pv][np.searchsorted(mesh.bins[pv], sf.elem)]
    xv = x_full.reshape(dm.n_nodes, ndof)
    h = mesh.tree.h()[sf.elem]
    jacS = (h / 2.0) ** (dim - 1)
    dscale = 2.0 / h
    dvec = geo.d.reshape(len(sf.elem) * nqf, dim)
    F_p = np.zeros(dim); F_v = np.zeros(dim); F_pen = np.zeros(dim)
    p_w = []; un_w = []; umag_w = []
    for fi in range(len(sf.elem)):
        f = int(sf.face[fi])
        un = xv[conn[fi], :dim]; pn = xv[conn[fi], dim]
        N = ftab.N[f]; dN = ftab.dN[f] * dscale[fi]
        for q in range(nqf):
            gp = fi * nqf + q
            w = ftab.w[q] * jacS[fi] * geo.corr[gp]
            n = geo.n[gp]
            gradu = dN[q].T @ un
            pq = float(N[q] @ pn); uq = N[q] @ un
            F_p += w * (pq * n)
            F_v += w * (-nu * (gradu @ n))
            shift = gradu @ dvec[gp]
            F_pen += w * (alpha * nu * dscale[fi] * (uq + shift))  # g=0
            p_w.append(pq); un_w.append(float(uq @ n)); umag_w.append(float(np.linalg.norm(uq)))
    return dict(F_p=F_p, F_v=F_v, F_pen=F_pen,
                p_w=np.array(p_w), un_w=np.array(un_w), umag_w=np.array(umag_w))


def report(tag, dm, sf, geo, x_full, nu, ndof, q, alpha):
    raw = surrogate_traction(dm, sf, geo, x_full, nu, ndof)
    cflux = sbm_consistent_flux(dm, sf, geo, x_full, nu, ndof, alpha=alpha)
    cflux_sym = sbm_consistent_flux(dm, sf, geo, x_full, nu, ndof, alpha=alpha,
                                    symmetric_grad=True)
    pc = _pieces(dm, sf, geo, x_full, nu, ndof, alpha)
    print(f"\n--- {tag} ---")
    print(f"  wall pressure  : mean={pc['p_w'].mean():+.4f}  "
          f"min={pc['p_w'].min():+.4f}  max={pc['p_w'].max():+.4f}")
    print(f"  wall u.n       : mean={pc['un_w'].mean():+.4f}  "
          f"max|u.n|={np.abs(pc['un_w']).max():.4f}")
    print(f"  wall |u|       : mean={pc['umag_w'].mean():.4f}  "
          f"max={pc['umag_w'].max():.4f}")
    print(f"  F_pressure     = [{pc['F_p'][0]:+.5f}, {pc['F_p'][1]:+.5f}]")
    print(f"  F_viscous      = [{pc['F_v'][0]:+.5f}, {pc['F_v'][1]:+.5f}]")
    print(f"  F_penalty      = [{pc['F_pen'][0]:+.5f}, {pc['F_pen'][1]:+.5f}]")
    print(f"  RAW traction   Fx={raw[0]:+.5f}  -> Cd={raw[0]/q:+.4f}")
    print(f"  CFLUX (lap)    Fx={cflux[0]:+.5f}  -> Cd={cflux[0]/q:+.4f}")
    print(f"  CFLUX (sym)    Fx={cflux_sym[0]:+.5f}  -> Cd={cflux_sym[0]/q:+.4f}")
    return dict(cd_raw=raw[0]/q, cd_cflux=cflux[0]/q, cd_cflux_sym=cflux_sym[0]/q)


def run(level=5, Re=40, half=0.125):
    fx = build_square_channel_2d(level, Re, half=half, offset=0, device="cpu")
    dm, sf, geo = fx["dm"], fx["sf"], fx["geo"]
    ndof, nu = fx["ndof"], fx["nu"]
    q = qref(fx)
    print(f"=== rung-B drag diagnostic  Re={Re} level={level} alpha={ALPHA} "
          f"qref={q:.5g} nu={nu:.5g} ===", flush=True)

    pr = march_projection(fx, dt=0.02, nsteps=600, rate_tol=5e-4, alpha=ALPHA,
                          log_every=0)
    mo = march_monolithic(fx, dt=0.02, nsteps=600, rate_tol=2e-4,
                          backflow_beta=0.5, boundary_vorticity=True,
                          alpha=ALPHA, log_every=0)
    print(f"\n march done: proj Cd(raw,in-loop)={pr['cd']:+.4f} steps={pr['steps']} "
          f"blew={pr['blew_up']}   mono Cd={mo['cd']:+.4f} steps={mo['steps']}")

    rp = report("PROJECTION", dm, sf, geo, pr["x_full"], nu, ndof, q, ALPHA)
    rm = report("MONOLITHIC", dm, sf, geo, mo["x_full"], nu, ndof, q, ALPHA)

    print(f"\n=== VERDICT (target mono raw Cd = {rm['cd_raw']:+.4f}) ===")
    print(f"  proj : raw={rp['cd_raw']:+.4f}  cflux={rp['cd_cflux']:+.4f}  "
          f"cflux_sym={rp['cd_cflux_sym']:+.4f}")
    print(f"  mono : raw={rm['cd_raw']:+.4f}  cflux={rm['cd_cflux']:+.4f}  "
          f"cflux_sym={rm['cd_cflux_sym']:+.4f}")


if __name__ == "__main__":
    import sys
    lvl = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run(level=lvl)
