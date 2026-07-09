"""M4 track (c): Wodo CMS-2012 Fig.-3 (1-D) replication smoke.

Blend 1:1 (phi_p = phi_f = 0.2, phi_s = 0.6 + 1% noise), Np = Nf = 5,
Ns = 1, chi = (1.0, 0.3, 0.3). Thin strip nx=4 x ny=128 (physics is
1-D vertical; lateral spinodal modes are suppressed by kappa at the
strip width). v1 mobility mapping (units D_s = L = h0 = 1, so time is
h0^2/D_s and Bi = k_e): M0 = D(phi0)/f''_ideal(phi0),
D = phi_s + 1e-3 (1 - phi_s)  [D_p = D_f = 1e-3 D_s].

Qualitative gates (paper Sec. 7.1):
 (i)   h(t) decreases monotonically; total drying time Bi=10 << Bi=0.1.
 (ii)  Bi=10: solvent-LEAN boundary layer at the TOP; separation
       initiates AT the surface and propagates DOWN.
 (iii) Bi=0.1: no through-film solvent gradient; separation initiates
       homogeneously along the height.
 (iv)  physical solute content h * Int(phi_i dtheta) conserved =>
       mapped mean fractions rise as h falls (enrichment).

Run:  python benchmarks/wodo_fig3.py [--quick]
"""

import os as _bos, sys as _bsys  # noqa: E402  (benchmark import bootstrap)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import _bench_bootstrap  # noqa: E402,F401
import argparse
import time

import numpy as np
import warp as wp

from diffsim.octree.build import build_uniform, Octree
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.wodo_film import WodoFilmStepper

PHI_P0, PHI_F0 = 0.2, 0.2
CHI = (1.0, 0.3, 0.3)
NCHAIN = (5.0, 5.0, 1.0)
KAPPA = (2e-4, 2e-4)
AMP_ON = 0.1          # |phi_p - phi_f| row amplitude: "separated"
AMP_SEED = 0.05


def build_strip(level=7, nx=4):
    tree0 = build_uniform(level, dim=2)
    keep = tree0.centers()[:, 0] < nx * 2.0 ** (-level)
    tree = Octree(tree0.keys[keep], tree0.levels[keep], dim=2,
                  periodic=tree0.periodic)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    return mesh, cons


def m0_from_params():
    """v1 constant-mobility mapping (see module docstring)."""
    phis0 = 1.0 - PHI_P0 - PHI_F0
    D0 = phis0 * 1.0 + (1.0 - phis0) * 1e-3
    fpp = 1.0 / (NCHAIN[0] * PHI_P0) + 1.0 / (NCHAIN[2] * phis0)
    return D0 / fpp


class Recorder:
    def __init__(self, st, mesh, dm, snap_h=(0.9, 0.75, 0.6, 0.45)):
        self.st, self.mesh, self.dm = st, mesh, dm
        yv = mesh.node_coords[:, 1]
        self.rowy, self.rowinv = np.unique(np.round(yv, 12),
                                           return_inverse=True)
        self.rowcnt = np.bincount(self.rowinv)
        self.nrow = len(self.rowy)
        self.width = mesh.node_coords[:, 0].max()
        self.t_cross = np.full(self.nrow, np.nan)
        self.snap_h = list(snap_h)
        self.snaps = []          # (h, t, phi_p row profile, phi_s rows)
        self.hist = []           # t, h, K, dt, iters, phis_top, phis_bot
        self.mass0 = None
        self.mass_drift = 0.0
        self.dphis_pre_onset = 0.0
        self.onset = None        # (t, theta of first crossing rows)

    def _rows(self, free_vec):
        full = np.asarray(self.st.Tc @ free_vec)
        return np.bincount(self.rowinv, weights=full) / self.rowcnt

    def _mass(self, free_vec):
        v, _ = self.st._gp(free_vec)
        m = 0.0
        for pv, b in self.dm.bins.items():
            h = self.mesh.tree.h()[self.mesh.bins[pv]]
            ne = len(self.mesh.conn_of[pv])
            wq = np.tile(self.dm.tables_by_p[pv].w, ne) \
                * np.repeat((h / 2) ** 2, b["nqp"])
            m += float((wq * v[pv]).sum())
        return m / self.width          # per-unit-width: Int phi dtheta

    def __call__(self, st, K, dt, iters):
        r1 = self._rows(st.x[0::4])
        r2 = self._rows(st.x[2::4])
        rs = 1.0 - r1 - r2
        amp = np.abs(r1 - r2)
        # solute content (physical) = h * Int phi dtheta
        P1 = self._mass(st.x[0::4])
        P2 = self._mass(st.x[2::4])
        if self.mass0 is None:
            self.mass0 = (st.h_curr * P1, st.h_curr * P2)
        d1 = abs(st.h_curr * P1 - self.mass0[0]) / self.mass0[0]
        d2 = abs(st.h_curr * P2 - self.mass0[1]) / self.mass0[1]
        self.mass_drift = max(self.mass_drift, d1, d2)
        if self.onset is None:
            self.dphis_pre_onset = max(self.dphis_pre_onset,
                                       rs[0] - rs[-1])
            if amp.max() > AMP_ON:
                seeded = self.rowy[amp > AMP_SEED]
                self.onset = (st.t, float(np.mean(self.rowy[amp > AMP_ON])),
                              float(seeded.min()), float(seeded.max()),
                              len(seeded) / self.nrow)
        new = np.isnan(self.t_cross) & (amp > AMP_ON)
        self.t_cross[new] = st.t
        self.hist.append((st.t, st.h_curr, K, dt, iters,
                          float(rs[-1]), float(rs[0]),
                          P1, P2, float(amp.max())))
        if self.snap_h and st.h_curr <= self.snap_h[0]:
            self.snaps.append((st.h_curr, st.t, r1.copy(), rs.copy()))
            self.snap_h.pop(0)


def run_case(Bi, mesh, cons, device, quick=False, device_bound=False):
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    M0 = m0_from_params()
    st = WodoFilmStepper(dm, chi=CHI, N=NCHAIN, M=(M0, 0.0, M0),
                         kappa=KAPPA, k_e=Bi, dt=1e-4,
                         use_device_assembly=device_bound)
    rng = np.random.default_rng(7)
    st.set_initial(
        lambda x: PHI_P0 + 0.01 * rng.standard_normal(len(x)),
        lambda x: PHI_F0 + 0.01 * rng.standard_normal(len(x)))
    rec = Recorder(st, mesh, dm)
    t0 = time.time()
    reason = st.march(h_min=0.42, phis_stop=0.05,
                      max_steps=300 if quick else 20000, callback=rec)
    wall = time.time() - t0
    return st, rec, reason, wall, M0


def report(Bi, st, rec, reason, wall):
    H = np.array([(h[0], h[1]) for h in rec.hist])   # t, h
    dh = np.diff(H[:, 1])
    print(f"\n===== Bi = {Bi} =====")
    print(f"steps={len(rec.hist)} rejects={st.n_reject} reason={reason} "
          f"wall={wall:.0f}s")
    print(f"t_final={H[-1, 0]:.4g}  h: 1.0 -> {H[-1, 1]:.3f}  "
          f"h monotone: {'YES' if (dh <= 1e-15).all() else 'NO'} "
          f"(n_increase={int((dh > 1e-15).sum())})")
    print(f"gate(iv) solute content h*Phi_i: max rel drift "
          f"{rec.mass_drift:.2e}; mapped means Phi_p {rec.hist[0][7]:.3f}"
          f" -> {rec.hist[-1][7]:.3f}, Phi_f {rec.hist[0][8]:.3f} -> "
          f"{rec.hist[-1][8]:.3f} (enrichment)")
    print(f"solvent stratification pre-onset: max[phis(bot)-phis(top)] "
          f"= {rec.dphis_pre_onset:+.4f}")
    if rec.onset:
        t_on, th_on, th_lo, th_hi, frac = rec.onset
        print(f"separation onset (row |phi_p-phi_f| > {AMP_ON}): "
              f"t={t_on:.4g}, theta={th_on:.2f}; rows already seeded "
              f"(>{AMP_SEED}): theta in [{th_lo:.2f},{th_hi:.2f}] "
              f"({100 * frac:.0f}% of rows)")
        crossed = ~np.isnan(rec.t_cross)
        if crossed.sum() > 5:
            th = rec.rowy[crossed]
            tc = rec.t_cross[crossed]
            c = np.corrcoef(th, tc)[0, 1]
            print(f"propagation: {crossed.sum()}/{rec.nrow} rows crossed;"
                  f" corr(theta, t_cross) = {c:+.2f} "
                  f"(negative => top first, propagating down); "
                  f"spread(t_cross)/t_final = "
                  f"{(np.nanmax(tc) - np.nanmin(tc)) / rec.hist[-1][0]:.2f}")
    else:
        print("no separation onset reached")
    for h, t, r1, rs in rec.snaps:
        idx = np.linspace(0, len(r1) - 1, 13).astype(int)
        prof = " ".join(f"{v:.2f}" for v in r1[idx])
        print(f"  phi_p(theta) @ h={h:.2f} t={t:.4g}: [{prof}] "
              f"(theta=0 -> 1); phis top={rs[-1]:.3f} bot={rs[0]:.3f}")
    return H[-1, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--level", type=int, default=7)
    ap.add_argument("--device-bound", action="store_true",
                    help="use_device_assembly=True (slot-map scatter "
                         "+ zero-copy cuDSS; M4 v1.2)")
    args = ap.parse_args()
    wp.init()
    device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    mesh, cons = build_strip(level=args.level)
    print(f"strip: {len(mesh.tree)} elements, {len(mesh.node_coords)} "
          f"nodes, device={device}, M0={m0_from_params():.4f}, "
          f"device_bound={args.device_bound}")
    results = {}
    for Bi in (10.0, 0.1):
        st, rec, reason, wall, M0 = run_case(Bi, mesh, cons, device,
                                             quick=args.quick,
                                             device_bound=args.device_bound)
        results[Bi] = (st, rec, reason, wall)
    tf = {}
    for Bi in (10.0, 0.1):
        st, rec, reason, wall = results[Bi]
        tf[Bi] = report(Bi, st, rec, reason, wall)
    print("\n===== cross-case gates =====")
    print(f"gate(i): t_final(Bi=10) = {tf[10.0]:.4g} vs t_final(Bi=0.1) "
          f"= {tf[0.1]:.4g} -> faster at high Bi: "
          f"{'PASS' if tf[10.0] < tf[0.1] else 'FAIL'}")
    r10, r01 = results[10.0][1], results[0.1][1]
    print(f"gate(ii): Bi=10 top solvent depletion {r10.dphis_pre_onset:+.3f}"
          f" (expect >> 0) and onset theta "
          f"{r10.onset[1] if r10.onset else float('nan'):.2f} (expect ~1)")
    print(f"gate(iii): Bi=0.1 stratification {r01.dphis_pre_onset:+.3f} "
          f"(expect ~0), onset seeded-row span "
          f"[{r01.onset[2] if r01.onset else float('nan'):.2f},"
          f"{r01.onset[3] if r01.onset else float('nan'):.2f}] "
          f"(expect wide = homogeneous)")


if __name__ == "__main__":
    main()
