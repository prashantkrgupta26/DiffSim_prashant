"""M4 track (c): Wodo CMS-2012 Figs. 6+7 (2-D evaporating-film
morphology regimes) — N-ladder and chi-asymmetry.

Domain: their Lx = 2.5 x Ly(=h0) = 1, phi_s0 = 0.75 (paper Sec. 7:
2-D value), blend 1:1 (phi_p0 = phi_f0 = 0.125). 96 x 48 elements:
level-7 unit-square octree strip [0, 0.75] x [0, 0.375], lateral
metric lat_scale = 2.5/0.75 = 10/3, vertical Ycomp = 0.375 mapped to
theta in [0,1] (wodo_film v1.1 generalized metric).

PARAMETER NOTE (deliberate deviation from the campaign one-liner):
the paper's Fig 6 varies Np ONLY with Nf = 5 FIXED ("when polymer
degree of polymerization is larger than fullerene['s], morphology
changes from percolated into multiple layered", Sec. 7.3) and Fig 7
states "Np = 100, Nf = 5 and Ns = 1" (Sec. 7.4). We follow the paper:
Np in {5, 20, 100}, Nf = 5. (The symmetric Np = Nf = 100 variant is
also thermodynamically wrong for this replication: at phi_s0 = 0.75 it
sits DEEP inside the p-f spinodal at t = 0 — antisymmetric curvature
2/(N phi) - 2 chi = -1.84 — giving instant bulk isotropic
decomposition, not the paper's evaporation-front-mediated layers.)

kappa scaled per their eps^2 ratios 3.57 : 2.62 : 2.05 normalized to
the N=5 value 2e-4 used by wodo_fig3.

MORPHOLOGY CLASSIFICATION per snapshot (h ~ 0.9 / 0.7 / 0.5 / final):
2-D structure factor of (phi_p - mean) on the PHYSICAL grid (vertical
stretched by h); A = E(ky-dominant modes) / E(kx-dominant modes):
A >> 1 -> horizontal stripes = MULTILAYERED; A ~ 1 isotropic/
percolated; A << 1 lateral stripes. Also reported: A_d — the same
metric on the exchange field (phi_p - phi_f), which cancels the common
solvent-enrichment vertical trend; layer count = sign crossings of the
laterally-averaged phi_p(theta); top-10% composition (phi_p vs phi_f).

GATES (regime-level):
 (i)   A_p(final, N=100) > 3 x A_p(final, N=5)
 (ii)  chi = (1,.3,.6) vs (1,.6,.3): OPPOSITE component enriched in
       the top 10% (the less-soluble component wets the solvent-lean
       free surface)
 (iii) all cases: physical solute content conserved < 1e-10 rel;
       simplex bounds respected (report min/max of phi_i).

Run:  python benchmarks/wodo_fig67.py [--quick] [--cases a,b,..]
"""

import os as _bos, sys as _bsys  # noqa: E402  (benchmark import bootstrap)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import _bench_bootstrap  # noqa: E402,F401
import argparse
import logging
import os
import time

import numpy as np
import warp as wp

from diffsim.octree.build import build_uniform, Octree
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.wodo_film import WodoFilmStepper

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "wodo_fig67")
LX = 2.5
PHI_S0 = 0.75
PHI_P0 = PHI_F0 = (1.0 - PHI_S0) / 2.0     # blend 1:1
KAP5 = 2e-4                                 # wodo_fig3 N=5 value
EPS2 = {5: 3.57, 20: 2.62, 100: 2.05}       # their eps^2 (1e-10 J/m)
SNAP_H = (0.9, 0.7, 0.5)
NS = 1.0

CASES = {
    # name: (Np, Nf, chi=(pf, ps, fs), Bi)
    "f6_n5":    (5.0,   5.0, (1.0, 0.3, 0.3), 0.4),
    "f6_n20":   (20.0,  5.0, (1.0, 0.3, 0.3), 0.4),
    "f6_n100":  (100.0, 5.0, (1.0, 0.3, 0.3), 0.4),
    "f7_sym":   (100.0, 5.0, (1.0, 0.3, 0.3), 0.3),
    "f7_chifs": (100.0, 5.0, (1.0, 0.3, 0.6), 0.3),  # fullerene less sol.
    "f7_chips": (100.0, 5.0, (1.0, 0.6, 0.3), 0.3),  # polymer less sol.
}
# gate-critical cases first; nice-to-have afterwards
ORDER = ["f6_n5", "f6_n100", "f7_chifs", "f7_chips", "f6_n20", "f7_sym"]


def kappa_of(Np):
    return KAP5 * EPS2[int(Np)] / EPS2[5]


def mobility_of(Np, Nf):
    """v1 frozen-mobility mapping (wodo_film docstring): M_i =
    D(phi0)/f''_ideal,i(phi0), units D_s = L = h0 = 1."""
    D0 = PHI_S0 * 1.0 + (1.0 - PHI_S0) * 1e-3
    fpp_p = 1.0 / (Np * PHI_P0) + 1.0 / (NS * PHI_S0)
    fpp_f = 1.0 / (Nf * PHI_F0) + 1.0 / (NS * PHI_S0)
    return D0 / fpp_p, D0 / fpp_f


# ---------------------------------------------------------------------
# interface-width check (their d = dphi_e * sqrt(eps^2 / Df_max))
# ---------------------------------------------------------------------
def _clip01(u):
    return np.clip(u, 1e-12, 1.0 - 1e-12)


def _binary_f(u, Np, Nf, chi):
    u = _clip01(u)
    return (u / Np * np.log(u) + (1.0 - u) / Nf * np.log(1.0 - u)
            + chi * u * (1.0 - u))


def interface_width(Np, Nf, chi_pf, kap):
    """Common-tangent construction on the fully-evaporated binary
    p/f system (smallest interface, their Sec. 7). Binodal via the
    lower convex hull of f (robust for deep quenches where the
    equilibrium compositions are e^-N-close to the simplex edge)."""
    uu = np.unique(np.concatenate([
        np.linspace(1e-9, 1 - 1e-9, 4001),
        np.geomspace(1e-9, 0.5, 2001),
        1.0 - np.geomspace(1e-9, 0.5, 2001)]))
    ff = _binary_f(uu, Np, Nf, chi_pf)
    # lower convex hull (monotone chain on the sorted grid)
    hull = []
    for i in range(len(uu)):
        while len(hull) > 1:
            i0, i1 = hull[-2], hull[-1]
            if ((ff[i1] - ff[i0]) * (uu[i] - uu[i1])
                    >= (ff[i] - ff[i1]) * (uu[i1] - uu[i0])):
                hull.pop()
            else:
                break
        hull.append(i)
    # widest hull gap = the miscibility gap
    gaps = np.diff(uu[hull])
    j = int(np.argmax(gaps))
    a, b = uu[hull[j]], uu[hull[j + 1]]
    fa = _binary_f(a, Np, Nf, chi_pf)
    s = (_binary_f(b, Np, Nf, chi_pf) - fa) / (b - a)
    um = np.linspace(a, b, 2001)[1:-1]
    barrier = np.max(_binary_f(um, Np, Nf, chi_pf) - (fa + s * (um - a)))
    return (b - a) * np.sqrt(kap / barrier)


def interface_width_ternary(Np, Nf, chi, kap, phis):
    """Same construction on the pseudo-binary p-f exchange at fixed
    phi_s (the LARGER early/mid-drying interface width)."""
    c12, c1s, c2s = chi
    c = 1.0 - phis

    def g(u):
        p1, p2 = _clip01(c * u), _clip01(c * (1.0 - u))
        return (p1 / Np * np.log(p1) + p2 / Nf * np.log(p2)
                + phis / NS * np.log(phis) + c12 * p1 * p2
                + c1s * p1 * phis + c2s * p2 * phis)

    uu = np.unique(np.concatenate([
        np.linspace(1e-9, 1 - 1e-9, 4001),
        np.geomspace(1e-9, 0.5, 2001),
        1.0 - np.geomspace(1e-9, 0.5, 2001)]))
    gg = g(uu)
    hull = []
    for i in range(len(uu)):
        while len(hull) > 1:
            i0, i1 = hull[-2], hull[-1]
            if ((gg[i1] - gg[i0]) * (uu[i] - uu[i1])
                    >= (gg[i] - gg[i1]) * (uu[i1] - uu[i0])):
                hull.pop()
            else:
                break
        hull.append(i)
    gaps = np.diff(uu[hull])
    j = int(np.argmax(gaps))
    a, b = uu[hull[j]], uu[hull[j + 1]]
    if b - a < 1e-3:
        return np.nan                      # single phase at this phi_s
    ga = g(a)
    s = (g(b) - ga) / (b - a)
    um = np.linspace(a, b, 2001)[1:-1]
    barrier = np.max(g(um) - (ga + s * (um - a)))
    if barrier <= 0:
        return np.nan
    # composition span in phi_p units = c*(b-a)
    return c * (b - a) * np.sqrt(kap / barrier)


# ---------------------------------------------------------------------
# morphology classification
# ---------------------------------------------------------------------
def anisotropy(grid, h, Lx=LX):
    """A = E(ky-dominant)/E(kx-dominant) of the 2-D structure factor on
    the physical grid (vertical spacing h/(ny-1)). grid[iy, ix]."""
    q = grid - grid.mean()
    ny, nx = q.shape
    E = np.abs(np.fft.fft2(q)) ** 2
    kx = np.abs(np.fft.fftfreq(nx, d=Lx / (nx - 1)))
    ky = np.abs(np.fft.fftfreq(ny, d=h / (ny - 1)))
    KX, KY = np.meshgrid(kx, ky)
    E[0, 0] = 0.0
    e_y = E[KY > KX * (1.0 + 1e-12)].sum()
    e_x = E[KX > KY * (1.0 + 1e-12)].sum()
    return e_y / max(e_x, 1e-300)


def layer_count(grid, thr=0.01):
    """Sign crossings of the laterally-averaged (phi_p - mean) vertical
    profile; |d| < thr treated as zero (noise floor)."""
    d = grid.mean(axis=1)
    d = d - d.mean()
    s = np.sign(d)
    s[np.abs(d) < thr] = 0
    s = s[s != 0]
    if len(s) < 2:
        return 0
    return int(np.sum(s[1:] != s[:-1]))


def top_comp(gp_, gf_, frac=0.1):
    ny = gp_.shape[0]
    i0 = int(np.floor((1.0 - frac) * (ny - 1)))
    return float(gp_[i0:].mean()), float(gf_[i0:].mean())


def classify(gp_, gf_, h):
    return dict(A_p=anisotropy(gp_, h), A_d=anisotropy(gp_ - gf_, h),
                layers=layer_count(gp_), top=top_comp(gp_, gf_))


# ---------------------------------------------------------------------
# mesh / recorder
# ---------------------------------------------------------------------
def build_strip(level=7, nx_c=96, ny_c=48):
    tree0 = build_uniform(level, dim=2)
    hc = 2.0 ** (-level)
    cen = tree0.centers()
    keep = (cen[:, 0] < nx_c * hc) & (cen[:, 1] < ny_c * hc)
    tree = Octree(tree0.keys[keep], tree0.levels[keep], dim=2,
                  periodic=tree0.periodic)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    return mesh, cons, hc


class Recorder2D:
    def __init__(self, st, mesh, dm, hc, case):
        self.st, self.mesh, self.dm, self.case = st, mesh, dm, case
        c = mesh.node_coords
        self.ix = np.round(c[:, 0] / hc).astype(int)
        self.iy = np.round(c[:, 1] / hc).astype(int)
        self.nxn, self.nyn = self.ix.max() + 1, self.iy.max() + 1
        self.snap_todo = list(SNAP_H)
        self.snaps = []                    # (tag, h, t, gp, gf)
        self.mass0 = None
        self.mass_drift = 0.0
        self.pmin, self.pmax = np.inf, -np.inf
        self.nstep = 0
        self.t0 = time.time()

    def grid(self, free_vec):
        full = np.asarray(self.st.Tc @ free_vec)
        g = np.zeros((self.nyn, self.nxn))
        g[self.iy, self.ix] = full
        return g

    def _mass(self, free_vec):
        v, _ = self.st._gp(free_vec)
        m = 0.0
        for pv, b in self.dm.bins.items():
            h = self.mesh.tree.h()[self.mesh.bins[pv]]
            ne = len(self.mesh.conn_of[pv])
            wq = np.tile(self.dm.tables_by_p[pv].w, ne) \
                * np.repeat((h / 2) ** 2, b["nqp"])
            m += float((wq * v[pv]).sum())
        return m

    def __call__(self, st, K, dt, iters):
        self.nstep += 1
        p1 = np.asarray(st.Tc @ st.x[0::4])
        p2 = np.asarray(st.Tc @ st.x[2::4])
        ps = 1.0 - p1 - p2
        self.pmin = min(self.pmin, p1.min(), p2.min(), ps.min())
        self.pmax = max(self.pmax, p1.max(), p2.max(), ps.max())
        P1 = self._mass(st.x[0::4])
        P2 = self._mass(st.x[2::4])
        if self.mass0 is None:
            self.mass0 = (st.h_curr * P1, st.h_curr * P2)
        self.mass_drift = max(
            self.mass_drift,
            abs(st.h_curr * P1 - self.mass0[0]) / self.mass0[0],
            abs(st.h_curr * P2 - self.mass0[1]) / self.mass0[1])
        if self.snap_todo and st.h_curr <= self.snap_todo[0]:
            tag = f"{self.snap_todo.pop(0):.2f}"
            self.snaps.append((tag, st.h_curr, st.t,
                               self.grid(st.x[0::4]),
                               self.grid(st.x[2::4])))
        if self.nstep % 50 == 0:
            gp_ = self.grid(st.x[0::4])
            gf_ = self.grid(st.x[2::4])
            A = anisotropy(gp_, st.h_curr)
            print(f"  [{self.case}] step {self.nstep:5d} t={st.t:8.4f} "
                  f"h={st.h_curr:.3f} phi_s={ps.mean():.3f} dt={dt:.2e} "
                  f"it={iters:2d} A={A:8.2f} lay={layer_count(gp_)} "
                  f"wall={time.time() - self.t0:5.0f}s", flush=True)

    def finalize(self, st):
        self.snaps.append(("final", st.h_curr, st.t,
                           self.grid(st.x[0::4]),
                           self.grid(st.x[2::4])))


# ---------------------------------------------------------------------
def run_case(case, mesh, cons, hc, device, quick=False, wall_cap=2400,
             linsolver="cudss", noise=1e-3):
    Np, Nf, chi, Bi = CASES[case]
    kap = kappa_of(Np)
    M11, M22 = mobility_of(Np, Nf)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    st = WodoFilmStepper(dm, chi=chi, N=(Np, Nf, NS),
                         M=(M11, 0.0, M22), kappa=(kap, kap), k_e=Bi,
                         dt=1e-4, lat_scale=LX / 0.75, linsolver=linsolver,
                         noise=noise, var_mob=True, b_reg=1e-3,
                         noise_seed=abs(hash(case + "q")) % 2**31)
    rng = np.random.default_rng(abs(hash(case)) % 2**31)
    st.set_initial(
        lambda x: PHI_P0 + 0.01 * rng.standard_normal(len(x)),
        lambda x: PHI_F0 + 0.01 * rng.standard_normal(len(x)))
    rec = Recorder2D(st, mesh, dm, hc, case)
    print(f"\n===== {case}: Np={Np:.0f} Nf={Nf:.0f} chi={chi} Bi={Bi} "
          f"kappa={kap:.3e} M=({M11:.3f},{M22:.3f}) =====", flush=True)
    d_bin = interface_width(Np, Nf, chi[0], kap)
    d_t3 = interface_width_ternary(Np, Nf, chi, kap, 0.3)
    dx_lat = hc * LX / 0.75            # physical lateral cell
    ny_c = round(0.375 / hc)           # vertical cells over theta [0,1]
    print(f"  interface check: delta(binary,final)={d_bin:.4f} "
          f"({d_bin / dx_lat:.1f} lat elems, "
          f"{d_bin / (0.30 / ny_c):.1f} vert elems @h=0.30) | "
          f"delta(ternary,phi_s=0.3)={d_t3:.4f} "
          f"({d_t3 / dx_lat:.1f} lat elems, "
          f"{d_t3 / (0.55 / ny_c):.1f} vert elems @h=0.55)", flush=True)
    t0 = time.time()
    reason = st.march(h_min=0.27, phis_stop=0.10,
                      max_steps=60 if quick else 20000,
                      callback=rec, wall_cap=wall_cap)
    rec.finalize(st)
    wall = time.time() - t0
    p1 = np.asarray(st.Tc @ st.x[0::4])
    p2 = np.asarray(st.Tc @ st.x[2::4])
    phis_end = float((1.0 - p1 - p2).mean())
    print(f"  done: reason={reason} steps={rec.nstep} "
          f"rejects={st.n_reject} h_end={st.h_curr:.3f} "
          f"phi_s_end={phis_end:.3f} wall={wall:.0f}s", flush=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    rows = []
    for tag, h, t, gp_, gf_ in rec.snaps:
        np.save(os.path.join(DATA_DIR, f"{case}_phip_h{tag}.npy"), gp_)
        c = classify(gp_, gf_, h)
        rows.append((tag, h, t, c))
    np.savez(os.path.join(DATA_DIR, f"{case}_final.npz"),
             phi_p=rec.snaps[-1][3], phi_f=rec.snaps[-1][4],
             h=st.h_curr, t=st.t)
    return dict(case=case, reason=reason, h_end=st.h_curr,
                phis_end=phis_end, wall=wall, rows=rows,
                mass_drift=rec.mass_drift, pmin=rec.pmin, pmax=rec.pmax,
                d_bin=d_bin, d_t3=d_t3)


def report(results):
    print("\n" + "=" * 78)
    print("RESULTS TABLE (A_p = anisotropy of phi_p; A_d = of "
          "phi_p - phi_f; A>>1 = MULTILAYER)")
    print("=" * 78)
    hdr = (f"{'case':9s} {'snap':6s} {'h':>5s} {'t':>8s} {'A_p':>9s} "
           f"{'A_d':>9s} {'lay':>3s} {'top phi_p':>9s} {'top phi_f':>9s}")
    print(hdr)
    for r in results:
        for tag, h, t, c in r["rows"]:
            print(f"{r['case']:9s} {tag:6s} {h:5.3f} {t:8.4f} "
                  f"{c['A_p']:9.2f} {c['A_d']:9.2f} {c['layers']:3d} "
                  f"{c['top'][0]:9.3f} {c['top'][1]:9.3f}")
        print(f"{'':9s} stop={r['reason']} phi_s_end={r['phis_end']:.3f} "
              f"mass_drift={r['mass_drift']:.2e} "
              f"phi in [{r['pmin']:+.4f}, {r['pmax']:.4f}] "
              f"wall={r['wall']:.0f}s")
    byc = {r["case"]: r for r in results}
    print("\n===== GATES =====")
    ok = {}
    if "f6_n5" in byc and "f6_n100" in byc:
        a5 = byc["f6_n5"]["rows"][-1][3]["A_p"]
        a100 = byc["f6_n100"]["rows"][-1][3]["A_p"]
        ok["i"] = a100 > 3.0 * a5
        print(f"gate(i)  N-ladder: A_p(final) N=100 {a100:.2f} vs "
              f"3 x N=5 {3 * a5:.2f} -> "
              f"{'PASS' if ok['i'] else 'FAIL'}")
    if "f7_chifs" in byc and "f7_chips" in byc:
        dfs = np.subtract(*byc["f7_chifs"]["rows"][-1][3]["top"])
        dps = np.subtract(*byc["f7_chips"]["rows"][-1][3]["top"])
        ok["ii"] = dfs * dps < 0
        print(f"gate(ii) chi-asymmetry: top-10% (phi_p - phi_f): "
              f"chi_fs=0.6 -> {dfs:+.3f} (expect fullerene-rich, <0); "
              f"chi_ps=0.6 -> {dps:+.3f} (expect polymer-rich, >0) -> "
              f"{'PASS' if ok['ii'] else 'FAIL'}")
    cons_ok = all(r["mass_drift"] < 1e-10 for r in results)
    bnd_ok = all(r["pmin"] > -0.05 and r["pmax"] < 1.05 for r in results)
    print(f"gate(iii) conservation < 1e-10: "
          f"{'PASS' if cons_ok else 'FAIL'} "
          f"(max {max(r['mass_drift'] for r in results):.2e}); "
          f"simplex bounds: {'PASS' if bnd_ok else 'FAIL'} "
          f"(phi in [{min(r['pmin'] for r in results):+.4f}, "
          f"{max(r['pmax'] for r in results):.4f}])")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cases", type=str, default=",".join(ORDER))
    ap.add_argument("--level", type=int, default=7)
    ap.add_argument("--wall-cap", type=float, default=2400.0)
    ap.add_argument("--linsolver", type=str, default="cudss")
    ap.add_argument("--noise", type=float, default=1e-3,
                    help="CHC conserved-flux noise amplitude (FDT shape)")
    args = ap.parse_args()
    logging.getLogger("nvmath").setLevel(logging.ERROR)
    wp.init()
    device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    sc = 2 ** (args.level - 7)
    mesh, cons, hc = build_strip(args.level, 96 * sc, 48 * sc)
    print(f"strip: {len(mesh.tree)} elements "
          f"({96 * sc}x{48 * sc}), {len(mesh.node_coords)} nodes, "
          f"physical {LX} x 1 (lat_scale={LX / 0.75:.4f}), "
          f"device={device}, solver={args.linsolver}")
    results = []
    for case in args.cases.split(","):
        case = case.strip()
        if case not in CASES:
            raise SystemExit(f"unknown case {case}")
        results.append(run_case(case, mesh, cons, hc, device,
                                quick=args.quick,
                                wall_cap=args.wall_cap,
                                linsolver=args.linsolver,
                                noise=args.noise))
        report(results)


if __name__ == "__main__":
    main()
