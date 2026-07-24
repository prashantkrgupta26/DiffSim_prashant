"""M4 track (c): Wodo CMS-2012 Nova campaign runner — parameterized
figs 3-7 cases + ONE reduced-3D stretch (cluster/wodo_campaign/).

Case tables from the paper (Sec. 7) with the agent-corrected fig67
parameters (Nf = 5 FIXED — see benchmarks/wodo_fig67.py's PARAMETER
NOTE; the symmetric Np = Nf = 100 variant is thermodynamically wrong):

  fig3 (1-D, Sec 7.1): Bi in {0.1, 1, 10}; blend 1:1, phi_s0 = 0.6,
       Np = Nf = 5, chi = (1,.3,.3). The committed fig3 gate
       configuration exactly (v1 constant-mobility mapping; NO CHC
       noise / var-mob / b-reg — 1-D physics needs none of them).
  fig4 (2-D, Sec 7.1): Bi in {0.03, 0.3, 3}; blend 1:1, Np = Nf = 5.
  fig5 (2-D, Sec 7.2): blends 1:1 and 1:0.8; Np = Nf = 5; Bi = 0.3
       (the paper does not restate Bi for Fig 5 — we use the Fig 4/7
       mid rate; DOCUMENTED CHOICE).
  fig6 (2-D, Sec 7.3): Np in {5, 20, 100}, Nf = 5 FIXED, Bi = 0.4.
  fig7 (2-D, Sec 7.4): chi in {(1,.3,.3), (1,.3,.6), (1,.6,.3)} at
       Np = 100, Nf = 5, Bi = 0.3.
  fig3d (H200 STRETCH): reduced 3-D 128x128x48 (~3.4M dofs; the paper
       runs 230x230x70 on 256 CPUs); phi_s0 = 0.66 (their 3-D value),
       Lx = Ly = 3.3, Np = Nf = 5, Bi = 0.3. try/except ALLOC — either
       outcome is a recorded result. Full-res 3-D needs the CH block
       preconditioner (recorded M4 item; cuDSS direct won't reach 15M
       dofs).

All 2-D cases carry the fig67 GATE-PASSING model configuration
exactly: CHC conserved Langevin noise = 1e-3 (FDT shape, local
sqrt(M)), var_mob = True (their D(phi) freeze-out, D_ratio = 1e-3),
b_reg = 1e-3 (their footnote-2 simplex regularizer). phi_s0 = 0.75
(paper Sec. 7 2-D value).

Resolutions: sweep = 96x48 (the fig67 gate mesh; level-7 strip);
full = 250x100 (the paper's own 2-D mesh; level-9 strip
[0, 250/512] x [0, 100/512], non-dyadic counts carried by the v1.1
generalized metric). fig3 strips: 4 x 128 (sweep) / 4 x 256 (full).

Run:  python benchmarks/wodo_nova.py --fig 6 --case n100 \
          --resolution full --to-phis 0.05 --device-bound
      python benchmarks/wodo_nova.py --list
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

from diffsim import default_device
from diffsim.octree.build import build_uniform, Octree
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.wodo_film import WodoFilmStepper

import wodo_fig3 as f3
import wodo_fig67 as f67

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "wodo_nova")
LX = f67.LX                      # 2.5 (2-D lateral extent)
PHI_S0_2D = f67.PHI_S0           # 0.75
NS = f67.NS                      # 1.0
SNAP_H = (0.9, 0.7, 0.6, 0.5, 0.4)      # the paper figs' row heights

# ---------------------------------------------------------------------
# case tables
# ---------------------------------------------------------------------
# fig3 (1-D): case -> Bi
FIG3 = {"bi0.1": 0.1, "bi1": 1.0, "bi10": 10.0}

# 2-D: (fig, case) -> params. blend = phi_f0 : phi_p0 ratio (1:blend).
FIG2D = {
    ("4", "bi0.03"): dict(Np=5.0, Nf=5.0, chi=(1.0, 0.3, 0.3),
                          Bi=0.03, blend=1.0),
    ("4", "bi0.3"):  dict(Np=5.0, Nf=5.0, chi=(1.0, 0.3, 0.3),
                          Bi=0.3, blend=1.0),
    ("4", "bi3"):    dict(Np=5.0, Nf=5.0, chi=(1.0, 0.3, 0.3),
                          Bi=3.0, blend=1.0),
    ("5", "blend11"):  dict(Np=5.0, Nf=5.0, chi=(1.0, 0.3, 0.3),
                            Bi=0.3, blend=1.0),
    ("5", "blend108"): dict(Np=5.0, Nf=5.0, chi=(1.0, 0.3, 0.3),
                            Bi=0.3, blend=0.8),
    ("6", "n5"):   dict(Np=5.0, Nf=5.0, chi=(1.0, 0.3, 0.3),
                        Bi=0.4, blend=1.0),
    ("6", "n20"):  dict(Np=20.0, Nf=5.0, chi=(1.0, 0.3, 0.3),
                        Bi=0.4, blend=1.0),
    ("6", "n100"): dict(Np=100.0, Nf=5.0, chi=(1.0, 0.3, 0.3),
                        Bi=0.4, blend=1.0),
    ("7", "sym"):    dict(Np=100.0, Nf=5.0, chi=(1.0, 0.3, 0.3),
                          Bi=0.3, blend=1.0),
    ("7", "chifs"):  dict(Np=100.0, Nf=5.0, chi=(1.0, 0.3, 0.6),
                          Bi=0.3, blend=1.0),
    ("7", "chips"):  dict(Np=100.0, Nf=5.0, chi=(1.0, 0.6, 0.3),
                          Bi=0.3, blend=1.0),
}

FIG3D = {"stretch": dict(Np=5.0, Nf=5.0, chi=(1.0, 0.3, 0.3), Bi=0.3,
                         phi_s0=0.66, lat=3.3, nxy=128, nz=48)}


def cases_of(fig):
    if fig == "3":
        return list(FIG3)
    if fig == "3d":
        return list(FIG3D)
    return [c for f, c in FIG2D if f == fig]


def mobility(Np, Nf, pp0, pf0, phis0):
    """v1 frozen-mobility mapping at the case's OWN initial blend
    (generalizes wodo_fig67.mobility_of): M_i = D(phi0)/f''_ideal,i,
    units D_s = L = h0 = 1 (the var-mob kernel then evolves D(phi))."""
    D0 = phis0 * 1.0 + (1.0 - phis0) * 1e-3
    fpp_p = 1.0 / (Np * pp0) + 1.0 / (NS * phis0)
    fpp_f = 1.0 / (Nf * pf0) + 1.0 / (NS * phis0)
    return D0 / fpp_p, D0 / fpp_f


def append_result(args, line):
    print("RESULT " + line, flush=True)
    if args.results_file:
        with open(args.results_file, "a") as fh:
            fh.write("RESULT " + line + "\n")


# ---------------------------------------------------------------------
# fig3 (1-D)
# ---------------------------------------------------------------------
def run_fig3(case, args, device):
    Bi = FIG3[case]
    level = 7 if args.resolution == "sweep" else 8
    mesh, cons = f3.build_strip(level=level)
    print(f"fig3/{case}: strip {len(mesh.tree)} elements "
          f"(4 x {2 ** level}), Bi={Bi}, device_bound="
          f"{args.device_bound}", flush=True)
    st, rec, reason, wall, M0 = f3.run_case(
        Bi, mesh, cons, device, device_bound=args.device_bound)
    t_final = f3.report(Bi, st, rec, reason, wall)
    on = rec.onset or (float("nan"),) * 5
    append_result(args, (
        f"fig=3 case={case} res={args.resolution} Bi={Bi} "
        f"reason={reason} steps={len(rec.hist)} t_final={t_final:.4g} "
        f"h_end={st.h_curr:.3f} strat={rec.dphis_pre_onset:+.4f} "
        f"onset_theta={on[1]:.2f} mass_drift={rec.mass_drift:.2e} "
        f"wall={wall:.0f}s"))


# ---------------------------------------------------------------------
# figs 4-7 (2-D)
# ---------------------------------------------------------------------
def run_fig2d(fig, case, args, device):
    p = FIG2D[(fig, case)]
    solute = 1.0 - PHI_S0_2D
    pp0 = solute * 1.0 / (1.0 + p["blend"])
    pf0 = solute * p["blend"] / (1.0 + p["blend"])
    kap = f67.kappa_of(p["Np"])
    M11, M22 = mobility(p["Np"], p["Nf"], pp0, pf0, PHI_S0_2D)
    if args.resolution == "sweep":
        level, nx, ny = 7, 96, 48
    else:                                   # the paper's own 2-D mesh
        level, nx, ny = 9, 250, 100
    mesh, cons, hc = f67.build_strip(level, nx, ny)
    lat_scale = LX / (nx * hc)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    tag = f"f{fig}_{case}_{args.resolution}"
    st = WodoFilmStepper(
        dm, chi=p["chi"], N=(p["Np"], p["Nf"], NS),
        M=(M11, 0.0, M22), kappa=(kap, kap), k_e=p["Bi"], dt=1e-4,
        lat_scale=lat_scale, linsolver="cudss", noise=args.noise,
        var_mob=True, b_reg=1e-3,
        noise_seed=abs(hash(tag + "q")) % 2 ** 31,
        use_device_assembly=args.device_bound)
    rng = np.random.default_rng(abs(hash(tag)) % 2 ** 31)
    st.set_initial(
        lambda x: pp0 + 0.01 * rng.standard_normal(len(x)),
        lambda x: pf0 + 0.01 * rng.standard_normal(len(x)))
    rec = f67.Recorder2D(st, mesh, dm, hc, tag)
    rec.snap_todo = list(SNAP_H)
    print(f"\n===== fig{fig}/{case} [{args.resolution} {nx}x{ny}]: "
          f"Np={p['Np']:.0f} Nf={p['Nf']:.0f} chi={p['chi']} "
          f"Bi={p['Bi']} blend 1:{p['blend']:g} "
          f"(phi_p0={pp0:.4f}, phi_f0={pf0:.4f}) kappa={kap:.3e} "
          f"M=({M11:.3f},{M22:.3f}) device_bound={args.device_bound} "
          f"=====", flush=True)
    d_bin = f67.interface_width(p["Np"], p["Nf"], p["chi"][0], kap)
    print(f"  interface check: delta(binary,final)={d_bin:.4f} = "
          f"{d_bin / (hc * lat_scale):.1f} lateral elems", flush=True)
    t0 = time.time()
    reason = st.march(h_min=0.25, phis_stop=args.to_phis,
                      max_steps=args.max_steps, callback=rec,
                      wall_cap=args.wall_cap)
    rec.finalize(st)
    wall = time.time() - t0
    p1 = np.asarray(st.Tc @ st.x[0::4])
    p2 = np.asarray(st.Tc @ st.x[2::4])
    phis_end = float((1.0 - p1 - p2).mean())
    outdir = args.outdir or DATA_DIR
    os.makedirs(outdir, exist_ok=True)
    for stag, h, t, gp_, gf_ in rec.snaps:
        np.save(os.path.join(outdir, f"{tag}_phip_h{stag}.npy"), gp_)
        c = f67.classify(gp_, gf_, h)
        print(f"  snap h={stag}: A_p={c['A_p']:.2f} A_d={c['A_d']:.2f} "
              f"layers={c['layers']} top=({c['top'][0]:.3f},"
              f"{c['top'][1]:.3f})", flush=True)
    np.savez(os.path.join(outdir, f"{tag}_final.npz"),
             phi_p=rec.snaps[-1][3], phi_f=rec.snaps[-1][4],
             h=st.h_curr, t=st.t)
    cf = f67.classify(rec.snaps[-1][3], rec.snaps[-1][4], st.h_curr)
    append_result(args, (
        f"fig={fig} case={case} res={args.resolution} reason={reason} "
        f"steps={rec.nstep} rejects={st.n_reject} h_end={st.h_curr:.3f} "
        f"phis_end={phis_end:.3f} A_p={cf['A_p']:.4g} "
        f"A_d={cf['A_d']:.4g} layers={cf['layers']} "
        f"top_p={cf['top'][0]:.3f} top_f={cf['top'][1]:.3f} "
        f"mass_drift={rec.mass_drift:.2e} "
        f"phi_range=[{rec.pmin:+.4f},{rec.pmax:.4f}] wall={wall:.0f}s"))


# ---------------------------------------------------------------------
# fig3d: the reduced-3D H200 stretch
# ---------------------------------------------------------------------
class Recorder3D:
    """Light 3-D recorder: cheap nodal diagnostics only (the GP mass
    quadrature at ~6M gauss points is host-side and would dominate)."""

    def __init__(self, st, every=10):
        self.st, self.every = st, every
        self.nstep = 0
        self.t0 = time.time()

    def __call__(self, st, K, dt, iters):
        self.nstep += 1
        if self.nstep % self.every == 0:
            p1 = np.asarray(st.Tc @ st.x[0::4])
            p2 = np.asarray(st.Tc @ st.x[2::4])
            ps = 1.0 - p1 - p2
            print(f"  [3d] step {self.nstep:5d} t={st.t:8.4f} "
                  f"h={st.h_curr:.3f} phi_s={ps.mean():.3f} "
                  f"dt={dt:.2e} it={iters:2d} "
                  f"phi_p=[{p1.min():+.3f},{p1.max():.3f}] "
                  f"wall={time.time() - self.t0:6.0f}s", flush=True)


def run_fig3d(case, args, device):
    p = FIG3D[case]
    nxy, nz = p["nxy"], p["nz"]
    try:
        tree0 = build_uniform(7, dim=3)               # 128^3
        hc = 2.0 ** -7
        cen = tree0.centers()
        keep = cen[:, 2] < nz * hc
        tree = Octree(tree0.keys[keep], tree0.levels[keep], dim=3,
                      periodic=tree0.periodic)
        mesh = build_mesh(tree, p=1)
        cons = build_constraints(mesh)
        ndof = 4 * len(mesh.node_coords)
        print(f"fig3d/{case}: {len(tree)} elements ({nxy}x{nxy}x{nz}), "
              f"{len(mesh.node_coords)} nodes, {ndof} dofs, "
              f"device_bound={args.device_bound}", flush=True)
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3),
                                  device)
        phis0 = p["phi_s0"]
        pp0 = pf0 = (1.0 - phis0) / 2.0
        M11, M22 = mobility(p["Np"], p["Nf"], pp0, pf0, phis0)
        kap = f67.kappa_of(p["Np"])
        st = WodoFilmStepper(
            dm, chi=p["chi"], N=(p["Np"], p["Nf"], NS),
            M=(M11, 0.0, M22), kappa=(kap, kap), k_e=p["Bi"], dt=1e-4,
            lat_scale=p["lat"], linsolver="cudss", noise=args.noise,
            var_mob=True, b_reg=1e-3, noise_seed=17,
            use_device_assembly=args.device_bound)
        rng = np.random.default_rng(17)
        st.set_initial(
            lambda x: pp0 + 0.01 * rng.standard_normal(len(x)),
            lambda x: pf0 + 0.01 * rng.standard_normal(len(x)))
        rec = Recorder3D(st)
        t0 = time.time()
        reason = st.march(h_min=0.30, phis_stop=args.to_phis,
                          max_steps=args.max_steps, callback=rec,
                          wall_cap=args.wall_cap)
        wall = time.time() - t0
        p1 = np.asarray(st.Tc @ st.x[0::4])
        p2 = np.asarray(st.Tc @ st.x[2::4])
        phis_end = float((1.0 - p1 - p2).mean())
        outdir = args.outdir or DATA_DIR
        os.makedirs(outdir, exist_ok=True)
        np.savez(os.path.join(outdir, f"f3d_{case}_final.npz"),
                 phi_p=p1, phi_f=p2, coords=mesh.node_coords,
                 h=st.h_curr, t=st.t)
        append_result(args, (
            f"fig=3d case={case} OUTCOME=RAN reason={reason} "
            f"steps={rec.nstep} rejects={st.n_reject} "
            f"h_end={st.h_curr:.3f} phis_end={phis_end:.3f} "
            f"wall={wall:.0f}s"))
    except (MemoryError, RuntimeError, Exception) as e:  # noqa: B014
        # ALLOC (host or device or cuDSS factor) is a RECORDED outcome:
        # it locates the direct-solver wall. Full-res 3-D (their
        # 230x230x70, ~15M dofs) needs the CH BLOCK PRECONDITIONER —
        # recorded M4 item.
        append_result(args, (
            f"fig=3d case={case} OUTCOME=ALLOC-OR-FAIL "
            f"err={type(e).__name__}: {str(e)[:200]}"))


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fig", type=str, required=False,
                    choices=["3", "4", "5", "6", "7", "3d"])
    ap.add_argument("--case", type=str, default="all",
                    help="case id (see --list), 'all', or comma list")
    ap.add_argument("--resolution", type=str, default="sweep",
                    choices=["sweep", "full"],
                    help="sweep=96x48 (fig67 gate mesh), "
                         "full=250x100 (the paper's mesh)")
    ap.add_argument("--to-phis", type=float, default=0.05,
                    help="stop when mean phi_s reaches this "
                         "(paper: 0.05)")
    ap.add_argument("--device-bound", action="store_true",
                    help="use_device_assembly=True (slot-map scatter + "
                         "zero-copy cuDSS; measured 8x vs host-cudss)")
    ap.add_argument("--noise", type=float, default=1e-3,
                    help="CHC conserved-flux amplitude (fig67 gate "
                         "value; fig3 1-D ignores it)")
    ap.add_argument("--wall-cap", type=float, default=10800.0,
                    help="per-case wall cap, seconds")
    ap.add_argument("--max-steps", type=int, default=200000)
    ap.add_argument("--results-file", type=str, default=None,
                    help="append one RESULT line per case here")
    ap.add_argument("--outdir", type=str, default=None,
                    help="snapshot/.npz output dir "
                         "(default benchmarks/data/wodo_nova)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print("fig3 (1-D):", ", ".join(FIG3))
        for f in ("4", "5", "6", "7"):
            print(f"fig{f} (2-D):", ", ".join(cases_of(f)))
        print("fig3d (3-D stretch):", ", ".join(FIG3D))
        return
    if not args.fig:
        raise SystemExit("--fig required (or --list)")
    logging.getLogger("nvmath").setLevel(logging.ERROR)
    wp.init()
    device = default_device()
    cases = (cases_of(args.fig) if args.case == "all"
             else [c.strip() for c in args.case.split(",")])
    for case in cases:
        if args.fig == "3":
            run_fig3(case, args, device)
        elif args.fig == "3d":
            run_fig3d(case, args, device)
        else:
            if (args.fig, case) not in FIG2D:
                raise SystemExit(f"unknown case fig{args.fig}/{case}; "
                                 f"try --list")
            run_fig2d(args.fig, case, args, device)


if __name__ == "__main__":
    main()
