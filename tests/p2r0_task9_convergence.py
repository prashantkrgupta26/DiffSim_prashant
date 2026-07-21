"""P2-R0 Task 9 — cylinder validation via a FULL MESH-CONVERGENCE study.

The headline validation of the composed projection+volumetric-SBM stepper
(``LeraySBMStepper``) against the literature. This is NOT a single-mesh number:
we march the Re20 cylinder to steady at MULTIPLE refinement levels, extract Cd
at each, and demonstrate Cd CONVERGES with refinement (h -> 0); likewise the
Re100 Strouhal.

REFERENCES
----------
[A] C.H. Yang, G. Scovazzi, A. Krishnamurthy, et al., "An immersed octree
    shifted-boundary method ...", J. Comput. Phys. 544 (2026) 114334
    (local_code_old/1-s2.0-S0021999125006163-main.pdf). Section 4.1.1 is a
    Re=100 UNSTEADY cylinder in a [0,30]x[0,20] domain, D=1 (5% blockage),
    cylinder at (10,10). Its mesh-convergence Table 1 (element sizes
    30/2^12, 30/2^13, 30/2^14):
        Cd = 1.384, 1.384, 1.386   (literature spread 1.310 - 1.453)
        St = 0.170, 0.170, 0.170   (literature spread 0.157 - 0.170)
    THIS IS THE SAME DISCRETIZATION FAMILY (octree + volumetric SBM, lambda=0.5,
    linear basis) as ours -> the primary target for the Re100 St cross-check.
    NOTE: [A] reports NO steady Re20 Cd; its 2D cylinder benchmark is the Re100
    shedding/St study.

[Classical Re20] Unbounded steady flow past a circular cylinder at Re_D=20:
    Cd ~ 2.0 (Tritton 1959; Sucker-Brauer correlation Cd(Re=20) ~ 2.05).
    CONFINED (Schaefer-Turek-like blockage) values run higher.

[M1b lock] Our own confined config ([0,1]^2, R=0.07 at (0.3,0.5), 14% blockage,
    no-slip walls) monolithic-solver locks: cylinder_re20_cd = 2.847 (measured,
    m1b_baselines), cylinder_re100_cd = 1.352, cylinder_re100_strouhal = 0.2059.
    The 14% blockage + no-slip walls elevate Cd well above the unbounded ~2.0.

STUDY DESIGN
------------
Two setups are run so the convergence can be judged against BOTH a directly
comparable literature reference AND our own monolithic lock:

  (1) CONFINED (M1b) config: R=0.07 at (0.3,0.5) in [0,1]^2, 14% blockage,
      no-slip walls. Reference = the M1b monolithic Cd=2.847. Tests that the
      projection+SBM path reproduces the monolithic solver as h->0.

  (2) [A]-MATCHED low-blockage config: a small cylinder in the unit box giving
      5% blockage with free-slip (symmetry) top/bottom walls and generous
      downstream length, so Cd is directly comparable to the unbounded /
      paper-[A] family. Reference = classical ~2.0 (Re20) and paper-[A] family
      (Re100 St ~0.170).

Per-level ALPHA PROBE: the Task-5 finding is a narrow stable Nitsche-penalty
window (alpha=100 stable at L5). As h shrinks the stable alpha may shift, so we
sweep a small alpha ladder per level and pick the converged march with the
smallest steady drag-rate (and finite, physical Cd). The chosen alpha is
recorded per level.

Results are written incrementally to
``tests/baselines/p2r0_task9_convergence.json`` so a multi-hour box run
captures intermediate levels as it goes.
"""
import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.geometry.csg import Sphere
from diffsim.sbm.surrogate import classify_lambda, extract_surrogate, GeometryData
from diffsim.steppers.leray_sbm import LeraySBMStepper
from p2r0_harness import make_strouhal_fn

U_IN = 1.0
OUT = os.path.join(os.path.dirname(__file__), "baselines",
                   "p2r0_task9_convergence.json")


def _device():
    try:
        import warp as wp
        return "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


# ---------------------------------------------------------------------------
# fixture builders
# ---------------------------------------------------------------------------
def build_case(level, *, R, ctr, Re, walls="noslip"):
    """Build a cylinder fixture in the unit box at a refinement ``level``.

    walls: "noslip"  -> top/bottom y-walls strong u=0 (confined, M1b)
           "freeslip"-> top/bottom y-walls left FREE (symmetry-like, so the
                        blockage is set by 2R/H_eff and the wake is not
                        wall-clamped) -> comparable to unbounded literature.
    Inflow x=0 strong u=(U_IN,0); outflow x=1 free (natural).
    """
    dim = 2
    D = 2.0 * R
    nu = U_IN * D / Re
    oracle = Sphere(ctr, R)
    tree = build_uniform(level, dim=2)
    ret, _ = classify_lambda(tree, oracle, 0.5, domain="outside")
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    device = _device()
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, 2),
                                domain="outside")
    coords = mesh.node_coords[cons.free_nodes]
    on = lambda v, c: np.abs(coords[:, c] - v) < 1e-12
    if walls == "noslip":
        strong = np.where(on(0.0, 0) | on(0.0, 1) | on(1.0, 1))[0]
    elif walls == "freeslip":
        # only the inflow face is strong; top/bottom + outflow left natural
        strong = np.where(on(0.0, 0))[0]
    else:
        raise ValueError(walls)
    strong_mask = np.zeros(len(coords), dtype=bool)
    strong_mask[strong] = True
    u_inf = np.zeros((len(coords), dim))
    inflow = strong[np.abs(coords[strong, 0]) < 1e-12]
    u_inf[inflow, 0] = U_IN
    h = 1.0 / (1 << level)
    return dict(oracle=oracle, dm=dm, sf=sf, geo=geo, strong_mask=strong_mask,
                u_inf=u_inf, mesh=mesh, cons=cons, R=R, D=D, nu=nu, h=h,
                level=level, n_free=dm.n_free, ndof=dm.dim + 1)


# ---------------------------------------------------------------------------
# Re20 Cd — per-level, per-alpha steady march with an alpha probe
# ---------------------------------------------------------------------------
def _march_steady_guarded(st, *, dt, D, max_steps, rate_tol, min_steps=10,
                          blowup=6.0):
    """Own step loop with an early BLOWUP bailout so diverging alphas terminate
    fast instead of running the full max_steps. Returns (cd, cl, steps) on
    convergence; raises RuntimeError on blowup or non-convergence."""
    qref = 0.5 * U_IN ** 2 * D
    cd_prev = None
    for step in range(max_steps):
        st.step()
        F = st.surrogate_traction()
        cd = F[0] / qref
        cl = F[1] / qref
        if not np.isfinite(cd) or abs(cd) > blowup:
            raise RuntimeError(f"blowup at step {step}: Cd={cd:.3f}")
        if cd_prev is not None and step >= min_steps:
            if abs(cd - cd_prev) / dt < rate_tol:
                return cd, cl, step + 1
        cd_prev = cd
    raise RuntimeError(f"no convergence in {max_steps} steps; last Cd={cd:.4f}")


def re20_cd_at_level(case, alphas, dt=0.05, max_steps=400, rate_tol=5e-3):
    """March to steady over an alpha ladder; return the best (converged,
    physical) result. 'Best' = converged march with smallest final drag-rate
    and Cd in a physical band, preferring the lowest stable alpha."""
    dim = 2

    def f_fn(x, t):
        return np.zeros((len(x), dim))

    results = []
    for alpha in alphas:
        t0 = time.time()
        try:
            st = LeraySBMStepper(
                case["oracle"], case["dm"], case["nu"], dt, f_fn,
                u_inf=case["u_inf"], strong_mask=case["strong_mask"],
                lam=0.5, domain="outside", order=1, picard_iters=2,
                solver="splu", ppe_finescale=False, alpha=float(alpha))
            st.set_initial(lambda c: np.zeros((len(c), dim)))
            cd, cl, nsteps = _march_steady_guarded(
                st, dt=dt, D=case["D"],
                max_steps=max_steps, rate_tol=rate_tol)
            blk = abs(st.surrogate_normal_flux()[0]) / U_IN
            ok = (np.isfinite(cd) and np.isfinite(cl) and 0.5 < cd < 8.0
                  and abs(cl) < 0.5 * abs(cd))
            results.append(dict(alpha=float(alpha), cd=float(cd),
                                cl=float(cl), steps=int(nsteps),
                                blockage=float(blk), converged=True,
                                physical=bool(ok),
                                wall_s=round(time.time() - t0, 1)))
        except Exception as e:  # divergence / no-steady-state / blowup
            results.append(dict(alpha=float(alpha), converged=False,
                                physical=False, error=str(e)[:200],
                                wall_s=round(time.time() - t0, 1)))
    # pick: among physical converged marches, the PLATEAU CENTER — the alpha
    # whose Cd is closest to the median physical Cd (robust to a lone
    # marginally-stable outlier). Also record the physical-Cd spread so the
    # stable-window sensitivity is visible.
    good = [r for r in results if r.get("physical")]
    if good:
        cds = np.array([r["cd"] for r in good])
        med = float(np.median(cds))
        best = min(good, key=lambda r: abs(r["cd"] - med))
        best = dict(best)
        best["plateau_median_cd"] = med
        best["plateau_cd_spread"] = float(cds.max() - cds.min())
        best["n_physical_alphas"] = len(good)
    else:
        conv = [r for r in results if r.get("converged")]
        best = conv[0] if conv else None
    return best, results


# ---------------------------------------------------------------------------
# Re100 Strouhal — transient march with a symmetry-breaking kick
# ---------------------------------------------------------------------------
def re100_strouhal_at_level(case, alpha, dt=0.02, t_end=40.0, kick=0.05,
                            kick_until=2.0):
    """Transient BDF2 march; break symmetry with a transverse inflow kick for
    the first ``kick_until`` time units, then record the Cl history and extract
    St via zero-crossing periods. Returns (St, amp, n_periods, wall_s)."""
    dim = 2
    D = case["D"]

    def f_fn(x, t):
        return np.zeros((len(x), dim))

    # kicked inflow: add a small transverse component to the inflow rows for
    # the first kick_until time units to seed the antisymmetric mode.
    u_inf_base = case["u_inf"].copy()
    strong = np.where(case["strong_mask"])[0]
    coords = case["mesh"].node_coords[case["cons"].free_nodes]
    inflow = strong[np.abs(coords[strong, 0]) < 1e-12]

    st = LeraySBMStepper(
        case["oracle"], case["dm"], case["nu"], dt, f_fn,
        u_inf=u_inf_base, strong_mask=case["strong_mask"],
        lam=0.5, domain="outside", order=2, picard_iters=2,
        solver="splu", ppe_finescale=False, alpha=float(alpha),
        beta_backflow=1.0)
    st.set_initial(lambda c: np.zeros((len(c), dim)))

    nsteps = int(round(t_end / dt))
    times, cl_hist = [], []
    t0 = time.time()
    for k in range(nsteps):
        t = (k + 1) * dt
        if t <= kick_until:
            st.u_inf[inflow, 1] = kick * U_IN
        else:
            st.u_inf[inflow, 1] = 0.0
        st.step()
        F = st.surrogate_traction()
        qref = 0.5 * U_IN ** 2 * D
        times.append(t)
        cl_hist.append(F[1] / qref)
    times = np.asarray(times)
    cl_hist = np.asarray(cl_hist)
    st_fn = make_strouhal_fn(D=D, U_in=U_IN)
    St, amp = st_fn(times, cl_hist, tail_frac=0.5)
    # count periods in the tail for confidence
    n0 = len(cl_hist) // 2
    c = cl_hist[n0:] - cl_hist[n0:].mean()
    sgn = np.sign(c)
    ncross = int((np.diff(sgn) > 0).sum())
    return (None if St is None else float(St)), float(amp), ncross, \
        round(time.time() - t0, 1), times, cl_hist


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _load():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return {}


def _save(d):
    d["_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, OUT)


def run_re20(levels, config, alphas, max_steps=200, rate_tol=1e-2):
    """config: 'confined' (M1b) or 'lowblock' ([A]-matched, 5% freeslip)."""
    data = _load()
    key = f"re20_cd_{config}"
    data.setdefault(key, {})
    if config == "confined":
        R, ctr, walls = 0.07, (0.3, 0.5), "noslip"
        ref = {"m1b_monolithic": 2.847, "unbounded_classical": 2.0,
               "note": "14% blockage, no-slip walls"}
    else:  # lowblock: 5% blockage, cylinder well upstream, free-slip walls
        R, ctr, walls = 0.025, (0.25, 0.5), "freeslip"
        ref = {"unbounded_classical": 2.0,
               "note": "5% blockage (2R=0.05), free-slip walls, D/downstream=0.75"}
    data[key]["reference"] = ref
    data[key].setdefault("levels", {})
    for lv in levels:
        print(f"\n=== Re20 Cd [{config}] level {lv} ===", flush=True)
        case = build_case(lv, R=R, ctr=ctr, Re=20, walls=walls)
        best, all_r = re20_cd_at_level(case, alphas, max_steps=max_steps,
                                       rate_tol=rate_tol)
        entry = dict(level=lv, h=case["h"], n_free=int(case["n_free"]),
                     n_surrogate_faces=int(case["sf"].elem.size),
                     best=best, alpha_probe=all_r)
        data[key]["levels"][str(lv)] = entry
        if best and best.get("physical"):
            err_mono = abs(best["cd"] - ref.get("m1b_monolithic", np.nan))
            print(f"  L{lv} h={case['h']:.5f}  Cd={best['cd']:.4f} "
                  f"alpha={best['alpha']} steps={best['steps']} "
                  f"blk={best['blockage']:.4f}  err_vs_mono={err_mono:.4f}",
                  flush=True)
        else:
            print(f"  L{lv}: NO physical converged march "
                  f"(alphas tried: {[a for a in alphas]})", flush=True)
        _save(data)
    return data


def run_re100_st(levels, config, alpha, t_end):
    data = _load()
    key = f"re100_strouhal_{config}"
    data.setdefault(key, {})
    if config == "confined":
        R, ctr, walls = 0.07, (0.3, 0.5), "noslip"
        ref = {"m1b_lock": 0.2059, "paper_A_family": 0.170,
               "note": "14% blockage no-slip walls; confinement elevates St"}
    else:
        R, ctr, walls = 0.025, (0.25, 0.5), "freeslip"
        ref = {"paper_A_family": 0.170, "unbounded_lit": 0.164,
               "note": "5% blockage free-slip; comparable to paper [A] Re100"}
    data[key]["reference"] = ref
    data[key].setdefault("levels", {})
    for lv in levels:
        print(f"\n=== Re100 St [{config}] level {lv} (t_end={t_end}) ===",
              flush=True)
        case = build_case(lv, R=R, ctr=ctr, Re=100, walls=walls)
        St, amp, ncross, wall, times, cl = re100_strouhal_at_level(
            case, alpha, t_end=t_end)
        entry = dict(level=lv, h=case["h"], n_free=int(case["n_free"]),
                     alpha=float(alpha), St=St, cl_amp=amp,
                     tail_crossings=ncross, t_end=t_end, wall_s=wall,
                     cl_tail=[round(float(x), 5) for x in cl[-50:]])
        data[key]["levels"][str(lv)] = entry
        print(f"  L{lv}: St={St}  amp={amp:.4f} crossings={ncross} "
              f"wall={wall}s", flush=True)
        _save(data)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True,
                    choices=["re20_confined", "re20_lowblock",
                             "re100_confined", "re100_lowblock"])
    ap.add_argument("--levels", type=int, nargs="+", required=True)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[50, 100, 200, 400])
    ap.add_argument("--alpha", type=float, default=1000.0,
                    help="single alpha for the Strouhal march")
    ap.add_argument("--t-end", type=float, default=40.0)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--rate-tol", type=float, default=1e-2)
    args = ap.parse_args()

    if args.study == "re20_confined":
        run_re20(args.levels, "confined", args.alphas,
                 max_steps=args.max_steps, rate_tol=args.rate_tol)
    elif args.study == "re20_lowblock":
        run_re20(args.levels, "lowblock", args.alphas,
                 max_steps=args.max_steps, rate_tol=args.rate_tol)
    elif args.study == "re100_confined":
        run_re100_st(args.levels, "confined", args.alpha, args.t_end)
    elif args.study == "re100_lowblock":
        run_re100_st(args.levels, "lowblock", args.alpha, args.t_end)
    print("\nDONE:", args.study, "->", OUT, flush=True)


if __name__ == "__main__":
    main()
