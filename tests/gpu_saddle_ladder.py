"""Track-A R2b iteration-ladder harness.

Task A2: measures per-step Krylov-iteration counts for the monolithic saddle
solve across four problem sizes ranging from ~0.3M DOFs (2-D r9) to ~9.24M
DOFs (3-D base-L7/band-r9).  The kill-gate is iteration growth ≤ ~2× over
1M→10M; if growth exceeds that threshold the campaign concludes negative.

Ladder points (copied verbatim from the reference GPU scripts listed below):

  2-D r9   (~0.3M DOF)   level=7, refine_to=9, wake_refine=9
    Ref: tests/gpu_re250_corrected.py (REFINE=9, WAKE=9)
         tests/gpu_dtladder_re250.py  (BASE dict: level=7, refine_to=9, wake_refine=9)
    Physics: corrected octree units — L=1/16, NU=U*L/Re, x_c=5/16

  2-D r11  (~1M+ DOF)    level=7, refine_to=11
    Ref: tests/gpu_traction_dissect.py  D3 leg (level=7, refine_to=11)
    Physics: same corrected units as r9

  3-D L6 uniform (~1.1M DOF)   level=6, refine_to=None
    Ref: tests/gpu_gh200_ladder.py  Phase 2 "r6b9" and GH200_CONFIG
         tests/p2r1c_thin_plate_flow_3d.py  GH200_CONFIG (level=6, nu=0.004, dt=0.005)
    Physics: PHASE2_NU=0.004, PHASE2_U_INF=1.0, PHASE2_DT=0.005

  3-D base-L7/band-r9 (~9.24M DOF, WP0 mesh)   level=7, refine_to=9
    Ref: tests/gpu_gh200_ladder.py  Phase 2 rung "r7b9" (base=7, refine_to=9)
    Physics: same as 3-D L6

The `__main__` block runs all points for SOLVERS (default "fgmres_bdiag", env
var SADDLE_SOLVERS comma-separated) and prints the results table; saves one
npz per point under results/.  Prints SADDLE-LADDER-OK sentinel at the end.

CPU gate: tests/test_saddle_ladder_cpu.py calls run_ladder_point with
  level=4, nsteps=2, fgmres_bdiag, cpu  — tiny and always runs in CI.

NOT pytest-collected (no test_ prefix in filename).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Ladder-point configurations (copied from reference scripts)
# ---------------------------------------------------------------------------

# --- 2-D corrected units (mirrors gpu_re250_corrected.py + gpu_dtladder_re250.py) ---
_L_INV_2D = 16          # plate = 1/16 of domain (both reference scripts)
_L_2D = 1.0 / _L_INV_2D        # plate length in octree units
_U_2D = 1.0
_RE_2D = 250.0
_NU_2D = _U_2D * _L_2D / _RE_2D           # NU = U*L/Re in corrected units

# Corrected 2-D plate config (same in both reference scripts)
_PLATE_2D = dict(
    plate_xc=5.0 / _L_INV_2D,     # 5 plate-lengths upstream  = 0.3125
    plate_yc=0.5,
    plate_L=_L_2D,                 # 1/16 = 0.0625
    nu=_NU_2D,
    U_inf=_U_2D,
    pert_eps=0.03,
    pert_t_end=0.5,
    alpha=50.0,
)

# 2-D r9: level=7, refine_to=9, wake_refine=9
# Source: gpu_re250_corrected.py (REFINE=9, WAKE=9); gpu_dtladder_re250.py BASE dict
LADDER_2D_R9 = dict(
    tag="2d-r9",
    dim=2,
    level=7,
    refine_to=9,
    wake_refine=9,
    dt=5e-4,         # from gpu_re250_corrected.py DT default
    nsteps=5,        # GPU run uses 8000; ladder only needs iters_per_step
    **_PLATE_2D,
)

# 2-D r11: level=7, refine_to=11, wake_refine=9
# Source: gpu_traction_dissect.py D3 leg (level=7, refine_to=11, wake_refine=9)
LADDER_2D_R11 = dict(
    tag="2d-r11",
    dim=2,
    level=7,
    refine_to=11,
    wake_refine=9,      # matches D3 reference leg config
    dt=5e-4,
    nsteps=5,
    **_PLATE_2D,
)

# --- 3-D configs (mirrors gpu_gh200_ladder.py Phase 2) ---
# PHASE2_NU=0.004, PHASE2_U_INF=1.0, PHASE2_DT=0.005 (from gpu_gh200_ladder.py)
_NU_3D = 0.004
_U_3D = 1.0
_DT_3D = 0.005

# 3-D plate geometry defaults from run_flow_past_3d / GH200_CONFIG
_PLATE_3D = dict(
    plate_xc=0.375,
    plate_yc=0.5,
    plate_zc=0.5,
    plate_half_y=0.125,
    plate_half_z=0.125,
    nu=_NU_3D,
    U_inf=_U_3D,
    alpha=50.0,
)

# 3-D L6 uniform (~1.1M DOF)
# Source: gpu_gh200_ladder.py Phase 2 rung "r6b9" uses (base=6, refine_to=9);
#         but the ~1.1M "uniform L6" point from the brief = level=6 uniform.
#         GH200_CONFIG: level=6, nu=0.004, dt=0.005 — that is the uniform L6 point.
LADDER_3D_L6 = dict(
    tag="3d-L6",
    dim=3,
    level=6,
    refine_to=None,   # uniform
    dt=_DT_3D,
    nsteps=5,
    **_PLATE_3D,
)

# 3-D L7 uniform (~8.6M DOF: 129^3 nodes x 4 dof/node)
# GH200 hold session (2026-07-29): uniform mesh => no hanging nodes =>
# identity_T => the device-assembly ChunkTable path applies (the band-refined
# 3d-L7r9 point hits the Warp 2^31 slot-array ceiling in the
# constraint-expansion scatter — see docs/dev/2026-07-28-track-a2-campaign.md
# §10.2).  This point separates device-assembly-at-scale from that blocker.
LADDER_3D_L7 = dict(
    tag="3d-L7",
    dim=3,
    level=7,
    refine_to=None,   # uniform
    dt=_DT_3D,
    nsteps=5,
    **_PLATE_3D,
)

# 3-D base-L7/band-r9 (~9.24M DOF, WP0 mesh)
# Source: gpu_gh200_ladder.py Phase 2 rung "r7b9" (base=7, refine_to=9)
LADDER_3D_L7R9 = dict(
    tag="3d-L7r9",
    dim=3,
    level=7,
    refine_to=9,
    dt=_DT_3D,
    nsteps=5,
    **_PLATE_3D,
)

# 3-D L8 uniform capacity rung (~67.9M DOF: 257^3 = 16,974,593 nodes x 4 dof)
# GH200 hold session (2026-07-29): uniform-mesh device-assembly capacity probe.
# Leg 4 measured ~34.8 GiB HBM at 8.58M DOF; naive scaling puts this rung at
# ~275 GiB > 95 GiB HBM — the DELIVERABLE is the spill/OOM behavior (Warp
# mempool on coupled Grace-Hopper memory), not an expected clean pass.
# nsteps=2 (capacity semantics); run with SADDLE_POINTS=3d-L8 only.
LADDER_3D_L8 = dict(
    tag="3d-L8",
    dim=3,
    level=8,
    refine_to=None,   # uniform
    dt=_DT_3D,
    nsteps=2,
    **_PLATE_3D,
)

# Ordered ladder points
ALL_POINTS = [LADDER_2D_R9, LADDER_2D_R11, LADDER_3D_L6, LADDER_3D_L7,
              LADDER_3D_L7R9, LADDER_3D_L8]


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_ladder_point(tag, solver, dim, nsteps=5, device="cpu", assembly=None,
                     pcd_inner=None, pcd_ap_inner=None, **cfg):
    """March nsteps of the MONOLITHIC driver and capture per-step iteration counts.

    Parameters
    ----------
    tag : str
        Short identifier for this ladder point (used in filenames and output).
    solver : str
        Monolithic solver backend key, e.g. "fgmres_bdiag", "cudss".
    dim : int
        Problem dimension: 2 calls run_flow_past; 3 calls run_flow_past_3d.
    nsteps : int
        Number of BDF steps to march.
    device : str
        Device string passed to the driver ("cpu", "cuda:0", etc.).
    assembly : str or None
        Assembly backend passed to the driver: "host" (default host path),
        "device" (DeviceNSAssembler), or None (don't pass the kwarg —
        preserves the driver's own default, i.e. today's behavior exactly).
        Set via env SADDLE_ASSEMBLY.
    pcd_inner : str or None
        PCD F-block (velocity) inner-solve backend: "jacobi" | "amgx" | None.
        None (default) omits the kwarg, preserving the driver's own default
        ("jacobi"). Set via env SADDLE_PCD_INNER.  Only relevant for
        solver="fgmres_pcd"; ignored by bdiag/cudss/etc.
    pcd_ap_inner : str or None
        PCD Ap-block (pressure Laplacian) inner-solve backend: "jacobi" | "amgx" | None.
        None (default) omits the kwarg, preserving the driver's own default
        ("jacobi"). Set via env SADDLE_PCD_AP_INNER.  Only relevant for
        solver="fgmres_pcd"; ignored by bdiag/cudss/etc.
    **cfg :
        Additional kwargs forwarded to the driver (level, refine_to, nu, dt, etc.).

    Returns
    -------
    dict with keys:
        tag, solver, dofs, iters_per_step (list), iters_mean, s_per_step,
        converged, inner_stats (dict or None — T1 telemetry from fgmres_pcd)
    """
    stats = []    # list to collect per-step iteration counts

    # Extract driver-specific kwargs from cfg
    if dim == 2:
        from p2r1a_thin_plate_flow import run_flow_past

        # Keys accepted by run_flow_past; forward only those present in cfg
        _2d_keys = {
            "level", "nsteps", "dt", "U_inf", "nu", "alpha",
            "plate_xc", "plate_yc", "plate_L",
            "refine_to", "wake_refine", "band_cells",
            "pert_eps", "pert_t_end",
            "verbose",
        }
        kw = {k: v for k, v in cfg.items() if k in _2d_keys}
        kw["nsteps"] = nsteps
        kw.setdefault("verbose", False)

        # assembly pass-through: only forward when caller requested it explicitly
        asm_kw = {} if assembly is None else {"assembly": assembly}
        # pcd_inner pass-through: only forward when explicitly set (T4 AMGX route)
        pcd_kw = {} if pcd_inner is None else {"pcd_inner": pcd_inner}
        # pcd_ap_inner pass-through: only forward when explicitly set (T5 AMG-on-Ap)
        pcd_ap_kw = {} if pcd_ap_inner is None else {"pcd_ap_inner": pcd_ap_inner}

        t0 = time.time()
        res = run_flow_past(
            mono_solver=solver,
            device=device,
            solver_stats=stats,
            **asm_kw,
            **pcd_kw,
            **pcd_ap_kw,
            **kw,
        )
        elapsed = time.time() - t0

        # DOF count: nfree * ndof (reconstruct from the returned cd shape + mesh)
        # Cheapest proxy: total free DOFs = n_nodes_free * 3 for 2-D (ndof=3).
        # We get this by building the mesh once — but to avoid a double build,
        # we read it from the driver result when available, or estimate from
        # the mesh size.  The simplest approach: build a tiny mesh to count DOFs
        # only for the GPU runs; for the CPU gate the exact count is not critical.
        # For the ladder table we need the actual DOF count: extract from result
        # if the driver exposes it, otherwise compute from known mesh params.
        # Since run_flow_past does not return nfree, we compute it separately
        # using _build_shell at the same config.
        try:
            dofs = _count_dofs_2d(**kw)
        except Exception:
            dofs = -1   # non-fatal: table shows -1 if count fails

    elif dim == 3:
        from p2r1c_thin_plate_flow_3d import run_flow_past_3d

        _3d_keys = {
            "level", "nsteps", "dt", "U_inf", "nu", "alpha",
            "plate_xc", "plate_yc", "plate_zc",
            "plate_half_y", "plate_half_z",
            "refine_to", "band_cells",
            "verbose",
        }
        kw = {k: v for k, v in cfg.items() if k in _3d_keys}
        kw["nsteps"] = nsteps
        kw.setdefault("verbose", False)

        # assembly pass-through: only forward when caller requested it explicitly
        asm_kw = {} if assembly is None else {"assembly": assembly}
        # pcd_inner pass-through: only forward when explicitly set (T4 AMGX route)
        pcd_kw = {} if pcd_inner is None else {"pcd_inner": pcd_inner}
        # pcd_ap_inner pass-through: only forward when explicitly set (T5 AMG-on-Ap)
        pcd_ap_kw = {} if pcd_ap_inner is None else {"pcd_ap_inner": pcd_ap_inner}

        t0 = time.time()
        res = run_flow_past_3d(
            mono_solver=solver,
            device=device,
            solver_stats=stats,
            **asm_kw,
            **pcd_kw,
            **pcd_ap_kw,
            **kw,
        )
        elapsed = time.time() - t0

        try:
            dofs = _count_dofs_3d(**kw)
        except Exception:
            dofs = -1

    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")

    cd = res.get("cd", np.array([]))
    converged = bool(np.all(np.isfinite(cd))) and len(cd) == nsteps

    iters_mean = float(np.mean(stats)) if stats else float("nan")
    s_per_step = elapsed / nsteps if nsteps > 0 else float("nan")

    # T1/T4 inner_stats: published by fgmres_pcd via _LAST_INNER_STATS global.
    # Only populated on the fgmres_pcd path; None on bdiag/cudss/etc.
    _inner_stats = None
    if solver == "fgmres_pcd":
        try:
            from diffsim.solvers.linsolve import _LAST_INNER_STATS
            _inner_stats = _LAST_INNER_STATS[0]
        except Exception:
            pass

    return dict(
        tag=tag,
        solver=solver,
        dim=dim,
        dofs=dofs,
        iters_per_step=stats,
        iters_mean=iters_mean,
        s_per_step=s_per_step,
        converged=converged,
        cd_last=float(cd[-1]) if len(cd) > 0 else float("nan"),
        nsteps_done=len(cd),
        inner_stats=_inner_stats,
    )


def _count_dofs_2d(level, nu=0.1, U_inf=1.0, alpha=50.0,
                   plate_xc=0.375, plate_yc=0.5, plate_L=0.25,
                   refine_to=None, wake_refine=None, band_cells=2,
                   nsteps=5, dt=0.01, **_):
    """Count free DOFs for a 2-D plate mesh configuration."""
    from p2r1a_thin_plate_flow import _build_shell
    fx = _build_shell(level, plate_xc, plate_yc, plate_L, dim=2,
                      refine_to=refine_to, wake_refine=wake_refine,
                      band_cells=band_cells)
    nfree = fx["cons"].T.shape[1]
    ndof = 3   # u_x, u_y, p
    return nfree * ndof


def _count_dofs_3d(level, nu=0.1, U_inf=1.0, alpha=50.0,
                   plate_xc=0.375, plate_yc=0.5, plate_zc=0.5,
                   plate_half_y=0.125, plate_half_z=0.125,
                   refine_to=None, band_cells=2,
                   nsteps=5, dt=0.01, **_):
    """Count free DOFs for a 3-D plate mesh configuration."""
    from p2r1c_thin_plate_flow_3d import _build_shell_3d
    fx = _build_shell_3d(level, plate_xc, plate_yc, plate_zc,
                         plate_half_y, plate_half_z,
                         refine_to=refine_to, band_cells=band_cells)
    nfree = fx["cons"].T.shape[1]
    ndof = 4   # u_x, u_y, u_z, p
    return nfree * ndof


# ---------------------------------------------------------------------------
# Result printing + npz save
# ---------------------------------------------------------------------------

def _print_row(r):
    iters = r["iters_per_step"]
    iters_str = (f"[{','.join(str(i) for i in iters)}]"
                 if iters else "[]")
    print(
        f"  {r['tag']:12s}  solver={r['solver']:16s}  "
        f"dofs={r['dofs']:>9}  "
        f"iters_mean={r['iters_mean']:>7.1f}  "
        f"s/step={r['s_per_step']:>7.3f}  "
        f"converged={r['converged']}  "
        f"iters={iters_str}",
        flush=True,
    )


def _save_npz(result, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    tag = result["tag"].replace("/", "_")
    sol = result["solver"]
    path = os.path.join(out_dir, f"saddle_ladder_{tag}_{sol}.npz")

    # Build extra kwargs for inner_stats (T1/T4 telemetry from fgmres_pcd).
    # Each block (F, Ap, Mp) is stored as a flat dict-of-scalars; we prefix
    # each key so the npz remains flat (no object dtypes or nested arrays).
    # When inner_stats is None (bdiag/cudss paths) these kwargs are empty.
    _ist_kw = {}
    ist = result.get("inner_stats")
    if ist is not None:
        for blk in ("F", "Ap", "Mp"):
            if blk in ist:
                for k, v in ist[blk].items():
                    _ist_kw[f"ist_{blk}_{k}"] = np.array(v)

    np.savez(
        path,
        tag=np.array(result["tag"]),
        solver=np.array(result["solver"]),
        dim=np.array(result["dim"]),
        dofs=np.array(result["dofs"]),
        iters_per_step=np.array(result["iters_per_step"], dtype=np.float64),
        iters_mean=np.array(result["iters_mean"]),
        s_per_step=np.array(result["s_per_step"]),
        converged=np.array(result["converged"]),
        cd_last=np.array(result["cd_last"]),
        **_ist_kw,
    )
    print(f"    saved -> {path}", flush=True)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warp as wp
    wp.init()

    DEVICE = os.environ.get("SADDLE_DEVICE", "cuda:0")
    SOLVER_ENV = os.environ.get("SADDLE_SOLVERS", "fgmres_bdiag")
    SOLVERS = [s.strip() for s in SOLVER_ENV.split(",") if s.strip()]
    NSTEPS = int(os.environ.get("SADDLE_NSTEPS", "5"))
    # SADDLE_ASSEMBLY: "device" or "host" => forward to run_ladder_point;
    # unset (or empty) => None, which preserves today's driver default ("host").
    _asm_env = os.environ.get("SADDLE_ASSEMBLY", "").strip()
    ASSEMBLY = _asm_env if _asm_env else None

    # SADDLE_PCD_INNER: "amgx" | "jacobi" => PCD F-block inner-solve backend (T4).
    # unset (or empty) => None, which preserves the driver's default ("jacobi").
    _pcd_inner_env = os.environ.get("SADDLE_PCD_INNER", "").strip()
    PCD_INNER = _pcd_inner_env if _pcd_inner_env else None

    # SADDLE_PCD_AP_INNER: "amgx" | "jacobi" => PCD Ap-block inner-solve backend (T5).
    # unset (or empty) => None, which preserves the driver's default ("jacobi").
    _pcd_ap_inner_env = os.environ.get("SADDLE_PCD_AP_INNER", "").strip()
    PCD_AP_INNER = _pcd_ap_inner_env if _pcd_ap_inner_env else None

    # SADDLE_POINTS: comma-separated subset of point tags to run in this process.
    # Used to run each ladder point in a SEPARATE PROCESS (the brief requirement:
    # a single-process multi-leg ladder previously accumulated cross-leg memory
    # and OOM'd).  Empty / absent => run all points.
    # Example: SADDLE_POINTS=2d-r9  SADDLE_POINTS=3d-L6,3d-L7r9
    POINTS_ENV = os.environ.get("SADDLE_POINTS", "")
    _point_tags = {t.strip() for t in POINTS_ENV.split(",") if t.strip()}
    ACTIVE_POINTS = [p for p in ALL_POINTS
                     if not _point_tags or p["tag"] in _point_tags]
    if not ACTIVE_POINTS:
        print(f"[saddle-ladder] SADDLE_POINTS={POINTS_ENV!r} matched no "
              f"known tags {[p['tag'] for p in ALL_POINTS]} — nothing to run.",
              flush=True)
        raise SystemExit(1)

    print(f"[saddle-ladder] device={DEVICE}  solvers={SOLVERS}  "
          f"nsteps={NSTEPS}  assembly={ASSEMBLY}  pcd_inner={PCD_INNER}  "
          f"pcd_ap_inner={PCD_AP_INNER}  "
          f"points={[p['tag'] for p in ACTIVE_POINTS]}",
          flush=True)
    print("=" * 78, flush=True)

    all_results = []
    for point in ACTIVE_POINTS:
        for solver in SOLVERS:
            tag = point["tag"]
            dim = point["dim"]
            print(f"\n--- {tag}  solver={solver} ---", flush=True)
            try:
                r = run_ladder_point(
                    tag=tag,
                    solver=solver,
                    dim=dim,
                    nsteps=NSTEPS,
                    device=DEVICE,
                    assembly=ASSEMBLY,
                    pcd_inner=PCD_INNER,
                    pcd_ap_inner=PCD_AP_INNER,
                    **{k: v for k, v in point.items()
                       if k not in ("tag", "dim", "nsteps")},
                )
                _print_row(r)
                _save_npz(r)
                all_results.append(r)
            except Exception as exc:
                import traceback
                print(f"  FAILED {tag}/{solver}: {type(exc).__name__}: {exc}",
                      flush=True)
                traceback.print_exc()
                all_results.append(dict(
                    tag=tag, solver=solver, dim=dim, dofs=-1,
                    iters_per_step=[], iters_mean=float("nan"),
                    s_per_step=float("nan"), converged=False,
                    cd_last=float("nan"), nsteps_done=0,
                    inner_stats=None,
                ))

    # Summary table
    print("\n" + "=" * 78, flush=True)
    print(" SADDLE ITERATION-LADDER SUMMARY", flush=True)
    print("=" * 78, flush=True)
    print(f"  {'tag':12s}  {'solver':16s}  {'dofs':>9s}  "
          f"{'iters_mean':>10s}  {'s/step':>7s}  {'converged':>9s}",
          flush=True)
    print("-" * 78, flush=True)
    for r in all_results:
        imean = f"{r['iters_mean']:.1f}" if np.isfinite(r['iters_mean']) else "N/A"
        sps = f"{r['s_per_step']:.3f}" if np.isfinite(r['s_per_step']) else "N/A"
        print(f"  {r['tag']:12s}  {r['solver']:16s}  {r['dofs']:>9}  "
              f"{imean:>10s}  {sps:>7s}  {str(r['converged']):>9s}",
              flush=True)

    # Kill-gate check: if any 2 consecutive rungs have iters_mean available,
    # check the growth factor.
    print("\n[saddle-ladder] KILL-GATE check (<=~2x growth across size rungs):",
          flush=True)
    by_solver = {}
    for r in all_results:
        by_solver.setdefault(r["solver"], []).append(r)
    for sol, rows in by_solver.items():
        valid = [(r["dofs"], r["iters_mean"])
                 for r in rows
                 if r["dofs"] > 0 and np.isfinite(r["iters_mean"])
                 and r["iters_mean"] > 0]
        if len(valid) < 2:
            print(f"  {sol}: insufficient valid points for kill-gate check", flush=True)
            continue
        min_mean = min(v for _, v in valid)
        max_mean = max(v for _, v in valid)
        growth = max_mean / min_mean if min_mean > 0 else float("inf")
        gate = "PASS" if growth <= 2.0 else "FAIL (>~2x — campaign concludes negative)"
        print(f"  {sol}: min_iters={min_mean:.1f}  max_iters={max_mean:.1f}  "
              f"growth={growth:.2f}x  -> {gate}", flush=True)

    print("\nSADDLE-LADDER-OK", flush=True)
