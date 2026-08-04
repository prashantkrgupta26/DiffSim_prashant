"""Traction-dissection probe (TD-3): term table + resolution/blockage legs.

Dissects the Nitsche weak-form reaction drag into its three constitutive
terms (consistency+adjoint, penalty, backflow) across four resolution/blockage
legs and computes the bridge ratio (cd_surr vs the consistency-class term).

GPU box:

    bash scripts/remote/gpubox-run.sh \
        ".venv/bin/python tests/gpu_traction_dissect.py" tractdissect

Legs:
  D1: level=7, refine_to=9  (baseline)
  D2: level=7, refine_to=10 (fine)
  D3: level=7, refine_to=11 (very fine; dt fallback on ConvergenceError/non-finite)
  D4: level=7, refine_to=10, plate_L_inv=32 (blockage doubler — L/32)

SIGN FACTS (TD-2):
  - Individual terms may be negative (adjoint, backflow) — do NOT assert positivity.
  - Σ_t == reaction_hist[:,0] at 1e-12 (partition gate).
  - cd_rxn_total uses the same positive-downstream convention as Cd_rxn.

BRIDGE RATIO: cd_surr / cd_rxn_terms["consistency+adjoint"]
  Tests whether bare σ·n ≈ the consistency-class term.
  Guard: if |denominator| < 1e-6 → ratio = NaN with printed diagnostic.

UNITS (octree): L = 1/L_INV, NU = U*L/250, X_C = 5/L_INV.
  RECOMPUTED PER LEG — D4 uses L=1/32 (not 1/16) so NU changes accordingly.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

U_INF = 1.0
RE    = 250.0
Y_C   = 0.5


# ---------------------------------------------------------------------------
# Parameterised reaction-set builder (mirrors _build_reaction_sets from the
# discriminator but accepts per-leg L/X_C/NU so D4 with L_inv=32 works
# without modifying gpu_leakdrag_discriminator.py).
# ---------------------------------------------------------------------------

def _build_reaction_sets_parameterised(level, refine_to, wake_refine,
                                       x_c, y_c, L, nu,
                                       radii=(1.5, 2.5),
                                       alpha=50.0):
    """Build plate-enclosing free-node index sets for the reaction arbiter.

    Functionally identical to gpu_leakdrag_discriminator._build_reaction_sets
    but accepts L/X_C/NU as explicit parameters so that any plate_L_inv is
    supported without modifying the discriminator file.

    Returns
    -------
    list of np.ndarray
        Free-node index arrays, one per radius, excluding BC nodes,
        guaranteed to include all SBM-coupled free-nodes.
    """
    from p2r1a_thin_plate_flow import _build_shell
    import scipy.sparse as sp
    from diffsim.sbm.vector import sbm_vector_dirichlet_twosided

    fx = _build_shell(level, x_c, y_c, L, refine_to=refine_to,
                      wake_refine=wake_refine)
    mesh  = fx["mesh"]
    cons  = fx["cons"]
    dm    = fx["dm"]
    free_coords = mesh.node_coords[cons.free_nodes]
    ndof  = 3  # u_x, u_y, p

    # Outer-boundary mask
    x_min, x_max = free_coords[:, 0].min(), free_coords[:, 0].max()
    y_min, y_max = free_coords[:, 1].min(), free_coords[:, 1].max()
    tol = 1e-10
    bc_mask = (
        (np.abs(free_coords[:, 0] - x_min) < tol) |
        (np.abs(free_coords[:, 0] - x_max) < tol) |
        (np.abs(free_coords[:, 1] - y_min) < tol) |
        (np.abs(free_coords[:, 1] - y_max) < tol)
    )

    # SBM-coupled free-nodes (pattern only — alpha arbitrary)
    T = cons.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    noslip_fn = lambda y: np.zeros((len(y), 2))
    Af_raw, _ = sbm_vector_dirichlet_twosided(
        dm, fx["sfp"], fx["gp"], fx["sfm"], fx["gm"],
        noslip_fn, nu, ndof, alpha=alpha)
    Af_c = (T_vec.T @ Af_raw @ T_vec).tocsr()
    sbm_free_nodes = set(np.unique(Af_c.nonzero()[0] // ndof).tolist())

    dist = np.sqrt((free_coords[:, 0] - x_c) ** 2 +
                   (free_coords[:, 1] - y_c) ** 2)

    sets = []
    for r in radii:
        radius_mask = (dist <= r * L) & ~bc_mask
        radius_nodes = set(np.where(radius_mask)[0].tolist())
        combined = np.array(sorted(radius_nodes | sbm_free_nodes), dtype=np.intp)
        sets.append(combined)
    return sets


# ---------------------------------------------------------------------------
# Core leg runner
# ---------------------------------------------------------------------------

def run_dissect_leg(
    tag,
    alpha=50.0,
    nsteps=8000,
    level=7,
    refine_to=9,
    wake_refine=9,
    plate_L_inv=16,
    dt=5e-4,
    device="cuda:0",
    mono_solver="cudss",
    assembly="device",
    t_start_lu=24.0,
):
    """Run one traction-dissection leg and return the analysis dict.

    Geometry is derived from plate_L_inv:
      L     = 1 / plate_L_inv
      X_C   = 5 / plate_L_inv
      NU    = U_INF * L / RE          (changes with L — NOT a constant!)

    Parameters
    ----------
    tag : str
        Short leg identifier (e.g. "D1", "D2") used in filenames.
    alpha : float
        SBM Nitsche penalty parameter.
    nsteps : int
        Number of time steps.
    level : int
        Uniform base octree level.
    refine_to : int or None
        Adaptive plate-refinement level (None → uniform).
    wake_refine : int or None
        Wake-band refinement level.
    plate_L_inv : int
        Inverse plate length (16 → L=1/16; 32 → L=1/32).
    dt : float
        Time-step size.
    device : str
        Device string ("cuda:0", "cpu", …).
    mono_solver : str
        Monolithic solver backend ("cudss", "splu", …).
    assembly : str
        Assembly backend ("device" or "host").
    t_start_lu : float
        Statistics start time in convective units (L/U).

    Returns
    -------
    dict with keys:
        tag, cd_rxn_total, cd_rxn_terms, cd_surr, bridge_ratio,
        St, dt_used, elapsed.
        Plus internal raw arrays (prefixed with _) used by the CPU gate:
        _reaction_hist, _reaction_terms_hist, _reaction_terms_names.
    """
    from p2r1a_thin_plate_flow import run_flow_past
    from diffsim.postproc.shedding import strouhal

    # Derived geometry (per-leg! NU changes with L)
    L    = 1.0 / float(plate_L_inv)
    X_C  = 5.0 / float(plate_L_inv)
    NU   = U_INF * L / RE

    t_start = t_start_lu * L / U_INF

    # Build reaction sets (one set — we use set 0 for term dissection)
    rxn_sets = _build_reaction_sets_parameterised(
        level, refine_to, wake_refine,
        x_c=X_C, y_c=Y_C, L=L, nu=NU, alpha=alpha,
    )

    # Surrogate Cd accumulator (post t_start)
    acc_surr  = 0.0
    n_avg     = 0

    def _cb(step, t, u_full, p_full, cd_step):
        nonlocal acc_surr, n_avg
        if t < t_start:
            return
        acc_surr += cd_step
        n_avg    += 1

    t0 = time.time()
    r = run_flow_past(
        level=level, refine_to=refine_to, wake_refine=wake_refine,
        nsteps=nsteps, dt=dt, nu=NU, U_inf=U_INF,
        plate_xc=X_C, plate_yc=Y_C, plate_L=L,
        pert_eps=0.03, pert_t_end=0.5, alpha=alpha,
        mono_solver=mono_solver, assembly=assembly,
        device=device, verbose=False, on_step=_cb,
        reaction_sets=rxn_sets,
        reaction_terms=True,
    )
    elapsed = time.time() - t0

    t_arr = np.arange(1, nsteps + 1) * dt
    try:
        St, _ = strouhal(t_arr, np.asarray(r["cl"]), U_INF, L)
        St = float(St)
    except Exception:
        St = float("nan")

    # Raw histories
    rh   = r["reaction_hist"]          # [nsteps, 1]
    rth  = r["reaction_terms_hist"]    # [nsteps, 3]
    names = r["reaction_terms_names"]  # list[str]

    # Time-average over t >= t_start (physical)
    t_mask = t_arr >= t_start
    if not np.any(t_mask):
        # No steps past t_start (tiny runs, t_start_lu=0 with nsteps=4)
        t_mask = np.ones(nsteps, dtype=bool)

    cd_rxn_total = float(np.mean(rh[t_mask, 0]))

    # Per-term time averages
    cd_rxn_terms = {}
    for i, name in enumerate(names):
        cd_rxn_terms[name] = float(np.mean(rth[t_mask, i]))

    # Surrogate Cd (post t_start — from on_step accumulator)
    cd_surr = acc_surr / max(n_avg, 1)

    # Bridge ratio: cd_surr / consistency+adjoint term
    con_term = cd_rxn_terms.get("consistency+adjoint", float("nan"))
    if abs(con_term) < 1e-6:
        print(
            f"[{tag}] bridge_ratio NaN-guarded: |consistency+adjoint|={abs(con_term):.3e} < 1e-6",
            flush=True,
        )
        bridge_ratio = float("nan")
    else:
        bridge_ratio = cd_surr / con_term

    # Save npz
    os.makedirs("results", exist_ok=True)
    np.savez(
        f"results/tractdissect_{tag}.npz",
        t=t_arr,
        cd=r["cd"],
        cl=r["cl"],
        reaction_hist=rh,
        reaction_terms_hist=rth,
        names=np.array(names, dtype=object),
    )

    return dict(
        tag=tag,
        cd_rxn_total=cd_rxn_total,
        cd_rxn_terms=cd_rxn_terms,
        cd_surr=cd_surr,
        bridge_ratio=bridge_ratio,
        St=St,
        dt_used=dt,
        elapsed=elapsed,
        # Internal raw arrays for CPU gate partition checks
        _reaction_hist=rh,
        _reaction_terms_hist=rth,
        _reaction_terms_names=names,
    )


# ---------------------------------------------------------------------------
# Main: D1, D2, D3 (with dt fallback), D4 (L/32@r10)
# ---------------------------------------------------------------------------

def _print_leg(out):
    """Print per-leg summary immediately with flush=True."""
    tag   = out["tag"]
    terms = out["cd_rxn_terms"]
    names_order = ["consistency+adjoint", "penalty", "backflow"]
    terms_str = "  ".join(
        f"{n}={terms.get(n, float('nan')):.4f}" for n in names_order
    )
    br = out["bridge_ratio"]
    br_str = f"{br:.4f}" if np.isfinite(br) else "NaN"
    print(
        f"[{tag}] cd_rxn_total={out['cd_rxn_total']:.4f}  "
        f"cd_surr={out['cd_surr']:.4f}  "
        f"bridge={br_str}  "
        f"St={out['St']:.4f}  "
        f"dt_used={out['dt_used']:.2e}  "
        f"elapsed={out['elapsed']:.1f}s",
        flush=True,
    )
    print(f"  terms: {terms_str}", flush=True)


if __name__ == "__main__":
    from diffsim.errors import ConvergenceError

    _device   = os.environ.get("DEVICE",     "cuda:0")
    _solver   = os.environ.get("MONO_SOLVER","cudss")
    _assembly = os.environ.get("ASSEMBLY",   "device")
    _nsteps   = int(os.environ.get("NSTEPS", "8000"))

    results = []

    # ------------------------------------------------------------------
    # D1: level=7, refine_to=9 (baseline)
    # ------------------------------------------------------------------
    print("\n[TD-3] D1: level=7, refine_to=9, L_inv=16, dt=5e-4 ...", flush=True)
    d1 = run_dissect_leg(
        tag="D1", nsteps=_nsteps, level=7, refine_to=9, wake_refine=9,
        plate_L_inv=16, dt=5e-4, device=_device, mono_solver=_solver,
        assembly=_assembly,
    )
    _print_leg(d1)
    results.append(d1)

    # ------------------------------------------------------------------
    # D2: level=7, refine_to=10
    # ------------------------------------------------------------------
    print("\n[TD-3] D2: level=7, refine_to=10, L_inv=16, dt=5e-4 ...", flush=True)
    d2 = run_dissect_leg(
        tag="D2", nsteps=_nsteps, level=7, refine_to=10, wake_refine=9,
        plate_L_inv=16, dt=5e-4, device=_device, mono_solver=_solver,
        assembly=_assembly,
    )
    _print_leg(d2)
    results.append(d2)

    # ------------------------------------------------------------------
    # D3: level=7, refine_to=11 — with dt fallback on failure
    # ------------------------------------------------------------------
    print("\n[TD-3] D3: level=7, refine_to=11, L_inv=16, dt=5e-4 ...", flush=True)
    _d3_dt_used = 5e-4
    _d3_nsteps  = _nsteps
    try:
        d3 = run_dissect_leg(
            tag="D3", nsteps=_d3_nsteps, level=7, refine_to=11, wake_refine=9,
            plate_L_inv=16, dt=_d3_dt_used, device=_device, mono_solver=_solver,
            assembly=_assembly,
        )
        # Check for non-finite cd
        if not np.isfinite(d3["cd_rxn_total"]):
            raise ValueError(f"D3 cd_rxn_total non-finite: {d3['cd_rxn_total']}")
    except (ConvergenceError, ValueError) as e:
        print(
            f"[TD-3] D3 dt=5e-4 failed ({type(e).__name__}: {e}); "
            "retrying with dt=2.5e-4, nsteps=16000 ...",
            flush=True,
        )
        _d3_dt_used = 2.5e-4
        _d3_nsteps  = 16000
        d3 = run_dissect_leg(
            tag="D3", nsteps=_d3_nsteps, level=7, refine_to=11, wake_refine=9,
            plate_L_inv=16, dt=_d3_dt_used, device=_device, mono_solver=_solver,
            assembly=_assembly,
        )
    d3 = dict(d3, dt_used=_d3_dt_used)  # ensure dt_used reflects actual dt
    _print_leg(d3)
    results.append(d3)

    # ------------------------------------------------------------------
    # D4: plate_L_inv=32, refine_to=10 (32 cells/plate), dt=2.5e-4
    # ------------------------------------------------------------------
    print(
        "\n[TD-3] D4: level=7, refine_to=10, L_inv=32, dt=2.5e-4 (blockage doubler) ...",
        flush=True,
    )
    d4 = run_dissect_leg(
        tag="D4", nsteps=_nsteps, level=7, refine_to=10, wake_refine=9,
        plate_L_inv=32, dt=2.5e-4, device=_device, mono_solver=_solver,
        assembly=_assembly,
    )
    _print_leg(d4)
    results.append(d4)

    # ------------------------------------------------------------------
    # Consolidated table
    # ------------------------------------------------------------------
    names_order = ["consistency+adjoint", "penalty", "backflow"]
    hdr = (
        f"{'Tag':>4} {'Cd_rxn':>8} "
        + " ".join(f"{n:>20}" for n in names_order)
        + f" {'Cd_surr':>8} {'bridge':>8} {'St':>7} {'dt':>9} {'s':>7}"
    )
    print("\n" + "=" * len(hdr), flush=True)
    print("TRACTION DISSECTION TABLE", flush=True)
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for r in results:
        t = r["cd_rxn_terms"]
        br = r["bridge_ratio"]
        br_str = f"{br:8.4f}" if np.isfinite(br) else "     NaN"
        row = (
            f"{r['tag']:>4} {r['cd_rxn_total']:8.4f} "
            + " ".join(f"{t.get(n, float('nan')):20.6f}" for n in names_order)
            + f" {r['cd_surr']:8.4f} {br_str} {r['St']:7.4f} "
            f"{r['dt_used']:9.2e} {r['elapsed']:7.1f}"
        )
        print(row, flush=True)
    print("=" * len(hdr), flush=True)
    print("TRACTDISSECT-OK", flush=True)
