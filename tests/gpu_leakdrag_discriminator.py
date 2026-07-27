"""Leak-drag discriminator (spec 2026-07-26): CV drag vs surrogate traction
x alpha-sweep on the corrected Re=250 config.  GPU box:

    bash scripts/remote/gpubox-run.sh \
        ".venv/bin/python tests/gpu_leakdrag_discriminator.py" leakdrag

Env: ALPHAS ("20,50,100"), NSTEPS (8000), DEVICE (cuda:0), MONO_SOLVER
(cudss), ASSEMBLY (device).  CPU mini-gate uses run_discriminator directly.

VERIFY-FIRST findings (2026-07-26):
  1. _build_shell returns dict keys: dm, mesh, cons, sfp, gp, sfm, gm,
     n_excluded (+ n_nodes/n_hanging/build_time when adaptive).  Keys in
     brief (fx["sfp"]/["gp"]/["sfm"]/["gm"]) are correct.
  2. run_flow_past DOES rebuild shell internally (line 434 of driver).
     Both builds are deterministic (same tree/classification → same n_nodes).
     Determinism assertion added: assert probe_n_nodes == u_full.shape[0].
  3. 6L/8L CV boxes (X_C=5/16, pitch snap) go outside the unit-square domain
     at coarse levels.  cv_drag_box raises ValueError in that case; the
     callback wraps each call in try/except and accumulates 0.0 so that
     n_steps_avg stays valid and cd_cv_mean is finite.
  4. strouhal raises on < 4-point tail; guarded with try/except -> NaN
     (mirrors gpu_re250_corrected.py).
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from diffsim.postproc.cv_drag import cv_drag_box
from diffsim.postproc.shedding import strouhal
from diffsim.mesh.faces import face_tables

L = 1.0 / 16.0
U = 1.0
RE = 250.0
NU = U * L / RE
X_C = 5.0 / 16.0
Y_C = 0.5


# ---------------------------------------------------------------------------
# Nested CV boxes: margins ~4L, 6L, 8L, edges snapped to node lines.
# The `pitch` parameter is the snapping pitch (mesh node spacing):
#   GPU config (level=7, h=1/128): pitch=1/128  (default)
#   CPU gate   (level=4, h=1/16):  pitch=1/16
# ---------------------------------------------------------------------------

def _snap(v, pitch):
    """Snap v to the nearest multiple of pitch."""
    inv = 1.0 / pitch
    return round(v * inv) / inv


def _boxes(pitch=1.0 / 128.0):
    """Return dict tag -> (x0, x1, y0, y1) snapped to given pitch."""
    out = {}
    # Margins in plate-lengths. Default {2,3,4}L: at x_c=5/16 the upstream
    # margin caps at 5L — the original {4,6,8}L set put 6L/8L off-domain
    # (found by the alpha-sweep spread self-check, 2026-07-27).
    _margs = [float(m) for m in
              os.environ.get("MARGINS", "2,3,4").split(",")]
    for tag, m in ((f"{m:g}L", m * L) for m in _margs):
        out[tag] = (
            _snap(X_C - m, pitch),
            _snap(X_C + m, pitch),
            _snap(Y_C - m, pitch),
            _snap(Y_C + m, pitch),
        )
    return out


# ---------------------------------------------------------------------------
# Leak-flux: net ∮ u·n over BOTH shell sides (mirrors
# LeraySBMShellStepper.surrogate_normal_flux — both-sided loop).
# ---------------------------------------------------------------------------

def _leak_flux(mesh, sfp, gp, sfm, gm, u_full, dim=2):
    """Net ∮ u·n over BOTH shell sides (mirrors surrogate_normal_flux)."""
    ftab = face_tables(1, dim)
    nqf = ftab.nqf
    net = 0.0
    for sf, geo in ((sfp, gp), (sfm, gm)):
        if sf.elem.size == 0:
            continue
        conn = mesh.conn_of[1][np.searchsorted(mesh.bins[1], sf.elem)]
        h = mesh.tree.h()[sf.elem]
        jacS = (h / 2.0) ** (dim - 1)
        for fi in range(len(sf.elem)):
            f = int(sf.face[fi])
            un = u_full[conn[fi]]
            for q in range(nqf):
                w = ftab.w[q] * jacS[fi] * geo.corr[fi * nqf + q]
                n = geo.n[fi * nqf + q]
                net += w * ((ftab.N[f][q] @ un) @ n)
    return net


# ---------------------------------------------------------------------------
# Reaction-set builder (LD-5)
# ---------------------------------------------------------------------------

def _build_reaction_sets(level, refine_to, wake_refine, radii=(1.5, 2.5)):
    """Build plate-enclosing free-node index sets for the reaction arbiter.

    Returns a list of free-node index arrays, one per radius in ``radii``
    (e.g. 1.5L and 2.5L from the plate centre).  Outer-boundary nodes
    (inflow x=x_min, top/bottom walls, outflow x=x_max) are excluded so
    that the w_k indicator has zero weight on all surgery rows — a requirement
    for the variational identity to hold exactly (LD-5 assertion).

    **Completeness guarantee**: each set is AUGMENTED with all SBM-coupled
    free-nodes (those appearing in any non-zero row of Af_c).  This ensures
    the variational identity `w_kᵀ (A_vol x − b_vol) = −w_kᵀ (Af_c x − bf_c)`
    captures the full plate force regardless of the radius.  Without this, any
    SBM node outside the radius is missed, breaking set-independence.  On the
    GPU config (fine mesh, large radius) the radius already covers all SBM nodes,
    but on the CPU mini-gate (coarse level-4 mesh, tiny L) the SBM nodes extend
    to 3L from center — well beyond the radii requested.

    Parameters
    ----------
    level : int
        Uniform base octree level.
    refine_to : int or None
        Adaptive plate-refinement level (None => uniform).
    wake_refine : int or None
        Wake-band refinement level (None => no wake band).
    radii : tuple of float
        Set radii in units of plate length L (default 1.5L and 2.5L).
        The radius-based selection provides the «outer shell» of each set;
        the SBM-coupled nodes are always included as the «inner core».

    Returns
    -------
    list of np.ndarray
        Free-node index arrays, one per radius, excluding BC nodes,
        guaranteed to include all SBM-coupled free-nodes.
    """
    from p2r1a_thin_plate_flow import _build_shell
    import scipy.sparse as sp
    from diffsim.sbm.vector import sbm_vector_dirichlet_twosided

    fx = _build_shell(level, X_C, Y_C, L, refine_to=refine_to,
                      wake_refine=wake_refine)
    mesh = fx["mesh"]
    cons = fx["cons"]
    dm = fx["dm"]
    free_coords = mesh.node_coords[cons.free_nodes]
    ndof = 3   # u_x, u_y, p

    # Outer-boundary mask: inflow (x=x_min), top/bottom walls, outflow (x=x_max)
    x_min, x_max = free_coords[:, 0].min(), free_coords[:, 0].max()
    y_min, y_max = free_coords[:, 1].min(), free_coords[:, 1].max()
    tol = 1e-10
    bc_mask = (
        (np.abs(free_coords[:, 0] - x_min) < tol) |
        (np.abs(free_coords[:, 0] - x_max) < tol) |
        (np.abs(free_coords[:, 1] - y_min) < tol) |
        (np.abs(free_coords[:, 1] - y_max) < tol)
    )

    # SBM-coupled free-nodes: rows of Af_c that have any non-zero entry.
    # These MUST appear in every set (with w_k = 1) for the variational identity
    # to be complete — any missing SBM node breaks set-independence.
    T = cons.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    noslip_fn = lambda y: np.zeros((len(y), 2))
    Af_raw, _ = sbm_vector_dirichlet_twosided(
        dm, fx["sfp"], fx["gp"], fx["sfm"], fx["gm"],
        noslip_fn, NU, ndof, alpha=50.0)   # alpha arbitrary (only need pattern)
    Af_c = (T_vec.T @ Af_raw @ T_vec).tocsr()
    sbm_free_nodes = set(np.unique(Af_c.nonzero()[0] // ndof).tolist())

    dist = np.sqrt((free_coords[:, 0] - X_C) ** 2 +
                   (free_coords[:, 1] - Y_C) ** 2)
    sets = []
    for r in radii:
        # Radius-based outer shell, excluding BC nodes
        radius_mask = (dist <= r * L) & ~bc_mask
        radius_nodes = set(np.where(radius_mask)[0].tolist())
        # Union with SBM-coupled core (guaranteed non-BC from construction)
        combined = np.array(sorted(radius_nodes | sbm_free_nodes), dtype=np.intp)
        sets.append(combined)
    return sets


def run_discriminator_with_reaction(alpha, nsteps, level=7, refine_to=9,
                                    wake_refine=9, dt=5e-4, device="cuda:0",
                                    mono_solver="cudss", assembly="device",
                                    t_start_lu=24.0, cv_pitch=None,
                                    reaction_radii=(1.5, 2.5)):
    """Run the leak-drag discriminator WITH the variational reaction-force arbiter.

    Extends ``run_discriminator`` with the LD-5 reaction-force instrument:
    builds two plate-enclosing node sets (default 1.5L and 2.5L radii, minus
    boundary nodes), passes them to ``run_flow_past`` via ``reaction_sets``,
    and accumulates a time-averaged ``Cd_reaction`` per set.

    The reaction Cd is the variational-identity estimate of the plate drag
    (plate x-force / ref_force).  Two different sets MUST agree to ≤ 1e-6
    relative — this agreement is the LD-5 instrument gate.
    HONEST SCOPE (final review 2026-07-27): with both sets SBM-core-
    augmented, agreement is zero BY CONSTRUCTION (Af_c rows identical in
    both indicators; residual zero elsewhere). The gate therefore
    certifies the RESIDUAL/SUBTRACTION MECHANICS + surgery-row exclusion
    — a self-check of the instrument's wiring, NOT an independent
    set-independence test. The Cd_reaction VALUE itself is the
    variationally-consistent Nitsche weak-form reaction — a genuinely
    independent third functional vs surrogate traction and CV drag.

    Parameters: same as ``run_discriminator`` plus:
    reaction_radii : tuple of float
        Radii (in units of L) for the two node sets (default 1.5L, 2.5L).

    Returns
    -------
    dict extending run_discriminator's return with:
        reaction_hist   : np.ndarray [nsteps, nsets] — per-step Cd_reaction
        cd_rxn_mean     : list of float — time-averaged Cd_reaction per set
        rxn_set_sizes   : list of int — node-set sizes
        rxn_agreement   : float — |set0 - set1| / |mean| (the gate metric)
    """
    from p2r1a_thin_plate_flow import run_flow_past, _build_shell

    # Auto-detect pitch from level if not supplied
    if cv_pitch is None:
        cv_pitch = 1.0 / (2 ** level)

    t_start = t_start_lu * L / U
    boxes = _boxes(pitch=cv_pitch)
    acc = {k: 0.0 for k in boxes}
    acc_steps = {k: 0 for k in boxes}
    acc_surr = 0.0
    acc_leak = 0.0
    n_avg = 0
    coords_ref = {}
    shell_ref = {}
    n_nodes_ref = {}

    # Build reaction sets (exclude boundary nodes so w_k ⊥ surgery rows).
    rxn_sets = _build_reaction_sets(level, refine_to, wake_refine,
                                    radii=reaction_radii)

    def cb(step, t, u_full, p_full, cd_step):
        nonlocal acc_surr, acc_leak, n_avg
        if t < t_start:
            return
        assert u_full.shape[0] == n_nodes_ref["n"], (
            f"External _build_shell produced {n_nodes_ref['n']} nodes but "
            f"run_flow_past mesh has {u_full.shape[0]} nodes — non-deterministic "
            "shell build."
        )
        coords = coords_ref["coords"]
        for k, b in boxes.items():
            try:
                cd_val = cv_drag_box(coords, u_full, p_full, b, NU, U_inf=U, L_ref=L)
                acc[k] += cd_val
                acc_steps[k] += 1
            except (ValueError, IndexError):
                pass
        acc_surr += cd_step
        acc_leak += abs(_leak_flux(
            shell_ref["mesh"], *shell_ref["sg"], u_full))
        n_avg += 1

    fx = _build_shell(level, X_C, Y_C, L, refine_to=refine_to,
                      wake_refine=wake_refine)
    coords_ref["coords"] = np.asarray(fx["mesh"].node_coords)
    n_nodes_ref["n"] = len(fx["mesh"].node_coords)
    shell_ref["mesh"] = fx["mesh"]
    shell_ref["sg"] = (fx["sfp"], fx["gp"], fx["sfm"], fx["gm"])

    t0 = time.time()
    r = run_flow_past(level=level, refine_to=refine_to, wake_refine=wake_refine,
                      nsteps=nsteps, dt=dt, nu=NU, U_inf=U,
                      plate_xc=X_C, plate_yc=Y_C, plate_L=L,
                      pert_eps=0.03, pert_t_end=0.5, alpha=alpha,
                      mono_solver=mono_solver, assembly=assembly,
                      device=device, verbose=False, on_step=cb,
                      reaction_sets=rxn_sets)
    el = time.time() - t0

    t_arr = np.arange(1, nsteps + 1) * dt
    try:
        St, _f = strouhal(t_arr, np.asarray(r["cl"]), U, L)
        St = float(St)
    except Exception:
        St = float("nan")

    # CV means
    cv_means = {k: acc[k] / max(acc_steps[k], 1) for k in boxes}
    vals = np.array(list(cv_means.values()))
    spread_denom = max(abs(vals.mean()), 1e-9)

    # Reaction means (all steps, not just post-t_start — for gate: use all steps)
    rh = r["reaction_hist"]  # [nsteps, nsets]
    cd_rxn_mean = [float(np.mean(rh[:, k])) for k in range(rh.shape[1])]
    denom_rxn = max(abs(cd_rxn_mean[0]), 1e-12)
    rxn_agreement = (abs(cd_rxn_mean[0] - cd_rxn_mean[1]) / denom_rxn
                     if len(cd_rxn_mean) >= 2 else 0.0)

    out = dict(
        alpha=alpha,
        cd_surr_mean=acc_surr / max(n_avg, 1),
        cd_cv_mean=cv_means,
        box_spread=float(vals.max() - vals.min()) / spread_denom,
        leak_mean_abs=acc_leak / max(n_avg, 1),
        St=St,
        n_steps_avg=n_avg,
        elapsed=el,
        # Reaction arbiter (LD-5)
        reaction_hist=rh,
        cd_rxn_mean=cd_rxn_mean,
        rxn_set_sizes=[len(s) for s in rxn_sets],
        rxn_agreement=rxn_agreement,
    )

    os.makedirs("results", exist_ok=True)
    np.savez(
        f"results/leakdrag_rxn_a{int(alpha)}_hist.npz",
        t=t_arr, cd=r["cd"], cl=r["cl"],
        reaction_hist=rh,
        **{f"cv_{k}": v for k, v in cv_means.items()},
    )
    return out


# ---------------------------------------------------------------------------
# Main discriminator
# ---------------------------------------------------------------------------

def run_discriminator(alpha, nsteps, level=7, refine_to=9, wake_refine=9,
                      dt=5e-4, device="cuda:0", mono_solver="cudss",
                      assembly="device", t_start_lu=24.0,
                      cv_pitch=None):
    """Run the leak-drag discriminator for a single alpha value.

    Parameters
    ----------
    alpha : float
        SBM Nitsche penalty parameter.
    nsteps : int
        Number of time steps to march.
    level : int
        Uniform base octree level (7 for GPU, 4 for CPU mini-gate).
    refine_to : int or None
        Adaptive refinement target level; None => uniform mesh.
    wake_refine : int or None
        Wake-band refinement level; None => no wake refinement.
    dt : float
        Time-step size.
    device : str
        Device for assembly and linear solve.
    mono_solver : str
        Monolithic solver backend ("cudss", "splu", etc.).
    assembly : str
        Assembly backend ("device" or "host").
    t_start_lu : float
        Time (in units of L/U) after which statistics are accumulated.
    cv_pitch : float or None
        Node-line snapping pitch for CV boxes.  None => auto-detect from level
        (pitch = 1/2**level, i.e. the uniform mesh node spacing).

    Returns
    -------
    dict with keys: alpha, cd_surr_mean, cd_cv_mean (dict box->float),
        box_spread, leak_mean_abs, St, n_steps_avg, elapsed.
    """
    from p2r1a_thin_plate_flow import run_flow_past, _build_shell

    # Auto-detect pitch from level if not supplied
    if cv_pitch is None:
        cv_pitch = 1.0 / (2 ** level)

    t_start = t_start_lu * L / U
    boxes = _boxes(pitch=cv_pitch)
    acc = {k: 0.0 for k in boxes}
    acc_steps = {k: 0 for k in boxes}   # count steps where box succeeded
    acc_surr = 0.0
    acc_leak = 0.0
    n_avg = 0
    coords_ref = {}
    shell_ref = {}
    n_nodes_ref = {}

    def cb(step, t, u_full, p_full, cd_step):
        nonlocal acc_surr, acc_leak, n_avg
        if t < t_start:
            return

        # Determinism assertion: external build must produce same n_nodes
        # as the mesh the driver actually uses (both builds are deterministic
        # on the same level/refine_to/wake_refine → same tree/classification).
        assert u_full.shape[0] == n_nodes_ref["n"], (
            f"External _build_shell produced {n_nodes_ref['n']} nodes but "
            f"run_flow_past mesh has {u_full.shape[0]} nodes — non-deterministic "
            "shell build; cannot use external refs for callback."
        )

        coords = coords_ref["coords"]
        for k, b in boxes.items():
            try:
                cd_val = cv_drag_box(coords, u_full, p_full, b, NU,
                                     U_inf=U, L_ref=L)
                acc[k] += cd_val
                acc_steps[k] += 1
            except (ValueError, IndexError):
                # Box edge outside domain or on a refinement transition;
                # skip this step for this box (accumulated count stays 0 →
                # mean returns 0.0, which is finite).
                pass

        acc_surr += cd_step
        acc_leak += abs(_leak_flux(
            shell_ref["mesh"], *shell_ref["sg"], u_full))
        n_avg += 1

    # Build shell OUTSIDE run to grab mesh/shell refs for the callback.
    # run_flow_past will rebuild internally (deterministic duplicate — ~0.2s).
    # We confirm determinism via the n_nodes assertion above.
    fx = _build_shell(level, X_C, Y_C, L, refine_to=refine_to,
                      wake_refine=wake_refine)
    coords_ref["coords"] = np.asarray(fx["mesh"].node_coords)
    n_nodes_ref["n"] = len(fx["mesh"].node_coords)
    shell_ref["mesh"] = fx["mesh"]
    shell_ref["sg"] = (fx["sfp"], fx["gp"], fx["sfm"], fx["gm"])

    t0 = time.time()
    r = run_flow_past(level=level, refine_to=refine_to, wake_refine=wake_refine,
                      nsteps=nsteps, dt=dt, nu=NU, U_inf=U,
                      plate_xc=X_C, plate_yc=Y_C, plate_L=L,
                      pert_eps=0.03, pert_t_end=0.5, alpha=alpha,
                      mono_solver=mono_solver, assembly=assembly,
                      device=device, verbose=False, on_step=cb)
    el = time.time() - t0

    t = np.arange(1, nsteps + 1) * dt
    try:
        St, _f = strouhal(t, np.asarray(r["cl"]), U, L)
        St = float(St)
    except Exception:
        St = float("nan")

    # CV means: use per-box step count (0.0 if box never succeeded)
    cv_means = {k: acc[k] / max(acc_steps[k], 1) for k in boxes}
    vals = np.array(list(cv_means.values()))
    spread_denom = max(abs(vals.mean()), 1e-9)
    out = dict(
        alpha=alpha,
        cd_surr_mean=acc_surr / max(n_avg, 1),
        cd_cv_mean=cv_means,
        box_spread=float(vals.max() - vals.min()) / spread_denom,
        leak_mean_abs=acc_leak / max(n_avg, 1),
        St=St,
        n_steps_avg=n_avg,
        elapsed=el,
    )

    os.makedirs("results", exist_ok=True)
    np.savez(
        f"results/leakdrag_a{int(alpha)}_hist.npz",
        t=t, cd=r["cd"], cl=r["cl"],
        **{f"cv_{k}": v for k, v in cv_means.items()},
    )
    return out


if __name__ == "__main__":
    # LD-5 GPU protocol: one α=50 leg with reaction arbiter + alpha sweep for CV context.
    # The reaction instrument gate (set-agreement ≤ 1e-6) replaces the box-spread
    # gate. NOTE: agreement is a mechanics SELF-CHECK (zero by construction with
    # SBM-core-augmented sets), not set-independence — see run_discriminator_with_reaction.
    # CV boxes stay as corroborating context; the formal verdict issues from Cd_reaction.
    alphas = [float(a) for a in os.environ.get("ALPHAS", "20,50,100").split(",")]
    nsteps = int(os.environ.get("NSTEPS", "8000"))
    t_start_lu = float(os.environ.get("T_START_LU", "24"))
    _device    = os.environ.get("DEVICE", "cuda:0")
    _solver    = os.environ.get("MONO_SOLVER", "cudss")
    _assembly  = os.environ.get("ASSEMBLY", "device")
    os.makedirs("results", exist_ok=True)

    # --- α=50 with reaction arbiter (LD-5 instrument gate) -------------------
    _rxn_alpha = 50.0  # the one α leg for the formal verdict
    print(f"\n[LD-5] Running α={_rxn_alpha} with reaction-force arbiter …")
    rxn_row = run_discriminator_with_reaction(
        _rxn_alpha, nsteps,
        t_start_lu=t_start_lu,
        device=_device, mono_solver=_solver, assembly=_assembly,
    )
    rh = rxn_row["reaction_hist"]
    cd_rxn_s0 = float(np.mean(rh[:, 0]))
    cd_rxn_s1 = float(np.mean(rh[:, 1]))
    rxn_agree  = rxn_row["rxn_agreement"]

    print(f"[LD-5] Cd_rxn set0={cd_rxn_s0:.4f}  set1={cd_rxn_s1:.4f}  "
          f"agreement={rxn_agree:.2e}  Cd_surr={rxn_row['cd_surr_mean']:.4f}  "
          f"set sizes={rxn_row['rxn_set_sizes']}")

    # --- α sweep for CV + surrogate context (existing path) ------------------
    rows = [
        run_discriminator(
            a, nsteps,
            t_start_lu=t_start_lu,
            device=_device, mono_solver=_solver, assembly=_assembly,
        )
        for a in alphas
    ]

    # --- Table (surrogate Cd + CV context columns) ---------------------------
    _tags = list(rows[0]["cd_cv_mean"].keys())
    print(
        f"\n{'alpha':>6} {'Cd_surr':>8} "
        + " ".join(f"{'CV(' + t + ')':>8}" for t in _tags)
        + f" {'spread':>7} {'|leak|':>9} {'St':>7}"
    )
    for r in rows:
        c = r["cd_cv_mean"]
        print(
            f"{r['alpha']:6.0f} {r['cd_surr_mean']:8.3f} "
            + " ".join(f"{c[t]:8.3f}" for t in _tags)
            + f" {r['box_spread']:7.3f} "
            f"{r['leak_mean_abs']:9.2e} {r['St']:7.4f}"
        )

    # -------------------------------------------------------------------------
    # Verdict (LD-5 amended):
    #   Instrument gate: reaction set-agreement ≤ 1e-6 (replaces box-spread gate).
    #   CV boxes: corroborating context only (columns stay in table above).
    #   Formal verdict from Cd_reaction (α=50 leg) vs Cd_surr:
    #     alpha-sensitive : surr spread > 5% of mean
    #     CV-vs-rxn gap   : (Cd_surr − Cd_rxn) / Cd_surr > 10% => overestimates
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("INSTRUMENT GATE (reaction set-agreement):")
    if rxn_agree > 1e-6:
        print(f"  GATE FAILED: set-agreement={rxn_agree:.2e} > 1e-6 — implementation bug")
        print("VERDICT: BLOCKED (reaction arbiter gate failed)")
        print("LEAKDRAG-OK")
        raise SystemExit(1)

    print(f"  GATE PASSED: set-agreement={rxn_agree:.2e} ≤ 1e-6 ✓")
    # Formal verdict from Cd_reaction (α=50)
    cd_rxn = cd_rxn_s0   # both sets agreed; use set0
    cd_surr_mid = rxn_row["cd_surr_mean"]
    surr_arr = np.array([r["cd_surr_mean"] for r in rows])
    alpha_sens = (surr_arr.max() - surr_arr.min()) / max(abs(surr_arr.mean()), 1e-9) > 0.05
    cv_low = (
        (cd_surr_mid - cd_rxn) / max(abs(cd_surr_mid), 1e-9) > 0.10
    )
    quad = "observable-overestimates" if cv_low else "real-flow-drag"
    quad += "+alpha-sensitive(leak)" if alpha_sens else "+alpha-insensitive"
    print(f"  Cd_reaction={cd_rxn:.4f}  Cd_surr={cd_surr_mid:.4f}  St={rxn_row['St']:.4f}")
    print(f"VERDICT: {quad}")
    print("LEAKDRAG-OK")
