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
    for tag, m in (("4L", 4 * L), ("6L", 6 * L), ("8L", 8 * L)):
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
    alphas = [float(a) for a in os.environ.get("ALPHAS", "20,50,100").split(",")]
    nsteps = int(os.environ.get("NSTEPS", "8000"))
    os.makedirs("results", exist_ok=True)
    rows = [
        run_discriminator(
            a, nsteps,
            device=os.environ.get("DEVICE", "cuda:0"),
            mono_solver=os.environ.get("MONO_SOLVER", "cudss"),
            assembly=os.environ.get("ASSEMBLY", "device"),
        )
        for a in alphas
    ]

    print(
        f"\n{'alpha':>6} {'Cd_surr':>8} {'CV(4L)':>8} {'CV(6L)':>8} "
        f"{'CV(8L)':>8} {'spread':>7} {'|leak|':>9} {'St':>7}"
    )
    for r in rows:
        c = r["cd_cv_mean"]
        print(
            f"{r['alpha']:6.0f} {r['cd_surr_mean']:8.3f} {c['4L']:8.3f} "
            f"{c['6L']:8.3f} {c['8L']:8.3f} {r['box_spread']:7.3f} "
            f"{r['leak_mean_abs']:9.2e} {r['St']:7.4f}"
        )

    # -------------------------------------------------------------------------
    # Verdict per spec thresholds (EXACT — do not soften):
    #   spread withheld:   box_spread > 5%
    #   alpha-sensitive:   surr spread > 5% of mean
    #   CV-vs-surr gap:    (surr - cv_mid) / surr > 10%  =>  observable-overestimates
    # -------------------------------------------------------------------------
    mid = rows[len(rows) // 2]
    cv_mid = np.mean(list(mid["cd_cv_mean"].values()))
    if mid["box_spread"] > 0.05:
        print("VERDICT: CV-UNRELIABLE (box spread >5%) — verdict withheld")
    else:
        surr = np.array([r["cd_surr_mean"] for r in rows])
        alpha_sens = (
            (surr.max() - surr.min()) / max(abs(surr.mean()), 1e-9) > 0.05
        )
        cv_low = (
            (mid["cd_surr_mean"] - cv_mid) / max(abs(mid["cd_surr_mean"]), 1e-9) > 0.10
        )
        quad = "observable-overestimates" if cv_low else "real-flow-drag"
        quad += "+alpha-sensitive(leak)" if alpha_sens else "+alpha-insensitive"
        print(f"VERDICT: {quad}")
    print("LEAKDRAG-OK")
