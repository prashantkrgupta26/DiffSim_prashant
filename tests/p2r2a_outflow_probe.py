"""P2-R2a outflow-Dirichlet PPE BC probe (Task 3b, Baskar outflow-BC route).

The bake-off (tests/p2r2a_bakeoff_3d.py, committed 556166d) found NO stable
config: the incremental pressure p* accumulates unboundedly and the weak-
divergence never decays; only non-incremental Chorin bounds it (wrong Cd).
Root cause (Baskar's call): the PPE pins a single arbitrary FREE node (node 0)
as "enclosed flow", but the sphere fixture is an EXTERNAL flow with a FREE
OUTFLOW at x=1. A single node-pin leaves the outflow pressure floating -> the
incremental p* drifts. The Taly ns_vms reference instead imposes a physical
Dirichlet pressure BC on the outlet nodes.

This probe marches the level-4 Re=1 Stokes sphere for NSTEPS under
``pressure_update="standard", ppe_fine_scale=True`` in TWO modes:

  (a) baseline    — node-0 pin (pressure_outflow_nodes=None)     [current path]
  (b) outflow-BC  — Dirichlet p=0 on the x=1 outflow free nodes  [Taly route]

and prints per step ||p_hat||, weak-divergence, Cd (surrogate_traction), plus
the monolithic same-mesh Cd reference (~0.381) for comparison, and a one-line
verdict per mode.

Runs on gpubox CPU (splu). Standalone script, NOT a pytest gate.

Usage:
    cd /path/to/DiffSim
    .venv/bin/python tests/p2r2a_outflow_probe.py 2>&1 | tee /tmp/outflow_probe.log

Modelled on tests/p2r2a_bakeoff_3d.py (Task 3).
"""
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap (mirrors the bake-off)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from p2r0_task10_sphere_derisk import (build_sphere_3d, monolithic_cd,       # noqa: E402
                                       outflow_free_nodes, R, U_IN)
from p2r2a_diagnostic_3d import _weak_divergence_3d                          # noqa: E402
from diffsim.steppers.leray_sbm import LeraySBMStepper                       # noqa: E402

# ---------------------------------------------------------------------------
# Constants (match the bake-off / Task-1 diagnostic)
# ---------------------------------------------------------------------------
DT = 0.05
LEVEL = 4
RE = 1.0        # Re=1: Stokes, isolates pressure coupling
ALPHA = 100.0
NSTEPS = 15


# ---------------------------------------------------------------------------
# Per-mode march
# ---------------------------------------------------------------------------

def _march(fx, pressure_outflow_nodes, label):
    """March NSTEPS on the level-4 sphere fixture in the given PPE-pin mode.

    pressure_outflow_nodes=None -> baseline node-0 pin.
    pressure_outflow_nodes=<indices> -> Taly outflow Dirichlet BC.

    Both use pressure_update="standard", ppe_fine_scale=True. Returns a dict
    with phat_norms, weakdiv, cds trajectories plus stability booleans.
    """
    dim = fx["dim"]

    def f_fn(x, t):
        return np.zeros((len(x), dim))

    st = LeraySBMStepper(
        fx["oracle"], fx["dm"], fx["nu"], DT, f_fn,
        u_inf=fx["u_inf"], strong_mask=fx["strong_mask"],
        lam=0.5, domain="outside", order=2, picard_iters=2,
        solver="splu", ppe_finescale=False, alpha=ALPHA,
        beta_backflow=1.0, velocity_update="consistent",
        pressure_update="standard", ppe_fine_scale=True,
        pressure_outflow_nodes=pressure_outflow_nodes,
    )
    st.set_initial(lambda c: np.zeros((len(c), dim)))

    q = 0.5 * U_IN ** 2 * np.pi * R ** 2   # dynamic-pressure reference for Cd

    phat_norms, weakdiv, cds = [], [], []
    for k in range(NSTEPS):
        _u, _p = st.step()

        phat_norms.append(float(np.linalg.norm(st.base.p_star)))
        w2, _winf = _weak_divergence_3d(st)
        weakdiv.append(float(w2))
        F = st.surrogate_traction()
        cds.append(float(F[0] / q))

        print(
            f"[{label}] step{k+1:2d}: "
            f"||p||={phat_norms[-1]:.4e}  "
            f"wdiv={weakdiv[-1]:.4e}  "
            f"Cd={cds[-1]:+.4f}",
            flush=True,
        )

    # --- stability predicates (same as the bake-off) ---
    bounded = bool(
        np.isfinite(phat_norms[-1])
        and phat_norms[-1] < 2.0 * (phat_norms[0] + 1e-30)
    )
    weakdiv_decays = bool(len(weakdiv) > 2 and weakdiv[-1] < 0.5 * weakdiv[2])
    stable = bool(bounded and weakdiv_decays and np.isfinite(cds[-1]))

    return dict(
        phat_norms=phat_norms, weakdiv=weakdiv, cds=cds,
        bounded=bounded, weakdiv_decays=weakdiv_decays, stable=stable,
    )


def _verdict_line(label, res, mono_cd):
    cd = res["cds"][-1]
    cd_rel = (abs(cd - mono_cd) / abs(mono_cd)) if np.isfinite(mono_cd) and mono_cd != 0 else float("nan")
    return (
        f"[{label}] VERDICT: bounded={res['bounded']}  "
        f"weakdiv_decays={res['weakdiv_decays']}  "
        f"stable={res['stable']}  "
        f"Cd_final={cd:+.4f}  Cd_mono={mono_cd:+.4f}  "
        f"|Cd-mono|/mono={cd_rel:.2%}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("[probe] Building level-4 sphere fixture at Re=1 (Stokes)...",
          flush=True)
    device = "cuda:0" if os.environ.get("DIFFSIM_CUDA") else "cpu"
    fx = build_sphere_3d(device, level=LEVEL, Re=RE)
    outflow = outflow_free_nodes(fx)
    print(
        f"[probe] fixture ready  n_free={len(fx['coords'])}  "
        f"n_outflow_free_nodes={len(outflow)}  ({time.time()-t0:.1f}s)",
        flush=True,
    )

    # --- monolithic same-mesh Cd reference (~0.381) ---
    print("\n=== monolithic same-mesh Cd reference ===", flush=True)
    mono = monolithic_cd(fx, ALPHA, DT, max_steps=120, rate_tol=5e-3)
    mono_cd = float(mono["cd"])
    print(f"[probe] monolithic Cd={mono_cd:+.4f} (steps={mono['steps']})",
          flush=True)

    # --- (a) baseline node-0 pin ---
    print("\n=== (a) baseline: node-0 pin (pressure_outflow_nodes=None) ===",
          flush=True)
    res_a = _march(fx, None, "baseline")

    # --- (b) outflow-BC Dirichlet pin ---
    print("\n=== (b) outflow-BC: Dirichlet p=0 on x=1 outflow free nodes ===",
          flush=True)
    res_b = _march(fx, outflow, "outflow-BC")

    # --- verdicts ---
    print("", flush=True)
    print(_verdict_line("baseline", res_a, mono_cd), flush=True)
    print(_verdict_line("outflow-BC", res_b, mono_cd), flush=True)
    print(f"\n[probe] done ({time.time()-t0:.1f}s total)", flush=True)


if __name__ == "__main__":
    main()
