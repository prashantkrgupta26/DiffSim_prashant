"""P2-R2a bake-off diagnostic — isolate the minimal stable lever set.

Sweeps (pressure_update, ppe_fine_scale) one lever at a time from the current
baseline plus the 2nd-order-preserving combos on the level-4 Re=1 Stokes
sphere (same fixture as p2r0_task10 / p2r2a_diagnostic_3d).  For each config
records per-step ‖p̂‖, weak-divergence, and Cd trajectories, and emits a
verdict naming the MINIMAL config with bounded ‖p̂‖ AND decaying weak-div.

Five distinct marches (configs 2 and 4 in the spec share the same
(standard, True) pair and are run ONCE, reused):

  # | label                        | pressure_update | ppe_fine_scale |
  --+------------------------------+-----------------+----------------+
  0 | baseline                     | standard        | False          |
  1 | chorin                       | chorin          | False          |
  2 | finescale-only               | standard        | True           |
  3 | rotational-only              | rotational      | False          |
  4 | incremental+finescale+rot    | rotational      | True           |

Verdict logic:
  - WINNER = minimal stable (bounded ‖p̂‖ AND decaying weak-div AND Cd finite),
    preferring fewest off-default levers, then standard over rotational.
  - If ONLY chorin stable  -> NEEDS_CONTEXT (1st-order; Task-5 escalation).
  - If NO config stable    -> NEEDS_CONTEXT.

Runs on gpubox CPU (splu).  Standalone script, NOT a pytest gate.
Writes tests/baselines/p2r2a_bakeoff_3d.json.

Usage:
    cd /path/to/DiffSim
    .venv/bin/python tests/p2r2a_bakeoff_3d.py 2>&1 | tee /tmp/bakeoff.log

Modelled on tests/p2r2a_diagnostic_3d.py (Task 1).
"""
import json
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap (mirrors Task-1 diagnostic)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from p2r0_task10_sphere_derisk import build_sphere_3d, R, U_IN          # noqa: E402
from p2r2a_diagnostic_3d import _weak_divergence_3d                      # noqa: E402
from diffsim.steppers.leray_sbm import LeraySBMStepper                   # noqa: E402

# ---------------------------------------------------------------------------
# Constants (match Task-1 diagnostic)
# ---------------------------------------------------------------------------
DT = 0.05
LEVEL = 4
RE = 1.0        # Re=1: Stokes, isolates pressure coupling
ALPHA = 100.0
NSTEPS = 15

# Five distinct (pressure_update, ppe_fine_scale) marches; configs 2 and 4 in
# the spec brief share (standard, True) — we run it once as "finescale-only"
# and reuse the result when emitting the full config matrix.
CONFIGS = [
    ("baseline",                          "standard",   False),
    ("chorin",                            "chorin",     False),
    ("finescale-only",                    "standard",   True),
    ("rotational-only",                   "rotational", False),
    ("incremental+finescale+rotational",  "rotational", True),
]

# The spec's six-row config_matrix duplicates "finescale-only" as
# "incremental+finescale" for narrative clarity; we carry both in the JSON.
CONFIG_MATRIX_LABELS = [
    ("baseline",                          "standard",   False),
    ("chorin",                            "chorin",     False),
    ("finescale-only",                    "standard",   True),
    ("rotational-only",                   "rotational", False),
    ("incremental+finescale",             "standard",   True),   # same as #2
    ("incremental+finescale+rotational",  "rotational", True),
]


# ---------------------------------------------------------------------------
# Per-config march
# ---------------------------------------------------------------------------

def _march(fx, pressure_update, ppe_fine_scale, label):
    """March NSTEPS on the level-4 sphere fixture with the given knob combo.

    Returns a per-config dict with phat_norms, weakdiv, cds trajectories plus
    the bounded/weakdiv_decays/stable booleans.
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
        pressure_update=pressure_update, ppe_fine_scale=ppe_fine_scale,
    )
    st.set_initial(lambda c: np.zeros((len(c), dim)))

    q = 0.5 * U_IN ** 2 * np.pi * R ** 2   # dynamic-pressure reference for Cd

    phat_norms, weakdiv, cds = [], [], []
    for k in range(NSTEPS):
        _u, _p = st.step()

        # ‖p̂‖ — p_star is the updated total pressure after step()
        phat_norms.append(float(np.linalg.norm(st.base.p_star)))

        # weak-div of the corrected velocity (independent B^T assembly)
        w2, _winf = _weak_divergence_3d(st)
        weakdiv.append(float(w2))

        # Cd — traction force normalised by dynamic pressure
        F = st.surrogate_traction()
        cds.append(float(F[0] / q))

        print(
            f"[{pressure_update},fs={ppe_fine_scale}] step{k+1:2d}: "
            f"||p||={phat_norms[-1]:.4e}  "
            f"wdiv={weakdiv[-1]:.4e}  "
            f"Cd={cds[-1]:+.4f}",
            flush=True,
        )

    # --- stability predicates ---
    # bounded: final ‖p̂‖ finite and < 2× initial (guard div by near-zero)
    bounded = bool(
        np.isfinite(phat_norms[-1])
        and phat_norms[-1] < 2.0 * (phat_norms[0] + 1e-30)
    )
    # weakdiv_decays: final < half the step-2 value (same index as spec: wdiv[2])
    weakdiv_decays = bool(len(weakdiv) > 2 and weakdiv[-1] < 0.5 * weakdiv[2])
    # stable: bounded pressure AND decaying div AND drag finite
    stable = bool(bounded and weakdiv_decays and np.isfinite(cds[-1]))

    return dict(
        phat_norms=phat_norms,
        weakdiv=weakdiv,
        cds=cds,
        bounded=bounded,
        weakdiv_decays=weakdiv_decays,
        stable=stable,
    )


# ---------------------------------------------------------------------------
# Verdict: minimal stable config
# ---------------------------------------------------------------------------

def _verdict(results_by_label):
    """Select the minimal stable config and format the verdict string.

    results_by_label: dict mapping label -> march result dict (with stable bool,
    pressure_update, ppe_fine_scale keys already merged in).
    """
    all_configs = list(results_by_label.values())
    stable_configs = [c for c in all_configs if c["stable"]]
    non_chorin_stable = [c for c in stable_configs if c["pressure_update"] != "chorin"]

    if non_chorin_stable:
        # prefer: fewest off-default levers, then standard over rotational
        def _rank(c):
            n_off = int(c["pressure_update"] != "standard") + int(c["ppe_fine_scale"])
            prefer_rot = int(c["pressure_update"] == "rotational")
            return (n_off, prefer_rot)

        win = sorted(non_chorin_stable, key=_rank)[0]
        verdict = (
            f"WINNER: {win['label']} "
            f"(pressure_update={win['pressure_update']}, "
            f"ppe_fine_scale={win['ppe_fine_scale']}) "
            f"— bounded ||p_hat|| and decaying weak-div"
        )
        winner = dict(
            label=win["label"],
            pressure_update=win["pressure_update"],
            ppe_fine_scale=win["ppe_fine_scale"],
        )
    elif stable_configs:
        # only Chorin is stable — 1st-order escalation
        verdict = (
            "NEEDS_CONTEXT — only Chorin (1st-order) stable; "
            "fine-scale consistency did not stabilize the incremental/2nd-order path"
        )
        winner = dict(label="chorin", pressure_update="chorin", ppe_fine_scale=False)
    else:
        verdict = "NEEDS_CONTEXT — no stable config"
        winner = None

    return verdict, winner


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("[bakeoff] Building level-4 sphere fixture at Re=1 (Stokes)...", flush=True)
    device = "cuda:0" if os.environ.get("DIFFSIM_CUDA") else "cpu"
    fx = build_sphere_3d(device, level=LEVEL, Re=RE)
    print(
        f"[bakeoff] fixture ready  n_free={len(fx['coords'])}  "
        f"({time.time()-t0:.1f}s)",
        flush=True,
    )

    # --- march each distinct config ---
    results_by_label = {}   # keyed by the CONFIGS label (five marches)
    for label, pu, fs in CONFIGS:
        print(f"\n=== {label}  (pressure_update={pu}, ppe_fine_scale={fs}) ===",
              flush=True)
        res = _march(fx, pu, fs, label)
        res.update(label=label, pressure_update=pu, ppe_fine_scale=fs)
        results_by_label[label] = res
        print(
            f"[bakeoff] {label}: "
            f"bounded={res['bounded']}  "
            f"weakdiv_decays={res['weakdiv_decays']}  "
            f"stable={res['stable']}",
            flush=True,
        )

    # --- build the six-row config_matrix (spec format) ---
    # Row 4 ("incremental+finescale") reuses the "finescale-only" march result.
    config_matrix = []
    for label, pu, fs in CONFIG_MATRIX_LABELS:
        # resolve the canonical march key (finescale-only covers incremental+finescale)
        if label == "incremental+finescale":
            src = results_by_label["finescale-only"].copy()
            src["label"] = label
        else:
            src = results_by_label[label].copy()
        config_matrix.append(src)

    # --- verdict ---
    verdict, winner = _verdict(results_by_label)
    print(f"\n[bakeoff] VERDICT: {verdict}", flush=True)

    # --- write JSON ---
    out = dict(
        _note=(
            "P2-R2a bake-off diagnostic. Sweeps (pressure_update, ppe_fine_scale) "
            "on the level-4 Re=1 Stokes sphere to isolate the minimal stable lever "
            "set. Driver: tests/p2r2a_bakeoff_3d.py."
        ),
        config=dict(
            level=LEVEL, Re=RE, dt=DT, alpha=ALPHA, nsteps=NSTEPS,
            R=R, U_IN=U_IN,
        ),
        config_matrix=config_matrix,
        winner=winner,
        verdict=verdict,
    )
    path = os.path.join(
        os.path.dirname(__file__), "baselines", "p2r2a_bakeoff_3d.json"
    )
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[bakeoff] wrote {path}  ({time.time()-t0:.1f}s total)", flush=True)


if __name__ == "__main__":
    main()
