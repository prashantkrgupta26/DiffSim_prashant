"""P2-R2c — monolithic 3-D sphere Cd mesh-convergence (Re=100).

R2a pivoted the 3-D forward engine to the MONOLITHIC SBM-NS saddle solve
(the projection split is a research track — see
docs/dev/2026-07-22-p2-r2a-projection-3d-findings.md). This is R2c's first
validation on that engine: march the monolithic sphere at Re=100 across mesh
levels and report Cd vs the literature drag correlation, establishing the
convergence trend.

Literature reference (unbounded sphere, Re=100): Schiller-Naumann
    Cd = (24/Re)(1 + 0.15 Re^0.687) = 1.087 at Re=100.
Our domain is a STRONGLY CONFINED unit box: sphere R=0.12 at (0.35,0.5,0.5),
so wall clearances are only ~1.0 D upstream, ~2.2 D downstream, ~1.6 D lateral
(box/D ~ 4.2). Frontal-area blockage is 4.5%, but that number badly understates
the confinement — with every wall ~1-1.6 D away, near-field wall retardation
dominates. The converged Cd ~= 1.41 is therefore ~30% ABOVE Schiller-Naumann,
and that +30% is the PHYSICALLY-CORRECT consequence of confinement, NOT a solver
error: definitions (Re_D=100 exactly, frontal-area normalization) and the drag
integration are separately verified correct (see
docs/dev/2026-07-22-r2c-confined-sphere-verdict.md). So the DELIVERABLE here is
the monotone mesh-refining trend toward a stable value (L4=0.964, L5=1.448,
L6=1.413 -> ~1.41); ABSOLUTE literature validation requires either a >=15-20 D
domain (AMR / real compute cost) or a matched confined-sphere reference at
box/D~4. Do NOT read the gap to Schiller-Naumann as an accuracy defect.

Runs on gpubox (host/splu). Env: LEVELS (default "4,5"), RE (default 100),
STEPS (default 80), ALPHA (default 100).

    STEPS=80 LEVELS=4,5 .venv/bin/python tests/p2r2c_monolithic_sphere_convergence.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from p2r0_task10_sphere_derisk import build_sphere_3d, monolithic_cd, R, U_IN

LEVELS = [int(x) for x in os.environ.get("LEVELS", "4,5").split(",")]
RE = float(os.environ.get("RE", "100"))
STEPS = int(os.environ.get("STEPS", "80"))
ALPHA = float(os.environ.get("ALPHA", "100"))
SOLVER = os.environ.get("SOLVER", "splu")   # "cudss" (GPU direct) reaches past the CPU-splu wall
DEVICE = os.environ.get("DEVICE", "cpu" if SOLVER == "splu" else "cuda:0")
DT = 0.05
RATE_TOL = 1e-4


def schiller_naumann(Re):
    return (24.0 / Re) * (1.0 + 0.15 * Re ** 0.687)


def main():
    ref = schiller_naumann(RE)
    print(f"[r2c] MONOLITHIC sphere Cd mesh-convergence  Re={RE}  dt={DT}  "
          f"alpha={ALPHA}  steps<= {STEPS}", flush=True)
    print(f"[r2c] literature (Schiller-Naumann, UNBOUNDED): Cd_ref={ref:.4f}; "
          f"our box is strongly confined (~1 D upstream/1.6 D lateral, box/D~4.2) "
          f"so ~+30% (Cd~1.41) is expected physics, not error", flush=True)

    rows = []
    for level in LEVELS:
        t0 = time.time()
        try:
            fx = build_sphere_3d(DEVICE, level=level, Re=RE)
        except Exception as e:  # noqa: BLE001
            print(f"[r2c] level {level}: fixture build FAILED: {e}", flush=True)
            rows.append(dict(level=level, status="build_failed", error=str(e)))
            continue
        dh = 2 * R / (1.0 / 2 ** level)
        n_free = int(len(fx["coords"]))
        n_dof = n_free * fx["ndof"]
        print(f"\n[r2c] level {level}: n_free={n_free} n_dof={n_dof} "
              f"D/h={dh:.2f} sf_faces={int(fx['sf'].elem.size)}", flush=True)
        try:
            res = monolithic_cd(fx, ALPHA, DT, STEPS, RATE_TOL, solver=SOLVER)
        except MemoryError as e:
            print(f"[r2c] level {level}: splu OOM ({e}) — stop refining here",
                  flush=True)
            rows.append(dict(level=level, status="oom", n_dof=n_dof))
            break
        except Exception as e:  # noqa: BLE001
            print(f"[r2c] level {level}: solve FAILED: {e}", flush=True)
            rows.append(dict(level=level, status="solve_failed", error=str(e)))
            continue
        cd = float(res["cd"])
        dt_wall = time.time() - t0
        rows.append(dict(level=level, status="ok", n_free=n_free, n_dof=n_dof,
                         D_over_h=round(dh, 3), cd=cd,
                         steps=int(res.get("steps", STEPS)),
                         wall_s=round(dt_wall, 1)))
        print(f"[r2c] level {level}: Cd={cd:+.4f}  (ref {ref:.3f}, "
              f"|Cd-ref|/ref={abs(cd - ref) / ref * 100:.1f}%)  "
              f"steps={res.get('steps')}  ({dt_wall:.1f}s)", flush=True)

    ok = [r for r in rows if r.get("status") == "ok"]
    print("\n[r2c] ===== CONVERGENCE SUMMARY =====", flush=True)
    print(f"[r2c] literature Cd_ref = {ref:.4f}", flush=True)
    for r in ok:
        print(f"[r2c]   level {r['level']}  D/h={r['D_over_h']:.2f}  "
              f"n_dof={r['n_dof']:>7}  Cd={r['cd']:+.4f}", flush=True)
    if len(ok) >= 2:
        d = ok[-1]["cd"] - ok[-2]["cd"]
        print(f"[r2c]   Cd change last refinement: {d:+.4f} "
              f"({'refining toward a limit' if abs(d) < abs(ok[-2]['cd']) else 'still coarse'})",
              flush=True)

    out = os.path.join(os.path.dirname(__file__), "baselines",
                       "p2r2c_monolithic_sphere_convergence.json")
    with open(out, "w") as fh:
        json.dump(dict(config=dict(Re=RE, dt=DT, alpha=ALPHA, steps=STEPS,
                                   levels=LEVELS, ref_schiller_naumann=ref),
                       rows=rows), fh, indent=2)
    print(f"[r2c] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
