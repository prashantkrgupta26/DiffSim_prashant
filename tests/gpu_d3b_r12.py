"""D3b: the r12 resolution rung (256 cells/plate) — does the reaction's
increment finally shrink?  Trend so far: 2.5996 (r9) -> 2.8098 (r10) ->
3.0483 (r11), increments +0.2102/+0.2385 (growing).  dt=1.25e-4 for CFL at
h=2^-12; nsteps=32000 keeps t_end=4.0 (64 L/U) matching D1-D3.

    bash scripts/remote/gpubox-run.sh ".venv/bin/python tests/gpu_d3b_r12.py" d3b-r12
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from gpu_traction_dissect import run_dissect_leg  # noqa: E402

r = run_dissect_leg(
    "D3b", alpha=50.0,
    nsteps=int(os.environ.get("NSTEPS", "32000")),
    level=7, refine_to=12, wake_refine=9, plate_L_inv=16,
    dt=float(os.environ.get("DT", "1.25e-4")),
    device=os.environ.get("DEVICE", "cuda:0"),
    mono_solver=os.environ.get("MONO_SOLVER", "cudss"),
    assembly=os.environ.get("ASSEMBLY", "device"),
    t_start_lu=24.0)
print(f"[D3b] cd_rxn_total={r['cd_rxn_total']:.4f}  cd_surr={r['cd_surr']:.4f}  "
      f"bridge={r['bridge_ratio']:.4f}  St={r['St']:.4f}  "
      f"dt_used={r['dt_used']:.2e}  elapsed={r['elapsed']:.1f}s", flush=True)
print("  terms:", {k: round(v, 6) for k, v in r["cd_rxn_terms"].items()},
      flush=True)
inc = r["cd_rxn_total"] - 3.0483
print(f"[D3b] increment vs r11: {inc:+.4f} "
      f"(prior increments +0.2102, +0.2385) -> "
      f"{'SHRINKING' if 0 < inc < 0.2385 else 'NOT SHRINKING' if inc >= 0.2385 else 'NON-MONOTONE'}",
      flush=True)
print("D3B-OK", flush=True)
