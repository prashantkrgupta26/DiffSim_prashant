"""benchmarks/chns/ab_interface.py — CH vs CAC interface A/B driver.

Reproduces the protocol described in
  docs/dev/2026-08-10-sp0-interface-decision.md  §"A/B evidence"

Setup
-----
- Level 5 (32×32), 2-D, BUBBLE_RISE_RE35_WE10, rho_ratio=10, Cn_override="2h".
- Monolithic BDF1, splu.
- Up to 200 steps at case.dt0 (gravity ON) for bubble-rise.
- Static drop (gravity OFF), 5 steps, for parasitic-current measurement.
- Two source variants for mass-conservation: no-source AND fixed Gaussian blob
  (A=2, radius=3h, centre=(0.5,0.5)) on the phi-row.

Metrics recorded per interface {ch, cac}
-----------------------------------------
bubble_rise:
  diverge_step    : first step where NaN or max|phi|>2.5 or singular (None = survived)
  mass_drift_nosrc: |sum(phi_T) - sum(phi_0)| / |sum(phi_0)|  (no source)
  mass_drift_src  : |sum(phi_T) - sum(phi_0) - T*dt*sum(s)| / |sum(phi_0)|
                     (with source, T = steps to divergence or 200)
  newton_avg      : mean Newton iters over completed steps
  newton_max      : max Newton iters over completed steps
  wall_per_step   : mean wall time per step (s)
  parasitic_umax  : max|u| after 5 steps, gravity OFF, static drop
  parasitic_ratio : parasitic_umax(cac) / parasitic_umax(ch)

Normalization convention (memo §"A/B evidence")
-------------------------------------------------
mass_drift_nosrc  : normalized by |Int phi_0| (via lumped-mass inner product M @ phi_0).
mass_drift_src    : |Int phi_T - Int phi_0 - T*dt*Int s| / |Int phi_0|.
  - Horizon: 200 steps (no-source) or steps-to-divergence / 200 (sourced).
  - Int phi via lumped mass M (partition-of-unity row sums), so Int s is M @ s(coords, 0).
  - "Static source": s(x,t) = A*exp(-|x-x0|^2/(2*rad^2)), evaluated once at t=0.

Usage
-----
  python benchmarks/chns/ab_interface.py [--out DIR]

Outputs (in DIR, default benchmarks/chns/results/ab_interface/):
  ab_interface.json  — full JSON of all recorded numbers
  ab_interface.txt   — printed table (also echoed to stdout)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path wiring (mirrors test_chns_forward.py bootstrap)
# ---------------------------------------------------------------------------
_CHNS_DIR = Path(__file__).resolve().parent    # benchmarks/chns/
_BENCH_DIR = _CHNS_DIR.parent                  # benchmarks/
_REPO_DIR = _BENCH_DIR.parent                  # repo root
for _p in [str(_BENCH_DIR), str(_REPO_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chns.cases import BUBBLE_RISE_RE35_WE10  # noqa: E402
from chns import metrics  # noqa: E402


# ---------------------------------------------------------------------------
# Mesh helper
# ---------------------------------------------------------------------------
def _make_dm(level: int, dim: int = 2):
    import warp as wp
    wp.init()
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    from diffsim import default_device

    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    device = default_device()
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), device)
    return dm, mesh, cons


# ---------------------------------------------------------------------------
# Source function (Gaussian blob on phi-row)
# ---------------------------------------------------------------------------
def _make_src_fns(coords, h, A=2.0, x0=0.5, y0=0.5, rad_h_mult=3.0, dim=2):
    """Fixed Gaussian blob source on the phi-row (field index dim+1=3 in 2-D)."""
    rad = rad_h_mult * h

    def src_phi(xq, t):
        rr2 = (xq[:, 0] - x0) ** 2 + (xq[:, 1] - y0) ** 2
        return A * np.exp(-rr2 / (2.0 * rad ** 2))

    blk = dim + 3
    src_fns = [None] * blk
    src_fns[dim + 1] = src_phi
    return src_fns, src_phi


# ---------------------------------------------------------------------------
# Run one bubble-rise experiment (up to max_steps), returning metric dict
# ---------------------------------------------------------------------------
def _run_bubble_rise(dm, mesh, cons, case, interface, max_steps,
                     Cn_override="2h", src_fns=None, src_phi_fn=None):
    """Run bubble rise for up to max_steps steps; return metric dict."""
    from diffsim.steppers.chns import CHNSStepper

    coords = mesh.node_coords
    st = CHNSStepper(dm, case, dt=case.dt0, mode="monolithic",
                     Cn_override=Cn_override, gravity=True,
                     interface=interface, src_fns=src_fns)

    # Light bubble centred at (0.5, 0.35), radius 0.2 (same as test suite)
    r = np.sqrt((coords[:, 0] - 0.5) ** 2 + (coords[:, 1] - 0.35) ** 2)
    phi0 = -np.tanh((r - 0.2) / (st.Cn * np.sqrt(2.0)))
    st.set_initial(phi0)

    M = st.lumped_mass()
    mass0 = float(M @ phi0)
    src_int = 0.0
    if src_phi_fn is not None:
        src_int = float(M @ src_phi_fn(coords, 0.0))

    newton_iters = []
    wall_times = []
    diverge_step = None

    for step_k in range(1, max_steps + 1):
        t0 = time.perf_counter()
        try:
            snap = st.step()
        except (RuntimeError, np.linalg.LinAlgError) as exc:
            print(f"    [{interface}] step {step_k}: exception {exc!r}")
            diverge_step = step_k
            break
        wall_times.append(time.perf_counter() - t0)

        phi_k = st.phi
        if not np.isfinite(phi_k).all() or float(np.abs(phi_k).max()) > 2.5:
            print(f"    [{interface}] step {step_k}: phi non-finite or |phi|>2.5 "
                  f"(max={float(np.abs(phi_k).max()):.3f})")
            diverge_step = step_k
            break

        newton_iters.append(int(snap.get("newton_iters", 0)))

    # Completed steps
    n_done = len(newton_iters)
    phi_T = st.phi
    mass_T = float(M @ phi_T)

    # mass drift no-source: |mass_T - mass0| / |mass0|
    mass_drift_nosrc = abs(mass_T - mass0) / max(abs(mass0), 1e-30)

    # mass drift with source: |mass_T - mass0 - n_done*dt*src_int| / |mass0|
    mass_drift_src = (abs(mass_T - mass0 - n_done * case.dt0 * src_int)
                      / max(abs(mass0), 1e-30))

    return {
        "interface": interface,
        "diverge_step": diverge_step,
        "steps_completed": n_done,
        "mass_drift_nosrc": float(mass_drift_nosrc),
        "mass_drift_src": float(mass_drift_src),
        "newton_avg": float(np.mean(newton_iters)) if newton_iters else float("nan"),
        "newton_max": int(max(newton_iters)) if newton_iters else 0,
        "wall_per_step": float(np.mean(wall_times)) if wall_times else float("nan"),
    }


# ---------------------------------------------------------------------------
# Static-drop parasitic current measurement (gravity off, 5 steps)
# ---------------------------------------------------------------------------
def _run_parasitic(dm, mesh, cons, case, interface, n_steps=5, Cn_override="2h"):
    """Run static drop (gravity off) for n_steps; return max|u|."""
    from diffsim.steppers.chns import CHNSStepper

    coords = mesh.node_coords
    st = CHNSStepper(dm, case, dt=1e-3, mode="monolithic",
                     Cn_override=Cn_override, gravity=False,
                     interface=interface)
    r = np.sqrt((coords[:, 0] - 0.5) ** 2 + (coords[:, 1] - 0.5) ** 2)
    phi0 = -np.tanh((r - 0.25) / (st.Cn * np.sqrt(2.0)))
    st.set_initial(phi0)
    for _ in range(n_steps):
        st.step()
    return float(np.abs(st.u).max())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(out_dir: str = "benchmarks/chns/results/ab_interface",
         level: int = 5,
         max_steps: int = 200):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    case = replace(BUBBLE_RISE_RE35_WE10, rho_ratio=10.0)

    print(f"A/B driver: level={level}, max_steps={max_steps}, "
          f"dt={case.dt0}, Cn_override=2h")
    print("Building mesh …")
    dm, mesh, cons = _make_dm(level=level)
    coords = mesh.node_coords
    h = float(dm.mesh.tree.h().min())
    Cn_eff = 2.0 * h
    print(f"  h={h:.5f}  Cn_eff(2h)={Cn_eff:.5f}")

    src_fns, src_phi_fn = _make_src_fns(coords, h, A=2.0, x0=0.5, y0=0.5,
                                        rad_h_mult=3.0, dim=2)

    results = {}

    for interface in ("ch", "cac"):
        print(f"\n--- {interface.upper()} bubble rise (no source) ---")
        r_nosrc = _run_bubble_rise(dm, mesh, cons, case, interface,
                                   max_steps=max_steps, src_fns=None,
                                   src_phi_fn=None)
        print(f"    diverge_step={r_nosrc['diverge_step']}  "
              f"completed={r_nosrc['steps_completed']}  "
              f"mass_drift={r_nosrc['mass_drift_nosrc']:.3e}  "
              f"newton={r_nosrc['newton_avg']:.1f}/{r_nosrc['newton_max']}  "
              f"wall/step={r_nosrc['wall_per_step']:.3f}s")

        print(f"--- {interface.upper()} bubble rise (with Gaussian src) ---")
        r_src = _run_bubble_rise(dm, mesh, cons, case, interface,
                                 max_steps=max_steps, src_fns=src_fns,
                                 src_phi_fn=src_phi_fn)
        print(f"    diverge_step={r_src['diverge_step']}  "
              f"completed={r_src['steps_completed']}  "
              f"mass_drift_src={r_src['mass_drift_src']:.3e}  "
              f"newton={r_src['newton_avg']:.1f}/{r_src['newton_max']}  "
              f"wall/step={r_src['wall_per_step']:.3f}s")

        print(f"--- {interface.upper()} parasitic drop (gravity off, 5 steps) ---")
        umax = _run_parasitic(dm, mesh, cons, case, interface)
        print(f"    max|u| = {umax:.3e}")

        results[interface] = {
            "bubble_rise_nosrc": r_nosrc,
            "bubble_rise_src": r_src,
            "parasitic_umax": umax,
        }

    # Parasitic ratio
    umax_ch = results["ch"]["parasitic_umax"]
    umax_cac = results["cac"]["parasitic_umax"]
    parasitic_ratio = umax_cac / umax_ch if umax_ch > 0 else float("nan")
    results["parasitic_ratio"] = parasitic_ratio

    # Mass-drift horizon note (from memo protocol):
    # no-source horizon = min(200, steps_completed) steps
    # sourced horizon   = min(200, steps to divergence or 200) steps
    results["normalization"] = (
        "mass_drift_nosrc: |Int phi_T - Int phi_0| / |Int phi_0|; "
        "horizon = 200 steps (or steps_completed if shorter). "
        "mass_drift_src: |Int phi_T - Int phi_0 - T*dt*Int s| / |Int phi_0|; "
        "T = steps completed before divergence or 200. "
        "Int via lumped-mass M (partition-of-unity row sums); "
        "Int s = M @ s(coords, t=0) (static Gaussian source)."
    )
    results["config"] = {
        "level": level,
        "max_steps": max_steps,
        "dt": case.dt0,
        "Cn_override": "2h",
        "Cn_eff": Cn_eff,
        "h": h,
        "rho_ratio": case.rho_ratio,
        "src_A": 2.0,
        "src_x0": 0.5,
        "src_y0": 0.5,
        "src_rad_h_mult": 3.0,
        "parasitic_drop_steps": 5,
        "parasitic_dt": 1e-3,
    }

    # Print summary table
    print("\n" + "=" * 70)
    print("A/B INTERFACE TABLE  (level 5, rho_ratio=10, Cn=2h, BDF1)")
    print("=" * 70)
    fmt = "{:<40s}  {:>12s}  {:>12s}"
    print(fmt.format("Axis", "CH", "CAC"))
    print("-" * 70)
    ch_br = results["ch"]["bubble_rise_nosrc"]
    cac_br = results["cac"]["bubble_rise_nosrc"]
    ch_brs = results["ch"]["bubble_rise_src"]
    cac_brs = results["cac"]["bubble_rise_src"]

    div_ch = str(ch_br["diverge_step"]) if ch_br["diverge_step"] else f"OK {max_steps}"
    div_cac = str(cac_br["diverge_step"]) if cac_br["diverge_step"] else f"OK {max_steps}"
    print(fmt.format("Bubble-rise robustness (diverge step)", div_ch, div_cac))
    print(fmt.format("Mass drift, no source (rel)",
                     f"{ch_br['mass_drift_nosrc']:.2e}",
                     f"{cac_br['mass_drift_nosrc']:.2e}"))
    print(fmt.format("Mass drift, with source (rel)",
                     f"{ch_brs['mass_drift_src']:.2e}",
                     f"{cac_brs['mass_drift_src']:.2e}"))
    print(fmt.format("Newton iters avg/max (bubble rise)",
                     f"{ch_br['newton_avg']:.1f}/{ch_br['newton_max']}",
                     f"{cac_br['newton_avg']:.1f}/{cac_br['newton_max']}"))
    print(fmt.format("Wall/step s (bubble rise)",
                     f"{ch_br['wall_per_step']:.3f}",
                     f"{cac_br['wall_per_step']:.3f}"))
    print(fmt.format("Parasitic max|u| (static drop, 5 steps)",
                     f"{umax_ch:.3e}",
                     f"{umax_cac:.3e}"))
    print(fmt.format("Parasitic ratio CAC/CH", "1.0", f"{parasitic_ratio:.1f}x"))
    print("=" * 70)
    print()
    print("Normalization (mass drift):")
    print(f"  no-source  : |Int phi_T - Int phi_0| / |Int phi_0|")
    print(f"  with-source: |Int phi_T - Int phi_0 - T*dt*Int_s| / |Int phi_0|")
    print(f"  horizon    : 200 steps (no-src) / steps-to-divergence or 200 (src)")
    print(f"  Int via    : lumped-mass M (partition-of-unity row sums)")
    print(f"  Int_s      : M @ s(coords, t=0) (static Gaussian A={results['config']['src_A']})")
    print()

    # Write JSON
    json_path = out_path / "ab_interface.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results written to {json_path}")

    # Write text table
    txt_path = out_path / "ab_interface.txt"
    with open(txt_path, "w") as fh:
        fh.write("A/B INTERFACE TABLE  (level 5, rho_ratio=10, Cn=2h, BDF1)\n")
        fh.write("=" * 70 + "\n")
        fh.write(fmt.format("Axis", "CH", "CAC") + "\n")
        fh.write("-" * 70 + "\n")
        fh.write(fmt.format("Bubble-rise robustness (diverge step)", div_ch, div_cac) + "\n")
        fh.write(fmt.format("Mass drift, no source (rel)",
                             f"{ch_br['mass_drift_nosrc']:.2e}",
                             f"{cac_br['mass_drift_nosrc']:.2e}") + "\n")
        fh.write(fmt.format("Mass drift, with source (rel)",
                             f"{ch_brs['mass_drift_src']:.2e}",
                             f"{cac_brs['mass_drift_src']:.2e}") + "\n")
        fh.write(fmt.format("Newton iters avg/max (bubble rise)",
                             f"{ch_br['newton_avg']:.1f}/{ch_br['newton_max']}",
                             f"{cac_br['newton_avg']:.1f}/{cac_br['newton_max']}") + "\n")
        fh.write(fmt.format("Wall/step s (bubble rise)",
                             f"{ch_br['wall_per_step']:.3f}",
                             f"{cac_br['wall_per_step']:.3f}") + "\n")
        fh.write(fmt.format("Parasitic max|u| (static drop, 5 steps)",
                             f"{umax_ch:.3e}",
                             f"{umax_cac:.3e}") + "\n")
        fh.write(fmt.format("Parasitic ratio CAC/CH", "1.0",
                             f"{parasitic_ratio:.1f}x") + "\n")
        fh.write("=" * 70 + "\n")
    print(f"Table written to {txt_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CH vs CAC A/B interface driver")
    parser.add_argument("--out", default="benchmarks/chns/results/ab_interface",
                        help="Output directory for JSON + table")
    parser.add_argument("--level", type=int, default=5, help="Mesh level")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max bubble-rise steps")
    args = parser.parse_args()
    main(out_dir=args.out, level=args.level, max_steps=args.max_steps)
