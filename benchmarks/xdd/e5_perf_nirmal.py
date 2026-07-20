"""SP-1 R0 Block E — E5 perf gate: the Nirmal ≥100× J(t) measurement.

Measures the wall-clock cost of one XDD forward J(t) trace at the Nirmal
device size on a single GPU, and computes the speedup vs the CPU baseline
(10–30 h on 36 cores from the Nirmal paper).

## Nirmal configuration (spec §7 R0 perf gate)
  - mesh: 513×129 over 400nm×100nm  →  66,177 nodes.
    THE SUBSTRATE CAVEAT (recorded, honest): `build_uniform` produces a SQUARE
    2^level grid on the unit square; there is no anisotropic (513×129) octree in
    this codebase (Block A/B build on it).  The per-step cost is governed by the
    linear-system SIZE (DOF count) and sparsity, both fixed by node count, NOT by
    the physical aspect ratio.  We therefore measure at level 8 = 257×257 =
    66,049 nodes — a 0.19% DOF match to the 66,177-node Nirmal mesh — and label
    the anisotropy proxy explicitly.  (The nondimensionalisation uses h_hat∈[0,1],
    so physical extent enters only through param scales, not mesh cost.)
  - excitation: light-modulated J(t), 10 GHz rectified sinusoid |sin(2πft)|,
    1 ns window  →  10 full optical cycles.
  - bias: short-circuit V̂=0.
  - params: PM6-class = XDDParams A1 defaults (PM6:Y6 canonical config.txt).
  - morphology: analytic bilayer via A2 signed_distance.

## What "GPU" means here (recorded, honest — the deferred-brainstorm input)
  Element assembly runs through Warp kernels on `cuda:N`.  The global 5-field
  Jacobian is a scipy CSR built on HOST.  The Newton linear solve backend is
  chosen with --linsolver:
    splu  (default) — HOST scipy `splu` (nonsymmetric sparse LU): the E-b
           baseline; GPU idle during the solve (0% util, host-LU bound at
           330k DOF — the documented E5 gate MISS).
    cudss — the E5 FIX: XDDSystem(linsolver="cudss") routes the assembled
           Jacobian through the in-tree cuDSS GPU sparse direct LU (nvmath,
           plan-once + refactorize per Newton iterate on the fixed sparsity).
           The factorization AND triangular solves run on the device.
  (The A3 closures remain HOST numpy either way — recorded; if transfer/closure
  eval dominates after the cudss fix, that is the next bottleneck to profile.)

## The march (per-step waveform, capturing J(t))
  The stock XDDRun PULSE/STEADY_PULSE paths use SQUARE-wave phases, not a
  per-step `generation.amplitude(t̂)` scaling.  For the 10 GHz rect-sin we drive a
  FIXED-dt BDF1 march directly: each step rescales the spatial generation by
  amplitude(t̂) = |sin(2πf·t̂·t0)| and captures the contact flux J(t).  dt is
  chosen to resolve the waveform (≥ ~20 steps/cycle → dt ≤ 5 ps → dt̂ = 5ps/t0).

Run ON THE BOX via the remote toolkit.  GPU device selected by --device
(default cuda:0).  Requires LD_LIBRARY_PATH=/usr/lib/wsl/lib on the WSL box so
Warp binds the passthrough libcuda (else CUDA error 100 — see the E-b report).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bench_bootstrap  # noqa

import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.poisson import gauss_points
from diffsim.physics.exciton_system import XDDSystem, bilayer_electrode_bcs
from diffsim.physics.exciton_closures import (
    LangevinRecombination, OnsagerBraunDissociation, Generation)
from diffsim.xdd.params import XDDParams
from diffsim.xdd.run import log_linear_ic, _flux_pair


# Nirmal physical mesh (recorded target) and the DOF-matched square proxy.
NIRMAL_NX, NIRMAL_NY = 513, 129
NIRMAL_NODES = NIRMAL_NX * NIRMAL_NY          # 66,177
NIRMAL_LX, NIRMAL_LY = 400e-9, 100e-9
NIRMAL_FREQ = 10e9                             # 10 GHz
NIRMAL_WINDOW = 1e-9                           # 1 ns


def bilayer_dist_gp(xq, height, h_axis=1):
    """Analytic bilayer signed-distance at GPs (A2 convention: neg=donor at the
    bottom half, pos=acceptor at the top half; interface at mid-height)."""
    dist_gp = {}
    for pv in xq:
        h_hat = xq[pv][:, h_axis]              # 0..1 along device height
        # signed distance to the mid-height interface, in metres
        dist_gp[pv] = (h_hat - 0.5) * height
    return dist_gp


def build_system(level: int, device: str, params: XDDParams,
                 *, h_axis: int = 1, regime: str = "marchable",
                 linsolver: str = "splu", assembly: str = "auto"):
    """Build the XDDSystem at the given uniform level on `device`.

    regime:
      "physical"  — PM6:Y6 A1 param scales verbatim: λ² = Debye² (Eg_hat≈42.5,
                    Debye≪h).  This is the true Nirmal device but hits the
                    documented CPU-mesh drive wall (E1(ii)): the BDF Newton
                    fails at iter 1 on any feasible uniform mesh, so per-step
                    timing is NOT representative (Newton bails early).
      "marchable" — E1's documented reduced-drive regime: λ²=1e-1, symmetric
                    μ̂, Eg_hat set by the caller.  The Newton runs its FULL
                    iteration count → the per-step cost (assembly + full LU +
                    closures) is REPRESENTATIVE of a converging device step.
                    The LINEAR-SYSTEM size/sparsity — hence the dominant LU
                    cost — is IDENTICAL to the physical regime (same mesh, same
                    5-field coupling); only the physical drive strength differs.
                    This is the honest s/step for the perf gate; the physical
                    mesh-wall is a SEPARATE recorded caveat.

    Returns (sysm, mesh, cons, dm, xq, dist_gp)."""
    pdeg = 1
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=pdeg)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(pdeg, dim=2), device)
    xq = gauss_points(mesh, dm.tables_by_p)

    s = params.scales()
    dist_gp = bilayer_dist_gp(xq, params.height, h_axis=h_axis)

    # Region-blended coefficient GP fields (RegionMobility physics, inline).
    from diffsim.xdd.morphology import tanh_mask
    mu0 = s.mu0
    mu_n = {}; mu_p = {}; mu_xd = {}; mu_xa = {}; eps = {}
    for pv in xq:
        d = dist_gp[pv]
        w_a = tanh_mask(d, params.interface_thk)     # →1 in acceptor (top)
        w_d = 1.0 - w_a
        if regime == "marchable":
            # symmetric, O(1) transport (E1 marchable regime) — full-Newton work
            mu_n[pv] = np.full(len(d), 0.5); mu_p[pv] = np.full(len(d), 0.5)
            mu_xd[pv] = np.full(len(d), 0.5); mu_xa[pv] = np.full(len(d), 0.5)
            eps[pv] = np.ones(len(d))
        else:
            mu_n[pv] = (params.mu_n / mu0) * (w_a + params.mu_ratio * w_d)
            mu_p[pv] = (params.mu_p / mu0) * (w_d + params.mu_ratio * w_a)
            mu_xd[pv] = (params.mu_x_donor / mu0) * (w_d + 1e-6 * w_a)
            mu_xa[pv] = (params.mu_x_acceptor / mu0) * (w_a + 1e-6 * w_d)
            eps[pv] = params.eps_D + (params.eps_A - params.eps_D) * w_a

    lang = LangevinRecombination(params, strategy="sum", zeta=1.0,
                                 spatial="uniform")
    ons = OnsagerBraunDissociation(params, width=params.interface_thk)
    tau_inv_d = s.t0 / params.tau_x_donor
    tau_inv_a = s.t0 / params.tau_x_acceptor
    lam2 = 1e-1 if regime == "marchable" else s.lambda2

    # E5 fix (2026-07-18): linsolver="cudss" routes the assembled 5-field
    # Jacobian through the in-tree cuDSS GPU sparse direct LU (plan-once +
    # refactorize per Newton iterate; see XDDSystem._cudss_linsolve).  The
    # host-splu default is GPU-idle at 330k DOF — the measured E-b bottleneck.
    sysm = XDDSystem(
        dm, lam2=lam2, eps_gp=eps, mu_n_gp=mu_n, mu_p_gp=mu_p,
        mu_xd_gp=mu_xd, mu_xa_gp=mu_xa, dist_gp=dist_gp,
        langevin=lang, onsager=ons, tau_inv_d=tau_inv_d, tau_inv_a=tau_inv_a,
        supg=1.0, carrier_vars="log", linsolver=linsolver, assembly=assembly)
    return sysm, mesh, cons, dm, xq, dist_gp


def measure(level: int, device: str, n_steps: int, dt_hat: float,
            params: XDDParams, *, h_axis: int = 1, verbose: bool = True,
            regime: str = "marchable", linsolver: str = "splu",
            assembly: str = "auto", profile: bool = False):
    """March `n_steps` fixed-dt BDF1 steps of the 10 GHz rect-sin drive,
    capturing J(t) and per-step wall time.  Returns a results dict.

    linsolver: "splu" (host scipy SuperLU, the E-b baseline) | "cudss" (the
    E5-fix GPU sparse direct LU — wired via XDDSystem(linsolver="cudss"))."""
    sysm, mesh, cons, dm, xq, dist_gp = build_system(
        level, device, params, h_axis=h_axis, regime=regime,
        linsolver=linsolver, assembly=assembly)
    s = params.scales()
    n_nodes = dm.n_nodes
    if verbose:
        print(f"[e5] device={device} level={level} nodes={n_nodes} "
              f"dofs={5*sysm.n_free} regime={regime} "
              f"(Nirmal target {NIRMAL_NODES})", flush=True)

    # Bilayer electrode BCs at V̂=0 (short-circuit) + log-linear marchable IC.
    Eg_hat = 4.0 if regime == "marchable" else params.E_g / s.phi0
    minority_ln = -Eg_hat if regime == "marchable" else -60.0
    bilayer_electrode_bcs(sysm, mesh, cons, Eg_hat=Eg_hat, V_app_hat=0.0,
                          h_axis=h_axis, minority_ln=minority_ln)
    state = log_linear_ic(mesh, Eg_hat, minority_ln, h_axis=h_axis)
    # Start at a nonzero waveform phase so the FIRST timed steps have amp>0
    # (|sin| at t̂=0 is 0 → a dark step; offset by a quarter cycle).
    phase_offset_hat = (0.25 / NIRMAL_FREQ) / s.t0

    # Generation: PM6-class constant spatial profile, rect-sin waveform 10 GHz.
    gen = Generation(params=params, profile="constant",
                     waveform="rect_sin", freq=NIRMAL_FREQ)
    # Precompute the static spatial generation (dist-only) once per bin.
    gd0 = {}; ga0 = {}
    for pv in dm.bins:
        Gd, Ga = gen.spatial(dist_gp[pv], xq[pv][:, h_axis])
        gd0[pv] = Gd; ga0[pv] = Ga

    t0 = s.t0
    t_hat = phase_offset_hat
    prev_state = {f: state[f].copy() for f in range(5)}
    step_times = []
    jt = []   # (t_seconds, Jny, Jpy)
    nk = {"max_iter": 12}

    # Warm-up: first step triggers Warp lazy JIT compile — time it separately.
    warm_t0 = time.time()
    amp = gen.amplitude(t_hat)
    sysm.set_generation({pv: gd0[pv] * amp for pv in dm.bins},
                        {pv: ga0[pv] * amp for pv in dm.bins})
    try:
        new_state, info = sysm.step_bdf(state, dt_hat, order=1,
                                        prev=prev_state, **nk)
        if info.get("converged"):
            prev_state = {f: state[f].copy() for f in range(5)}
            state = new_state
            t_hat += dt_hat
    except Exception as exc:
        info = {"converged": False, "reason": str(exc)}
    warm_wall = time.time() - warm_t0
    if verbose:
        print(f"[e5] warm-up step (JIT compile incl.): {warm_wall:.3f}s "
              f"converged={info.get('converged')} its={info.get('iters')}",
              flush=True)

    # Task #35: measured assembly-budget numbers (the 100M-dof planning line).
    budget = None
    if sysm._assembly_device:
        budget = sysm.device_assembler().budget()
        if verbose:
            print(f"[e5] device-assembly budget: nnz={budget['nnz']} "
                  f"nnz/dof={budget['nnz_per_dof']:.2f} "
                  f"bytes/dof={budget['bytes_per_dof']:.1f} "
                  f"csr={budget['csr_gb']:.3f} GB "
                  f"wide={budget['index_wide']}", flush=True)

    # Optional cProfile of ONE representative step (the G4 attribution).
    if profile:
        import cProfile
        import pstats
        import io as _io
        amp = gen.amplitude(t_hat)
        sysm.set_generation({pv: gd0[pv] * amp for pv in dm.bins},
                            {pv: ga0[pv] * amp for pv in dm.bins})
        pr = cProfile.Profile()
        pr.enable()
        try:
            new_state, info = sysm.step_bdf(state, dt_hat, order=1,
                                            prev=prev_state, **nk)
        except Exception as exc:
            info = {"converged": False, "reason": str(exc)}
            new_state = state
        pr.disable()
        if info.get("converged"):
            prev_state = {f: state[f].copy() for f in range(5)}
            state = new_state
            t_hat += dt_hat
        buf = _io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(30)
        print("[e5] === cProfile of one step (cumulative, top 30) ===",
              flush=True)
        print(buf.getvalue(), flush=True)

    # Timed steps.
    for k in range(n_steps):
        amp = gen.amplitude(t_hat)
        sysm.set_generation({pv: gd0[pv] * amp for pv in dm.bins},
                            {pv: ga0[pv] * amp for pv in dm.bins})
        st0 = time.time()
        try:
            new_state, info = sysm.step_bdf(state, dt_hat, order=1,
                                            prev=prev_state, **nk)
        except Exception as exc:
            info = {"converged": False, "reason": str(exc)}
            new_state = state
        dt_wall = time.time() - st0
        step_times.append(dt_wall)
        if info.get("converged"):
            prev_state = {f: state[f].copy() for f in range(5)}
            state = new_state
            t_hat += dt_hat
        jny, jpy = _flux_pair(sysm, state, h_axis=h_axis)
        jt.append((t_hat * t0, float(jny), float(jpy), int(info.get("iters", 0)),
                   bool(info.get("converged", False))))
        if verbose and (k < 3 or k % 25 == 0):
            print(f"[e5] step {k:4d} wall={dt_wall:.3f}s its={info.get('iters')} "
                  f"conv={info.get('converged')} amp={amp:.3f} "
                  f"J=({jny:.3e},{jpy:.3e})", flush=True)

    st = np.array(step_times)
    s_per_step = float(np.median(st))
    s_per_step_mean = float(np.mean(st))

    # Steps to cover the 1 ns window at this dt.
    dt_seconds = dt_hat * t0
    steps_for_window = int(math.ceil(NIRMAL_WINDOW / dt_seconds))
    end_to_end_proj = s_per_step * steps_for_window

    n_converged = sum(1 for r in jt if r[4])
    return dict(
        device=device, level=level, nodes=n_nodes, dofs=5 * sysm.n_free,
        regime=regime, n_converged=n_converged,
        n_steps_timed=n_steps, dt_hat=dt_hat, dt_seconds=dt_seconds,
        t0=t0, Eg_hat=Eg_hat, lam2=s.lambda2,
        warmup_wall_s=warm_wall,
        s_per_step_median=s_per_step, s_per_step_mean=s_per_step_mean,
        s_per_step_min=float(np.min(st)), s_per_step_max=float(np.max(st)),
        steps_for_window=steps_for_window,
        end_to_end_projected_s=end_to_end_proj,
        jt_trace=jt,
        linsolver=linsolver,
        assembly=sysm.assembly,
        assembly_budget=budget,
        solver=("cudss GPU sparse direct LU (nvmath, plan-once + "
                "refactorize per Newton iterate; warp assembly on GPU)"
                if linsolver == "cudss" else
                "host_scipy_splu (cuDSS not wired; warp assembly on GPU)"),
    )


def gate_arithmetic(res: dict, baseline_hours_lo=10.0, baseline_hours_hi=30.0):
    """Speedup = CPU baseline / our end-to-end.  Honest: report vs both ends of
    the 10–30 h/36-core Nirmal baseline band."""
    e2e = res["end_to_end_projected_s"]
    lo = (baseline_hours_lo * 3600.0) / e2e
    hi = (baseline_hours_hi * 3600.0) / e2e
    return dict(
        baseline_hours_lo=baseline_hours_lo, baseline_hours_hi=baseline_hours_hi,
        end_to_end_s=e2e, end_to_end_hours=e2e / 3600.0,
        speedup_vs_10h=lo, speedup_vs_30h=hi,
        target=100.0,
        pass_vs_10h=bool(lo >= 100.0), pass_vs_30h=bool(hi >= 100.0),
        provenance=("PROJECTION: s/step (median of timed steps) × "
                    "steps_for_1ns_window; end-to-end not run to full 1 ns "
                    "(see driver)."),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--level", type=int, default=8)   # 257×257 = 66,049 nodes
    ap.add_argument("--n-steps", type=int, default=40)
    ap.add_argument("--dt-ps", type=float, default=5.0,
                    help="fixed step in picoseconds (5 ps → 20 steps/cycle at 10 GHz)")
    ap.add_argument("--regime", default="marchable",
                    choices=("marchable", "physical"))
    ap.add_argument("--linsolver", default="splu",
                    choices=("splu", "cudss"),
                    help="linear-solve backend: splu (host, E-b baseline) | "
                         "cudss (GPU sparse direct LU, the E5 perf fix)")
    ap.add_argument("--assembly", default="auto",
                    choices=("auto", "host", "device"),
                    help="Jacobian/residual assembly backend (Task #35): "
                         "auto = device on CUDA, host on CPU")
    ap.add_argument("--profile", action="store_true",
                    help="cProfile ONE representative step (G4 attribution)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    params = XDDParams()   # PM6:Y6 A1 defaults
    s = params.scales()
    dt_hat = (args.dt_ps * 1e-12) / s.t0

    print(f"[e5] Nirmal perf gate — PM6:Y6 A1 defaults", flush=True)
    print(f"[e5] t0={s.t0:.4e}s  dt={args.dt_ps}ps  dt_hat={dt_hat:.4e}  "
          f"1ns window = {NIRMAL_WINDOW/(args.dt_ps*1e-12):.0f} steps", flush=True)

    res = measure(args.level, args.device, args.n_steps, dt_hat, params,
                  regime=args.regime, linsolver=args.linsolver,
                  assembly=args.assembly, profile=args.profile)
    gate = gate_arithmetic(res)
    res_out = {k: v for k, v in res.items() if k != "jt_trace"}
    res_out["jt_trace_len"] = len(res["jt_trace"])
    res_out["jt_trace_head"] = res["jt_trace"][:5]
    out = {"result": res_out, "gate": gate}

    print("\n" + "=" * 70, flush=True)
    print(f"[e5] s/step (median) = {res['s_per_step_median']:.4f} s  "
          f"(mean {res['s_per_step_mean']:.4f}, min {res['s_per_step_min']:.4f})",
          flush=True)
    print(f"[e5] steps for 1 ns window = {res['steps_for_window']}", flush=True)
    print(f"[e5] end-to-end PROJECTED = {gate['end_to_end_s']:.1f} s "
          f"({gate['end_to_end_hours']:.3f} h)", flush=True)
    print(f"[e5] speedup vs 10h/36c baseline = {gate['speedup_vs_10h']:.1f}×  "
          f"vs 30h = {gate['speedup_vs_30h']:.1f}×  (target ≥100×)", flush=True)
    print(f"[e5] solver = {res['solver']}", flush=True)
    print("=" * 70, flush=True)
    print(json.dumps(out, indent=2, default=str), flush=True)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f"[e5] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
