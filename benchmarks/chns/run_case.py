#!/usr/bin/env python
r"""benchmarks/chns/run_case.py — SP-0 Task 9 end-to-end benchmark pipeline.

config -> march (CHNSStepper, the monolithic winner) -> npz snapshots ->
VTU/PVD (diffsim.viz.export) -> interface movie (movie.py) -> metrics JSON ->
overlay plot vs reference (matplotlib PNG).

CLI
---
  run_case.py --case <name> [--full] [--level N] [--max-wall-min M]
              [--tstep bdf1|bdf2] [--out DIR]

  --case   bubble_rise_re35_we10 | bubble_rise_re35_we125 | dam_break | rt | mms
  --full   production level / t_end (with the spike runner's per-run wall cap
           discipline; caps are RECORDED in the output JSON).  Default is
           CI-sized: one level below production, truncated t_end.
  --level  override the refinement level.
  --max-wall-min  per-run wall cap in minutes (default 25, the spike discipline).

Domain convention (documented, matches the whole SP-0 CHNS stack): the mesh is
the unit square [0,1]^2 (build_uniform); the non-dimensional CHNSCase
parameters (Re, We, Cn, Pe, Fr, ratios) drive the physics, and the interface is
resolved with Cn_override="2h" (the coupling-decision memo's guard finding: the
benchmark must resolve Cn >~ h, else the monolithic clamp-guard fires).  ICs are
built directly on unit coords with the resolvable Cn width — the spike-runner
pattern.  (domain_aspect is carried on the case for provenance but the stepper
operates on the unit square; a non-unit box is a later mesh-builder concern.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_DIR = os.path.normpath(os.path.join(_HERE, ".."))
_REPO_DIR = os.path.normpath(os.path.join(_HERE, "..", ".."))
for _p in (_BENCH_DIR, _REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chns import cases as _cases        # noqa: E402
from chns import metrics                 # noqa: E402
from chns import movie                   # noqa: E402


WALL_CAP_DEFAULT_MIN = 25.0              # spike-runner per-run discipline

# case-registry: name -> (CHNSCase, production_level, ci_level, prod_t_end,
#                         ci_t_end, ic_builder(coords, Cn) -> phi0,
#                         bubble_sign)
# bubble_sign: +1 if the tracked blob is phi>0, -1 if phi<0 (bubble = light).


def _bubble_ic(coords, Cn, xc=0.5, yc=0.35, radius=0.2):
    r = np.sqrt((coords[:, 0] - xc) ** 2 + (coords[:, 1] - yc) ** 2)
    return -np.tanh((r - radius) / (Cn * np.sqrt(2.0)))   # phi=-1 light bubble


def _dambreak_ic(coords, Cn, x_iface=0.4):
    # heavy column left of x_iface (phi=+1), light right (phi=-1)
    return -np.tanh((coords[:, 0] - x_iface) / (Cn * np.sqrt(2.0)))


def _rt_ic(coords, Cn, y_iface=0.5, amp=0.05):
    y_pert = y_iface + amp * np.cos(2.0 * np.pi * coords[:, 0])
    return np.tanh((coords[:, 1] - y_pert) / (Cn * np.sqrt(2.0)))  # heavy on top


CASE_REGISTRY = {
    "bubble_rise_re35_we10": dict(
        case=_cases.BUBBLE_RISE_RE35_WE10, prod_level=6, ci_level=5,
        prod_t_end=1.5, ci_t_end=0.15, ic=_bubble_ic, blob_sign=-1,
        gravity=True),
    "bubble_rise_re35_we125": dict(
        case=_cases.BUBBLE_RISE_RE35_WE125, prod_level=6, ci_level=5,
        prod_t_end=1.5, ci_t_end=0.15, ic=_bubble_ic, blob_sign=-1,
        gravity=True),
    "dam_break": dict(
        case=_cases.DAM_BREAK_2D, prod_level=6, ci_level=5,
        prod_t_end=0.5, ci_t_end=0.06, ic=_dambreak_ic, blob_sign=+1,
        gravity=True),
    "rt": dict(
        case=_cases.RT_2D, prod_level=7, ci_level=5,
        prod_t_end=1.0, ci_t_end=0.1, ic=_rt_ic, blob_sign=+1,
        gravity=True),
}


def make_dm(level, dim=2):
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
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim),
                              default_device())
    return dm, mesh, cons


def run_case(name, full=False, level=None, max_wall_min=WALL_CAP_DEFAULT_MIN,
             tstep="bdf1", out_dir=None, snap_target=40, dt=None,
             cap_override=None):
    """Run one benchmark case end to end.  Returns the metrics dict."""
    from diffsim.steppers.chns import CHNSStepper

    if name not in CASE_REGISTRY:
        raise ValueError(f"unknown case {name!r}; choices "
                         f"{sorted(CASE_REGISTRY)}")
    reg = CASE_REGISTRY[name]
    case = reg["case"]
    lvl = level if level is not None else (reg["prod_level"] if full
                                           else reg["ci_level"])
    t_end = reg["prod_t_end"] if full else reg["ci_t_end"]
    wall_cap_s = max_wall_min * 60.0

    if out_dir is None:
        out_dir = os.path.join(_HERE, "results", name
                               + ("_full" if full else "_ci"))
    os.makedirs(out_dir, exist_ok=True)

    dm, mesh, cons = make_dm(lvl, dim=2)
    coords = mesh.node_coords
    dt = float(dt) if dt is not None else float(case.dt0)
    if full and tstep == "bdf2":
        dt = dt * 4.0        # BDF2 tolerates a larger step (documented)

    st = CHNSStepper(dm, case, dt=dt, mode="auto", tstep=tstep,
                     Cn_override="2h", gravity=reg["gravity"])
    phi0 = reg["ic"](coords, st.Cn)
    st.set_initial(phi0)

    sign = reg["blob_sign"]
    h = float(st.h)
    mass0 = st.mass_phi()
    E0 = st.energy()

    steps_full = int(round(t_end / dt))
    snap_every = max(1, steps_full // snap_target)

    # --- marching loop with per-step recording + wall cap -------------------
    t_series, cy_series, mass_series, etot_series = [], [], [], []
    snaps_phi, snaps_t = [], []
    npz_paths = []
    nit_series, clamp_series, wall_series = [], [], []
    died, death_reason = False, None
    t_wall0 = time.perf_counter()
    k = 0
    cap_hit = False
    while st.t < t_end - 1e-12:
        if time.perf_counter() - t_wall0 > wall_cap_s:
            cap_hit = True
            break
        if cap_override is not None and k >= cap_override:
            cap_hit = True
            break
        try:
            st.step()
        except RuntimeError as e:
            died, death_reason = True, f"{type(e).__name__}: {e}"
            break
        k += 1
        if not np.isfinite(st.phi).all() or not np.isfinite(st.u).all():
            died, death_reason = True, f"NaN at step {k}"
            break
        if k % snap_every == 0 or st.t >= t_end - 1e-12:
            t_series.append(float(st.t))
            cy_series.append(float(metrics.centroid_y(sign * st.phi, coords)))
            mass_series.append(st.mass_phi())
            etot_series.append(st.energy()["total"])
            nit_series.append(int(st.last_newton_iters))
            clamp_series.append(int(st.last_clamped))
            wall_series.append(float(st.last_wall))
            snaps_phi.append(st.phi.copy())
            snaps_t.append(float(st.t))
            # npz snapshot
            npz = os.path.join(out_dir, f"snap_{len(snaps_t) - 1:04d}.npz")
            np.savez_compressed(npz, coords=coords, phi=st.phi, u=st.u,
                                p=st.p, t=float(st.t), h=h)
            npz_paths.append(npz)
    total_wall = time.perf_counter() - t_wall0
    steps_run = k

    # --- reductions / metrics ----------------------------------------------
    def _drift(a):
        a = np.asarray(a, float)
        return float(np.max(np.abs(a - a[0]))) if a.size else float("nan")

    mass_drift = (float(np.max(np.abs(np.asarray(mass_series) - mass0)))
                  if mass_series else float("nan"))
    rise_peak = float("nan")
    if len(cy_series) >= 2:
        rv = metrics.rise_velocity(np.asarray(cy_series), snap_every * dt)
        rise_peak = float(np.nanmax(np.abs(rv)))
    min_circ = float("nan")
    try:
        if snaps_phi:
            circs = [metrics.circularity(sign * p, coords, h)
                     for p in snaps_phi]
            min_circ = float(np.nanmin(circs))
    except Exception:
        pass
    # energy non-increasing between consecutive unforced steps? report the
    # worst positive jump (should be <= tiny slack for the closed unforced box)
    e_arr = np.asarray(etot_series, float)
    max_energy_rise = (float(np.max(np.diff(e_arr))) if e_arr.size >= 2
                       else float("nan"))

    metrics_out = {
        "case": name,
        "full": bool(full),
        "level": int(lvl),
        "tstep": tstep,
        "dt": dt,
        "t_end_target": t_end,
        "t_reached": float(t_series[-1]) if t_series else 0.0,
        "steps_run": steps_run,
        "steps_full_t_end": steps_full,
        "wall_capped": bool(cap_hit),
        "wall_cap_min": max_wall_min,
        "total_wall_s": total_wall,
        "wall_per_step_mean": (float(np.mean(wall_series))
                               if wall_series else float("nan")),
        "died": died,
        "death_reason": death_reason,
        "n_nodes": int(dm.n_nodes),
        "h": h,
        "Cn_effective": float(st.Cn),
        "case_params": {"Re": case.Re, "We": case.We, "Pe": case.Pe,
                        "Fr": case.Fr, "rho_ratio": case.rho_ratio,
                        "eta_ratio": case.eta_ratio,
                        "Cn_case": case.Cn},
        "mass_initial": float(mass0),
        "mass_drift_max_abs": mass_drift,
        "energy_initial": E0["total"],
        "energy_final": (etot_series[-1] if etot_series else float("nan")),
        "max_energy_rise_between_snaps": max_energy_rise,
        "centroid_final": (cy_series[-1] if cy_series else float("nan")),
        "rise_velocity_peak": rise_peak,
        "circularity_min": min_circ,
        "newton_iters_mean": (float(np.mean(nit_series))
                              if nit_series else float("nan")),
        "clamp_total": int(np.sum(clamp_series)) if clamp_series else 0,
        "series": {
            "t": [float(x) for x in t_series],
            "centroid_y": [float(x) for x in cy_series],
            "mass_drift": [float(m - mass0) for m in mass_series],
            "energy_total": [float(x) for x in etot_series],
        },
        "reference_note": (
            "Physical references (Hysing et al. 2009, IJNMF 60:1259 for bubble "
            "rise; Martin & Moyce 1952, Phil.Trans.R.Soc.A 244:312 for dam "
            "break) are the DIRECTION; the CI-sized non-dimensional runs on the "
            "unit square with Cn=2h resolvability are gated on INTERNAL "
            "reproducibility vs a committed provisional baseline (see "
            "tests/test_chns_benchmarks.py REFERENCES)."),
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_out, fh, indent=2)

    # --- VTU / PVD time series ----------------------------------------------
    try:
        from diffsim.viz.export import export_vtu
        vtu_files = []
        want_vtu = full          # CI keeps only npz + JSON + PNG + gif
        if want_vtu:
            for i, (p, tt) in enumerate(zip(snaps_phi, snaps_t)):
                vp = os.path.join(out_dir, f"field_{i:04d}.vtu")
                export_vtu(mesh, vp, fields={"phi": p})
                vtu_files.append(vp)
            if vtu_files:
                from diffsim.viz.export import _write_pvd
                _write_pvd(vtu_files[0], vtu_files)
    except Exception as e:
        print(f"[run_case] VTU export skipped: {e}")

    # --- interface movie ----------------------------------------------------
    try:
        if snaps_phi:
            movie.write_interface_movie(
                snaps_phi, coords,
                os.path.join(out_dir, "interface.gif"), fps=8)
    except Exception as e:
        print(f"[run_case] movie skipped: {e}")

    # --- overlay plot vs reference ------------------------------------------
    try:
        _overlay_plot(name, metrics_out, out_dir)
    except Exception as e:
        print(f"[run_case] overlay plot skipped: {e}")

    print(f"[run_case] {name} {'FULL' if full else 'CI'} L{lvl} tstep={tstep} "
          f"steps={steps_run}/{steps_full} wall={total_wall:.1f}s "
          f"cap={'Y' if cap_hit else 'N'} died={died} "
          f"mass_drift={mass_drift:.2e} cy_final={metrics_out['centroid_final']}")
    return metrics_out


def _overlay_plot(name, m, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = m["series"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].plot(s["t"], s["centroid_y"], "-o", ms=3, label="this run")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("centroid_y (tracked blob)")
    ax[0].set_title(f"{name}: centroid trajectory")
    ax[0].grid(True, alpha=0.3); ax[0].legend()
    ax[1].plot(s["t"], s["energy_total"], "-s", ms=3, color="C3")
    ax[1].set_xlabel("t"); ax[1].set_ylabel("discrete free energy")
    ax[1].set_title(f"{name}: energy (K+interface+bulk)")
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "overlay.png"), dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(CASE_REGISTRY))
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--max-wall-min", type=float, default=WALL_CAP_DEFAULT_MIN)
    ap.add_argument("--tstep", default="bdf1", choices=["bdf1", "bdf2"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    run_case(args.case, full=args.full, level=args.level,
             max_wall_min=args.max_wall_min, tstep=args.tstep,
             out_dir=args.out)


if __name__ == "__main__":
    main()
