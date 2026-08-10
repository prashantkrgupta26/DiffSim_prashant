#!/usr/bin/env python
"""benchmarks/chns/spike_run.py — SP-0 Task 8 coupling-decision spike RUNNER.

Compares the two CHNS coupling prototypes to EACH OTHER (no reference-curve
comparison here — that is Task 9's transcription job):

    CHNSStaggeredStepper  (Task 6: CH -> NS projection split)
    CHNSMonolithicStepper (Task 7: fully-implicit coupled Newton)

on BUBBLE_RISE_RE35_WE10 at level 6 (64x64) across a density-ratio sweep.

MATRIX
------
  {staggered, monolithic} x {rho_ratio 10, rho_ratio 100}   (full t_end, capped)
  {staggered, monolithic} x {rho_ratio 1000}                (stress probe, 50 steps)

RUNTIME PRAGMATICS (binding)
----------------------------
  * Before the matrix, 5 steps of each stepper are timed at level 6 and the
    per-step wall extrapolated.
  * Every run is capped so no single run exceeds ~WALL_CAP_S wall. If the full
    t_end does not fit, the step count is capped and the cap RECORDED
    (steps_run vs steps_full_t_end).
  * Progress printed every 25 steps.
  * Both steppers raise on Newton failure; that is caught, recorded
    (robustness_events / died), and the matrix continues. A dying run at
    ratio 1000 IS evidence, not a script bug.

ENERGY PROXY
------------
  Lumped-node kinetic energy  E_kin = 0.5 * sum_i rho(phi_i) * |u_i|^2 / n
  (documented lumped-node approximation; the GP-weighted form is not surfaced
  by the staggered stepper's public API, so the lumped form is used uniformly
  across BOTH steppers for a like-for-like series). Monotone-decay assessment
  is left to the memo; the series is only recorded here.

OUTPUTS (under --out, default benchmarks/chns/results/spike/)
-------------------------------------------------------------
  spike_<stepper>_r<ratio>.json   one lean JSON per run (downsampled series)
  summary.json                    combined headline numbers for all runs
  centroid_r<ratio>.png           centroid_y vs t overlay (both steppers)
  energy_r<ratio>.png             E_kin vs t overlay (both steppers)
  evidence_table.md               headline markdown table

Run:
  .venv/bin/python benchmarks/chns/spike_run.py --out benchmarks/chns/results/spike/
  (optional) --only staggered:10   to run a single matrix cell
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import replace

import numpy as np

# --- bootstrap: benchmarks/ + repo root on sys.path (mirror the tests) -----
_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_DIR = os.path.normpath(os.path.join(_HERE, ".."))
_REPO_DIR = os.path.normpath(os.path.join(_HERE, "..", ".."))
for _p in (_BENCH_DIR, _REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chns.cases import BUBBLE_RISE_RE35_WE10  # noqa: E402
from chns import metrics  # noqa: E402

# --- run configuration ------------------------------------------------------
LEVEL = 6
WALL_CAP_S = 25.0 * 60.0          # ~25 min hard cap per run
STRESS_STEPS = 50                 # ratio-1000 stress-probe cap (spec)
PROGRESS_EVERY = 25
SERIES_POINTS = 50                # downsample target for recorded series
CN_OVERRIDE = "2h"               # resolvability convention (mirrors the gates)


# ---------------------------------------------------------------------------
# Mesh helper (uniform p=1, constraints.T == identity) — mirrors the tests.
# ---------------------------------------------------------------------------
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


def _downsample_idx(n, k):
    """Return up to k evenly spaced indices into range(n) (always incl. last)."""
    if n <= k:
        return list(range(n))
    idx = np.linspace(0, n - 1, k).round().astype(int)
    return sorted(set(idx.tolist()))


def _rho_of_phi(phi, rho_h, rho_l):
    """Lumped-node density from nodal phi (linear mix, [-1,1] -> [l,h])."""
    t = np.clip((phi + 1.0) * 0.5, 0.0, 1.0)
    return rho_l + (rho_h - rho_l) * t


def _ekin_lumped(phi, u, rho_h, rho_l):
    """E_kin = 0.5 * sum_i rho(phi_i) |u_i|^2 / n  (lumped-node proxy)."""
    rho = _rho_of_phi(np.asarray(phi), rho_h, rho_l)
    speed2 = np.sum(np.asarray(u) ** 2, axis=1)
    n = len(phi)
    return float(0.5 * np.sum(rho * speed2) / max(n, 1))


# ---------------------------------------------------------------------------
# One matrix cell.
# ---------------------------------------------------------------------------
def run_cell(stepper_name, ratio, dm, mesh, cons, wall_cap_s):
    """Run one (stepper x ratio) cell. Marches step-by-step (not via .march())
    so a mid-run Newton failure can be caught, recorded, and the partial series
    still returned. Returns a lean result dict."""
    from diffsim.steppers.chns import (CHNSStaggeredStepper,
                                        CHNSMonolithicStepper)

    cls = {"staggered": CHNSStaggeredStepper,
           "monolithic": CHNSMonolithicStepper}[stepper_name]

    case = replace(BUBBLE_RISE_RE35_WE10, rho_ratio=float(ratio))
    dt = case.dt0
    coords = mesh.node_coords            # uniform: free_nodes == all nodes
    steps_full = int(round(case.t_end / dt))

    st = cls(dm, case, dt=dt, Cn_override=CN_OVERRIDE)
    # IC: case geometry (centre (0.5, 0.25), radius 0.25, phi=-1 light bubble
    # inside) but tanh width = the STEPPER's Cn (the smoke-test harness
    # pattern). The case ic_fn's own Cn=0.01 profile is under-resolved at the
    # level-6 h=1/64 and produces an unphysical sinking artifact (verified);
    # the gates' Cn_override="2h" resolvability convention is applied to the
    # IC width as well.
    r_ic = np.sqrt((coords[:, 0] - 0.5) ** 2 + (coords[:, 1] - 0.25) ** 2)
    phi0 = np.tanh((r_ic - 0.25) / (st.Cn * np.sqrt(2.0)))
    st.set_initial(phi0)

    rho_h, rho_l = 1.0, 1.0 / float(ratio)
    h = float(st.h)
    mass0 = st.mass_phi()

    # --- decide the step cap ------------------------------------------------
    if ratio >= 1000:
        cap = STRESS_STEPS
        cap_reason = "stress-probe (50 steps, spec)"
    else:
        # time 1 pilot step already paid by set_initial? no — time it live.
        t_pilot = time.perf_counter()
        info0 = st.step()
        pilot_wall = time.perf_counter() - t_pilot
        # budget the remaining wall
        budget_left = max(wall_cap_s - pilot_wall, 0.0)
        remaining_cap = int(budget_left / max(pilot_wall, 1e-6))
        cap = min(steps_full, 1 + remaining_cap)
        cap_reason = ("full t_end fits" if cap >= steps_full
                      else f"wall-capped @ ~{wall_cap_s/60:.0f}min")
        # we already took 1 step; fold its bookkeeping in below

    # --- marching loop with per-step recording ------------------------------
    t_series, cy_series, ekin_series, mass_series = [], [], [], []
    newton_iters, clamp_counts, wall_series = [], [], []
    robustness_events = []
    died = False
    death_reason = None
    nan_detected = False

    def _record(step_idx):
        nonlocal nan_detected
        phi = st.phi
        u = st.u
        p = st.p
        finite = (np.isfinite(phi).all() and np.isfinite(u).all()
                  and np.isfinite(p).all())
        if not finite:
            nan_detected = True
        t_series.append(float(st.t))
        # bubble is phi<0 -> pass -phi to centroid_y (tests' convention)
        cy_series.append(float(metrics.centroid_y(-phi, coords))
                         if finite else float("nan"))
        ekin_series.append(_ekin_lumped(phi, u, rho_h, rho_l)
                           if finite else float("nan"))
        mass_series.append(st.mass_phi() if finite else float("nan"))
        newton_iters.append(int(st.last_newton_iters))
        clamp_counts.append(int(st.last_clamped))
        wall_series.append(float(st.last_wall))

    t_wall0 = time.perf_counter()

    # if ratio<1000 we already took step 1 above; record it, then continue
    start_k = 0
    if ratio < 1000:
        _record(1)
        start_k = 1
        if nan_detected:
            died = True
            death_reason = "NaN after first step"

    k = start_k
    while (not died) and k < cap:
        # wall guard (in case extrapolation under-estimated)
        if time.perf_counter() - t_wall0 > wall_cap_s:
            robustness_events.append(
                {"step": k, "event": "wall-cap hit mid-run, truncating"})
            break
        try:
            st.step()
        except RuntimeError as e:
            died = True
            death_reason = f"{type(e).__name__}: {e}"
            robustness_events.append({"step": k + 1, "event": death_reason})
            break
        k += 1
        _record(k)
        if nan_detected and not died:
            died = True
            death_reason = f"NaN detected at step {k}"
            robustness_events.append({"step": k, "event": death_reason})
            break
        if k % PROGRESS_EVERY == 0:
            el = time.perf_counter() - t_wall0
            print(f"    [{stepper_name} r{int(ratio)}] step {k}/{cap} "
                  f"t={st.t:.4f} cy={cy_series[-1]:.5f} "
                  f"nit={newton_iters[-1]} clamp={clamp_counts[-1]} "
                  f"wall={el:.0f}s", flush=True)

    steps_run = k
    total_wall = time.perf_counter() - t_wall0

    # --- reductions ---------------------------------------------------------
    wall_arr = np.asarray(wall_series, float)
    nit_arr = np.asarray(newton_iters, float)
    mass_arr = np.asarray(mass_series, float)
    t_arr = np.asarray(t_series, float)

    def _safe(fn, arr, default=float("nan")):
        arr = arr[np.isfinite(arr)]
        return float(fn(arr)) if arr.size else default

    mass_drift = mass_arr - mass0
    drift_finite = mass_drift[np.isfinite(mass_drift)]
    max_abs_drift = float(np.max(np.abs(drift_finite))) if drift_finite.size else float("nan")

    # rise velocity peak + min circularity (compute circularity on final finite)
    rise_peak = float("nan")
    if t_arr.size >= 2 and np.isfinite(cy_series).sum() >= 2:
        cy_fin = np.asarray(cy_series, float)
        good = np.isfinite(cy_fin)
        if good.sum() >= 2:
            rv = metrics.rise_velocity(cy_fin[good], dt)
            rise_peak = float(np.nanmax(rv))
    min_circ = float("nan")
    # circularity of a few snapshots (cheap; needs full phi -> recompute last)
    try:
        if np.isfinite(st.phi).all():
            min_circ = float(metrics.circularity(-st.phi, coords, h))
    except Exception:
        min_circ = float("nan")

    # --- downsample series for the JSON ------------------------------------
    idx = _downsample_idx(len(t_series), SERIES_POINTS)
    ds = lambda a: [float(a[i]) for i in idx]

    result = {
        "stepper": stepper_name,
        "rho_ratio": float(ratio),
        "case": {
            "name": case.name, "level": LEVEL, "dim": case.dim,
            "Re": case.Re, "We": case.We, "Cn_override": CN_OVERRIDE,
            "Cn_effective": float(st.Cn), "Pe": case.Pe, "Fr": case.Fr,
            "eta_ratio": case.eta_ratio, "t_end": case.t_end, "dt0": dt,
            "n_nodes": int(dm.n_nodes), "h": h,
        },
        "steps_run": int(steps_run),
        "steps_full_t_end": int(steps_full),
        "capped": bool(steps_run < steps_full),
        "cap_reason": cap_reason,
        "t_reached": float(t_series[-1]) if t_series else 0.0,
        "survived": (not died),
        "died": bool(died),
        "death_reason": death_reason,
        "nan_detected": bool(nan_detected),
        "robustness_events": robustness_events,
        "wall_per_step_mean": _safe(np.mean, wall_arr),
        "wall_per_step_p95": _safe(lambda a: np.percentile(a, 95), wall_arr),
        "total_wall_s": float(total_wall),
        "newton_iters_mean": _safe(np.mean, nit_arr),
        "newton_iters_max": int(np.nanmax(nit_arr)) if nit_arr.size else 0,
        "clamp_total": int(np.nansum(clamp_counts)),
        "clamp_max": int(np.nanmax(clamp_counts)) if clamp_counts else 0,
        "mass_initial": float(mass0),
        "mass_final": float(mass_arr[-1]) if mass_arr.size else float("nan"),
        "mass_drift_max_abs": max_abs_drift,
        "rise_velocity_peak": rise_peak,
        "circularity_final": min_circ,
        "energy_proxy": "lumped-node E_kin = 0.5*sum(rho(phi_i)|u_i|^2)/n",
        "series": {
            "t": ds(t_series),
            "centroid_y": ds(cy_series),
            "E_kin": ds(ekin_series),
            "mass_drift": [float(mass_series[i] - mass0) for i in idx],
            "newton_iters": [int(newton_iters[i]) for i in idx],
        },
    }
    return result


# ---------------------------------------------------------------------------
# Plots.
# ---------------------------------------------------------------------------
def make_plots(results_by_ratio, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for ratio, cells in sorted(results_by_ratio.items()):
        # centroid overlay
        fig, ax = plt.subplots(figsize=(6, 4))
        for r in cells:
            s = r["series"]
            lbl = (f"{r['stepper']}"
                   + ("" if r["survived"]
                      else f" (died @ step {r['steps_run']})"))
            ax.plot(s["t"], s["centroid_y"], marker=".", ms=3, label=lbl)
        ax.set_xlabel("t"); ax.set_ylabel("centroid_y")
        ax.set_title(f"BUBBLE_RISE_RE35_WE10  L{LEVEL}  rho_ratio={int(ratio)}")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"centroid_r{int(ratio)}.png"),
                    dpi=110)
        plt.close(fig)

        # energy overlay
        fig, ax = plt.subplots(figsize=(6, 4))
        for r in cells:
            s = r["series"]
            ax.plot(s["t"], s["E_kin"], marker=".", ms=3, label=r["stepper"])
        ax.set_xlabel("t"); ax.set_ylabel("E_kin (lumped-node proxy)")
        ax.set_title(f"kinetic energy  L{LEVEL}  rho_ratio={int(ratio)}")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"energy_r{int(ratio)}.png"), dpi=110)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Evidence table.
# ---------------------------------------------------------------------------
def write_evidence_table(all_results, out_dir):
    lines = []
    lines.append("# SP-0 Task 8 — coupling-decision spike evidence")
    lines.append("")
    lines.append(f"BUBBLE_RISE_RE35_WE10, level {LEVEL} (64x64), "
                 f"Cn_override={CN_OVERRIDE}. Steppers compared to EACH OTHER "
                 "(no reference-curve comparison — Task 9).")
    lines.append("")
    hdr = ("| stepper | ratio | steps (run/full) | capped | survived | "
           "wall/step mean (s) | wall/step p95 (s) | Newton mean/max | "
           "clamp total | max\\|drift\\| | rise-vel peak | circ final |")
    sep = ("|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines.append(hdr); lines.append(sep)
    for r in all_results:
        lines.append(
            f"| {r['stepper']} | {int(r['rho_ratio'])} | "
            f"{r['steps_run']}/{r['steps_full_t_end']} | "
            f"{'yes' if r['capped'] else 'no'} | "
            f"{'YES' if r['survived'] else 'NO'} | "
            f"{r['wall_per_step_mean']:.3f} | {r['wall_per_step_p95']:.3f} | "
            f"{r['newton_iters_mean']:.2f}/{r['newton_iters_max']} | "
            f"{r['clamp_total']} | {r['mass_drift_max_abs']:.2e} | "
            f"{r['rise_velocity_peak']:.4f} | {r['circularity_final']:.4f} |")
    lines.append("")

    # centroid agreement (staggered vs monolithic) at matching ratios
    lines.append("## Centroid agreement (staggered vs monolithic)")
    lines.append("")
    lines.append("Both downsampled centroid_y series are linearly interpolated "
                 "onto the COMMON time support [0, min(t_max)] (the two runs "
                 "cover different spans when one is capped or dies); "
                 "max|delta| and the value of each at the common t_max are "
                 "reported.")
    lines.append("")
    lines.append("| ratio | common t_max | max\\|Δcentroid_y\\| | "
                 "centroid @ common t_max (stag / mono) |")
    lines.append("|---|---|---|---|")
    by_ratio = {}
    for r in all_results:
        by_ratio.setdefault(int(r["rho_ratio"]), {})[r["stepper"]] = r
    for ratio in sorted(by_ratio):
        cell = by_ratio[ratio]
        if "staggered" in cell and "monolithic" in cell:
            sd = cell["staggered"]["series"]
            md = cell["monolithic"]["series"]
            ta, ca = np.asarray(sd["t"], float), np.asarray(
                sd["centroid_y"], float)
            tb, cb = np.asarray(md["t"], float), np.asarray(
                md["centroid_y"], float)
            ga, gb = np.isfinite(ca), np.isfinite(cb)
            if ga.sum() < 2 or gb.sum() < 2:
                lines.append(f"| {ratio} | n/a | n/a | n/a |")
                continue
            ta, ca, tb, cb = ta[ga], ca[ga], tb[gb], cb[gb]
            t_max = float(min(ta[-1], tb[-1]))
            tg = np.linspace(float(max(ta[0], tb[0])), t_max, 200)
            ia = np.interp(tg, ta, ca)
            ib = np.interp(tg, tb, cb)
            maxdiff = float(np.max(np.abs(ia - ib)))
            lines.append(
                f"| {ratio} | {t_max:.4f} | {maxdiff:.3e} | "
                f"{ia[-1]:.5f} / {ib[-1]:.5f} |")
    lines.append("")
    path = os.path.join(out_dir, "evidence_table.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def build_matrix():
    """Return the list of (stepper, ratio) cells in run order."""
    cells = []
    for ratio in (10, 100):
        for stp in ("staggered", "monolithic"):
            cells.append((stp, ratio))
    for stp in ("staggered", "monolithic"):
        cells.append((stp, 1000))
    return cells


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/chns/results/spike/")
    ap.add_argument("--only", default=None,
                    help="single cell 'stepper:ratio', e.g. staggered:10")
    ap.add_argument("--wall-cap-min", type=float, default=25.0,
                    help="per-run wall cap in minutes (default 25)")
    args = ap.parse_args(argv)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    wall_cap_s = args.wall_cap_min * 60.0

    cells = build_matrix()
    if args.only:
        stp, rt = args.only.split(":")
        cells = [(stp.strip(), int(rt))]

    print(f"[spike] out={out_dir}")
    print(f"[spike] level={LEVEL} wall_cap={args.wall_cap_min:.0f}min "
          f"cells={cells}", flush=True)

    print("[spike] building level-6 mesh...", flush=True)
    dm, mesh, cons = make_dm(LEVEL, dim=2)
    print(f"[spike] n_nodes={dm.n_nodes} h={float(dm.mesh.tree.h().min()):.5f}",
          flush=True)

    all_results = []
    for (stp, ratio) in cells:
        print(f"\n[spike] === {stp} x rho_ratio={ratio} ===", flush=True)
        try:
            res = run_cell(stp, ratio, dm, mesh, cons, wall_cap_s)
        except Exception as e:
            # a genuine script/setup error (not a stiffness death, which
            # run_cell catches internally) — record and continue.
            print(f"[spike] !! cell crashed hard: {e}", flush=True)
            traceback.print_exc()
            res = {
                "stepper": stp, "rho_ratio": float(ratio),
                "hard_error": f"{type(e).__name__}: {e}",
                "survived": False, "died": True,
                "steps_run": 0, "steps_full_t_end": 0, "capped": False,
                "wall_per_step_mean": float("nan"),
                "wall_per_step_p95": float("nan"),
                "newton_iters_mean": float("nan"), "newton_iters_max": 0,
                "clamp_total": 0, "mass_drift_max_abs": float("nan"),
                "rise_velocity_peak": float("nan"),
                "circularity_final": float("nan"),
                "series": {"t": [], "centroid_y": [], "E_kin": [],
                           "mass_drift": [], "newton_iters": []},
            }
        # per-run JSON
        pj = os.path.join(out_dir, f"spike_{stp}_r{int(ratio)}.json")
        with open(pj, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"[spike] wrote {pj}  survived={res['survived']} "
              f"steps={res['steps_run']}/{res.get('steps_full_t_end','?')} "
              f"wall/step={res['wall_per_step_mean']:.3f}s", flush=True)
        all_results.append(res)

    # combined summary
    summary = {
        "config": {"level": LEVEL, "case": "bubble_rise_Re35_We10",
                   "Cn_override": CN_OVERRIDE,
                   "wall_cap_min": args.wall_cap_min,
                   "energy_proxy":
                       "lumped-node E_kin = 0.5*sum(rho(phi_i)|u_i|^2)/n"},
        "runs": all_results,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # plots + table (group by ratio)
    by_ratio = {}
    for r in all_results:
        if r.get("series", {}).get("t"):
            by_ratio.setdefault(int(r["rho_ratio"]), []).append(r)
    try:
        make_plots(by_ratio, out_dir)
        print("[spike] wrote plots", flush=True)
    except Exception as e:
        print(f"[spike] plot step failed (non-fatal): {e}", flush=True)
    tbl = write_evidence_table(all_results, out_dir)
    print(f"[spike] wrote {tbl}", flush=True)
    print("[spike] DONE", flush=True)


if __name__ == "__main__":
    main()
