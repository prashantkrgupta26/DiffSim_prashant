"""OrgElMorph course - Computational C7 driver (harness-based, verified).

Profiles ONE complete Cahn-Hilliard time step and the device primitives that
make it up, then prints a PASS/FAIL gate table:

  1. STEP timing (CUDA-synced warm step) + Newton iterations.
  2. ASYNC pitfall + kernel-launch overhead.
  3. BANDWIDTH vs arithmetic intensity (device triad vs PCIe copies).
  4. PLAN/FACTOR reuse (LU factor vs solve) + per-step D2H copy volume.
  5. DEVICE-MEMORY high-water (driver footprint + Ae-buffer breakdown).
  6. NSIGHT SYSTEMS: profile the one-step workload and read back the NVTX
     timeline (real capture).  Kernel/memcpy-level rows need a CUPTI matching
     the CUDA driver; the commands are taught either way.

    PYTHONPATH=<repo>/src <repo>/.venv/bin/python run.py --device cuda:0 \\
        --solver splu --output outputs/c7 --overwrite --mode reference

Config is the canonical record; results.json is tolerance-checked against
baseline.yaml; provenance goes to metadata.json.  gen_figures.py renders the
document's figures + numbers/c7.tex from the saved run.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                os.pardir, os.pardir)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import config as cfgmod                          # noqa: E402
from common.run_base import build_parser, run_tutorial        # noqa: E402

import profile_ch as prof                                     # noqa: E402

SCHEMA = cfgmod.ConfigSchema(name="c7", fields={
    "level": cfgmod.Field(int, default=6, min=4, max=8),
    "bw_nelems": cfgmod.Field(int, default=1 << 20, min=1 << 16),
    "launch_reps": cfgmod.Field(int, default=2000, min=100),
    "run_nsys": cfgmod.Field(bool, default=True),
    "precision": cfgmod.Field(str, default="fp64"),
})


def _run_nsys(here, device, level, run_dir, log):
    """Profile the one-step workload under Nsight Systems and read back the
    NVTX range summary.  Best effort: if nsys is absent or the report cannot be
    produced, return an availability record with the taught commands and let
    the numeric gates carry the chapter."""
    nsys = shutil.which("nsys")
    workload = os.path.join(here, "nsys_workload.py")
    rep = os.path.join(run_dir, "step_profile")
    out = {"nsys_path": nsys, "captured": False, "nvtx": {},
           "kernel_rows": False}
    if nsys is None:
        log("   nsys not found on PATH -- run these on your box (see chapter)")
        return out
    env = dict(os.environ)
    try:
        subprocess.run(
            [nsys, "profile", "--trace=cuda,nvtx", "--sample=none",
             "--cpuctxsw=none", "--force-overwrite=true", "-o", rep,
             sys.executable, workload, "--device", device,
             "--level", str(level)],
            check=True, capture_output=True, text=True, timeout=600, env=env)
        stats = subprocess.run(
            [nsys, "stats", "--report", "nvtx_sum", "--report",
             "cuda_gpu_kern_sum", "--format", "json", rep + ".nsys-rep"],
            capture_output=True, text=True, timeout=600, env=env)
        txt = stats.stdout
        out["captured"] = os.path.exists(rep + ".nsys-rep")
        out["kernel_rows"] = "does not contain CUDA kernel data" not in \
            (stats.stdout + stats.stderr)
        # parse the NVTX range durations (ns) for warmup / ch_step
        for blob in _iter_json_arrays(txt):
            for row in blob:
                name = str(row.get("Range", row.get("Name", "")))
                if name in ("warmup", "ch_step"):
                    ns = row.get("Total Time (ns)") or row.get("Total Time")
                    if ns is not None:
                        out["nvtx"][name] = float(ns) / 1e6   # -> ms
        log(f"   nsys captured={out['captured']} kernel_rows="
            f"{out['kernel_rows']} nvtx={ {k: round(v,1) for k,v in out['nvtx'].items()} }")
    except Exception as e:                          # pragma: no cover
        log(f"   nsys capture failed ({type(e).__name__}); teaching commands "
            f"only")
    return out


def _iter_json_arrays(text):
    """Yield top-level JSON arrays from concatenated nsys --format json output."""
    dec = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] not in "[{":
            i += 1
        if i >= n:
            break
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, list):
            yield obj
        i = end


def c7_run(cfg, ctx):
    dev = ctx.device
    lvl = cfg["level"]
    here = os.path.dirname(os.path.abspath(__file__))
    ctx.provenance.update(
        mesh={"family": "uniform box 2D", "level": lvl},
        time_integrator="BDF1 (single-step profiling)",
        note="C7 CUDA profiling and memory")

    ctx.log("1. STEP timing (one complete, CUDA-synced warm step)")
    step = prof.step_timing(lvl, dev)
    ctx.log(f"   level {lvl}: {step['step_ms']:.1f} ms/step, "
            f"{step['newton_iters']} Newton iters, dofs={step['dofs']}, "
            f"nnz={step['nnz']}")

    ctx.log("2. ASYNC pitfall + kernel-launch overhead")
    lo = prof.launch_overhead(dev, cfg["bw_nelems"], cfg["launch_reps"])
    ctx.log(f"   launch overhead {lo['us_per_launch_sync']:.1f} us; "
            f"single launch unsynced {lo['async_unsynced_ms']:.3f} ms vs "
            f"synced {lo['async_synced_ms']:.3f} ms "
            f"({lo['async_ratio']:.0f}x)")

    ctx.log("3. BANDWIDTH vs arithmetic intensity")
    bw = prof.bandwidth(dev, cfg["bw_nelems"])
    ctx.log(f"   device triad {bw['device_gbs']:.0f} GB/s "
            f"(AI {bw['device_ai']:.3f} flop/byte); PCIe copy "
            f"{bw['pcie_gbs']:.1f} GB/s -- {bw['device_gbs']/bw['pcie_gbs']:.0f}x"
            f" slower")

    ctx.log("4. PLAN/FACTOR reuse (LU factor vs solve)")
    fr = prof.factor_reuse(lvl, dev)
    ctx.log(f"   factor {fr['factor_ms']:.1f} ms vs solve {fr['solve_ms']:.3f} "
            f"ms ({fr['factor_over_solve']:.0f}x); step refactors "
            f"{fr['newton_iters']}x; D2H {fr['d2h_per_step_mb']:.1f} MB/step")

    ctx.log("5. DEVICE-MEMORY high-water")
    mem = prof.device_memory(lvl, dev)
    fp = mem["footprint_mb"]
    ctx.log(f"   driver footprint {mem['driver_mb']} MB (mempool-quantised); "
            f"analytical accounting {mem['analytical_mb']:.2f} MB; "
            f"Ae buffer {mem['ae_buffer_mb']:.2f} MB dominates "
            f"({mem['n_elements']} elements)")

    ctx.log("6. NSIGHT SYSTEMS capture (one step)")
    nsys = _run_nsys(here, dev, lvl, ctx.paths["root"], ctx.log) \
        if cfg["run_nsys"] else {"captured": False, "nvtx": {}, "kernel_rows": False}

    # figures history ------------------------------------------------------
    ctx.history["bw_copy_mb"] = np.asarray([c["mb"] for c in bw["copies"]])
    ctx.history["bw_h2d"] = np.asarray([c["h2d_gbs"] for c in bw["copies"]])
    ctx.history["bw_d2h"] = np.asarray([c["d2h_gbs"] for c in bw["copies"]])
    ctx.history["bw_device_gbs"] = np.asarray([bw["device_gbs"]])
    # a small step-time breakdown estimate (factor-dominated host time)
    factor_total = fr["factor_ms"] * fr["newton_iters"]
    solve_total = fr["solve_ms"] * fr["newton_iters"]
    other = max(step["step_ms"] - factor_total - solve_total, 0.0)
    ctx.history["break_labels"] = np.asarray(
        ["LU factor x%d" % fr["newton_iters"], "LU solve", "assembly+copies+host"])
    ctx.history["break_ms"] = np.asarray([factor_total, solve_total, other])
    # device-memory analytical breakdown (drop the 'total' aggregate)
    mparts = {k: v for k, v in mem["parts_mb"].items() if k != "total"}
    ctx.history["mem_labels"] = np.asarray(list(mparts.keys()))
    ctx.history["mem_mb"] = np.asarray(list(mparts.values()))

    results = {
        "step_ms": step["step_ms"],
        "newton_iters": step["newton_iters"],
        "dofs": step["dofs"], "nnz": step["nnz"],
        "launch_us": lo["us_per_launch_sync"],
        "async_unsynced_ms": lo["async_unsynced_ms"],
        "async_synced_ms": lo["async_synced_ms"],
        "async_ratio": lo["async_ratio"],
        "device_gbs": bw["device_gbs"],
        "device_ai": bw["device_ai"],
        "pcie_gbs": bw["pcie_gbs"],
        "bw_ratio": bw["device_gbs"] / bw["pcie_gbs"],
        "factor_ms": fr["factor_ms"],
        "solve_ms": fr["solve_ms"],
        "factor_over_solve": fr["factor_over_solve"],
        "d2h_per_step_mb": fr["d2h_per_step_mb"],
        "footprint_mb": fp,
        "driver_mb": mem["driver_mb"],
        "analytical_mb": mem["analytical_mb"],
        "ae_buffer_mb": mem["ae_buffer_mb"],
        "n_elements": mem["n_elements"],
        "nsys_captured": bool(nsys.get("captured")),
        "nsys_kernel_rows": bool(nsys.get("kernel_rows")),
        "nvtx_warmup_ms": nsys.get("nvtx", {}).get("warmup"),
        "nvtx_step_ms": nsys.get("nvtx", {}).get("ch_step"),
    }

    gates = {
        "one step timed (>0), Newton iters in [1,20]":
            step["step_ms"] > 0 and 1 <= step["newton_iters"] <= 20,
        "device bandwidth >> PCIe copy bandwidth (>10x)":
            bw["device_gbs"] / bw["pcie_gbs"] > 10.0,
        "streaming kernel is memory-bound (AI < 0.5 flop/byte)":
            bw["device_ai"] < 0.5,
        "LU factor costs >> solve (>5x)":
            fr["factor_over_solve"] > 5.0,
        "device-memory footprint measured (>0)":
            fp is not None and fp > 0.0,
        "kernel-launch overhead measured (us range)":
            0.0 < lo["us_per_launch_sync"] < 1000.0,
        "nsys captured the one-step NVTX timeline":
            bool(nsys.get("captured")) and "ch_step" in nsys.get("nvtx", {}),
    }
    ctx.log("--- self-check gates ---")
    for name, ok in gates.items():
        ctx.log(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    # nsys gate is advisory (a matched box may differ): report but do not fail
    core = {k: v for k, v in gates.items() if not k.startswith("nsys")}
    ctx.log(f"ALL CORE GATES: {'PASS' if all(core.values()) else 'FAIL'}")
    results["all_gates_pass"] = bool(all(core.values()))
    results["nsys_gate"] = bool(gates["nsys captured the one-step NVTX timeline"])
    return results


def main():
    args = build_parser("OrgElMorph C7 (CUDA profiling)").parse_args()
    here = os.path.dirname(__file__)
    baseline = (os.path.join(here, "baseline.yaml")
                if args.mode == "reference" else None)
    run_tutorial(c7_run, schema=SCHEMA, args=args,
                 default_output=os.path.join(here, "outputs", "c7"),
                 baseline=baseline, default_solver="splu")


if __name__ == "__main__":
    main()
