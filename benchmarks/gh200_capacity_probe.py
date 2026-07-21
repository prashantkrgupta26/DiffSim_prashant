#!/usr/bin/env python3
"""GH200 coherent-capacity probe (M1d #34, index-widening #33 GPU smoke).

A thin driver over the film front-end that marches the ternary-film
``blockch_dev`` device-resident path at a chosen resolution for a few
accepted steps, then distils the run's ``runlog.jsonl`` into ONE compact
ladder row:

    dofs, nnz, nnz/dof, GPU GB (measured peak), host forecast GB,
    s/step (median of accepted-step deltas), index-width mode
    (narrow|wide auto), mass_drift, and a verdict.

It changes NOTHING in the solver: ``index_width`` defaults to "auto" all
the way down, so the wide mixed-width CSR path (#33) self-selects once nnz
crosses ~90% of 2^31 (~18 M dofs at the film's ~106 nnz/dof). This probe
merely reports which path fired (from the preflight ``index-width`` check)
and confirms the physics stays sane (mass drift ~1e-14 class).

Design: reuse the committed wodo3d-hero 3-D extension EXACTLY (physics
from ``wodo2012_fig6_n5.yaml``; the only overrides are the 3-D domain +
resolution + ``blockch_dev`` + step budget). Sizes scale the in-plane
mesh keeping the film aspect (Lx==Ly, thin Z), anchored on the paper's
230x230x70 = 15.15 M-dof hero. ``--no-strict`` is carried because the
interface-resolution rule FAILs off-anchor (documented deviation, same as
the a100_wodo3d_hero kit) — we are measuring capacity, not validating
morphology.

Usage (one size per sbatch, tee the stdout to the probe results dir):

    python benchmarks/gh200_capacity_probe.py --res 230 230 70 \
        --max-steps 4 --outdir /work/.../gh200-probe2/n230

    # oversubscription driver mode: emit the ladder row for a size that
    # is expected to spill past HBM (the driver does NOT set any UM env
    # var itself — pass allocator env in the sbatch wrapper; this probe
    # just records whether the march completed and its s/step).

The ladder row is printed as a ``PROBE_ROW <json>`` line (grep-friendly)
and also as a human table line, so an sbatch loop can collect many sizes
into one findings table.
"""
import argparse
import json
import os
import subprocess
import sys
import time


def _film_argv(cfg, outdir, res, max_steps, extra_sets):
    """Build the film front-end argv (sans the python -m prefix).

    Mirrors cluster/wodo_campaign/a100_wodo3d_hero.sbatch's 3-D
    extension of the fig6_n5 config verbatim, parametrised on resolution.
    """
    nx, ny, nz = res
    argv = [
        cfg,
        "--outdir", outdir,
        "--set", "domain.dim=3",
        "--set", "domain.Lx=3.3",
        "--set", "domain.Ly=3.3",
        "--set", f"domain.resolution=[{nx},{ny},{nz}]",
        "--set", "physics.blend={ratio: '1:1', phi_s0: 0.66}",
        "--set", "numerics.linsolver=blockch_dev",
        "--set", "numerics.device_assembly=true",
        "--set", f"stop.max_steps={max_steps}",
        # give the probe a clean stop: max_steps, not h/phis milestones
        "--set", "stop.h_min=0.0",
        "--set", "stop.phis_stop=0.0",
        "--no-strict",
    ]
    for s in extra_sets:
        argv += ["--set", s]
    return argv


def _film_cmd(cfg, outdir, res, max_steps, extra_sets):
    """Subprocess argv: default (mempool/default warp allocator) path."""
    return [sys.executable, "-m", "diffsim.film"] + \
        _film_argv(cfg, outdir, res, max_steps, extra_sets)


def _install_managed_allocator(device="cuda:0"):
    """OVERSUBSCRIPTION MECHANISM (M1d item 2), zero solver change.

    Warp 1.15 instantiates ``CudaManagedAllocator`` (cudaMallocManaged)
    but never wires it to ``current_allocator`` -- the built-in choice is
    mempool-or-default, both HBM-bound. The PUBLIC hook
    ``wp.set_device_allocator(dev, wp.CudaManagedAllocator())`` routes
    every subsequent ``wp.zeros``/``wp.array`` device allocation through
    managed memory, which on GH200's NVLink-C2C coherent fabric spills
    transparently into the 480 GB Grace LPDDR5X pool past the 96 GB HBM3.
    The dominant film buffers (``vals_d`` nnz x f64, slot/offset arrays)
    are all warp-owned, so this swap makes the WHOLE device-resident march
    oversubscribable without touching diffsim. Called in-process before
    the film run allocates anything.
    """
    import warp as wp
    wp.init()
    dev = wp.get_device(device)
    if not dev.is_managed_memory_supported:
        raise RuntimeError(
            f"device {dev} reports is_managed_memory_supported=False -- "
            "managed oversubscription unavailable")
    # NB: CudaManagedAllocator() takes no ctor args -- device-agnostic;
    # warp pushes the target context before each allocate()
    wp.set_device_allocator(dev, wp.CudaManagedAllocator())
    print(f"PROBE_MANAGED installed CudaManagedAllocator on {dev} "
          f"(concurrent_managed_access="
          f"{dev.is_concurrent_managed_access_supported})", flush=True)


def _distil(outdir):
    """Read runlog.jsonl written by the film run -> one ladder dict."""
    path = os.path.join(outdir, "runlog.jsonl")
    prov = pf = final = None
    attempts = []
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            t = rec.get("type")
            if t == "provenance":
                prov = rec
            elif t == "preflight":
                pf = rec
            elif t == "attempt" and rec.get("accepted"):
                attempts.append(rec)
            elif t == "final":
                final = rec

    # -- preflight-derived fields -------------------------------------
    # dofs/nnz are recovered from the check detail strings (the jsonl
    # provenance record does NOT carry them): the index-width detail
    # opens with "nnz=<N> = ..%", and the memory-forecast detail closes
    # with "for <M.MM>M dofs".
    import re
    idx_mode = mem_txt = None
    dofs = nnz = None
    chunked = None
    if pf is not None:
        for c in pf["checks"]:
            if c["rule"] == "index-width":
                d = c["detail"]
                if "wide (auto)" in d:
                    idx_mode = "wide-auto"
                elif "wide (forced)" in d:
                    idx_mode = "wide-forced"
                elif "narrow" in d:
                    idx_mode = "narrow"
                m = re.search(r"nnz=(\d+)", d)
                if m:
                    nnz = int(m.group(1))
                # Task #38 block-row ChunkedCSR: preflight reports
                # "block-row chunked CSR (C=<n> chunks, auto|forced ...)"
                m = re.search(r"chunked CSR \(C=(\d+) chunks", d)
                chunked = f"C={m.group(1)}" if m else "off"
            if c["rule"] == "memory-forecast":
                mem_txt = c["detail"]
                m = re.search(r"for ([\d.]+)M dofs", d if False else c["detail"])
                if m:
                    dofs = int(round(float(m.group(1)) * 1e6))

    # -- s/step from accepted-step wall deltas -------------------------
    # ``wall`` is cumulative seconds since march start; the per-step cost
    # is the successive difference. Drop step 1 (carries symbolic pair
    # setup + first Warp compile) and report the median of the rest.
    walls = [a["wall"] for a in attempts]
    per = [walls[i] - walls[i - 1] for i in range(1, len(walls))]
    per_sorted = sorted(per)
    s_per_step = (per_sorted[len(per_sorted) // 2] if per_sorted
                  else (walls[0] if walls else None))
    first_step_s = walls[0] if walls else None

    # -- peak measured GPU GB -----------------------------------------
    gpu_peak = max((a.get("gpu_gb") or 0.0) for a in attempts) \
        if attempts else None

    mass_drift = final.get("mass_drift") if final else None
    reason = final.get("reason") if final else "no-final"
    steps = final.get("steps") if final else len(attempts)

    return {
        "dofs": dofs,
        "nnz": nnz,
        "nnz_per_dof": (round(nnz / dofs, 1) if dofs and nnz else None),
        "gpu_gb_peak": (round(gpu_peak, 1) if gpu_peak else None),
        "s_per_step": (round(s_per_step, 2)
                       if s_per_step is not None else None),
        "first_step_s": (round(first_step_s, 1)
                         if first_step_s is not None else None),
        "index_width": idx_mode,
        "chunked": chunked,
        "mass_drift": mass_drift,
        "steps": steps,
        "reason": reason,
        "mem_forecast": mem_txt,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--res", nargs=3, type=int, required=True,
                    metavar=("NX", "NY", "NZ"),
                    help="film mesh resolution (keep Lx==Ly aspect)")
    ap.add_argument("--config", default=None,
                    help="film config (default: fig6_n5 in-package)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-steps", type=int, default=4,
                    help="accepted-step budget (3-4 suffices)")
    ap.add_argument("--set", action="append", default=[], dest="sets",
                    metavar="KEY=VALUE",
                    help="extra film --set overrides (repeatable)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the film argv and exit")
    ap.add_argument("--managed", action="store_true",
                    help="OVERSUBSCRIPTION mode: install warp's "
                         "CudaManagedAllocator (cudaMallocManaged) BEFORE "
                         "the film run and drive it in-process, so device "
                         "arrays spill into the Grace pool past HBM. No "
                         "solver change.")
    ap.add_argument("--balloon-gb", type=float, default=0.0,
                    help="pin this many GB of HBM with a plain (non-"
                         "managed) torch CUDA allocation before the film "
                         "run, shrinking effective HBM so a managed run of "
                         "known working set spills a controlled fraction "
                         "into the Grace pool (oversubscription-"
                         "degradation experiment at fixed problem size)")
    ap.add_argument("--chunking", choices=("auto", "off", "force"),
                    default=None,
                    help="Task #38 block-row ChunkedCSR knob, set via a "
                         "probe-side WodoFilmStepper class attribute "
                         "(same idiom as --force-narrow). Default: leave "
                         "the stepper's 'auto' (chunking fires with the "
                         "wide auto-switch at ~90%% of 2^31 nnz). "
                         "'off' past 2^31 FAILs loudly (warp array "
                         "ceiling); 'force' chunks at any size.")
    ap.add_argument("--force-narrow", action="store_true",
                    help="force index_width='narrow' via a probe-side "
                         "WodoFilmStepper class attribute (legal below "
                         "2^31 nnz). Needed in the 90-100%% of 2^31 band "
                         "where 'auto' picks wide but warp's int32 array-"
                         "shape ceiling (types.py check_array_shape) "
                         "rejects >2^31-element nnz buffers.")
    ap.add_argument("--val-dtype", choices=("fp64", "fp32"), default=None,
                    help="Task #36 (8j) mixed-precision value storage, set "
                         "via a probe-side WodoFilmStepper class attribute "
                         "(same idiom as --force-narrow). Default: leave the "
                         "stepper's 'fp64' (unchanged). 'fp32' stores the "
                         "CSR values as an fp32 round-on-store snapshot fed "
                         "to a cuDSS fp32 factorization, recovered to fp64 by "
                         "iterative refinement (residual in fp64 vs the fp64 "
                         "vals_d). Measures s/step + peak GB vs fp64 (G4, "
                         "measurement-only — the §8j Ada net-loss prior "
                         "stands). NB: the fp32-FACTOR+IR path lives in the "
                         "cuDSS solve — you MUST pair with "
                         "'--set numerics.linsolver=cudss' to exercise it. "
                         "On the default blockch_dev path fp32 has NO effect: "
                         "the fp32 snapshot is allocated but UNUSED (the "
                         "iterative blockch SpMV reads the fp64 vals_d — "
                         "fp32-factor+IR targets cuDSS only, per #36 scope), "
                         "so it neither halves SpMV bytes nor changes the "
                         "solve there.")
    args = ap.parse_args(argv)

    cfg = args.config
    if cfg is None:
        here = os.path.dirname(os.path.abspath(__file__))
        cfg = os.path.join(here, os.pardir, "src", "diffsim", "film",
                           "configs", "wodo2012_fig6_n5.yaml")
        cfg = os.path.normpath(cfg)

    os.makedirs(args.outdir, exist_ok=True)
    film_argv = _film_argv(cfg, args.outdir, args.res, args.max_steps,
                           args.sets)
    in_process = (args.managed or args.force_narrow
                  or args.balloon_gb > 0 or args.chunking is not None
                  or args.val_dtype is not None)
    print("PROBE_CMD python -m diffsim.film " + " ".join(film_argv)
          + (" [in-process:"
             + ("managed" if args.managed else "")
             + (f" balloon={args.balloon_gb}GB" if args.balloon_gb else "")
             + (" force-narrow" if args.force_narrow else "") + "]"
             if in_process else ""), flush=True)
    if args.dry_run:
        return 0

    t0 = time.time()
    if in_process:
        balloon = None
        if args.balloon_gb > 0:
            # plain torch cudaMalloc -> HBM-resident, NOT spillable:
            # shrinks the HBM the managed working set can occupy.
            import torch
            balloon = torch.empty(int(args.balloon_gb * 2 ** 30),
                                  dtype=torch.uint8, device="cuda:0")
            free_b, tot_b = torch.cuda.mem_get_info("cuda:0")
            print(f"PROBE_BALLOON {args.balloon_gb} GB pinned; HBM free "
                  f"{free_b / 2**30:.1f} / {tot_b / 2**30:.1f} GB",
                  flush=True)
        if args.managed:
            _install_managed_allocator(device="cuda:0")
        if args.force_narrow:
            from diffsim.physics.wodo_film import WodoFilmStepper
            WodoFilmStepper._index_width = "narrow"
            print("PROBE_FORCE_NARROW WodoFilmStepper._index_width="
                  "'narrow' (probe-side class attr)", flush=True)
        if args.chunking is not None:
            from diffsim.physics.wodo_film import WodoFilmStepper
            WodoFilmStepper._chunking = args.chunking
            print(f"PROBE_CHUNKING WodoFilmStepper._chunking="
                  f"{args.chunking!r} (probe-side class attr)",
                  flush=True)
        if args.val_dtype is not None:
            from diffsim.physics.wodo_film import WodoFilmStepper
            WodoFilmStepper._val_dtype = args.val_dtype
            print(f"PROBE_VAL_DTYPE WodoFilmStepper._val_dtype="
                  f"{args.val_dtype!r} (probe-side class attr; #36 8j "
                  f"fp32-storage + FP64-IR)", flush=True)
        from diffsim.film.__main__ import main as film_main
        try:
            rc = film_main(film_argv)
        except Exception as e:                       # OOM / alloc failures
            print(f"PROBE_INPROC_EXC {e!r}", flush=True)
            rc = 1
        del balloon
    else:
        rc = subprocess.run(_film_cmd(
            cfg, args.outdir, args.res, args.max_steps, args.sets)).returncode
    wall = time.time() - t0

    row = {"res": args.res, "wall_total_s": round(wall, 1),
           "film_rc": rc, "managed": args.managed,
           "balloon_gb": args.balloon_gb,
           "force_narrow": args.force_narrow,
           "chunking_arg": args.chunking,
           "val_dtype_arg": args.val_dtype}
    try:
        row.update(_distil(args.outdir))
    except (FileNotFoundError, ValueError) as e:
        row["distil_error"] = repr(e)
    if args.force_narrow:
        # preflight reads params (config-level 'auto'), not the probe's
        # stepper-class override -- report what actually ran
        row["index_width"] = "narrow-forced(probe)"
    if args.chunking is not None and args.chunking != "auto":
        # same params-vs-stepper-attr caveat for the chunking override
        row["chunked"] = f"{args.chunking}(probe)"
    # #36: the val_dtype override is a probe-side stepper class attr (not a
    # film param), so record what actually ran.
    row["val_dtype"] = (f"{args.val_dtype}(probe)"
                        if args.val_dtype is not None else "fp64")

    # verdict: physical if it finished max_steps and mass_drift is small
    md = row.get("mass_drift")
    if rc == 0 and md is not None and md < 1e-10:
        row["verdict"] = "PHYSICAL"
    elif rc == 0:
        row["verdict"] = "COMPLETED-CHECK-DRIFT"
    else:
        row["verdict"] = "FAILED"

    print("PROBE_ROW " + json.dumps(row), flush=True)
    print(
        "PROBE_TABLE "
        f"res={args.res} dofs={row.get('dofs')} nnz={row.get('nnz')} "
        f"gpu_gb={row.get('gpu_gb_peak')} s/step={row.get('s_per_step')} "
        f"idx={row.get('index_width')} chunked={row.get('chunked')} "
        f"val_dtype={row.get('val_dtype')} "
        f"drift={row.get('mass_drift')} "
        f"managed={args.managed} verdict={row.get('verdict')}", flush=True)
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
