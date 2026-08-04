"""GH200 large-DOF adaptive NS-SBM ladder + oversubscription probe.

Task 8b — two-phase capacity ladder on Nova GH200 (95 GiB HBM3):

PHASE 1 (primary): BLUFF-BODY cube-in-channel at UNIFORM levels L4→L5→L6→L7.
  - Reuses build_cube_channel_3d + march_monolithic_3d from the Task 7 ladder
    (ladder_fixtures.py + ladder_rung3d_cube.py).  No modifications to those files.
  - HOST assembly + DEVICE cuDSS solve (this is the correct config for that driver;
    device assembly is not wired for the bluff-body path — stated explicitly in
    the summary table).
  - ~10 steps per rung.  Stop at first hard failure (cuDSS ALLOC / OOM), then run
    exactly the L7 past-wall probe regardless.
  - Oversubscription: the past-wall L7 rung under default allocation IS the probe;
    the gh200_capacity_probe.py --managed lever uses wp.set_device_allocator on
    Warp arrays (the film path).  cuDSS allocates via its own CUDA-stream pools,
    NOT via Warp — so the managed-allocator hook does NOT transfer to this path.
    We report this decision and rely on GH200 C2C hardware behavior (UVM/managed
    pages may help if the CUDA driver opts into C2C coherence at OOM, but we do
    not force it).  The first failing rung's timing and error class ARE the result.

PHASE 1b (adaptive bluff, scope increment 2026-07-25): band-refined cube-channel
  rungs (base L5, band r8), (base L6, band r9), and — only if still under the
  wall — (base L6, band r10).  Same 10-step protocol/measurements via the same
  march (fixture: tests/adaptive_cube_channel.py, CPU-gated by
  tests/test_adaptive_cube_channel.py).  The adaptive rungs reach high near-wall
  resolution at far lower DOF than uniform; DOF is reported alongside the
  effective finest h so the uniform-vs-adaptive comparison is explicit in the
  summary table.

PHASE 2 (secondary, time-permitting): thin-plate adaptive rungs — TRIMMED to
  (5,8), (6,9), (7,9): Phase 1b outranks Phase 2 in the 4h window; the small
  sanity rungs are covered by the GPU smoke + Phase 1.  Skipped entirely if
  >3h have elapsed, so the summary table + sentinel always print before the
  wall-clock kill.

Run:
    CUDA_VISIBLE_DEVICES=0 \\
    PYTHONPATH=src:tests \\
    python tests/gpu_gh200_ladder.py

Not pytest-collected (no test_ prefix).
"""
import os
import subprocess
import sys
import time
import traceback

# ---------------------------------------------------------------------------
# CUDA sanity gate
# ---------------------------------------------------------------------------
import warp as wp
wp.init()
_cuda_dev = wp.get_device("cuda:0")
assert _cuda_dev.is_cuda, (
    "No CUDA device found — gpu_gh200_ladder.py requires a CUDA GPU. "
    "Set CUDA_VISIBLE_DEVICES and check the driver."
)
print(f"[ladder] CUDA OK: {_cuda_dev}  "
      f"mem_total={_cuda_dev.total_memory/2**30:.1f} GiB", flush=True)

# ---------------------------------------------------------------------------
# Imports (PYTHONPATH must include src:tests)
# ---------------------------------------------------------------------------
import numpy as np

# Phase 1 imports: bluff-body cube (uniform + adaptive band-refined)
from ladder_fixtures import build_cube_channel_3d, U_IN
from ladder_rung3d_cube import march_monolithic_3d, ALPHA, FN1_GRADDIV_GAMMA
from adaptive_cube_channel import build_adaptive_cube_channel_3d

# Phase 2 imports: thin-plate adaptive
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from p2r1c_thin_plate_flow_3d import run_flow_past_3d
from p2r1c_thin_plate_flow_3d_projection import run_flow_past_3d_projection

# ---------------------------------------------------------------------------
# GPU memory helper
# ---------------------------------------------------------------------------

def _gpu_mem_mib():
    """Return (used_MiB, total_MiB) via subprocess nvidia-smi (point-in-time)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=10, text=True).strip()
        parts = out.split(",")
        return int(parts[0].strip()), int(parts[1].strip())
    except Exception:
        return None, None


def _torch_peak_mib():
    """Return torch CUDA peak memory in MiB if torch is available."""
    try:
        import torch
        return round(torch.cuda.max_memory_allocated(0) / 2**20)
    except Exception:
        return None


def _reset_torch_peak():
    try:
        import torch
        torch.cuda.reset_peak_memory_stats(0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Oversubscription mechanism assessment
# ---------------------------------------------------------------------------
_OSUB_DECISION = (
    "OVERSUBSCRIPTION NOTE: gh200_capacity_probe.py --managed installs "
    "wp.CudaManagedAllocator() on Warp device arrays (film path). "
    "cuDSS allocates its factor matrices via CUDA stream memory pools "
    "(not Warp), so the managed-allocator hook does NOT transfer to the "
    "torch/cuDSS path. The past-wall L7 rung under DEFAULT allocation IS "
    "the oversubscription probe: GH200's NVLink-C2C coherent fabric may "
    "allow the CUDA UVM driver to page-in Grace LPDDR5X transparently, "
    "but we do not force it. Timing + error class of the L7 rung are the "
    "result."
)

# ---------------------------------------------------------------------------
# Phase 1: bluff-body cube-in-channel ladder
# ---------------------------------------------------------------------------

PHASE1_RUNGS = [
    # (level, label, note)
    (4, "L4", "sanity, seconds"),
    (5, "L5", "~1k fluid cells, warm-up"),
    (6, "L6", "~8k cells, ~1.1M dof — expected 95 GiB wall zone"),
    (7, "L7", "~64k cells — PAST-WALL probe"),
]

PHASE1_NSTEPS = 10
PHASE1_DT = 0.02
PHASE1_RE = 40
PHASE1_HALF = 0.125
PHASE1_OFFSET = 0.05   # genuine SBM shift (rung-C' config)
PHASE1_SOLVER = "cudss"
PHASE1_DEVICE = "cuda:0"


def _n_nodes_from_fx(fx):
    return len(fx["mesh"].node_coords)


def _saddle_dof(fx):
    return fx["cons"].free_nodes.shape[0] * fx["ndof"]


def run_phase1():
    """Bluff-body cube ladder: L4→L5→L6→L7 (stop at first hard failure,
    then run L7 past-wall regardless)."""
    print("\n" + "="*72, flush=True)
    print(" PHASE 1: bluff-body cube-in-channel ladder (HOST assembly, cuDSS solve)")
    print(" HOST-ASSEMBLY CAVEAT: device assembly is not wired for this driver.")
    print("="*72 + "\n", flush=True)

    rows = []
    wall_level = None     # level of first hard failure
    past_wall_done = False

    for level, label, note in PHASE1_RUNGS:
        is_past_wall = (level == 7)
        if wall_level is not None and not is_past_wall:
            # stop after first failure, skip intermediate rungs; only L7 runs
            continue

        print(f"\n--- Phase 1 rung {label} ({note}) ---", flush=True)
        print(f"    build_cube_channel_3d(level={level}, Re={PHASE1_RE}, "
              f"half={PHASE1_HALF}, offset={PHASE1_OFFSET})", flush=True)

        _reset_torch_peak()
        smi_used_before, smi_total = _gpu_mem_mib()
        t_start = time.time()

        row = dict(phase=1, label=label, level=level, note=note,
                   host_assembly=True, solver=PHASE1_SOLVER)
        try:
            t_build = time.time()
            fx = build_cube_channel_3d(
                level, PHASE1_RE, half=PHASE1_HALF,
                offset=PHASE1_OFFSET, device=PHASE1_DEVICE)
            build_time = time.time() - t_build
            n_nodes = _n_nodes_from_fx(fx)
            n_fluid_cells = fx["n_fluid_cells"]
            saddle_dof = _saddle_dof(fx)
            print(f"    mesh built in {build_time:.1f}s  "
                  f"n_nodes={n_nodes}  n_fluid_cells={n_fluid_cells}  "
                  f"saddle_dof={saddle_dof}", flush=True)

            t_march = time.time()
            result = march_monolithic_3d(
                fx, dt=PHASE1_DT, nsteps=PHASE1_NSTEPS,
                rate_tol=None, log_every=5,
                solver=PHASE1_SOLVER, device=PHASE1_DEVICE,
                strong_obstacle=False,   # weak Nitsche + SBM shift (rung C' config)
                alpha=ALPHA, graddiv_gamma=FN1_GRADDIV_GAMMA,
                backflow_beta=0.5)
            march_time = time.time() - t_march
            steps_done = result["steps"]
            s_per_step = march_time / steps_done if steps_done > 0 else float("nan")
            cd_final = result["cd"]
            cd_finite = np.isfinite(cd_final)

            smi_used_after, _ = _gpu_mem_mib()
            torch_peak = _torch_peak_mib()
            elapsed = time.time() - t_start

            row.update(dict(
                n_nodes=n_nodes, n_fluid_cells=n_fluid_cells,
                saddle_dof=saddle_dof, build_time_s=round(build_time, 2),
                s_per_step=round(s_per_step, 2), steps_done=steps_done,
                cd_final=round(float(cd_final), 5) if cd_finite else None,
                cd_finite=cd_finite,
                smi_used_before_mib=smi_used_before,
                smi_used_after_mib=smi_used_after,
                smi_total_mib=smi_total,
                torch_peak_mib=torch_peak,
                elapsed_s=round(elapsed, 1),
                status="OK",
            ))
            if not cd_finite:
                row["status"] = "NAN-CD"
            print(f"    => Cd={cd_final:+.5f}  s/step={s_per_step:.2f}  "
                  f"smi_after={smi_used_after}MiB  torch_peak={torch_peak}MiB  "
                  f"elapsed={elapsed:.1f}s  status={row['status']}", flush=True)

        except Exception as exc:
            elapsed = time.time() - t_start
            exc_class = type(exc).__name__
            exc_msg = str(exc)[:200]
            tb_tail = traceback.format_exc()[-400:]
            print(f"    WALL at {label}: {exc_class}: {exc_msg}", flush=True)
            print(f"    traceback tail:\n{tb_tail}", flush=True)
            row.update(dict(
                n_nodes=None, saddle_dof=None,
                status="WALL", exc_class=exc_class, exc_msg=exc_msg,
                elapsed_s=round(elapsed, 1),
            ))
            if wall_level is None:
                wall_level = level
                print(f"    [WALL recorded at level {level}]", flush=True)
        finally:
            rows.append(row)
            if is_past_wall:
                past_wall_done = True

    # run L7 past-wall probe if we haven't already (only reached if wall hit earlier)
    if wall_level is not None and not past_wall_done:
        print("\n--- Phase 1 PAST-WALL probe: L7 (forced, regardless of wall) ---",
              flush=True)
        level, label, note = 7, "L7", "PAST-WALL probe"
        _reset_torch_peak()
        t_start = time.time()
        row = dict(phase=1, label=label, level=level, note=note,
                   host_assembly=True, solver=PHASE1_SOLVER,
                   past_wall_probe=True)
        try:
            fx = build_cube_channel_3d(
                level, PHASE1_RE, half=PHASE1_HALF,
                offset=PHASE1_OFFSET, device=PHASE1_DEVICE)
            n_nodes = _n_nodes_from_fx(fx)
            saddle_dof = _saddle_dof(fx)
            n_fluid_cells = fx["n_fluid_cells"]
            t_march = time.time()
            result = march_monolithic_3d(
                fx, dt=PHASE1_DT, nsteps=PHASE1_NSTEPS,
                rate_tol=None, log_every=5,
                solver=PHASE1_SOLVER, device=PHASE1_DEVICE,
                strong_obstacle=False, alpha=ALPHA,
                graddiv_gamma=FN1_GRADDIV_GAMMA, backflow_beta=0.5)
            march_time = time.time() - t_march
            steps_done = result["steps"]
            s_per_step = march_time / steps_done if steps_done > 0 else float("nan")
            cd_final = result["cd"]
            cd_finite = np.isfinite(cd_final)
            smi_used_after, _ = _gpu_mem_mib()
            torch_peak = _torch_peak_mib()
            elapsed = time.time() - t_start
            row.update(dict(
                n_nodes=n_nodes, n_fluid_cells=n_fluid_cells,
                saddle_dof=saddle_dof,
                s_per_step=round(s_per_step, 2), steps_done=steps_done,
                cd_final=round(float(cd_final), 5) if cd_finite else None,
                cd_finite=cd_finite,
                smi_used_after_mib=smi_used_after,
                torch_peak_mib=torch_peak,
                elapsed_s=round(elapsed, 1),
                status="OK" if cd_finite else "NAN-CD",
            ))
            print(f"    => Cd={cd_final:+.5f}  s/step={s_per_step:.2f}  "
                  f"smi_after={smi_used_after}MiB  torch_peak={torch_peak}MiB  "
                  f"elapsed={elapsed:.1f}s  status={row['status']}", flush=True)
        except Exception as exc:
            elapsed = time.time() - t_start
            exc_class = type(exc).__name__
            exc_msg = str(exc)[:200]
            print(f"    PAST-WALL WALL: {exc_class}: {exc_msg}", flush=True)
            print(traceback.format_exc()[-400:], flush=True)
            row.update(dict(
                status="WALL", exc_class=exc_class, exc_msg=exc_msg,
                elapsed_s=round(elapsed, 1),
            ))
        rows.append(row)

    return rows, wall_level


# ---------------------------------------------------------------------------
# Phase 1b: ADAPTIVE bluff-body cube ladder (band-refined octree)
# ---------------------------------------------------------------------------

PHASE1B_RUNGS = [
    # (base, refine_to, label, note, only_if_under_wall)
    (5, 8, "a5r8", "adaptive band r8 (h_fine=1/256)", False),
    (6, 9, "a6r9", "adaptive band r9 (h_fine=1/512)", False),
    (6, 10, "a6r10", "adaptive band r10 — only if still under the wall", True),
]


def run_phase1b():
    """Adaptive band-refined cube-channel ladder (scope increment): high
    near-wall resolution at far lower DOF than the uniform Phase-1 rungs.
    Stops at first hard failure; the (6,10) rung runs only if no wall yet."""
    print("\n" + "="*72, flush=True)
    print(" PHASE 1b: ADAPTIVE bluff-body cube ladder "
          "(band-refined, HOST assembly + cuDSS solve)")
    print(" DOF is reported alongside the effective finest h for the explicit")
    print(" uniform-vs-adaptive comparison.")
    print("="*72 + "\n", flush=True)

    rows = []
    wall_label = None

    for base, refine_to, label, note, only_if_under_wall in PHASE1B_RUNGS:
        if wall_label is not None:
            # stop at first hard failure; the only_if_under_wall rung (6,10)
            # is by definition also skipped once the wall is hit.
            print(f"\n--- Phase 1b rung {label} SKIPPED "
                  f"(wall already hit at {wall_label}) ---", flush=True)
            continue

        h_fine = 1.0 / 2**refine_to
        print(f"\n--- Phase 1b rung {label} (base=L{base}, band=r{refine_to}, "
              f"h_fine={h_fine:.6f}) [{note}] ---", flush=True)

        _reset_torch_peak()
        smi_used_before, smi_total = _gpu_mem_mib()
        t_start = time.time()
        row = dict(phase="1b", label=label, base=base, refine_to=refine_to,
                   h_fine=h_fine, note=note, host_assembly=True,
                   solver=PHASE1_SOLVER)
        try:
            t_build = time.time()
            fx = build_adaptive_cube_channel_3d(
                base_level=base, refine_to=refine_to,
                Re=PHASE1_RE, half=PHASE1_HALF, offset=PHASE1_OFFSET,
                device=PHASE1_DEVICE)
            build_time = time.time() - t_build
            n_nodes = fx["n_nodes"]
            n_hanging = fx["n_hanging"]
            n_fluid_cells = fx["n_fluid_cells"]
            saddle_dof = _saddle_dof(fx)
            print(f"    mesh built in {build_time:.1f}s  n_nodes={n_nodes}  "
                  f"n_hanging={n_hanging}  n_fluid_cells={n_fluid_cells}  "
                  f"saddle_dof={saddle_dof}  h_fine={h_fine:.6f}", flush=True)

            t_march = time.time()
            result = march_monolithic_3d(
                fx, dt=PHASE1_DT, nsteps=PHASE1_NSTEPS,
                rate_tol=None, log_every=5,
                solver=PHASE1_SOLVER, device=PHASE1_DEVICE,
                strong_obstacle=False, alpha=ALPHA,
                graddiv_gamma=FN1_GRADDIV_GAMMA, backflow_beta=0.5)
            march_time = time.time() - t_march
            steps_done = result["steps"]
            s_per_step = march_time / steps_done if steps_done > 0 else float("nan")
            cd_final = result["cd"]
            cd_finite = np.isfinite(cd_final)

            smi_used_after, _ = _gpu_mem_mib()
            torch_peak = _torch_peak_mib()
            elapsed = time.time() - t_start
            row.update(dict(
                n_nodes=n_nodes, n_hanging=n_hanging,
                n_fluid_cells=n_fluid_cells, saddle_dof=saddle_dof,
                build_time_s=round(build_time, 2),
                s_per_step=round(s_per_step, 2), steps_done=steps_done,
                cd_final=round(float(cd_final), 5) if cd_finite else None,
                cd_finite=cd_finite,
                smi_used_before_mib=smi_used_before,
                smi_used_after_mib=smi_used_after,
                smi_total_mib=smi_total,
                torch_peak_mib=torch_peak,
                elapsed_s=round(elapsed, 1),
                status="OK" if cd_finite else "NAN-CD",
            ))
            print(f"    => Cd={cd_final:+.5f}  s/step={s_per_step:.2f}  "
                  f"smi_after={smi_used_after}MiB  torch_peak={torch_peak}MiB  "
                  f"elapsed={elapsed:.1f}s  status={row['status']}", flush=True)
        except Exception as exc:
            elapsed = time.time() - t_start
            exc_class = type(exc).__name__
            exc_msg = str(exc)[:200]
            print(f"    WALL at {label}: {exc_class}: {exc_msg}", flush=True)
            print(traceback.format_exc()[-400:], flush=True)
            row.update(dict(
                status="WALL", exc_class=exc_class, exc_msg=exc_msg,
                elapsed_s=round(elapsed, 1),
            ))
            if wall_label is None:
                wall_label = label
        finally:
            rows.append(row)

    return rows, wall_label


# ---------------------------------------------------------------------------
# Phase 2: thin-plate adaptive ladder (time-permitting)
# ---------------------------------------------------------------------------
# TRIMMED (scope increment): Phase 1b outranks Phase 2 in the 4h window; the
# small sanity rungs (4,6)/(5,7)/(6,8) are covered by the GPU smoke + Phase 1.

PHASE2_RUNGS = [
    # (base, refine_to, label, note)
    (5, 8, "r5b8", "~100k+"),
    (6, 9, "r6b9", "~1M-2M DOF, expected 95 GiB wall zone"),
    (7, 9, "r7b9", "PAST-WALL probe"),
]

PHASE2_NSTEPS = 10
PHASE2_NSTEPS_PROJ = 5
PHASE2_DT = 0.005
PHASE2_NU = 0.004
PHASE2_U_INF = 1.0


def run_phase2():
    """Thin-plate adaptive ladder: (base,refine) rungs from the Task 8b table."""
    print("\n" + "="*72, flush=True)
    print(" PHASE 2: thin-plate adaptive ladder (device assembly + cuDSS solve)")
    print("="*72 + "\n", flush=True)

    rows = []
    wall_rung = None
    past_wall_label = "r7b9"

    succeeded_rungs = []   # for projection secondary leg

    for base, refine_to, label, note in PHASE2_RUNGS:
        is_past_wall = (label == past_wall_label)
        if wall_rung is not None and not is_past_wall:
            continue   # stop at first failure; only past-wall runs after

        print(f"\n--- Phase 2 rung {label} (base={base}, refine_to={refine_to}) "
              f"[{note}] ---", flush=True)

        _reset_torch_peak()
        t_start = time.time()
        row = dict(phase=2, label=label, base=base, refine_to=refine_to,
                   note=note, assembly="device", solver="cudss")

        try:
            result = run_flow_past_3d(
                level=base, refine_to=refine_to,
                nsteps=PHASE2_NSTEPS, dt=PHASE2_DT,
                nu=PHASE2_NU, U_inf=PHASE2_U_INF,
                mono_solver="cudss", assembly="device",
                device="cuda:0", verbose=False)
            elapsed = time.time() - t_start
            cd_arr = result["cd"]
            cd_final = float(cd_arr[-1])
            cd_finite = bool(np.isfinite(cd_arr).all())
            n_excluded = result.get("n_excluded")
            smi_used, _ = _gpu_mem_mib()
            torch_peak = _torch_peak_mib()
            s_per_step = elapsed / PHASE2_NSTEPS

            row.update(dict(
                n_excluded=n_excluded,
                s_per_step=round(s_per_step, 2),
                cd_final=round(cd_final, 5) if np.isfinite(cd_final) else None,
                cd_finite=cd_finite,
                smi_used_mib=smi_used,
                torch_peak_mib=torch_peak,
                elapsed_s=round(elapsed, 1),
                status="OK" if cd_finite else "NAN-CD",
            ))
            print(f"    => Cd={cd_final:+.5f}  s/step={s_per_step:.2f}  "
                  f"n_excluded={n_excluded}  "
                  f"smi={smi_used}MiB  torch_peak={torch_peak}MiB  "
                  f"elapsed={elapsed:.1f}s  status={row['status']}", flush=True)
            if row["status"] == "OK":
                succeeded_rungs.append((base, refine_to, label))

        except Exception as exc:
            elapsed = time.time() - t_start
            exc_class = type(exc).__name__
            exc_msg = str(exc)[:200]
            print(f"    WALL at {label}: {exc_class}: {exc_msg}", flush=True)
            print(traceback.format_exc()[-400:], flush=True)
            row.update(dict(
                status="WALL", exc_class=exc_class, exc_msg=exc_msg,
                elapsed_s=round(elapsed, 1),
            ))
            if wall_rung is None:
                wall_rung = label
        finally:
            rows.append(row)

    # ---- projection secondary leg: 2 largest succeeded rungs ----
    proj_rungs = succeeded_rungs[-2:] if len(succeeded_rungs) >= 2 else succeeded_rungs
    if proj_rungs:
        print(f"\n--- Phase 2 projection secondary leg "
              f"({len(proj_rungs)} rung(s)) ---", flush=True)
        print("    NOTE: ~40% Cd magnitude gap vs monolithic is EXPECTED "
              "(documented 3-D projection defect).", flush=True)
        for base, refine_to, label in proj_rungs:
            print(f"\n    projection rung {label} (base={base}, "
                  f"refine_to={refine_to})", flush=True)
            _reset_torch_peak()
            t_start = time.time()
            prow = dict(phase="2-proj", label=label, base=base,
                        refine_to=refine_to, solver_pred="cudss",
                        ppe_solver="gpu_cg")
            try:
                pr = run_flow_past_3d_projection(
                    level=base, refine_to=refine_to,
                    nsteps=PHASE2_NSTEPS_PROJ, dt=PHASE2_DT,
                    nu=PHASE2_NU, U_inf=PHASE2_U_INF,
                    predictor_solver="cudss", ppe_solver="gpu_cg",
                    device="cuda:0", verbose=False)
                elapsed = time.time() - t_start
                cd_arr = pr["cd"]
                cd_final = float(cd_arr[-1])
                cd_finite = bool(np.isfinite(cd_arr).all())
                bounded = cd_finite and np.abs(cd_arr).max() < 1e6
                smi_used, _ = _gpu_mem_mib()
                torch_peak = _torch_peak_mib()
                s_per_step = elapsed / PHASE2_NSTEPS_PROJ
                prow.update(dict(
                    cd_final=round(cd_final, 5) if np.isfinite(cd_final) else None,
                    cd_finite=cd_finite, cd_bounded=bounded,
                    s_per_step=round(s_per_step, 2),
                    smi_used_mib=smi_used, torch_peak_mib=torch_peak,
                    elapsed_s=round(elapsed, 1),
                    status="OK" if bounded else "UNBOUNDED",
                ))
                print(f"    => Cd={cd_final:+.5f}  finite={cd_finite}  "
                      f"bounded={bounded}  s/step={s_per_step:.2f}  "
                      f"status={prow['status']}", flush=True)
            except Exception as exc:
                elapsed = time.time() - t_start
                exc_class = type(exc).__name__
                exc_msg = str(exc)[:200]
                print(f"    PROJ WALL {label}: {exc_class}: {exc_msg}", flush=True)
                print(traceback.format_exc()[-400:], flush=True)
                prow.update(dict(
                    status="WALL", exc_class=exc_class, exc_msg=exc_msg,
                    elapsed_s=round(elapsed, 1),
                ))
            rows.append(prow)

    return rows, wall_rung


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _fmt_row(r):
    ph = r.get("phase", "?")
    lb = r.get("label", "?")
    status = r.get("status", "?")
    cd = r.get("cd_final")
    cd_str = f"{cd:+.4f}" if cd is not None else "N/A"
    sps = r.get("s_per_step")
    sps_str = f"{sps:.2f}" if sps is not None else "N/A"
    mem = r.get("smi_used_after_mib") or r.get("smi_used_mib")
    mem_str = f"{mem}MiB" if mem else "N/A"
    tp = r.get("torch_peak_mib")
    tp_str = f"{tp}MiB" if tp else "N/A"
    el = r.get("elapsed_s")
    el_str = f"{el:.1f}s" if el else "?"
    exc = r.get("exc_class", "")
    exc_str = f" [{exc}]" if exc else ""
    # DOF + effective finest h (the uniform-vs-adaptive comparison columns)
    dof = r.get("saddle_dof")
    dof_str = f"{dof}" if dof else "N/A"
    hf = r.get("h_fine")
    hf_str = f"{hf:.6f}" if hf else (f"{1.0/2**r['level']:.6f}"
                                     if r.get("level") else "N/A")
    return (f"  Ph{ph}  {lb:8s}  status={status:12s}  dof={dof_str:9s}  "
            f"h_fine={hf_str:9s}  Cd={cd_str:10s}  s/step={sps_str:7s}  "
            f"smi={mem_str:12s}  torch_peak={tp_str:12s}  "
            f"elapsed={el_str}{exc_str}")


def _print_summary(p1_rows, p1_wall, p1b_rows, p1b_wall, p2_rows, p2_wall,
                   p2_skipped_reason=None):
    print("\n" + "#"*72, flush=True)
    print(" GH200 LADDER SUMMARY TABLE", flush=True)
    print("#"*72, flush=True)
    print("  PHASE 1: UNIFORM bluff-body cube-in-channel (HOST assembly + cuDSS solve)", flush=True)
    for r in p1_rows:
        print(_fmt_row(r), flush=True)
    if p1_wall:
        print(f"  PHASE 1 WALL: first hard failure at level {p1_wall}", flush=True)
    else:
        print("  PHASE 1 WALL: none (all rungs completed)", flush=True)

    print("", flush=True)
    print("  PHASE 1b: ADAPTIVE bluff-body (band-refined; same march, "
          "lower DOF per finest h)", flush=True)
    for r in p1b_rows:
        print(_fmt_row(r), flush=True)
    if p1b_wall:
        print(f"  PHASE 1b WALL: first hard failure at rung {p1b_wall}", flush=True)
    elif p1b_rows:
        print("  PHASE 1b WALL: none (all rungs completed)", flush=True)
    else:
        print("  PHASE 1b: NOT RUN", flush=True)

    print("", flush=True)
    if p2_rows:
        print("  PHASE 2: thin-plate adaptive (device assembly + cuDSS + proj)", flush=True)
        for r in p2_rows:
            print(_fmt_row(r), flush=True)
        if p2_wall:
            print(f"  PHASE 2 WALL: first hard failure at rung {p2_wall}", flush=True)
        else:
            print("  PHASE 2 WALL: none (all rungs completed)", flush=True)
    elif p2_skipped_reason:
        print(f"  PHASE 2: SKIPPED — {p2_skipped_reason}", flush=True)
    else:
        print("  PHASE 2: NOT RUN", flush=True)

    print("", flush=True)
    print(_OSUB_DECISION, flush=True)
    print("", flush=True)
    print("GH200-LADDER-OK", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Skip Phase 2 if this much wall-clock has elapsed (sbatch limit is 4h; leave
# headroom so the summary table + sentinel always print before the kill).
PHASE2_TIME_BUDGET_S = 3.0 * 3600


def main():
    t_total = time.time()

    # Phase 1: uniform bluff-body ladder (+ L7 past-wall probe)
    p1_rows, p1_wall = run_phase1()

    # Phase 1b: adaptive band-refined bluff-body ladder (outranks Phase 2)
    p1b_rows, p1b_wall = run_phase1b()

    # Phase 2: thin-plate adaptive (secondary) — only if time remains
    elapsed = time.time() - t_total
    p2_skipped_reason = None
    if elapsed > PHASE2_TIME_BUDGET_S:
        p2_rows, p2_wall = [], None
        p2_skipped_reason = (f"time budget: {elapsed/3600:.2f}h elapsed > "
                             f"{PHASE2_TIME_BUDGET_S/3600:.1f}h cutoff "
                             f"(Phase 1b outranks Phase 2)")
        print(f"\n[ladder] SKIPPING Phase 2 — {p2_skipped_reason}", flush=True)
    else:
        p2_rows, p2_wall = run_phase2()

    _print_summary(p1_rows, p1_wall, p1b_rows, p1b_wall, p2_rows, p2_wall,
                   p2_skipped_reason=p2_skipped_reason)
    print(f"\n[ladder] total elapsed: {(time.time()-t_total)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
