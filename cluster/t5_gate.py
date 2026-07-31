"""T5 GATE LEG — the truck march through the ramp toward truck-arrival.

The campaign's biggest moment: corrected units + plain bdiag + loose-ramp tol
schedule + DEVICE_CSR handoff (P1) + warm-start (P2) + the T-B4 static-set-empty
finding (incremental assembly STOPPED, full fill retained).  NSTEPS=800 marches
the flow front from the inlet to the truck (~step 500 at dt_unit=6.25e-4, truck
front x~0.3125) and ~300 steps past it — the moment of first contact where
cd_react goes from an exact 0 (pre-arrival) to nonzero physics.

Unit-table step counts (dt_unit = dt_phys*scale = 0.01/16 = 6.25e-4):
  truck arrival : x_front=0.3125 / U=1 -> t_unit=0.3125 -> step ~500
  Re ramp mid   : t_phys=50 -> t_unit=3.125   -> step 5000  (NOT reached at 800)
  Re ramp end   : t_phys=51 -> t_unit=3.1875  -> step 5100  (NOT reached at 800)
So Re stays ~1000 for the whole 800-step leg; the tol schedule stays LOOSE
(5e-4) throughout (arrival happens deep inside the ramp).

Records per step: s/step, iters/step (warm-start effect), RSS, cd_react
(0 -> NONZERO at arrival), cd_surr (startup pathology decay), boundedness.
Viz: Q-iso / centerline / surface-Cp frames every 20 steps + .vtu every 200.

Env: SADDLE_DEVICE_CSR=1, assembly=device, fgmres_bdiag, saddle_x0=extrap,
     equilibrate OFF (T4b: plain Jacobi is ~200x tighter on the mass-dominated
     corrected saddle).  timeout 21600 (6 h).
"""
import os, sys, time, resource, subprocess, threading, json, pathlib

os.environ.setdefault("SADDLE_DEVICE_CSR", "1")
os.environ.setdefault("DIFFSIM_ASM_PROFILE", "1")

sys.path.insert(0, "/work/mech-ai/baskarg/DiffSim/tests")
sys.path.insert(0, "/work/mech-ai/baskarg/DiffSim/src")

CONF = ("/work/mech-ai/baskarg/DiffSim/local_code_old/truck_4case_fresh_inputs/"
        "NewRun-no-shell-slope0p25/config.txt")
BASE_LEVEL = int(os.environ.get("BASE_LEVEL", "7"))
BAND_TO    = int(os.environ.get("BAND_TO",    "12"))
NSTEPS     = int(os.environ.get("NSTEPS",     "800"))
VIZ_INT    = int(os.environ.get("VIZ_INTERVAL", "20"))
CKPT_INT   = int(os.environ.get("VIZ_CKPT_INTERVAL", "200"))
EQUIL      = os.environ.get("SADDLE_EQUILIBRATE", "0") == "1"
# assembly path: "device" (P1 device-CSR handoff, needs a warp install with
# native/ headers so the NS element kernel can cold-compile) or "host" (scipy
# assembly; no new kernel compile).  Default device; set TRUCK_ASSEMBLY=host to
# fall back on the ARM venv where warp lacks native/ headers (JIT of a
# module="unique" kernel raises "cannot open source file builtin.h").
ASSEMBLY   = os.environ.get("TRUCK_ASSEMBLY", "device")
VIZ_DIR    = os.environ.get(
    "VIZ_DIR", "/work/mech-ai/baskarg/DiffSim/results/truck-t5-frames")

_rss_peak = [0]
def _rss_sampler():
    while True:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if rss > _rss_peak[0]:
            _rss_peak[0] = rss
        time.sleep(2)
threading.Thread(target=_rss_sampler, daemon=True).start()

_smi_peak = [0]
def _smi_sampler():
    while True:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"], timeout=5).decode().strip()
            mem = int(out.split('\n')[0].strip())
            if mem > _smi_peak[0]:
                _smi_peak[0] = mem
        except Exception:
            pass
        time.sleep(5)
threading.Thread(target=_smi_sampler, daemon=True).start()

print("=" * 72, flush=True)
print(f"[T5] GATE LEG  base={BASE_LEVEL} band={BAND_TO} nsteps={NSTEPS} "
      f"viz_interval={VIZ_INT} ckpt={CKPT_INT} equilibrate={EQUIL} "
      f"assembly={ASSEMBLY}", flush=True)

import torch, warp as wp
wp.init()
print(f"[T5] torch {torch.__version__}  cuda={torch.cuda.is_available()}",
      flush=True)
if torch.cuda.is_available():
    print(f"[T5] GPU: {torch.cuda.get_device_name(0)}", flush=True)

from diffsim.cases.truck_config import load_truck_config
cfg = load_truck_config(CONF)
scale = cfg.domain_scale                   # 1/16
dt_phys = float(cfg.dt_v[1])               # 0.01 physical
dt = dt_phys * scale                       # 6.25e-4 unit
ramp_end_phys = float(cfg.re_ramping[-1])  # 51 physical s
ramp_end_unit = ramp_end_phys * scale      # 3.1875 unit
arrival_step = int(round((0.3125) / dt))   # front to truck (~500)
print(f"[T5] scale={scale} dt_unit={dt} ramp_end_unit={ramp_end_unit} "
      f"-> ramp_end_step={ramp_end_unit/dt:.0f}  arrival~step{arrival_step}",
      flush=True)

from truck_flow import run_truck, make_nu_schedule
nu_sched = make_nu_schedule(cfg, U_inf=1.0, L_ref=1.0, scale=scale)
# C++-faithful low-Re startup (ReSolverRampInitial=10 heritage): env RE_START>0
# overrides the schedule's early phase — Re ramps RE_START -> config Re over
# RE_RAMP_STEPS steps (linear in step count), then hands off to the config
# schedule. Zero solver changes; pure nu(t) shaping.
_re_start = float(os.environ.get("RE_START", "0") or 0)
_re_ramp_steps = int(os.environ.get("RE_RAMP_STEPS", "250"))
if _re_start > 0:
    _cfg_sched = nu_sched
    _dtu = dt
    def nu_sched(t_unit, _s=_cfg_sched, _r0=_re_start,
                 _n=_re_ramp_steps, _dt=_dtu, _scale=scale):
        step = t_unit / _dt
        nu_cfg = _s(t_unit)                      # config Re at this time
        re_cfg = _scale / nu_cfg                 # invert: nu_unit = scale/Re
        frac = min(1.0, step / max(_n, 1))
        re_now = _r0 + frac * (re_cfg - _r0)
        return _scale / re_now
    print(f"[T5] RE_START={_re_start} ramp over {_re_ramp_steps} steps -> config Re", flush=True)
print(f"[T5] nu_unit(0)={nu_sched(0.0):.3e} effRe={scale/nu_sched(0.0):.0f}",
      flush=True)

# T5 leg-2 knob: the leg-1 march died at step 25 (post-arrival transient) with
# relres=8.87e-4 vs the 5e-4 ramp tol — only 1.8x above.  TOL_RAMP loosens the
# in-ramp tol (e.g. 1e-3) so the march survives the impulsive-arrival transient;
# post-ramp stays tight (1e-6).
TOL_RAMP = float(os.environ.get("TOL_RAMP", "5e-4"))

# TU5R SOFT-START LEG: the corrected physics is that the startup spike is the
# incompressible (elliptic) pressure response to an IMPULSIVE inlet, NOT a
# front-arrival event.  A short inlet amplitude ramp lets that pressure
# response develop smoothly.  SOFT_START = ramp length in dt-units (default 30;
# set 0 to disable).  Combined with a STEP-BASED tol schedule: loose through
# the startup transient, tight after step TOL_TIGHTEN_STEP (~200).
SOFT_START = float(os.environ.get("SOFT_START", "30"))
TOL_LOOSE = float(os.environ.get("TOL_LOOSE", "5e-4"))
TOL_TIGHT = float(os.environ.get("TOL_TIGHT", "1e-6"))
TOL_TIGHTEN_STEP = int(os.environ.get("TOL_TIGHTEN_STEP", "200"))
# tighten-step -> unit time boundary (t_new = (step+1)*dt is passed in)
_tol_tighten_t = (TOL_TIGHTEN_STEP + 1) * dt

def tol_sched(t_unit):
    # loose through startup (step < TOL_TIGHTEN_STEP), tight after.  The Re ramp
    # is far longer than the leg, so this is the operative schedule.
    return TOL_LOOSE if t_unit < _tol_tighten_t else TOL_TIGHT

print(f"[T5] soft_start={SOFT_START} dt-units  tol_loose={TOL_LOOSE} -> "
      f"tol_tight={TOL_TIGHT} at step>{TOL_TIGHTEN_STEP}", flush=True)

# DT-SEGMENTED STARTUP (the C++ dt_V startup ladder, r120b escalation): the
# clean r120b leg proved the developing-flow solve exceeds scalar-Jacobi at
# full dt (24000 inner iters, relres 7.6e-4 at step 12).  T4b forensics: what
# makes Jacobi work is sigma-dominance (sigma = b0/dt).  DT_START_FACTOR>0
# runs the first DT_START_STEPS steps at dt/DT_START_FACTOR (sigma x FACTOR),
# then returns to full dt via the variable-step BDF2 table.  All t-based
# schedules (nu ramp, tol, soft_start) stay physical-time consistent because
# run_truck accumulates t_new under a dt_schedule.
DT_START_FACTOR = float(os.environ.get("DT_START_FACTOR", "0") or 0)
DT_START_STEPS = int(os.environ.get("DT_START_STEPS", "0") or 0)
dt_sched = None
if DT_START_FACTOR > 0 and DT_START_STEPS > 0:
    def dt_sched(step, _dt=dt, _f=DT_START_FACTOR, _n=DT_START_STEPS):
        return _dt / _f if step < _n else _dt
    print(f"[T5] DT ladder: dt/{DT_START_FACTOR} for first {DT_START_STEPS} "
          f"steps (sigma x{DT_START_FACTOR}), then dt={dt}", flush=True)

# PCD PER-STEP FALLBACK (five-leg Jacobi-class verdict): bdiag stays the fast
# primary; any step that exhausts its budget is re-solved with the Track-A
# fgmres_pcd (Cahouet-Chabard Schur, AMGX inners per A4/A2).  SADDLE_FALLBACK=
# pcd enables; PCD_F_INNER / PCD_AP_INNER pick the inner backends.
SADDLE_FALLBACK = os.environ.get("SADDLE_FALLBACK", "") or None
PCD_F_INNER = os.environ.get("PCD_F_INNER", "amgx")
PCD_AP_INNER = os.environ.get("PCD_AP_INNER", "amgx")
if SADDLE_FALLBACK:
    print(f"[T5] saddle_fallback={SADDLE_FALLBACK} "
          f"(F={PCD_F_INNER}, Ap={PCD_AP_INNER})", flush=True)

# TAU_DT (Baskar directive): decouple tau_m's transient term from the marching
# dt so a dt-ladder startup doesn't collapse the pressure stabilization.
#   unset/"" -> None (tau follows marching dt, as today)
#   "base"   -> freeze tau at the BASE dt (exact once ladder reaches full dt)
#   "0"      -> drop the transient term (steady tau)
#   <float>  -> freeze tau at that dt_unit
_tau_env = os.environ.get("TAU_DT", "").strip()
TAU_DT = (None if _tau_env == "" else
          dt if _tau_env == "base" else float(_tau_env))
if TAU_DT is not None:
    print(f"[T5] TAU_DT={TAU_DT} (tau_m transient frozen; marching dt free)",
          flush=True)

# ACCEPT-MISS window (Baskar directive): solver misses during the initial
# transient are logged and accepted (truncated iterate marches on); strict
# semantics return at this step.  Blow-up sentinel always active.
ACCEPT_MISS_UNTIL = int(os.environ.get("ACCEPT_MISS_UNTIL", "0") or 0)
U_CAP = float(os.environ.get("U_CAP", "50"))
if ACCEPT_MISS_UNTIL > 0:
    print(f"[T5] accept-miss window: steps < {ACCEPT_MISS_UNTIL} "
          f"(u_cap={U_CAP})", flush=True)

# STAGED BC (Baskar): strong no-slip on the carved-out boundary through the
# transient; SBM (true-geometry Nitsche) from SBM_START_STEP on.  Put the
# switch inside the accept-miss window.
SBM_START_STEP = int(os.environ.get("SBM_START_STEP", "0") or 0)
if SBM_START_STEP > 0:
    print(f"[T5] staged BC: strong carved-out no-slip until step "
          f"{SBM_START_STEP}, then SBM", flush=True)

# tauM_scale (C++ heritage, config-faithful): default = the case config's
# value (0.1 for the truck); TAU_M_SCALE env overrides (set 1.0 to reproduce
# the pre-fix marches).
TAU_M_SCALE = float(os.environ.get("TAU_M_SCALE", "") or cfg.tau_m_scale)
print(f"[T5] tau_m_scale={TAU_M_SCALE} (config {cfg.tau_m_scale})", flush=True)

from diffsim.solvers import linsolve
_step_times = []
_last_step_t = [time.time()]
_rows = []
_arrival = [None]
def on_step(step, info):
    now = time.time()
    dt_step = now - _last_step_t[0]
    _step_times.append(dt_step)
    _last_step_t[0] = now
    cd = info.get("cd", float("nan"))
    cds = info.get("cd_surr", float("nan"))
    fs = info.get("F_surr_raw", float("nan"))
    fr = info.get("F_react_raw", float("nan"))
    rf = info.get("ref_force", float("nan"))
    iters = linsolve._LAST_ITERS[0]
    if _arrival[0] is None and abs(cd) > 1e-6:
        _arrival[0] = step
        print(f"[T5] *** FIRST CONTACT at step {step}: cd_react={cd:+.5f} ***",
              flush=True)
    _rows.append(dict(step=step, cd=cd, cd_surr=cds, F_surr_raw=fs,
                      F_react_raw=fr, ref_force=rf, iters=iters,
                      t_step=dt_step))
    _um = info.get("umax")
    _ul = info.get("umax_loc")
    _um_s = (f" umax={_um:.3f}@({_ul[0]:.4f},{_ul[1]:.4f},{_ul[2]:.4f})"
             if _um is not None and _ul is not None else
             (f" umax={_um:.3f}" if _um is not None else ""))
    print(f"[T5] step {step:4d} cd_react={cd:+.5f} cd_surr={cds:+.2f} "
          f"iters={iters} t={dt_step:.1f}s rss={_rss_peak[0]/1024/1024:.1f}G "
          f"smi={_smi_peak[0]}MiB{_um_s}", flush=True)

t_run = time.time()
res = run_truck(
    cfg, NSTEPS,
    device="cuda",
    assembly=ASSEMBLY,
    mono_solver="fgmres_bdiag",
    saddle_x0="extrap",
    saddle_equilibrate=EQUIL,
    nu_schedule=nu_sched,
    saddle_restart=(int(os.environ["SADDLE_RESTART"]) if os.environ.get("SADDLE_RESTART") else None),
    nu=None,
    on_step=on_step,
    verbose=False,
    base_level=BASE_LEVEL,
    dt=dt,
    truck_band_to=BAND_TO,
    band_cells=3,
    region_refine=True,
    linsolve_tol=5e-4,
    linsolve_tol_schedule=tol_sched,
    viz_interval=VIZ_INT,
    viz_dir=VIZ_DIR,
    viz_checkpoint_interval=CKPT_INT,
    viz_Q_thresh=0.5,
    soft_start=(SOFT_START if SOFT_START > 0 else None),
    dt_schedule=dt_sched,
    saddle_fallback=SADDLE_FALLBACK,
    pcd_f_inner=PCD_F_INNER,
    pcd_ap_inner=PCD_AP_INNER,
    tau_dt=TAU_DT,
    accept_miss_until=(ACCEPT_MISS_UNTIL if ACCEPT_MISS_UNTIL > 0 else None),
    u_cap=U_CAP,
    sbm_start_step=(SBM_START_STEP if SBM_START_STEP > 0 else None),
    tau_m_scale=TAU_M_SCALE,
)
t_total = time.time() - t_run

print("=" * 72, flush=True)
print(f"[T5] GATE RESULTS", flush=True)
cd = list(res["cd"])
print(f"  bounded     : {'YES' if all(abs(x) < 1e3 for x in cd) else 'NO'}",
      flush=True)
print(f"  cells       : {res['n_cells']}", flush=True)
print(f"  steps done  : {len(_rows)}", flush=True)
print(f"  arrival step: {_arrival[0]}", flush=True)
if _arrival[0] is not None:
    print(f"  cd_react @arrival+: "
          f"{[round(r['cd'],5) for r in _rows if r['step']>=_arrival[0]][:10]}",
          flush=True)
print(f"  t_total     : {t_total:.1f}s", flush=True)
if _step_times:
    print(f"  s/step avg  : {sum(_step_times)/len(_step_times):.1f}s  "
          f"min {min(_step_times):.1f}s", flush=True)
print(f"  RSS peak    : {_rss_peak[0]/1024/1024:.1f} GB", flush=True)
print(f"  GPU SMI peak: {_smi_peak[0]} MiB", flush=True)

out_dir = pathlib.Path(VIZ_DIR)
out_dir.mkdir(parents=True, exist_ok=True)
summary = dict(
    base_level=BASE_LEVEL, band_to=BAND_TO, nsteps=NSTEPS,
    viz_interval=VIZ_INT, ckpt_interval=CKPT_INT, equilibrate=EQUIL,
    scale=scale, dt_unit=dt, ramp_end_unit=ramp_end_unit,
    arrival_step=_arrival[0], n_cells=int(res["n_cells"]),
    steps_done=len(_rows),
    cd_react=[float(x) for x in cd],
    cd_surr=[float(x) for x in res["cd_surr"]],
    rows=_rows, t_total=t_total, step_times=_step_times,
    rss_peak_gb=_rss_peak[0] / 1024 / 1024, gpu_smi_peak_mib=_smi_peak[0],
    bounded=all(abs(x) < 1e3 for x in cd),
)
with open(out_dir / "t5_gate_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
# frame directory listing for the Mac fetch
frame_files = sorted(str(p) for p in out_dir.rglob("*.vtp")) \
    + sorted(str(p) for p in out_dir.rglob("*.vtu"))
print(f"  frames written: {len(frame_files)}", flush=True)
print(f"  summary -> {out_dir / 't5_gate_summary.json'}", flush=True)
print("=" * 72, flush=True)
