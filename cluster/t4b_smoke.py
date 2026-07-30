"""T4b re-run smoke: 50-step GH200 truck smoke with the T4b fixes.

Changes vs T4's leg2_smoke.py:
  * UNIT MAPPING (fix): dt_unit = dt_phys * scale (=0.01/16=6.25e-4);
    make_nu_schedule(scale=cfg.domain_scale) => nu_unit=s/Re, ramp in unit time.
  * EQUILIBRATION: saddle_equilibrate=True (breaks the scalar-Jacobi floor).
  * TOL SCHEDULE: loose 5e-4 during the Re ramp (t_unit < ramp_end), tight 1e-6
    post-ramp.
  * cd_surr ARITHMETIC: prints raw surrogate + reaction forces + ref_force.

Timeout: 7200s. Solver: fgmres_bdiag + SADDLE_DEVICE_CSR=1 + warm-start.
"""
import os, sys, time, resource, subprocess, threading, json, pathlib

os.environ.setdefault("SADDLE_DEVICE_CSR", "1")
os.environ.setdefault("DIFFSIM_ASM_PROFILE", "1")

sys.path.insert(0, "/work/mech-ai/baskarg/DiffSim/tests")
sys.path.insert(0, "/work/mech-ai/baskarg/DiffSim/src")

CONF = "/work/mech-ai/baskarg/DiffSim/local_code_old/truck_4case_fresh_inputs/NewRun-no-shell-slope0p25/config.txt"
BASE_LEVEL = int(os.environ.get("BASE_LEVEL", "7"))
BAND_TO    = int(os.environ.get("BAND_TO",    "12"))
NSTEPS     = int(os.environ.get("NSTEPS",     "50"))
EQUIL      = os.environ.get("SADDLE_EQUILIBRATE", "1") == "1"
VIZ_DIR    = os.environ.get("VIZ_DIR", "/work/mech-ai/baskarg/DiffSim/results/truck-t4b-frames")

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
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                timeout=5).decode().strip()
            mem = int(out.split('\n')[0].strip())
            if mem > _smi_peak[0]:
                _smi_peak[0] = mem
        except Exception:
            pass
        time.sleep(5)
threading.Thread(target=_smi_sampler, daemon=True).start()

print("=" * 70, flush=True)
print(f"[T4b] 50-step smoke  base={BASE_LEVEL}  band={BAND_TO}  nsteps={NSTEPS}  "
      f"equilibrate={EQUIL}", flush=True)

import torch, warp as wp
wp.init()
print(f"[T4b] torch {torch.__version__}  cuda={torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"[T4b] GPU: {torch.cuda.get_device_name(0)}", flush=True)

from diffsim.cases.truck_config import load_truck_config
cfg = load_truck_config(CONF)
scale = cfg.domain_scale                  # 1/16
dt_phys = float(cfg.dt_v[1])              # 0.01 (physical)
dt = dt_phys * scale                      # UNIT time: 6.25e-4
ramp_end_phys = float(cfg.re_ramping[-1]) # 51 physical s
ramp_end_unit = ramp_end_phys * scale     # 3.1875 unit
print(f"[T4b] scale={scale}  dt_phys={dt_phys}  dt_unit={dt}  "
      f"ramp_end_unit={ramp_end_unit}", flush=True)

from truck_flow import run_truck, make_nu_schedule
nu_sched = make_nu_schedule(cfg, U_inf=1.0, L_ref=1.0, scale=scale)
print(f"[T4b] nu_unit(0)={nu_sched(0.0):.3e} -> effRe={scale/nu_sched(0.0):.0f}  "
      f"(unit-frame Re with L=1)", flush=True)

# tolerance schedule: loose during ramp, tight after (unit time)
def tol_sched(t_unit):
    return 5e-4 if t_unit < ramp_end_unit else 1e-6

_step_times = []
_last_step_t = [time.time()]
_rows = []
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
    _rows.append(dict(step=step, cd=cd, cd_surr=cds, F_surr_raw=fs,
                      F_react_raw=fr, ref_force=rf, t_step=dt_step))
    # cd_surr arithmetic (item B): raw forces + shared ref_force
    print(f"[T4b] step {step:3d}  cd_react={cd:+.4f}  cd_surr={cds:+.4f}  "
          f"| F_react_raw={fr:+.4e}  F_surr_raw={fs:+.4e}  ref_force={rf:.4e}  "
          f"t_step={dt_step:.1f}s", flush=True)
    if step >= 10 and (abs(cd) > 1e3):
        print(f"[T4b] EARLY EXIT at step {step}: |cd_react|>1e3", flush=True)
        sys.exit(42)

t_run = time.time()
res = run_truck(
    cfg, NSTEPS,
    device="cuda",
    mono_solver="fgmres_bdiag",
    saddle_x0="extrap",
    saddle_equilibrate=EQUIL,
    nu_schedule=nu_sched,
    nu=None,
    on_step=on_step,
    verbose=True,
    base_level=BASE_LEVEL,
    dt=dt,
    truck_band_to=BAND_TO,
    band_cells=3,
    region_refine=True,
    linsolve_tol=5e-4,
    linsolve_tol_schedule=tol_sched,
    viz_interval=10,
    viz_dir=VIZ_DIR,
    viz_checkpoint_interval=25,
    viz_Q_thresh=0.5,
)
t_total = time.time() - t_run

print("=" * 70, flush=True)
print(f"[T4b] SMOKE RESULTS  equilibrate={EQUIL}", flush=True)
print(f"  bounded       : {'YES' if all(abs(x) < 1e3 for x in res['cd']) else 'NO'}", flush=True)
print(f"  cells         : {res['n_cells']}", flush=True)
print(f"  cd_react[:]   : {list(res['cd'])}", flush=True)
print(f"  cd_surr[:]    : {list(res['cd_surr'])}", flush=True)
print(f"  t_total       : {t_total:.1f}s", flush=True)
if _step_times:
    print(f"  s/step (avg)  : {sum(_step_times)/len(_step_times):.1f}s", flush=True)
    print(f"  s/step (min)  : {min(_step_times):.1f}s", flush=True)
print(f"  RSS peak      : {_rss_peak[0]/1024:.0f} MB", flush=True)
print(f"  GPU SMI peak  : {_smi_peak[0]} MiB", flush=True)

out_dir = pathlib.Path(VIZ_DIR)
out_dir.mkdir(parents=True, exist_ok=True)
summary = dict(
    base_level=BASE_LEVEL, band_to=BAND_TO, nsteps=NSTEPS,
    equilibrate=EQUIL, scale=scale, dt_unit=dt, ramp_end_unit=ramp_end_unit,
    n_cells=int(res["n_cells"]),
    cd_react=list(float(x) for x in res["cd"]),
    cd_surr=list(float(x) for x in res["cd_surr"]),
    rows=_rows, t_total=t_total, step_times=_step_times,
    rss_peak_mb=_rss_peak[0]/1024, gpu_smi_peak_mib=_smi_peak[0],
    bounded=all(abs(x) < 1e3 for x in res["cd"]),
)
with open(out_dir / "t4b_smoke_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"  summary -> {out_dir / 't4b_smoke_summary.json'}", flush=True)
print("=" * 70, flush=True)
