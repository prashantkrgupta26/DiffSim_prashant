"""AHMED GATE LEG — Rung 2 of the solver-escalation campaign.

Clean-geometry full-amplitude march: Ahmed body (no stilts, resolvable
clearance) in the unit channel, PCD-primary by default.  Pass bar (spec
§6): >=500 consecutive steps at amplitude 1.0 with zero misses
post-transient, umax bounded (< U_CAP/10), finite cd_react.
"""
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, "/work/mech-ai/baskarg/DiffSim/src")

CONF = ("/work/mech-ai/baskarg/DiffSim/local_code_old/"
        "truck_4case_fresh_inputs/NewRun-no-shell-slope0p25/config.txt")

BASE_LEVEL = int(os.environ.get("BASE_LEVEL", "7"))
BAND_TO = int(os.environ.get("BAND_TO", "11"))
NSTEPS = int(os.environ.get("NSTEPS", "900"))
DT = float(os.environ.get("DT", "6.25e-4"))
RE_TARGET = float(os.environ.get("RE_TARGET", "250"))
SOFT_START = float(os.environ.get("SOFT_START", "300"))
U_CAP = float(os.environ.get("U_CAP", "5000"))
MONO = os.environ.get("MONO_SOLVER", "fgmres_pcd")
F_INNER = os.environ.get("PCD_F_INNER", "amgx")
AP_INNER = os.environ.get("PCD_AP_INNER", "amgx")
TOL = float(os.environ.get("TOL", "5e-4"))
A_LEN = float(os.environ.get("AHMED_LENGTH", "0.06"))
A_CLR = float(os.environ.get("AHMED_CLEARANCE", "0.003"))
A_XF = float(os.environ.get("AHMED_XFRONT", "0.32"))
ASSEMBLY = os.environ.get("TRUCK_ASSEMBLY", "device")
CKPT_DIR = os.environ.get("MARCH_CKPT", "") or None
RESUME = os.environ.get("RESUME", "0") == "1"
OUT = pathlib.Path(os.environ.get(
    "OUT_DIR", "/work/mech-ai/baskarg/DiffSim/results/ahmed-r2"))
OUT.mkdir(parents=True, exist_ok=True)

from diffsim.cases.truck_config import load_truck_config
from diffsim.cases.truck import run_truck
from diffsim.cases.ahmed import ahmed_merged
from diffsim.solvers import linsolve

cfg = load_truck_config(CONF)
merged = ahmed_merged(length=A_LEN, clearance=A_CLR, x_front=A_XF,
                      band_level=BAND_TO)
# nu in unit-cube coords: nu_unit = domain_scale / Re  (mirror T5/truck physics)
nu_unit = cfg.domain_scale / RE_TARGET

print("=" * 72, flush=True)
print(f"[AHMED] base={BASE_LEVEL} band={BAND_TO} nsteps={NSTEPS} "
      f"Re={RE_TARGET} nu_unit={nu_unit:.3e} mono={MONO} "
      f"F={F_INNER} Ap={AP_INNER} soft_start={SOFT_START} "
      f"L={A_LEN} clr={A_CLR}", flush=True)

_rows = []
_miss_post = [0]


def on_step(step, info):
    miss = bool(info.get("accepted_miss", False))
    if step >= SOFT_START and miss:
        _miss_post[0] += 1
    _rows.append(dict(step=step, cd=info.get("cd"),
                      umax=info.get("umax"), miss=miss,
                      iters=linsolve._LAST_ITERS[0]))
    if step % 10 == 0:
        r = _rows[-1]
        print(f"[AHMED] step {step:4d} cd={r['cd']:+.5f} "
              f"umax={r['umax']:.2f} iters={r['iters']} miss={miss}",
              flush=True)
        with (OUT / "rows.jsonl").open("a") as _f:
            _f.write("\n".join(json.dumps(q) for q in _rows[-10:]) + "\n")


t0 = time.time()
res = run_truck(cfg, NSTEPS,
                device="cuda",
                assembly=ASSEMBLY,
                base_level=BASE_LEVEL,
                truck_band_to=BAND_TO,
                band_cells=2,
                merged=merged,
                region_refine=False,
                nu=nu_unit,
                dt=DT,
                mono_solver=MONO,
                pcd_f_inner=F_INNER,
                pcd_ap_inner=AP_INNER,
                linsolve_tol=TOL,
                soft_start=(SOFT_START if SOFT_START > 0 else None),
                u_cap=U_CAP,
                on_step=on_step,
                checkpoint_interval=(200 if CKPT_DIR else None),
                checkpoint_dir=CKPT_DIR,
                resume=RESUME,
                verbose=False)
post = [r for r in _rows if r["step"] >= SOFT_START]
_umax_final = post[-1]["umax"] if post else float("nan")
print(f"[AHMED] DONE {time.time()-t0:.0f}s steps={len(_rows)} "
      f"post-transient={len(post)} misses_post={_miss_post[0]} "
      f"umax_final={_umax_final}",
      flush=True)
_steps_ok = len(post) >= 500
_miss_ok = _miss_post[0] == 0
_umax_ok = _umax_final < U_CAP / 10.0
print(f"[AHMED] PASS-BAR: steps_at_amp1={len(post)}>=500? {_steps_ok} "
      f"misses=0? {_miss_ok} "
      f"umax<U_CAP/10? {_umax_ok}",
      flush=True)
