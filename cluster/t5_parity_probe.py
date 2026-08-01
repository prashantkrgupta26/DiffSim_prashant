"""TU5R miscompile sentinel: CPU(host) vs GPU(device) parity on the truck tiny
case, against the CLEAN warp reinstall.

If the graft-free reinstalled warp kernels silently miscompiled, the device
force trajectory would diverge from the CPU reference.  This runs the SAME tiny
truck case on both paths and asserts the reaction/surrogate drag match tightly.
Small, fast, fully logged (via the t5_hdr.sh-style wrapper that launches it)."""
import os, sys, time
os.environ["SADDLE_DEVICE_CSR"] = "1"
sys.path.insert(0, "/work/mech-ai/baskarg/DiffSim/src")
import numpy as np
import torch, warp as wp
wp.init()
print(f"[parity] WARP {wp.__version__} cuda={torch.cuda.is_available()} "
      f"cache={os.environ.get('WARP_CACHE_PATH')}", flush=True)

from diffsim.cases.truck_config import load_truck_config
from diffsim.geometry.merged_trimesh import MergedTriMesh, _read_stl
from diffsim.cases.truck import run_truck

CONF = ("/work/mech-ai/baskarg/DiffSim/local_code_old/truck_4case_fresh_inputs/"
        "NewRun-no-shell-slope0p25/config.txt")
cfg = load_truck_config(CONF)

# tiny one-tire body (same construction as the CPU gate test)
v, t = _read_stl(os.path.join(cfg.config_dir, "tire_1.stl"))
v = v - v.mean(0); v = v / np.abs(v).max()
v = v * 0.05 + np.asarray((0.35, 0.0625, 0.0625), np.float64)
merged = MergedTriMesh(v, t)

common = dict(nsteps=3, base_level=5, truck_band_to=6, band_cells=2,
              merged=merged, region_refine=False, nu=1.0 / 50.0, dt=0.02,
              mono_solver="splu", saddle_x0=None, verbose=False)

t0 = time.time()
res_h = run_truck(cfg, device="cpu", assembly="host", **common)
print(f"[parity] host  done t={time.time()-t0:.1f}s cd={list(res_h['cd'])}",
      flush=True)
t0 = time.time()
res_d = run_truck(cfg, device="cuda", assembly="device", **common)
print(f"[parity] device done t={time.time()-t0:.1f}s cd={list(res_d['cd'])}",
      flush=True)

cd_h, cd_d = np.asarray(res_h["cd"]), np.asarray(res_d["cd"])
cds_h, cds_d = np.asarray(res_h["cd_surr"]), np.asarray(res_d["cd_surr"])
dr = float(np.max(np.abs(cd_h - cd_d)))
ds = float(np.max(np.abs(cds_h - cds_d)))
print(f"[parity] max|cd_react_h - cd_react_d| = {dr:.3e}", flush=True)
print(f"[parity] max|cd_surr_h  - cd_surr_d | = {ds:.3e}", flush=True)
TOL = 1e-6
ok = dr < TOL and ds < TOL
print(f"[parity] {'PASS' if ok else 'FAIL'} (tol {TOL:.0e}) -- "
      f"miscompile sentinel {'clear' if ok else 'TRIPPED'}", flush=True)
sys.exit(0 if ok else 1)
