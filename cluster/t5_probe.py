import os,sys,time
os.environ["SADDLE_DEVICE_CSR"]="1"
sys.path.insert(0,"/work/mech-ai/baskarg/DiffSim/src")
import torch,warp as wp; wp.init()
print("cuda",torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
from diffsim.cases.truck_config import load_truck_config
from diffsim.cases.truck import run_truck, make_nu_schedule
CONF="/work/mech-ai/baskarg/DiffSim/local_code_old/truck_4case_fresh_inputs/NewRun-no-shell-slope0p25/config.txt"
cfg=load_truck_config(CONF); scale=cfg.domain_scale; dt=0.01*scale
nu_sched=make_nu_schedule(cfg,scale=scale)
t0=time.time()
def on_step(s,i):
    print(f"  step{s} cd={i['cd']:+.5f} cd_surr={i['cd_surr']:+.1f} t={time.time()-t0:.1f}",flush=True)
res=run_truck(cfg,3,device="cuda",assembly="device",mono_solver="fgmres_bdiag",
    saddle_x0="extrap",nu_schedule=nu_sched,nu=None,on_step=on_step,verbose=True,
    base_level=6,dt=dt,truck_band_to=8,band_cells=3,region_refine=True,
    linsolve_tol=5e-4)
print("GPU-DEVICE-PATH OK cells=",res["n_cells"],"cd=",list(res["cd"]),flush=True)
