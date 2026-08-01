"""Dump the FINAL SEALED truck mesh (q-leg recipe) for the Rung-0a
geometry QA gate (solver-escalation spec §4).

Recipe recovered from the q-leg launch log (t5-faired6-nohup.out):
  base 7, band 12, consolidated Truck.stl, CARVE_DELTA=1.0,
  WALLS_BANDS 9:0.0156 + 8:0.05, GROUND_LVL=9 (band 0.0156),
  SEAL_UNDERBODY y<0.006, SEAL_BOXES [0.33,0.35]x[0,0.017]x[0.057,0.068].
Expected (q-leg): 2,648,772 cells single fluid domain, 36 pockets dropped.

Runs the REAL production mesh path (build_truck_mesh), so every automated
check of spec §4 prints here: seal excisions, pocket count, single-domain
cell count, surrogate-face sanity.

Usage (Nova, CPU-only, ~minutes):
  python cluster/mesh_dump_sealed.py [OUT.npz]
"""
import sys

import numpy as np

sys.path.insert(0, "/work/mech-ai/baskarg/DiffSim/src")

from diffsim.cases.truck_config import load_truck_config, BodySpec
from diffsim.cases.truck import build_truck_mesh
from diffsim.octree import morton

CONF = ("/work/mech-ai/baskarg/DiffSim/local_code_old/truck_4case_fresh_inputs/"
        "NewRun-no-shell-slope0p25/config.txt")
TRUCK_STL = "/work/mech-ai/baskarg/DiffSim/local_code_old/truck/Truck.stl"
OUT = sys.argv[1] if len(sys.argv) > 1 else \
    "/work/mech-ai/baskarg/DiffSim/results/mesh_sealed_final.npz"

cfg = load_truck_config(CONF)
bodies = [BodySpec(mesh_path=TRUCK_STL, position=(0.0, -0.002, -7.0),
                   refine_lvl=12, is_static=True)]

fx = build_truck_mesh(
    cfg, base_level=7, region_refine=True, truck_band_to=12, band_cells=3,
    device="cpu", bodies=bodies, carve_lam=1.0,
    ground_refine_to=9, ground_band=0.0156,
    walls_refine_to=[(9, 0.0156), (8, 0.05)],
    carve_delta=1.0,
    seal_underbody=True, seal_y=0.006,
    seal_boxes=[(0.33, 0.35, 0.0, 0.017, 0.057, 0.068)])

ret = fx["tree"]
L = morton.lmax(3)
anchors = (ret.anchors() * 2.0 ** -L).astype(np.float32)
np.savez_compressed(OUT, anchors=anchors,
                    levels=ret.levels.astype(np.int8),
                    h=ret.h().astype(np.float32))

print(f"[mesh-dump-sealed] cells={fx['n_cells']} "
      f"pockets={fx['n_pockets']} ({fx['n_pocket_cells']} cells) "
      f"excluded={fx['n_excluded']} slab_cut={fx['n_slab_cut']} "
      f"sf_faces={len(fx['sf'].face_cells) if hasattr(fx['sf'], 'face_cells') else 'see-log'}",
      flush=True)
# Rung-0a automated checks (spec §4): fail loudly here, not at review time
assert fx["n_slab_cut"] == 0, "slab carve not dyadic-exact"
assert fx["n_cells"] > 2_000_000, "cell count implausibly low vs q-leg 2.65M"
print(f"[mesh-dump-sealed] wrote {OUT}", flush=True)
print("done rc=0", flush=True)
