"""TRUCK case driver package (volumetric one-sided SBM on merged-STL triangle soup).

Code was split from tests/truck_flow.py (pure motion).  Four sub-modules:
  truck_mesh  — incomplete-octree mesh construction (slab carve, refine, truck carve)
  truck_bc    — boundary conditions (strong BC masks, reaction set, soft-start)
  truck_ckpt  — mesh-sequenced checkpoint interpolation
  truck_march — BDF2 transient march loop (run_truck, make_nu_schedule)
"""
from .truck_mesh import (
    _channel_box,
    slab_carve,
    refine_region_boxes,
    refine_truck_band,
    refine_walls,
    refine_ground,
    _face_components,
    flood_fill_retain,
    build_truck_mesh,
)
from .truck_bc import (
    truck_bc_masks,
    truck_strong_bc,
    _pressure_pin,
    soft_start_amp,
    truck_reaction_set,
)
from .truck_ckpt import interpolate_checkpoint
from .truck_march import (
    _gp_field,
    _gp_history_fq,
    run_truck,
    make_nu_schedule,
)

__all__ = [
    # mesh
    "_channel_box",
    "slab_carve",
    "refine_region_boxes",
    "refine_truck_band",
    "refine_walls",
    "refine_ground",
    "_face_components",
    "flood_fill_retain",
    "build_truck_mesh",
    # bc
    "truck_bc_masks",
    "truck_strong_bc",
    "_pressure_pin",
    "soft_start_amp",
    "truck_reaction_set",
    # ckpt
    "interpolate_checkpoint",
    # march
    "_gp_field",
    "_gp_history_fq",
    "run_truck",
    "make_nu_schedule",
]
