"""Compatibility shim — truck driver moved to diffsim.cases.truck.

The implementation has been split into four modules under
src/diffsim/cases/truck/:
  truck_mesh.py   — mesh construction (slab carve, refine, truck carve)
  truck_bc.py     — boundary conditions and reaction-set helpers
  truck_ckpt.py   — mesh-sequenced checkpoint interpolation
  truck_march.py  — BDF2 transient march (run_truck, make_nu_schedule)

This shim re-exports the full public API so that old references (ledger
snippets, results scripts, cluster scripts not yet updated) continue to work.
New code should import from diffsim.cases.truck directly.
"""
from diffsim.cases.truck import (
    _channel_box,
    slab_carve,
    refine_region_boxes,
    refine_truck_band,
    refine_walls,
    refine_ground,
    _face_components,
    flood_fill_retain,
    build_truck_mesh,
    truck_bc_masks,
    truck_strong_bc,
    _pressure_pin,
    soft_start_amp,
    truck_reaction_set,
    interpolate_checkpoint,
    _gp_field,
    _gp_history_fq,
    run_truck,
    make_nu_schedule,
)
