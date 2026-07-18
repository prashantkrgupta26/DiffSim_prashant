"""XDD — excitonic drift-diffusion solver package (SP-1).

Block A (this package): pure numpy/stdlib — no warp or torch imports.
Block B will add GPU kernels as separate submodules.
"""
from diffsim.xdd.params import (
    XDDParams,
    XDDScales,
    read_spec,
    ret_factor_from_spectra,
    generation_from_spectra,
)
from diffsim.xdd.morphology import (
    Morphology,
    read_cpu_cloud,
    write_cpu_cloud,
    signed_distance,
    from_film_npz,
    tanh_mask,
    region_weights,
    interface_mask,
    descriptors,
)

__all__ = [
    # A1 — params
    "XDDParams",
    "XDDScales",
    "read_spec",
    "ret_factor_from_spectra",
    "generation_from_spectra",
    # A2 — morphology
    "Morphology",
    "read_cpu_cloud",
    "write_cpu_cloud",
    "signed_distance",
    "from_film_npz",
    "tanh_mask",
    "region_weights",
    "interface_mask",
    "descriptors",
]
