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

__all__ = [
    "XDDParams",
    "XDDScales",
    "read_spec",
    "ret_factor_from_spectra",
    "generation_from_spectra",
]
