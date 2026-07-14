"""diffsim.diagnostics — reusable, verified scientific-diagnostic library.

A Warp-free (numpy-only) toolkit the OrgElMorph course tutorials and the
research code share, so every chapter reports the SAME measured quantity the
same way (spec Phase 0). Each function documents its UNITS and is covered by a
unit test in ``tests/test_diagnostics.py``.

Submodules
----------
conservation   quadrature mass, component content, boundary flux, moving-domain
               balance, transfer mass-change.
energy         Ginzburg-Landau energy split (bulk/grad/wall/cryst/coupling),
               stepwise increments, discrete monotonicity.
morphology     structure factor, peak & first-moment wavelength, two-point
               correlation, interfacial area, phase fractions, anisotropy.
admissibility  field bounds, simplex residual, clipped/projected fraction.
convergence    observed order of accuracy, Richardson extrapolation, GCI.
stochastic     ensemble aggregation, bootstrap CI, first-passage, event prob.
profiling      CUDA-synced stage timers, peak device memory.
provenance     environment + run fingerprint for metadata.json.
"""
from __future__ import annotations

from . import (admissibility, conservation, convergence, energy, morphology,
               profiling, provenance, stochastic)

__all__ = [
    "conservation", "energy", "morphology", "admissibility", "convergence",
    "stochastic", "profiling", "provenance",
]
