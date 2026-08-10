"""benchmarks/chns — CHNS benchmark cases, Hysing metrics, and movie writer.

Submodules:
    cases   — CHNSCase dataclass + BUBBLE_RISE_RE35_WE10, BUBBLE_RISE_RE35_WE125,
               DAM_BREAK_2D, RT_2D case instances from legacy configs.
    metrics — centroid_y, rise_velocity, circularity (Hysing et al. 2009 definitions).
    movie   — write_interface_movie: phi=0 contour gif via matplotlib PillowWriter.
"""

from benchmarks.chns.cases import (
    CHNSCase,
    BUBBLE_RISE_RE35_WE10,
    BUBBLE_RISE_RE35_WE125,
    DAM_BREAK_2D,
    RT_2D,
)
from benchmarks.chns import metrics
from benchmarks.chns import movie

__all__ = [
    "CHNSCase",
    "BUBBLE_RISE_RE35_WE10",
    "BUBBLE_RISE_RE35_WE125",
    "DAM_BREAK_2D",
    "RT_2D",
    "metrics",
    "movie",
]
