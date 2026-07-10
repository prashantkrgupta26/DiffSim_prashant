"""diffsim.film — the user-facing evaporating-film front end.

Run a new material system by editing four numbers (chi, N, blend, Bi)
in a YAML config; validated named cases live in diffsim/film/configs/.

    from diffsim.film import FilmParams, FilmRun
    summary = FilmRun(FilmParams.from_yaml("case.yaml")).run("out/")

CLI:  python -m diffsim.film <config.yaml> [--outdir DIR] [--set k=v]
"""
import os

from .params import FilmParams, ResolvedFilm, kappa_from_eps2
from .preflight import run_preflight, PreflightReport, PreflightError
from .run import FilmRun

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "configs")

__all__ = ["FilmParams", "ResolvedFilm", "FilmRun", "run_preflight",
           "PreflightReport", "PreflightError", "kappa_from_eps2",
           "CONFIG_DIR"]
