"""Import-path bootstrap for the grouped benchmark scripts.

The benchmarks live in physics-grouped subfolders (navier-stokes/, heat-mass/,
phase-field/, ...) but several cross-import each other by bare module name
(e.g. hero_h3 imports hero_h1; coupled_nu imports heated_cylinder) and some
import test helpers from tests/. To keep those imports working regardless of
which subfolder a script lives in — and regardless of the current working
directory — each script does, right after its docstring:

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import _bench_bootstrap  # noqa

Importing this module registers every benchmark subfolder plus tests/ on
sys.path. Idempotent; safe to import many times.
"""
import os
import sys
import glob

_BENCH = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_BENCH)

_dirs = [os.path.normpath(d) for d in glob.glob(os.path.join(_BENCH, "*", ""))]
_dirs += [_BENCH, os.path.join(_ROOT, "tests")]
for _d in _dirs:
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)
