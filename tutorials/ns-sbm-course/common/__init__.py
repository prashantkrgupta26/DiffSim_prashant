"""NS-SBM course — shared run/config/provenance/check foundation.

Every module imports from here so a run is a *repeatable workflow*: a YAML
config (canonical record), a provenance ``metadata.json``, a ``results.json``
checked against a tolerance baseline, and a standard output layout. This is a
light, engine-agnostic copy of the sibling ``orgelmorph-course/common`` harness
(the config/check/provenance/run_base pattern is shared verbatim; the NS
modules differ only in their ``run_fn`` and schema).
"""
from __future__ import annotations

from . import check_results, config, provenance, run_base

__all__ = ["config", "provenance", "check_results", "run_base"]
