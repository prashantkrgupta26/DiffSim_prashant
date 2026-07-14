"""OrgElMorph course — shared run/config/provenance/check foundation.

Every tutorial imports from here so a run is a *repeatable workflow*: a YAML
config (canonical record), a provenance ``metadata.json``, a ``results.json``
checked against a tolerance baseline, and a standard output layout. See
``docs/dev/2026-07-14-course-v2-evaluation.md`` (Phase 0).
"""
from __future__ import annotations

from . import check_results, config, provenance, run_base

__all__ = ["config", "provenance", "check_results", "run_base"]
