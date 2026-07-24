import os
import sys

import numpy as np
import pytest

# Make sibling helper modules in tests/ (e.g. ladder_fixtures, the ladder rung
# drivers) importable by bare name from any test module, regardless of pytest's
# collection order / import mode. Without this, the alphabetically-first direct
# importer (test_backflow_stab) can be imported before tests/ lands on sys.path
# under `pythonpath=["."]` + prepend importmode, giving a spurious
# ModuleNotFoundError: ladder_fixtures during whole-suite collection.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Warp is optional here so the "Warp-free" CI tiers (numpy-only error-path and
# diagnostics tests) can collect without it installed. Warp-dependent fixtures
# skip when it is absent.
try:
    import warp as wp
    wp.init()
except ModuleNotFoundError:
    wp = None

@pytest.fixture(scope="session")
def device():
    if wp is None:
        pytest.skip("Warp not installed (Warp-free CI tier)")
    from diffsim import default_device
    return default_device()

@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(20260702)
