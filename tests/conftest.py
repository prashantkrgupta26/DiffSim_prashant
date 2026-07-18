import numpy as np
import pytest

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
    return "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"

@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(20260702)
