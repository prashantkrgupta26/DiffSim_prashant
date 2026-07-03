import numpy as np
import pytest
import warp as wp

wp.init()

@pytest.fixture(scope="session")
def device():
    return "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"

@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(20260702)
