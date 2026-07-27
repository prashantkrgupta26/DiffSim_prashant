"""CPU smoke for the leak-drag probe machinery (tiny config, splu/host)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def test_probe_runs_tiny_cpu():
    from gpu_leakdrag_discriminator import run_discriminator
    out = run_discriminator(
        alpha=50.0, nsteps=4, level=4, refine_to=None,
        wake_refine=None, dt=0.01, device="cpu",
        mono_solver="splu", assembly="host",
        t_start_lu=0.0,
    )
    assert np.isfinite(out["cd_surr_mean"])
    for v in out["cd_cv_mean"].values():
        assert np.isfinite(v)
    assert np.isfinite(out["leak_mean_abs"])
    assert out["n_steps_avg"] == 4
