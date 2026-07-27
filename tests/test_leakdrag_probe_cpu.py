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


def test_reaction_arbiter_tiny_cpu():
    """CPU gate for the reaction-force arbiter (LD-5).

    Two node sets (1.5L and 2.5L from plate centre, minus strong-BC nodes)
    must:
      1. Produce finite reaction_hist [nsteps, 2].
      2. Agree to <= 1e-6 relative (the variational-identity self-check).
      3. Sign of Cd_reaction matches sign of Cd_surr (positive = downstream drag).
    Tiny config: level=4, 6 steps, splu/host.  Wall time: ~2 s.
    """
    from gpu_leakdrag_discriminator import run_discriminator_with_reaction
    out = run_discriminator_with_reaction(
        alpha=50.0, nsteps=6, level=4, refine_to=None,
        wake_refine=None, dt=0.01, device="cpu",
        mono_solver="splu", assembly="host",
        t_start_lu=0.0,
    )
    rh = out["reaction_hist"]       # [nsteps, 2]
    cd_surr = out["cd_surr_mean"]   # float (time-averaged surrogate Cd)

    # 1. Finite reaction_hist
    assert np.all(np.isfinite(rh)), f"reaction_hist has non-finite values: {rh}"

    # 2. Set agreement <= 1e-6 relative
    cd_rxn_0 = np.mean(rh[:, 0])
    cd_rxn_1 = np.mean(rh[:, 1])
    denom = max(abs(cd_rxn_0), 1e-12)
    rel_diff = abs(cd_rxn_0 - cd_rxn_1) / denom
    assert rel_diff <= 1e-6, (
        f"Reaction-set agreement {rel_diff:.2e} > 1e-6: "
        f"set0={cd_rxn_0:.6g}, set1={cd_rxn_1:.6g}"
    )

    # 3. Sign: Cd_reaction and Cd_surr have the same sign
    # (both should be positive for downstream drag on the plate)
    assert np.sign(cd_rxn_0) == np.sign(cd_surr) or abs(cd_rxn_0) < 1e-10, (
        f"Sign mismatch: Cd_reaction={cd_rxn_0:.4g}, Cd_surr={cd_surr:.4g}"
    )
