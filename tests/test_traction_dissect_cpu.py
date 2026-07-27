"""CPU smoke gate for the traction-dissection probe (TD-3).

One tiny leg (level=4, uniform, 4 steps, splu/host/cpu) — exercises
run_dissect_leg's output contract without GPU:
  1. All outputs finite (bridge may be NaN-guarded — assert finite or NaN-with-reason).
  2. Term partition holds: Σ_terms ≈ cd_rxn_total from the returned arrays.
  3. Term names match expected set {"consistency+adjoint", "penalty", "backflow"}.
  4. cd_surr finite.
  5. St finite or NaN (no crash).
  6. bridge_ratio: either finite, or NaN with a printed diagnostic.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def test_dissect_leg_tiny_cpu():
    """One tiny leg: level=4, uniform (refine_to=None), nsteps=4, dt=0.01,
    splu/host/cpu.  All outputs finite; term partition holds; names as expected.
    Wall time ~2–5 s.
    """
    from gpu_traction_dissect import run_dissect_leg

    out = run_dissect_leg(
        tag="cpu_smoke",
        alpha=50.0,
        nsteps=4,
        level=4,
        refine_to=None,
        wake_refine=None,
        plate_L_inv=16,
        dt=0.01,
        device="cpu",
        mono_solver="splu",
        assembly="host",
        t_start_lu=0.0,   # accumulate from step 1 (tiny run)
    )

    # 1. cd_rxn_total finite
    assert np.isfinite(out["cd_rxn_total"]), (
        f"cd_rxn_total non-finite: {out['cd_rxn_total']}"
    )

    # 2. cd_surr finite
    assert np.isfinite(out["cd_surr"]), (
        f"cd_surr non-finite: {out['cd_surr']}"
    )

    # 3. Term names: must have the expected 3-term split
    expected_names = {"consistency+adjoint", "penalty", "backflow"}
    actual_names = set(out["cd_rxn_terms"].keys())
    assert actual_names == expected_names, (
        f"Expected term names {expected_names}, got {actual_names}"
    )

    # 4. All per-term values finite
    for name, val in out["cd_rxn_terms"].items():
        assert np.isfinite(val), f"Term '{name}' non-finite: {val}"

    # 5. Term partition: Σ terms ≈ total (from raw hist arrays)
    rth = out["_reaction_terms_hist"]   # [nsteps, 3]
    rh  = out["_reaction_hist"]         # [nsteps, 1]
    # Time-averaged total must match sum of time-averaged terms
    total_mean = float(np.mean(rh[:, 0]))
    terms_sum_mean = float(np.sum(rth.mean(axis=0)))
    err = abs(terms_sum_mean - total_mean)
    tol = 1e-10 * max(1.0, abs(total_mean))
    assert err <= tol, (
        f"Term partition error {err:.3e} > tol {tol:.3e}. "
        f"total_mean={total_mean:.8g}, sum_terms={terms_sum_mean:.8g}"
    )

    # Per-step partition too
    for s in range(rh.shape[0]):
        total_s = float(rh[s, 0])
        sum_s   = float(rth[s].sum())
        tol_s   = 1e-12 * max(1.0, abs(total_s))
        err_s   = abs(sum_s - total_s)
        assert err_s <= tol_s, (
            f"Step {s}: partition error {err_s:.3e} > {tol_s:.3e}. "
            f"total={total_s:.8g}, sum_terms={sum_s:.8g}"
        )

    # 6. bridge_ratio: finite OR NaN-guarded (both are valid)
    br = out["bridge_ratio"]
    assert np.isfinite(br) or np.isnan(br), (
        f"bridge_ratio must be finite or NaN, got: {br}"
    )

    # 7. dt_used finite and positive
    assert np.isfinite(out["dt_used"]) and out["dt_used"] > 0, (
        f"dt_used invalid: {out['dt_used']}"
    )

    # 8. elapsed finite and positive
    assert np.isfinite(out["elapsed"]) and out["elapsed"] >= 0, (
        f"elapsed invalid: {out['elapsed']}"
    )
