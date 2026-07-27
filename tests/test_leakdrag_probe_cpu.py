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


# ---------------------------------------------------------------------------
# TD-2: per-term consistent-reaction history (reaction_terms)
# ---------------------------------------------------------------------------

def test_reaction_terms_parity():
    """Default reaction_terms=False must be bit-for-bit identical to the
    legacy march: same cd values and no new keys in the return dict.

    Uses reaction_sets=None to avoid any surgery-row node complications and
    to verify the reaction_terms flag itself has zero overhead.
    """
    from p2r1a_thin_plate_flow import run_flow_past

    kw = dict(level=4, nsteps=4, dt=0.01, nu=0.1, alpha=50.0)

    # reference: no reaction_terms kwarg at all (legacy path)
    out_ref = run_flow_past(**kw)
    # with explicit reaction_terms=False (must be bit-for-bit identical)
    out_default = run_flow_past(**kw, reaction_terms=False)

    # bit-for-bit cd equality
    assert np.array_equal(out_ref["cd"], out_default["cd"]), (
        "reaction_terms=False broke bit-for-bit parity on cd"
    )

    # No new keys beyond what's in the reference output
    ref_keys = set(out_ref.keys())
    new_keys = set(out_default.keys())
    extra_keys = new_keys - ref_keys
    assert not extra_keys, (
        f"reaction_terms=False added unexpected keys: {extra_keys}"
    )


def test_reaction_terms_hist_shape_and_partition():
    """reaction_terms=True with reaction_sets → hist [nsteps, 3], stable names,
    per-step Σ==total at 1e-12, all finite.

    Term count 3 is the TD-1 split: consistency+adjoint, penalty, backflow.
    TDD RED: run_flow_past does not yet accept reaction_terms kwarg.
    """
    from p2r1a_thin_plate_flow import run_flow_past

    level = 4
    nsteps = 4
    x_c, y_c, L = 0.375, 0.5, 0.25

    # Build a simple free-node set (all interior nodes, avoid surgery rows)
    from diffsim.octree.build import build_uniform
    from diffsim.geometry.csg import Segment
    from diffsim.sbm.surrogate import classify_shell_intercepted
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    import scipy.sparse as sp

    a = (x_c, y_c - L / 2.0)
    b_pt = (x_c, y_c + L / 2.0)
    seg = Segment(a, b_pt)
    tree = build_uniform(level, dim=2)
    ret, _ = classify_shell_intercepted(tree, seg)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    coords = mesh.node_coords[cons.free_nodes]

    # Interior nodes only: not on any boundary face (avoids surgery rows)
    tol = 1e-10
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    interior_mask = (
        (coords[:, 0] > x_min + tol) &
        (coords[:, 0] < x_max - tol) &
        (coords[:, 1] > y_min + tol) &
        (coords[:, 1] < y_max - tol)
    )
    interior_nodes = np.where(interior_mask)[0].astype(np.intp)
    assert len(interior_nodes) > 0, "No interior nodes found"

    reaction_sets = [interior_nodes]

    out = run_flow_past(
        level=level, nsteps=nsteps, dt=0.01, nu=0.1, alpha=50.0,
        reaction_sets=reaction_sets,
        reaction_terms=True,
    )

    # 1. hist shape [nsteps, 3]
    assert "reaction_terms_hist" in out, "reaction_terms_hist missing from output"
    hist = out["reaction_terms_hist"]    # [nsteps, nterms]
    assert hist.shape == (nsteps, 3), (
        f"Expected hist shape ({nsteps}, 3), got {hist.shape}"
    )

    # 2. term names present and stable
    assert "reaction_terms_names" in out, "reaction_terms_names missing from output"
    names = out["reaction_terms_names"]
    assert len(names) == 3, f"Expected 3 term names, got {len(names)}: {names}"
    expected_names = {"consistency+adjoint", "penalty", "backflow"}
    assert set(names) == expected_names, (
        f"Expected names {expected_names}, got {set(names)}"
    )

    # 3. all finite
    assert np.all(np.isfinite(hist)), f"reaction_terms_hist has non-finite: {hist}"

    # 4. reaction_hist still present (set 0 only): [nsteps, 1]
    assert "reaction_hist" in out, "reaction_hist missing from output"
    rh = out["reaction_hist"]
    assert rh.shape == (nsteps, 1), f"reaction_hist shape: {rh.shape}"

    # 5. per-step partition: Σ_t f_x_t == f_x_total at 1e-12
    total_cd = rh[:, 0]  # [nsteps] — total reaction Cd for set 0
    term_sum_cd = hist.sum(axis=1)  # [nsteps] — sum across terms
    for s in range(nsteps):
        tol_gate = 1e-12 * max(1.0, abs(float(total_cd[s])))
        err = abs(float(term_sum_cd[s]) - float(total_cd[s]))
        assert err <= tol_gate, (
            f"Step {s}: partition error {err:.3e} > 1e-12·max(1,|total|) "
            f"({tol_gate:.3e}). total={total_cd[s]:.8g}, "
            f"sum_terms={term_sum_cd[s]:.8g}"
        )


def test_reaction_terms_requires_reaction_sets():
    """reaction_terms=True without reaction_sets must raise ValueError.

    TDD RED: run_flow_past does not yet accept reaction_terms kwarg.
    """
    from p2r1a_thin_plate_flow import run_flow_past
    import pytest

    with pytest.raises(ValueError, match="reaction_sets"):
        run_flow_past(
            level=4, nsteps=2, dt=0.01,
            reaction_sets=None,
            reaction_terms=True,
        )
