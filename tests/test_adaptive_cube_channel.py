"""CPU gate for the adaptive band-refined cube-channel fixture (Task 8b scope
increment: adaptive bluff-body rungs for the GH200 ladder).

Three always-on checks (Mac CPU, splu, small levels — must stay fast):
  1. structure  — base L3 -> band r5: hanging nodes exist, surrogate faces
                  nonzero, excluded (carved) cells nonzero;
  2. march      — 3 monolithic steps (splu) on the adaptive mesh -> finite Cd;
  3. anchor     — uniform L4 (existing fixture) vs adaptive L3/r4 with the same
                  physical params: same Cd sign and within a loose 2x band
                  (sanity gate, NOT a convergence claim).
"""
import numpy as np
import pytest

from adaptive_cube_channel import build_adaptive_cube_channel_3d
from ladder_fixtures import build_cube_channel_3d
from ladder_rung3d_cube import march_monolithic_3d

RE = 40
HALF = 0.125
OFFSET = 0.05
DT = 0.02
NSTEPS = 3


@pytest.fixture(scope="module")
def adaptive_l3r5():
    return build_adaptive_cube_channel_3d(
        base_level=3, refine_to=5, Re=RE, half=HALF, offset=OFFSET,
        device="cpu")


def test_structure(adaptive_l3r5):
    fx = adaptive_l3r5
    assert fx["n_hanging"] > 0, "band refinement must create hanging nodes"
    assert fx["sf"].elem.size > 0, "carve must expose surrogate faces"
    assert fx["n_excluded"] > 0, "cube must exclude (carve) cells"
    assert fx["n_nodes"] == len(fx["mesh"].node_coords)
    # adaptive must be cheaper than uniform at the fine level:
    # L5 uniform = 32^3 = 32768 cells; the band mesh must be far below that.
    assert fx["n_fluid_cells"] < 32768


def test_march_finite_cd(adaptive_l3r5):
    fx = adaptive_l3r5
    res = march_monolithic_3d(
        fx, dt=DT, nsteps=NSTEPS, rate_tol=None, log_every=0,
        solver="splu", device="cpu", strong_obstacle=False)
    assert np.isfinite(res["cd"]), f"adaptive-mesh Cd not finite: {res['cd']}"
    assert all(np.isfinite(c) for c in res["cd_hist"])


def test_anchor_uniform_l4_vs_adaptive_l3r4():
    """Same physical params, 3 steps each: uniform L4 vs adaptive L3/r4.
    Same finest h => Cd must agree in SIGN and within a loose 2x factor.
    This is a sanity gate, not convergence."""
    fx_u = build_cube_channel_3d(4, RE, half=HALF, offset=OFFSET, device="cpu")
    res_u = march_monolithic_3d(
        fx_u, dt=DT, nsteps=NSTEPS, rate_tol=None, log_every=0,
        solver="splu", device="cpu", strong_obstacle=False)

    fx_a = build_adaptive_cube_channel_3d(
        base_level=3, refine_to=4, Re=RE, half=HALF, offset=OFFSET,
        device="cpu")
    res_a = march_monolithic_3d(
        fx_a, dt=DT, nsteps=NSTEPS, rate_tol=None, log_every=0,
        solver="splu", device="cpu", strong_obstacle=False)

    cd_u, cd_a = res_u["cd"], res_a["cd"]
    assert np.isfinite(cd_u) and np.isfinite(cd_a)
    assert np.sign(cd_u) == np.sign(cd_a), (
        f"sign mismatch: uniform L4 Cd={cd_u:+.5f} vs adaptive L3/r4 "
        f"Cd={cd_a:+.5f}")
    ratio = abs(cd_a) / abs(cd_u)
    assert 0.5 <= ratio <= 2.0, (
        f"loose 2x sanity band violated: uniform L4 Cd={cd_u:+.5f}, "
        f"adaptive L3/r4 Cd={cd_a:+.5f}, ratio={ratio:.3f}")
