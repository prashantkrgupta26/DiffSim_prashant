"""M1a locked numerical baselines (plan Task 11; spec S9.1 'physics
acceptance ... locked as numerical regression baselines', rtol 1e-6 per
S9.2 — ~100x above cross-machine reduction noise, ~1e4x below real-regression
signal). Regenerate deliberately with:
    .venv/bin/python tests/test_m1a_baselines.py
"""
import json
import os

import numpy as np
import pytest

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "baselines",
                             "m1a_baselines.json")

pytestmark = pytest.mark.tier5


def compute_baselines(device):
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_ad_gradients import _forward, THETA0, U_t
    from diffsim.sbm.adjoint import solve_adjoint, shape_gradient
    from test_sbm_poisson import sbm_setup, U2, F2, ZERO
    from test_sbm_neumann import (_solve_neumann, _band, _q_of, R, CTR2)
    from diffsim.geometry.csg import Sphere
    from diffsim.sbm.poisson import SBMPoisson
    from diffsim.sbm.surrogate import (classify_lambda, extract_surrogate,
                                       GeometryData)
    from diffsim.mesh.faces import face_tables
    from diffsim.octree.build import build_uniform
    from diffsim.physics.poisson import l2_error_masked

    out = {}

    def mms(level, p, lam, kappa=1.0, domain="inside", r=0.3):
        oracle = Sphere((0.5, 0.5), r)
        dm, geo, sf = sbm_setup(oracle, level, p, lam, 2, device,
                                domain=domain)
        prob = SBMPoisson(dm, geo, sf, g_fn=U2, kappa=kappa)
        u = prob.solve(f_fn=lambda x: kappa * F2(x), g_outer_fn=U2)
        sgn = -1.0 if domain == "inside" else 1.0
        return l2_error_masked(dm, u, U2,
                               lambda x: sgn * oracle.classify(x) > 0)

    out["disk_p1_l5"] = mms(5, 1, 0.0)
    out["disk_p2_l5"] = mms(5, 2, 0.0)
    out["exterior_p1_l5"] = mms(5, 1, 0.0, domain="outside", r=0.25)

    # var-kappa MMS at level 5
    from test_sbm_poisson import _kap_lin
    oracle = Sphere((0.5, 0.5), 0.3)
    dm, geo, sf = sbm_setup(oracle, 5, 1, 0.0, 2, device)

    def fvk(x):
        s = np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
        ux = np.pi * np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
        uy = np.pi * np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
        return -(1.0 * ux + 2.0 * uy) + 2.0 * np.pi ** 2 * _kap_lin(x) * s
    u = SBMPoisson(dm, geo, sf, g_fn=U2, kappa=_kap_lin).solve(f_fn=fvk)
    out["varkappa_p1_l5"] = l2_error_masked(
        dm, u, U2, lambda x: oracle.classify(x) < 0)

    # Neumann band3 at level 5
    _, e_neu, _, _, _, _ = _solve_neumann(5, _band(3), device)
    out["neumann_band3_l5"] = e_neu

    # corrected staircase perimeter, level 6 (geometry pipeline lock)
    o2 = Sphere((0.5, 0.5), 0.3)
    tree = build_uniform(6, dim=2)
    ret, _ = classify_lambda(tree, o2, 0.0)
    sfc = extract_surrogate(ret)
    ftab = face_tables(1, 2)
    geo2 = GeometryData.evaluate(o2, ret, sfc, ftab)
    h = ret.h()[sfc.elem]
    dS = np.repeat(h, ftab.nqf) * np.tile(ftab.w, len(sfc.elem)) / 2.0
    out["corrected_perimeter_l6"] = float((geo2.corr * dS).sum())

    # the three-way-verified shape gradient (adjoint values)
    fw = _forward(THETA0, device)
    lam = solve_adjoint(fw["A"], fw["dJdu"])
    shape_gradient(fw["prob"], fw["u_all"], lam, fw["oracle"], fw["meta"],
                   g_fn_torch=U_t)
    out["shape_grad_cx"] = float(fw["oracle"].center.grad[0])
    out["shape_grad_cy"] = float(fw["oracle"].center.grad[1])
    out["shape_grad_r"] = float(fw["oracle"].radius.grad)
    return {k: float(v) for k, v in out.items()}


def test_m1a_baselines_locked(device):
    assert os.path.exists(BASELINE_PATH), (
        "baseline file missing — generate with "
        "`python tests/test_m1a_baselines.py`")
    with open(BASELINE_PATH) as fh:
        ref = json.load(fh)
    cur = compute_baselines(device)
    mismatches = []
    for k, vref in ref.items():
        vcur = cur.get(k)
        denom = max(abs(vref), 1e-300)
        rel = abs(vcur - vref) / denom
        if rel > 1e-6:
            mismatches.append(f"  {k}: locked {vref:.12e}  current "
                              f"{vcur:.12e}  rel {rel:.3e}")
    assert not mismatches, ("M1a baseline regression:\n" +
                            "\n".join(mismatches))


if __name__ == "__main__":
    from diffsim import default_device
    vals = compute_baselines(default_device())
    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    with open(BASELINE_PATH, "w") as fh:
        json.dump(vals, fh, indent=2, sort_keys=True)
    print(f"wrote {BASELINE_PATH}:")
    for k, v in sorted(vals.items()):
        print(f"  {k} = {v:.12e}")
