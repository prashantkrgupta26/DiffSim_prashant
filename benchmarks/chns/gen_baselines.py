#!/usr/bin/env python
r"""benchmarks/chns/gen_baselines.py — regenerate the CI provisional baselines.

Writes benchmarks/chns/results/ci_baselines.json — the committed
internal-reproducibility targets the CI benchmark gates
(tests/test_chns_benchmarks.py) assert against with doubled tolerance.

These are PROVISIONAL and NOT physical reference values (the CI runs are the
non-dimensional CHNS on the unit square with Cn=2h resolvability and a truncated
step count).  See the REFERENCES block in the test module for the Hysing 2009 /
Martin-Moyce 1952 physical direction.  Run this after any INTENDED physics
change; commit the JSON delta with the code change.

  .venv/bin/python benchmarks/chns/gen_baselines.py
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "..")):
    _p = os.path.normpath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chns import cases as C          # noqa: E402
from chns import metrics             # noqa: E402


def _make_dm(level):
    import warp as wp
    wp.init()
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    from diffsim import default_device
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    return DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2),
                                default_device()), mesh


def bubble_ic(coords, Cn, yc=0.35, rad=0.2):
    r = np.sqrt((coords[:, 0] - 0.5) ** 2 + (coords[:, 1] - yc) ** 2)
    return -np.tanh((r - rad) / (Cn * np.sqrt(2.0)))


def dam_ic(coords, Cn, xi=0.4):
    return -np.tanh((coords[:, 0] - xi) / (Cn * np.sqrt(2.0)))


def rt_ic(coords, Cn, yi=0.5, amp=0.05):
    yp = yi + amp * np.cos(2.0 * np.pi * coords[:, 0])
    return np.tanh((coords[:, 1] - yp) / (Cn * np.sqrt(2.0)))


def _march(case, level, dt, nsteps, ic, sign):
    from diffsim.steppers.chns import CHNSStepper
    dm, mesh = _make_dm(level)
    coords = mesh.node_coords
    st = CHNSStepper(dm, case, dt=dt, mode="auto", Cn_override="2h",
                     gravity=True)
    st.set_initial(ic(coords, st.Cn))
    m0 = st.mass_phi()
    h = st.h
    cy, front, tip = [], [], []
    for _ in range(nsteps):
        st.step()
        cy.append(metrics.centroid_y(sign * st.phi, coords))
        front.append(metrics.surge_front_x(st.phi, coords))
        tip.append(metrics.spike_tip_y(st.phi, coords))
    rv = metrics.rise_velocity(np.asarray(cy), dt)
    return dict(
        st=st, coords=coords, h=h, sign=sign, cy=cy, front=front, tip=tip,
        cy_final=float(cy[-1]),
        rise_peak=float(np.max(np.abs(rv))),
        circ_min=float(metrics.circularity(sign * st.phi, coords, h)),
        mass_drift=float(abs(st.mass_phi() - m0)),
        energy_final=float(st.energy()["total"]))


def main():
    base = {"_README": (
        "SP-0 Task 9 PROVISIONAL internal-reproducibility baselines for the CI "
        "benchmark gates (tests/test_chns_benchmarks.py). NOT physical "
        "reference values (unit-square non-dimensional CHNS, Cn=2h, truncated "
        "steps). See the test module REFERENCES for Hysing 2009 / Martin-Moyce "
        "1952 direction. Regenerate with benchmarks/chns/gen_baselines.py.")}

    b10 = _march(C.BUBBLE_RISE_RE35_WE10, 5,
                 C.BUBBLE_RISE_RE35_WE10.dt0, 40, bubble_ic, -1)
    base["bubble_rise_re35_we10"] = dict(
        level=5, dt=C.BUBBLE_RISE_RE35_WE10.dt0, nsteps=40, gravity=True,
        cy_final=b10["cy_final"], rise_peak=b10["rise_peak"],
        circ_min=b10["circ_min"], mass_drift=b10["mass_drift"],
        energy_final=b10["energy_final"])

    b125 = _march(C.BUBBLE_RISE_RE35_WE125, 5,
                  C.BUBBLE_RISE_RE35_WE125.dt0, 40, bubble_ic, -1)
    base["bubble_rise_re35_we125"] = dict(
        level=5, dt=C.BUBBLE_RISE_RE35_WE125.dt0, nsteps=40, gravity=True,
        cy_final=b125["cy_final"], rise_peak=b125["rise_peak"],
        circ_min=b125["circ_min"], mass_drift=b125["mass_drift"],
        energy_final=b125["energy_final"])

    dam = _march(C.DAM_BREAK_2D, 4, C.DAM_BREAK_2D.dt0, 60, dam_ic, 1)
    # surge front at 3 sampled step indices (1/3, 2/3, end)
    idx3 = [len(dam["front"]) // 3 - 1, 2 * len(dam["front"]) // 3 - 1,
            len(dam["front"]) - 1]
    base["dam_break"] = dict(
        level=4, dt=C.DAM_BREAK_2D.dt0, nsteps=60, gravity=True,
        surge_front_x_at=[int(i) for i in idx3],
        surge_front_x=[float(dam["front"][i]) for i in idx3],
        circ_min=dam["circ_min"], mass_drift=dam["mass_drift"],
        energy_final=dam["energy_final"])

    rt = _march(C.RT_2D, 4, C.RT_2D.dt0, 60, rt_ic, 1)
    idx2 = [len(rt["tip"]) // 2 - 1, len(rt["tip"]) - 1]
    base["rt"] = dict(
        level=4, dt=C.RT_2D.dt0, nsteps=60, gravity=True,
        spike_tip_y_at=[int(i) for i in idx2],
        spike_tip_y=[float(rt["tip"][i]) for i in idx2],
        circ_min=rt["circ_min"], mass_drift=rt["mass_drift"],
        energy_final=rt["energy_final"])

    out = os.path.join(_HERE, "results", "ci_baselines.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(base, fh, indent=2)
    print(f"[gen_baselines] wrote {out}")
    print(json.dumps({k: v for k, v in base.items()
                      if not k.startswith("_")}, indent=2))


if __name__ == "__main__":
    main()
