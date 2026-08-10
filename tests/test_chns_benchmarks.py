r"""tests/test_chns_benchmarks.py — SP-0 Task 9 CHNS benchmark gates.

All gates are CI-SIZED (one level below production, truncated step count,
tolerance doubled per the plan) and run on the monolithic CHNSStepper (the
SP-0 coupling winner).  The mesh is the unit square [0,1]^2; the interface is
resolved with Cn_override="2h" (the coupling-decision memo's guard finding:
benchmarks must resolve Cn >~ h, else the monolithic clamp-guard fires).

================================================================================
REFERENCES (physical direction; sourcing decision recorded)
================================================================================
Bubble rise — Hysing, S., Turek, S., Kuzmin, D., Parolini, N., Burman, E.,
  Ganesan, S., Tobiska, L. (2009). "Quantitative benchmark computations of
  two-dimensional bubble dynamics." Int. J. Numer. Meth. Fluids 60(11):1259.
  Test case 1 (rho 1000/100, mu 10/1, sigma 24.5): min circularity 0.9013 @
  t=1.90, max rise velocity 0.2417 @ t=0.924, final centroid 1.0799 @ t=3.
  Test case 2 (rho 1000/1, mu 10/0.1, sigma 1.96): min circularity 0.6901,
  max rise velocity 0.2502, final centroid 0.9154 @ t=2.
  Source: onlinelibrary.wiley.com/doi/10.1002/fld.1934,
          www.karlin.mff.cuni.cz/~hron/NMMO403/1934_ftp.pdf (open copy).

Dam break — Martin, J.C., Moyce, W.J. (1952). "An experimental study of the
  collapse of liquid columns on a rigid horizontal plane." Phil. Trans. R. Soc.
  Lond. A 244:312.  Dimensionless surge-front Z=x/L vs T=t*sqrt(2g/L).

SOURCING DECISION (honest, per the brief's rule): the Hysing/Martin-Moyce
numbers are DIMENSIONAL benchmarks (domain [1x2], specific rho/mu/g/sigma);
our runs are the Khanwale NON-DIMENSIONAL form on the UNIT SQUARE at COARSE CI
resolution with Cn=2h.  A truly comparable number is NOT obtainable for our
nondim/coarse setup, so — exactly as the brief permits — these gates assert
INTERNAL REPRODUCIBILITY against a committed PROVISIONAL baseline
(benchmarks/chns/results/ci_baselines.json), produced by this same code
(benchmarks/chns/gen_baselines.py).  The physical references above are the
DIRECTION, not the gate.  See sp0-task-9-report.md for the full rationale and
the dam-break/RT surge-dynamics escalation (the coarse CI runs do not develop
the surge front / spike within the wall budget at the stiff Re; a production
--full run is required — quantitative Martin-Moyce agreement remains out of
scope for the nondim setup).

The MMS gate is a genuine (source-forced) DESIGN-ORDER convergence gate — not a
reproducibility gate — and asserts L2 order >= 1.8 for u and phi.
"""
import json
import os as _os
import sys as _sys

import numpy as np
import pytest

_BENCH_DIR = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..",
                  "benchmarks"))
if _BENCH_DIR not in _sys.path:
    _sys.path.insert(0, _BENCH_DIR)
_REPO_DIR = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
if _REPO_DIR not in _sys.path:
    _sys.path.insert(0, _REPO_DIR)

from chns import cases as C           # noqa: E402
from chns import metrics              # noqa: E402
from dataclasses import replace       # noqa: E402

_BASELINE = json.load(open(_os.path.join(
    _BENCH_DIR, "chns", "results", "ci_baselines.json")))


# ---------------------------------------------------------------------------
def _make_dm(level, dim=2):
    import warp as wp
    wp.init()
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    from diffsim import default_device
    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim),
                              default_device())
    return dm, mesh, cons


def _bubble_ic(coords, Cn, yc=0.35, rad=0.2):
    r = np.sqrt((coords[:, 0] - 0.5) ** 2 + (coords[:, 1] - yc) ** 2)
    return -np.tanh((r - rad) / (Cn * np.sqrt(2.0)))


def _dam_ic(coords, Cn, xi=0.4):
    return -np.tanh((coords[:, 0] - xi) / (Cn * np.sqrt(2.0)))


def _rt_ic(coords, Cn, yi=0.5, amp=0.05):
    yp = yi + amp * np.cos(2.0 * np.pi * coords[:, 0])
    return np.tanh((coords[:, 1] - yp) / (Cn * np.sqrt(2.0)))


def _march(case, level, dt, nsteps, ic, sign):
    """Return the stepper + tracked series after nsteps.  Structural
    diagnostics (mass drift, energy) are captured too."""
    from diffsim.steppers.chns import CHNSStepper
    dm, mesh, cons = _make_dm(level)
    coords = mesh.node_coords
    st = CHNSStepper(dm, case, dt=dt, mode="auto", Cn_override="2h",
                     gravity=True)
    st.set_initial(ic(coords, st.Cn))
    m0 = st.mass_phi()
    cy, front, tip, energy = [], [], [], [st.energy()["total"]]
    for _ in range(nsteps):
        st.step()
        assert np.isfinite(st.phi).all() and np.isfinite(st.u).all(), \
            "NaN encountered during march"
        cy.append(metrics.centroid_y(sign * st.phi, coords))
        front.append(metrics.surge_front_x(st.phi, coords))
        tip.append(metrics.spike_tip_y(st.phi, coords))
        energy.append(st.energy()["total"])
    return dict(st=st, coords=coords, h=st.h, sign=sign, m0=m0,
                cy=cy, front=front, tip=tip, energy=energy,
                mass_drift=float(abs(st.mass_phi() - m0)))


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


# ===========================================================================
# Structural diagnostics (shared assertions on every driven run)
# ===========================================================================
def _assert_structural(res, mass_tol=1e-8):
    # mass drift < 1e-8 (no source; conservative CH advection is machine-exact)
    assert res["mass_drift"] < mass_tol, \
        f"mass drift {res['mass_drift']:.3e} >= {mass_tol}"


# ===========================================================================
# Bubble rise Re35/We10 and Re35/We125
# ===========================================================================
@pytest.mark.parametrize("name,case,sign", [
    ("bubble_rise_re35_we10",
     replace(C.BUBBLE_RISE_RE35_WE10, rho_ratio=10.0), -1),
    ("bubble_rise_re35_we125", C.BUBBLE_RISE_RE35_WE125, -1),
])
def test_bubble_rise(name, case, sign):
    """CI-sized bubble rise: terminal centroid + max rise velocity + min
    circularity reproduce the committed provisional baseline (doubled tol),
    plus machine-exact mass conservation.  Physical direction: Hysing 2009
    (see module REFERENCES)."""
    b = _BASELINE[name]
    res = _march(case, b["level"], b["dt"], b["nsteps"], _bubble_ic, sign)
    _assert_structural(res)
    cy_final = res["cy"][-1]
    rv = metrics.rise_velocity(np.asarray(res["cy"]), b["dt"])
    rise_peak = float(np.max(np.abs(rv)))
    circ_min = float(metrics.circularity(sign * res["st"].phi,
                                         res["coords"], res["h"]))
    # doubled-tolerance reproducibility (2% on centroid, 10% on velocity/circ)
    assert _rel(cy_final, b["cy_final"]) < 0.02, \
        f"{name} centroid {cy_final:.5f} vs baseline {b['cy_final']:.5f}"
    assert _rel(rise_peak, b["rise_peak"]) < 0.10, \
        f"{name} rise peak {rise_peak:.5f} vs baseline {b['rise_peak']:.5f}"
    assert _rel(circ_min, b["circ_min"]) < 0.10, \
        f"{name} circ {circ_min:.5f} vs baseline {b['circ_min']:.5f}"
    # light bubble must rise (physical direction sanity)
    assert res["cy"][-1] > res["cy"][0], "bubble did not rise"


# ===========================================================================
# Dam break — surge-front reproducibility (Martin-Moyce direction)
# ===========================================================================
def test_dam_break():
    """CI-sized dam break: surge-front x(t) at 3 sampled times reproduces the
    committed baseline, plus machine-exact mass conservation.  Physical
    direction: Martin-Moyce 1952 (see module REFERENCES; the coarse CI run does
    not develop the surge within the wall budget at Re=280000 — the front
    position is a reproducibility regression guard, not a Martin-Moyce number;
    see the report escalation)."""
    b = _BASELINE["dam_break"]
    res = _march(C.DAM_BREAK_2D, b["level"], b["dt"], b["nsteps"], _dam_ic, +1)
    _assert_structural(res)
    for idx, ref in zip(b["surge_front_x_at"], b["surge_front_x"]):
        got = res["front"][idx]
        if np.isnan(ref):
            continue
        assert _rel(got, ref) < 0.10, \
            f"dam surge front @{idx} {got:.4f} vs baseline {ref:.4f}"
    circ_min = float(metrics.circularity(res["st"].phi, res["coords"],
                                         res["h"]))
    assert _rel(circ_min, b["circ_min"]) < 0.10, \
        f"dam circ {circ_min:.5f} vs baseline {b['circ_min']:.5f}"


# ===========================================================================
# Rayleigh-Taylor — spike-tip reproducibility
# ===========================================================================
def test_rt():
    """CI-sized RT: spike-tip y at 2 sampled times reproduces the committed
    internal baseline (labeled provisional), plus machine-exact mass
    conservation.  (Khanwale-figure numbers are not obtainable at this coarse
    nondim CI resolution — internal baseline is the gate, per the brief.)"""
    b = _BASELINE["rt"]
    res = _march(C.RT_2D, b["level"], b["dt"], b["nsteps"], _rt_ic, +1)
    _assert_structural(res)
    for idx, ref in zip(b["spike_tip_y_at"], b["spike_tip_y"]):
        got = res["tip"][idx]
        if np.isnan(ref):
            continue
        assert _rel(got, ref) < 0.10, \
            f"RT spike tip @{idx} {got:.4f} vs baseline {ref:.4f}"


# ===========================================================================
# Structural diagnostics: energy non-increase + parasitic currents
# (no gravity, no source — the setting where the free energy IS monotone)
# ===========================================================================
def test_energy_nonincreasing_and_parasitic():
    """No-gravity relaxing drop: the discrete free energy (kinetic+interface+
    bulk) is non-increasing between consecutive unforced steps (tol 1e-10
    slack), mass drift is machine-exact, and parasitic currents on the static
    drop stay < 1e-3 (adapts the mirror's stationary-drop setup onto the Warp
    stepper)."""
    from diffsim.steppers.chns import CHNSStepper
    dm, mesh, cons = _make_dm(5)
    coords = mesh.node_coords
    case = replace(C.BUBBLE_RISE_RE35_WE10, rho_ratio=10.0)
    st = CHNSStepper(dm, case, dt=1e-3, mode="auto", Cn_override="2h",
                     gravity=False)
    r = np.sqrt((coords[:, 0] - 0.5) ** 2 + (coords[:, 1] - 0.5) ** 2)
    st.set_initial(-np.tanh((r - 0.25) / (st.Cn * np.sqrt(2.0))))
    m0 = st.mass_phi()
    E = [st.energy()["total"]]
    for _ in range(5):
        st.step()
        E.append(st.energy()["total"])
    dE = np.diff(E)
    assert np.all(dE < 1e-10), \
        f"free energy increased between unforced steps: max dE={dE.max():.3e}"
    assert abs(st.mass_phi() - m0) < 1e-8, \
        f"mass drift {abs(st.mass_phi() - m0):.3e} >= 1e-8"
    umax = float(np.abs(st.u).max())
    assert umax < 1e-3, f"parasitic current max|u|={umax:.3e} >= 1e-3"


# ===========================================================================
# MMS — sympy-forced design-order convergence (L2 order >= 1.8 for u and phi)
# ===========================================================================
def test_mms_convergence_order():
    """Manufactured-solution design-order gate (brief §4).  Fields:
      u=(sin pi x cos pi y, -cos pi x sin pi y) g(t),  p=cos pi x cos pi y g(t),
      phi=tanh((y-0.5-0.1 sin(2 pi x) g(t))/(sqrt2 Cn)),  g(t)=cos t (transient
      forcing available; the convergence GATE uses the STEADY g=1 setting so the
      BDF time error does not contaminate the very-smooth velocity order — see
      benchmarks/chns/mms.py).  Forcing is computed symbolically (sympy) and
      injected via src_fns (CH rows) + body_fn (NS body-force channel; enters
      r_mom for SUPG/PSPG consistency).  Assert L2 order >= 1.8 for u and phi
      over levels 4->5->6 at fixed Cn (Cn FIXED so the manufactured field is the
      same across levels)."""
    import math
    sympy = pytest.importorskip("sympy")            # noqa: F841
    from chns.mms import build_mms
    from diffsim.steppers.chns import CHNSStepper

    Re, We, Pe, rr, er, Cn = 10.0, 10.0, 100.0, 2.0, 2.0, 0.12

    def l2(level):
        dm, mesh, cons = _make_dm(level)
        coords = mesh.node_coords
        case = replace(C.BUBBLE_RISE_RE35_WE10, Re=Re, We=We, Pe=Pe, Cn=Cn,
                       rho_ratio=rr, eta_ratio=er, Fr=1.0)
        m = build_mms(case, Cn, Re, We, Pe, rr, er, gravity=False, steady=True)
        st = CHNSStepper(dm, case, dt=1e6, mode="auto", Cn_override=Cn,
                         gravity=False, tstep="bdf1",
                         src_fns=m["src_fns"], body_fn=m["body_fn"])
        st.bc_u_fn = m["u_fn"]
        st.bc_phi_fn = m["phi_fn"]
        st.bc_mu_fn = m["mu_fn"]
        # exact IC; huge dt => the steady spatial residual dominates and Newton
        # converges to the discrete steady state (pure spatial error).
        st.set_initial(m["phi_fn"](coords, 0.0), u0=m["u_fn"](coords, 0.0))
        st._mu = m["mu_fn"](coords, 0.0)
        st._p = m["p_fn"](coords, 0.0)
        st.phi_n = st._phi.copy()
        st.u_n = st._u.copy()
        for _ in range(4):
            st.step()
        h = st.h
        ue = m["u_fn"](coords, 0.0)
        phie = m["phi_fn"](coords, 0.0)
        eu = np.sqrt(np.sum((st.u - ue) ** 2) * h * h)
        ephi = np.sqrt(np.sum((st.phi - phie) ** 2) * h * h)
        return eu, ephi

    e = {lv: l2(lv) for lv in (4, 5, 6)}
    ou = [math.log(e[a][0] / e[b][0], 2) for a, b in ((4, 5), (5, 6))]
    op = [math.log(e[a][1] / e[b][1], 2) for a, b in ((4, 5), (5, 6))]
    print(f"[MMS] u errors {[f'{e[lv][0]:.3e}' for lv in (4,5,6)]} "
          f"orders {[f'{o:.2f}' for o in ou]}")
    print(f"[MMS] phi errors {[f'{e[lv][1]:.3e}' for lv in (4,5,6)]} "
          f"orders {[f'{o:.2f}' for o in op]}")
    assert min(ou) >= 1.8, f"u convergence order {ou} < 1.8"
    assert min(op) >= 1.8, f"phi convergence order {op} < 1.8"
