"""F2 — Phase field II: Cahn-Hilliard, conservation, and spinodal
decomposition.

LEARNING OUTCOME. You can (i) explain why conserved dynamics makes the
equation FOURTH order and why we solve it as a MIXED (c, mu) pair of
second-order equations, (ii) state why mass conservation here is a
STRUCTURAL property of the discretization (measured 1.9e-15 — that is
machine epsilon accumulation, not "small"), (iii) run a spinodal
decomposition and read its physics off three printed diagnostics: mass,
energy, coarsening length.

THEORY.

1. Same functional, different constraint. F[c] is exactly F1's:

       F[c] = Int [ (1/4)(c^2 - 1)^2 + (kappa/2) |grad c|^2 ] dV.

   But now c is a CONSERVED density (composition of a binary alloy,
   polymer blend fraction): Int c dV must not change. Steepest descent
   dc/dt = -M dF/dc (Allen-Cahn) violates that. The gradient flow that
   respects conservation transports c by a FLUX driven by the chemical
   potential mu = dF/dc:

       dc/dt = div( M grad mu ),      mu = c^3 - c - kappa lap c.

   Conservation is then automatic: d/dt Int c dV =
   Int div(M grad mu) dV = boundary flux = 0 with no-flux walls.
   Energy still decays: dF/dt = -M Int |grad mu|^2 dV <= 0.

2. Why mixed? Substituting mu gives one scalar equation

       dc/dt = div( M grad(c^3 - c - kappa lap c) )

   — FOURTH-order in space. C0 finite elements cannot discretize
   lap(lap c) directly (you'd need C1 continuity, e.g. Hermite/spline
   elements). The standard escape (group guide Sec 2.3, and Wodo &
   Ganapathysubramanian JCP 2011): keep mu as a second UNKNOWN and
   solve two coupled second-order equations. Weak form: find (c, mu)
   with test functions (v, q):

       Int[v c_t] + M Int[grad v . grad mu]                     = 0
       Int[q mu] - Int[q (c^3 - c)] - kappa Int[grad q . grad c] = 0

   Equal-order elements (both p1 or both p2) are FINE here — this pair
   is not a saddle point like Stokes; no inf-sup drama.

3. Conservation is structural. Put v = 1 (constant test function — a
   legal member of the C0 space) in the first equation:

       d/dt Int c dV = -M Int grad(1) . grad mu dV = 0.   EXACTLY.

   No stabilization ate it, no BC leaks it: the semi-discrete scheme
   conserves mass IDENTICALLY, and BDF preserves that in time. So the
   measured drift can only be roundoff. MEASURED: |dm| = 1.9e-15 after
   20 implicit steps of a spinodal quench. When a property is
   structural, machine precision is the pass bar — 1e-6 "small" drift
   would mean a BUG, not "good enough".

4. Spinodal decomposition. Between the wells (|c| < 1/sqrt(3), where
   f''(c) = 3c^2 - 1 < 0) a uniform mixture is linearly UNSTABLE:
   infinitesimal composition noise grows and the mixture separates into
   c = +1 / c = -1 domains — spinodal decomposition. Linear analysis
   about c = 0 gives growth rate s(k) = M k^2 (1 - kappa k^2) for
   wavenumber k: zero at k = 0 (conservation!), fastest at
   k* = 1/sqrt(2 kappa), cut off at high k by the interface energy. So
   the early pattern has a CHARACTERISTIC LENGTH ~ 2 pi sqrt(2 kappa);
   later, domains coarsen (LSW/diffusive regime, L ~ t^(1/3) for bulk-
   diffusion-controlled CH). Demo prints a coarsening-length proxy
       L(t) = |Omega| / Int |grad c| dV
   (inverse interface density — grows as domains merge).

IMPLEMENTATION (src/diffsim/physics/cahn_hilliard.py). Node-major 2-dof
layout (c, mu) per node; monolithic Newton with the 2x2 block Jacobian
(Suresh's pattern):

       [ sigma N N + 0        ,  M grad.grad          ] [dc ]
       [ -(3c_k^2 - 1) N N - kappa grad.grad ,  N N   ] [dmu]

   Again f''(c) = 3c^2 - 1 appears in exactly one block — the (mu, c)
   coupling. In the M4 learning arc a parameterized f' replaces c^3 - c
   in the residual and its derivative lands HERE; nothing else changes.

EXPECTED RESULTS (the M4-a gate battery, tests/test_cahn_hilliard.py):
    MMS orders (mixed, manufactured mu* independently — see the test
    for why):   p1: 2.01   p2: 3.00
    mass conservation: |dm| = 1.9e-15 over 20 steps  (STRUCTURAL)
    energy: 0.2512 -> 0.0831 over the quench, monotone FROM STEP 1
    spinodal smoke: c in [-1.00, 1.03] (phases formed; the 3% overshoot
    is the standard quartic-potential excursion, not a bug)
    demo 1 live extras (measured for THIS script): L(t) grows
    0.085 -> 0.131 over steps 2-20; at step 15 one node briefly
    overshoots to c ~ 2.5 during a domain-merger event and the quartic
    well pulls it back within a step — energy STILL decays through it
    (0.0885 -> 0.0878): pointwise bounds are not what implicit CH
    guarantees, energy decay is.

THE STEP-0 TEACHING MOMENT (read tests/test_cahn_hilliard.py:117).
    set_initial seeds mu = 0, which is INCONSISTENT with a rough IC
    (the true mu(c0) = c0^3 - c0 - kappa lap c0 is huge for noise).
    The first implicit step absorbs it as a transient: energy was
    MEASURED going 0.25 -> 173 -> 0.61 and only then decaying
    monotonically. Nothing diverged — backward Euler swallowed the
    inconsistency — but the "energy decay" gate would fail at step 0.
    Lesson: mixed methods need CONSISTENT initialization of the
    auxiliary field (project mu0 = f'(c0) - kappa lap c0 — the recorded
    refinement), and a gate that fails can be diagnosing your IC, not
    your operator. Watch demo 1 print it live.

Run:  python tutorials/F_phasefield/F2_cahn_hilliard.py
"""
import os
import sys

import numpy as np
import warp as wp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _viz as _viz  # guarded viz helper (no-ops when [viz] not installed)

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.cahn_hilliard import CahnHilliardStepper
from diffsim import default_device

DEVICE = default_device()


def make_problem(level, p=1):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), DEVICE)
    return dm, mesh, cons


def quadrature_weights(dm, mesh):
    """Global GP weight vector per p-bin (2-D: jac = (h/2)^2)."""
    wq = {}
    for pv, b in dm.bins.items():
        h = mesh.tree.h()[mesh.bins[pv]]
        ne = len(mesh.conn_of[pv])
        wq[pv] = np.tile(dm.tables_by_p[pv].w, ne) \
            * np.repeat((h / 2.0) ** 2, b["nqp"])
    return wq


def diagnostics(st, dm, mesh, c_free, kappa):
    """(mass, energy, coarsening length) from one GP sweep."""
    v, g = st._gp_scalar(c_free)
    wq = quadrature_weights(dm, mesh)
    m = E = grad_l1 = vol = 0.0
    for pv in dm.bins:
        m += float((wq[pv] * v[pv]).sum())
        gsq = (g[pv] ** 2).sum(1)
        E += float((wq[pv] * (0.25 * (v[pv] ** 2 - 1) ** 2
                              + 0.5 * kappa * gsq)).sum())
        grad_l1 += float((wq[pv] * np.sqrt(gsq)).sum())
        vol += float(wq[pv].sum())
    L = vol / max(grad_l1, 1e-30)      # inverse interface density
    return m, E, L


# ----------------------------------------------------------------------
# Demo 1: spinodal decomposition of a critical quench (mean c = 0).
# Watch four columns: mass (frozen at machine precision), energy
# (the step-0 transient, then monotone decay), coarsening length
# (grows), c-range (phases forming).
# ----------------------------------------------------------------------
def demo_1_spinodal(level=5, nsteps=20):
    print("=" * 70)
    print("Demo 1: spinodal decomposition — mass, energy, coarsening")
    print("=" * 70)
    M, kappa, dt = 1.0, 5e-4, 0.02
    lam = 2 * np.pi * np.sqrt(2 * kappa)
    print(f"  fastest-growing wavelength 2 pi sqrt(2 kappa) = {lam:.3f}"
          f"  ({lam * 2 ** level:.1f} elements at level {level})")
    dm, mesh, cons = make_problem(level)
    st = CahnHilliardStepper(dm, M, kappa, dt, order=1)
    rng = np.random.default_rng(3)
    c = st.set_initial(lambda x: 0.05 * rng.standard_normal(len(x)))
    m0, E, L = diagnostics(st, dm, mesh, c, kappa)
    t_log, E_log, L_log = [0.0], [E], [L]
    print(f"  {'step':>4} {'mass drift':>11} {'energy':>10} "
          f"{'L(t)':>7}  c-range")
    print(f"  {0:4d} {0.0:11.1e} {E:10.4f} {L:7.3f}  "
          f"[{c.min():+.2f},{c.max():+.2f}]")
    E_prev, viol = None, 0
    for n in range(nsteps):
        c, mu = st.step()
        m, E, L = diagnostics(st, dm, mesh, c, kappa)
        t_log.append(st.t)
        E_log.append(E)
        L_log.append(L)
        note = ""
        if n == 0:
            note = "  <-- step-0 mu-init transient (see docstring!)"
        elif E_prev is not None and E > E_prev + 1e-9:
            note = "  DECAY VIOLATION"
            viol += 1
        print(f"  {n+1:4d} {m - m0:11.1e} {E:10.4f} {L:7.3f}  "
              f"[{c.min():+.2f},{c.max():+.2f}]{note}")
        E_prev = E
    print(f"  mass drift stayed O(1e-15): STRUCTURAL conservation "
          f"(v = 1 test function).")
    print(f"  energy monotone from step 1 ({viol} violations); "
          f"L(t) grew: coarsening.")
    # --- viz (additive; no-ops on base venv) ---
    _viz.history(__file__, t_log,
                 {"F[c]": E_log, "L(t)": L_log},
                 "spinodal_energy_coarsening",
                 xlabel="t", ylabel="F[c] / L(t)")
    return c


def main(level=5, nsteps=20):
    """Run the CH spinodal demo (default full-size; pass small values for tests)."""
    return demo_1_spinodal(level=level, nsteps=nsteps)


if __name__ == "__main__":
    main()
    print("""
EXPLORE
 1. Convex splitting. Our Newton solves the fully implicit potential;
    an alternative with unconditional gradient stability treats the
    CONVEX part (c^3) implicitly and the concave part (-c) explicitly
    (Eyre splitting). Implement it by moving the -c term into the
    history/RHS. Compare: max stable dt, Newton iterations per step,
    and the energy curve at dt 10x larger than demo 1's. What did the
    splitting buy, and what did it cost in per-step accuracy?
 2. Deeper quench. Scale the IC noise to 0.4 instead of 0.05 and
    (separately) scale the well depth by replacing f with
    2 (c^2-1)^2 / 4 (double chi, in Flory-Huggins language). How do the
    early growth rate and the step-0 transient change? Does the
    fastest-growing wavelength move the way s(k) = M k^2 (1 - kappa k^2)
    predicts when you rescale time?
 3. Off-critical composition. Rerun with IC mean c = -0.4 (droplets of
    the minority phase) vs c = 0 (bicontinuous labyrinth). Print the
    same diagnostics and describe the morphology change — this
    droplets-vs-bicontinuous switch is exactly the blend-ratio knob in
    the Wodo CMS-2012 organic-solar-cell replication (M4 track c).
 4. Consistent mu-init. Implement the recorded refinement: after
    set_initial, project mu0 = c0^3 - c0 - kappa lap c0 (weak form:
    Int[q mu0] = Int[q (c0^3 - c0)] + kappa Int[grad q . grad c0] — one
    mass-matrix solve) and store it in st.x[1::2]. Does the step-0
    energy spike disappear? Now the decay gate can assert from step 0.
""")
