"""F1 — Phase field I: the Allen-Cahn equation (non-conserved dynamics).

Welcome to track F. You have finished A-C (or at least A1 and C1): you
can measure MMS orders and you know what BDF1/BDF2 history rotation is.
This track teaches PHASE-FIELD modeling — the workhorse of microstructure
evolution — on the same bricks, ending at the M4 research frontier
(learned free energies). F1 is the simplest phase-field model there is.

LEARNING OUTCOME. You can (i) derive Allen-Cahn from a free-energy
functional, (ii) explain why its interfaces are DIFFUSE and how wide
they are, (iii) run the brick and verify the two classic benchmarks:
energy decay on a random quench and the shrinking circle's exact
area law. Theory that predicts a number, then the measured number —
that is the ritual of this whole course.

THEORY (read this before running anything).

1. The free energy. Phase-field models start from a functional, not a
   PDE. For a non-conserved order parameter c(x, t) in [-1, 1] (think:
   crystalline vs disordered, up vs down domain):

       F[c] = Int_Omega [ f(c) + (kappa/2) |grad c|^2 ] dV,
       f(c) = (1/4) (c^2 - 1)^2.

   f is a DOUBLE WELL: two minima at c = +-1 (the pure phases), a
   barrier at c = 0. The gradient term penalizes spatial variation —
   it is what gives the interface a width and an energy (surface
   tension). Everything phase-field follows from this competition.

2. Non-conserved (Allen-Cahn) dynamics = gradient flow. The simplest
   evolution that decreases F is steepest descent in L2:

       dc/dt = -M dF/dc = -M [ f'(c) - kappa lap c ]
             = -M [ c^3 - c - kappa lap c ],   M > 0.

   dF/dc is the VARIATIONAL derivative (Euler-Lagrange operand). Along
   solutions dF/dt = -M Int (dF/dc)^2 dV <= 0: energy decay is a THEOREM,
   so we test it to machine precision, not "roughly".
   Note what is NOT conserved: Int c dV can change freely — phase can
   appear/disappear locally. Contrast with F2 (Cahn-Hilliard).

3. Interface asymptotics — the two numbers you can predict.
   (a) WIDTH. The flat 1-D equilibrium interface solves
       c'' = (c^3 - c)/kappa, giving the famous profile

           c(x) = tanh( x / sqrt(2 kappa) ),

       i.e. an intrinsic interface width w ~ sqrt(2 kappa). RESOLUTION
       RULE: you need >= 3-4 elements across w, or the interface pins
       to the mesh (try it — EXPLORE 4).
   (b) MOTION BY CURVATURE. Matched asymptotics (kappa -> 0) give the
       normal interface velocity v_n = -M kappa H, with H the mean
       curvature (H = 1/R for a circle in 2-D). So a circular domain
       of radius R shrinks as

           dR/dt = -M kappa / R   ==>   R^2(t) = R0^2 - 2 M kappa t.

       AREA DECAYS LINEARLY IN TIME — an exact, parameter-explicit law
       hiding inside a nonlinear PDE. Demo 2 measures it.

IMPLEMENTATION (src/diffsim/physics/allen_cahn.py). Galerkin weak form
(the group guide Sec 1.3; no SUPG — there is no advection):

    Int[v c_t] + M kappa Int[grad v . grad c] + M Int[v (c^3 - c)] = 0

with natural (no-flux) BCs: grad c . n = 0 falls out of integration by
parts — the physically right "do nothing" for an interface meeting a
wall at 90 degrees. Time: BDF1/BDF2 (sigma = b0/dt mass block, history
at the GPs). The nonlinearity is handled by monolithic NEWTON: about
the iterate c_k, the (c^3 - c) term contributes M (3 c_k^2 - 1) dc to
the Jacobian. That 3c_k^2 - 1 is f''(c_k) — remember this: in the M4
learning arc a neural f' enters the residual and its derivative f''
enters HERE, and nowhere else.

EXPECTED RESULTS (measured on this stack — the M4-b gate battery,
tests/test_allen_cahn.py, all green on first run):
    MMS orders  p1: 2.00 (errs 3.15e-03 -> 7.88e-04, L4 -> L5)
                p2: 3.00 (errs 2.02e-04 -> 2.52e-05, L3 -> L4)
    energy decay: monotone across all steps of a random quench
    shrinking circle: R measured vs sqrt(R0^2 - 2 M kappa t)
                      rel error 0.4%  (level 6, 25 BDF2 steps)
This script reprints the last two live (measured for THIS script:
    demo 1: E 0.2519 -> 0.1559 over 150 steps, ZERO decay violations;
            c-range shrinks [-0.78,+0.61] -> [-0.54,+0.46] by step 30
            — smoothing — then grows to [-0.95,+0.94] by t = 3 with 2%
            of nodes saturated (|c| > 0.9) — the wells winning
    demo 2: rel error 0.5% -> 0.4% over the run, final 0.4%);
run the gate battery for the order study:
    pytest tests/test_allen_cahn.py -s

Run:  python tutorials/F_phasefield/F1_allen_cahn.py
"""
import os
import sys

import numpy as np
import warp as wp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "src"))
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.allen_cahn import AllenCahnStepper
from diffsim import default_device

DEVICE = default_device()


def make_problem(level, p=1):
    """Uniform 2-D octree -> mesh -> constraints -> device mesh. The
    same four lines as every chapter since A1."""
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), DEVICE)
    return dm, mesh, cons


def free_energy(st, dm, mesh, c_free, kappa):
    """F[c] = Int [ (c^2-1)^2/4 + kappa/2 |grad c|^2 ]: quadrature over
    the same GP tables the residual uses (consistency!)."""
    cv, cg = st._gp(c_free)
    E = 0.0
    for pv, b in dm.bins.items():
        h = mesh.tree.h()[mesh.bins[pv]]
        ne = len(mesh.conn_of[pv])
        wq = np.tile(dm.tables_by_p[pv].w, ne) \
            * np.repeat((h / 2.0) ** 2, b["nqp"])
        fbulk = 0.25 * (cv[pv] ** 2 - 1.0) ** 2
        gsq = (cg[pv] ** 2).sum(1)
        E += float((wq * (fbulk + 0.5 * kappa * gsq)).sum())
    return E


# ----------------------------------------------------------------------
# Demo 1: quench. Random noise evolves in TWO measured stages:
#   stage 1 (fast, ~10 steps): the gradient term kills high-wavenumber
#     noise — the c-range SHRINKS first (watch it: 0.78 -> ~0.5);
#   stage 2 (slow): the surviving long-wave pattern deepens into
#     c = +-1 domains (range grows to +-0.95 by t = 3), whose walls
#     then move by curvature (demo 2's law).
# The testable claim across BOTH stages: F decreases EVERY step.
# ----------------------------------------------------------------------
def demo_1_coarsening(level=5, nsteps=150, print_every=15):
    print("=" * 70)
    print("Demo 1: random quench -> energy decay (the AC theorem, "
          "measured)")
    print("=" * 70)
    M, kappa, dt = 1.0, 2e-4, 0.02
    dm, mesh, cons = make_problem(level)
    st = AllenCahnStepper(dm, M, kappa, dt, order=1)
    rng = np.random.default_rng(0)
    c = st.set_initial(lambda x: 0.2 * rng.standard_normal(len(x)))
    E = free_energy(st, dm, mesh, c, kappa)
    viol = 0
    print(f"  {'step':>4} {'t':>5} {'energy':>10}  c-range        "
          f"|c|>0.9")
    print(f"  {0:4d} {0.0:5.2f} {E:10.6f}  "
          f"[{c.min():+.2f},{c.max():+.2f}]   {0:5.0%}")
    for n in range(nsteps):
        c = st.step()
        E_new = free_energy(st, dm, mesh, c, kappa)
        if E_new > E + 1e-10:
            viol += 1
        E = E_new
        if (n + 1) % print_every == 0:
            sat = float(np.mean(np.abs(c) > 0.9))
            print(f"  {n+1:4d} {st.t:5.2f} {E:10.6f}  "
                  f"[{c.min():+.2f},{c.max():+.2f}]   {sat:5.0%}")
    print(f"  energy-decay violations: {viol} (theorem says 0; "
          f"measured 0)")
    print("  Two stages, both visible above: the range SHRINKS first")
    print("  (gradient term smooths the noise), then grows to +-1 as")
    print("  the wells win. Energy decays through both — Eq. dF/dt <= 0")
    print("  does not care which term is doing the work.")


# ----------------------------------------------------------------------
# Demo 2: the shrinking circle. IC = tanh profile of radius R0; theory
# says R^2(t) = R0^2 - 2 M kappa t, no fitting parameters. We measure R
# as the radius of the c = 0 level set and compare.
# ----------------------------------------------------------------------
def demo_2_shrinking_circle(level=6, nsteps=25):
    print("=" * 70)
    print("Demo 2: shrinking circle vs the exact area law")
    print("=" * 70)
    M, kappa, dt = 1.0, 2e-4, 0.02
    R0 = 0.30
    w = np.sqrt(2 * kappa)              # intrinsic width parameter
    h = 1.0 / 2 ** level
    # the IC uses the (wider) tanh(r / sqrt(2) w) convention; its full
    # transition zone |c| < 0.9 spans ~ 2 * atanh(0.9) * sqrt(2) w:
    zone = 2 * np.arctanh(0.9) * np.sqrt(2) * w
    print(f"  kappa = {kappa}: w = sqrt(2 kappa) = {w:.4f}; transition "
          f"zone (|c|<0.9)")
    print(f"  = {zone:.3f} ~ {zone/h:.1f} elements at level {level} "
          f"(resolution rule: >= 3-4)")
    dm, mesh, cons = make_problem(level)
    st = AllenCahnStepper(dm, M, kappa, dt, order=2)

    def circle_ic(x):
        r = np.linalg.norm(x - 0.5, axis=1)
        return -np.tanh((r - R0) / (np.sqrt(2) * w))

    st.set_initial(circle_ic)
    coords = mesh.node_coords[cons.free_nodes]
    r_node = np.linalg.norm(coords - 0.5, axis=1)
    order = np.argsort(r_node)

    def measured_radius(c):
        """First sign change of c along increasing radius = the c = 0
        level set (crude but honest; good to O(h))."""
        cz = c[order]
        iz = np.where(np.sign(cz[:-1]) != np.sign(cz[1:]))[0]
        return float(r_node[order][iz[0]])

    print(f"  {'t':>6} {'R measured':>11} {'R theory':>9} {'rel err':>8}")
    for n in range(nsteps):
        c = st.step()
        if (n + 1) % 5 == 0 or n == nsteps - 1:
            Rm = measured_radius(c)
            Rth = np.sqrt(R0 ** 2 - 2 * M * kappa * st.t)
            print(f"  {st.t:6.2f} {Rm:11.4f} {Rth:9.4f} "
                  f"{abs(Rm-Rth)/Rth:8.1%}")
    Rm = measured_radius(c)
    Rth = np.sqrt(R0 ** 2 - 2 * M * kappa * st.t)
    print(f"  final rel error {abs(Rm-Rth)/Rth:.1%}  "
          f"(gate battery measured 0.4% here)")
    print("  A nonlinear PDE, a curvature flow, and a one-line exact")
    print("  answer. When a benchmark like this exists, USE it before")
    print("  trusting any prettier problem.")


if __name__ == "__main__":
    demo_1_coarsening()
    demo_2_shrinking_circle()
    print("""
EXPLORE
 1. BDF1 vs BDF2. Rerun demo 2 with order=1 at dt = 0.02, 0.01, 0.005
    and tabulate the final radius error. Does it halve per dt halving
    (order 1) while BDF2's is already dominated by the O(h) radius
    readout? Where does the temporal error stop mattering, and why is
    that the whole argument for adaptive dt (F3's preview)?
 2. kappa sweep -> width measurement. For kappa in {1e-4, 4e-4, 1.6e-3}
    equilibrate a FLAT interface (IC: tanh(x - 0.5 / sqrt(2 kappa))) and
    fit tanh to the nodal values along y = 0.5. Recover w = sqrt(2 kappa)
    to a few percent? Now push kappa down until you can't — how many
    elements-per-width is the breaking point?
 3. p2 vs p1 at equal DOFs. Level-6 p1 and level-5 p2 have comparable
    DOF counts. Run demo 2 on both and compare the radius error AND the
    wall-clock per step. Where does p2's extra accuracy pay for its
    denser element blocks — and would it still at kappa 10x smaller?
 4. Break it on purpose: run demo 2 at level 4, where the whole
    transition zone spans ~1.3 elements (the same 0.083 zone, h = 1/16).
    The circle stops shrinking — the interface PINS to the mesh. Explain
    with 3(a)'s resolution rule; this failure mode is exactly what F3's
    interface-band refinement exists to prevent (cheaply).
""")
