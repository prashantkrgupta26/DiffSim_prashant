"""OrgElMorph course - Physics P3: substrate/surface energy + boundary
conditions for binary Cahn-Hilliard.

Importable core.  Real organic films are cast on a SUBSTRATE (and dry
against AIR): the two interfaces are not neutral -- one component
usually prefers the substrate (or the free surface), and that
preference sets the VERTICAL stratification of the morphology, which in
a solar cell decides whether the right material sits next to the right
electrode.  This concept adds that physics through the boundary
condition, and lays out the BC taxonomy.

We use the (M, K)-generic brick in its simplest form
(diffsim.physics.multiphase.MultiPhaseStepper with M=1, K=0): one
conserved composition phi in (0,1) (the eliminated "solvent" is the
second component, phi_s = 1 - phi), Flory-Huggins bulk, natural no-flux
sides -- exactly binary Cahn-Hilliard -- PLUS an optional SUBSTRATE WALL
FREE ENERGY on the bottom edge.

WALL FREE ENERGY (the A2 term).  A surface energy on the substrate
Gamma_w,
    F_w = INT_{Gamma_w} f_w(phi) dS,   f_w(phi) = g phi + h phi^2,
adds a NATURAL boundary term to the chemical-potential equation: the
variational (total-variation) boundary condition is
    kap dphi/dn = -f_w'(phi) = -(g + 2 h phi)   on Gamma_w.
g < 0 makes f_w DECREASE with phi -> the wall ATTRACTS the component
(enrichment); g > 0 REPELS it (depletion); h tunes the curvature.  This
is a boundary term only -- it changes where material wants to be, not
the conservation law, so total mass is still conserved.

BC TAXONOMY (what you can put on a phase-field boundary):
  * NATURAL / no-flux (default):  grad mu . n = grad phi . n = 0.
    Nothing crosses; the wall is neutral.  Mass conserved.
  * WALL ENERGY (this concept): a Neumann-type condition on grad phi
    set by the surface energy f_w'(phi).  Still no MASS flux (the mu
    equation's flux BC is unchanged), so mass is conserved, but the
    equilibrium is stratified.
  * DIRICHLET: pin phi (or mu) to a prescribed value on the boundary
    (a reservoir / fixed-composition contact).  This DOES exchange mass
    and is used mainly for manufactured-solution verification -- see the
    Computational track's boundary-condition concept.
This concept demonstrates the first two on the same quench.
"""
import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.multiphase import MultiPhaseStepper


def build_mesh_dm(level=6, p=1, device="cuda:0"):
    """Uniform 2-D box, LATERALLY PERIODIC (x), non-periodic in y so the
    y=0 edge is a real substrate.  2^level cells per side."""
    tree = build_uniform(level, dim=2, periodic=(True, False))
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), device)
    return dm, mesh, cons


def _chi_aa(chi):
    """2x2 FULL-species chi (includes eliminated solvent, index 1)."""
    m = np.zeros((2, 2))
    m[0, 1] = m[1, 0] = chi
    return m


def run(dm, mesh, cons, wall_g=0.0, wall_h=0.0, chi=2.2, kappa=1e-3,
        phi0=0.5, amp=0.02, dt=2e-4, t_end=0.4, seed=5, dt_max=0.02,
        device="cuda:0", snap_times=None):
    """March a binary quench with an optional substrate wall energy.
    The wall free energy is the bounded QUADRATIC well
    f_w = g phi + h phi^2, whose minimum sets a preferred surface
    composition phi* = -g/(2h) (keep phi* strictly in (0,1) so the
    projection never clips -- clipping would inject mass).  wall_g < 0
    attracts the component to the y=0 substrate.  Low Onsager mobility
    (0.2) and a bounded well keep Newton at ~4 iterations/step.  Returns
    field snapshots and the vertical enrichment profile phi(y)."""
    if snap_times is None:
        snap_times = [0.0, t_end / 4, t_end]
    st = MultiPhaseStepper(
        dm, M=1, K=0, chi_aa=_chi_aa(chi), N=[1.0, 1.0],
        onsager=[[0.2]], kappa=[kappa], dt=dt,
        wall_g=[wall_g], wall_h=[wall_h], wall_face=(1, 0),
        newton_tol=1e-8, newton_max=30)
    rng = np.random.default_rng(seed)
    ic = phi0 + amp * rng.standard_normal(st.nfree)
    st.set_initial([lambda x: ic])

    # robust node->(ix, iy) map on the FULL (all-node) grid, so
    # grid[ix, iy] and "substrate row iy=0 = y minimum" are exact,
    # independent of the internal DOF ordering.
    fcoords = mesh.node_coords
    xs = np.unique(np.round(fcoords[:, 0], 12))
    ys = np.unique(np.round(fcoords[:, 1], 12))
    nx, ny = len(xs), len(ys)
    ix = np.searchsorted(xs, np.round(fcoords[:, 0], 12))
    iy = np.searchsorted(ys, np.round(fcoords[:, 1], 12))
    side = nx

    def grab():
        full = np.asarray(cons.T @ st.phi(0))
        g = np.zeros((nx, ny))
        g[ix, iy] = full
        return g                             # g[ix, iy]; iy=0 -> substrate

    snaps = {}
    tlog, wall_phi, bulk_phi = [], [], []

    def record():
        g = grab()
        tlog.append(st.t)
        wall_phi.append(float(g[:, 0].mean()))    # substrate row (y min)
        bulk_phi.append(float(g.mean()))

    record()
    if snap_times[0] <= 1e-12:
        snaps[0.0] = grab()
    next_snaps = [s for s in snap_times if s > 1e-12]
    st.dt = dt
    while st.t < t_end - 1e-12:
        st.march(t_end=min(st.t + 0.02, t_end), dt_max=dt_max,
                 max_steps=2000, dt_min=1e-8)
        record()
        while next_snaps and st.t >= next_snaps[0] - 1e-9:
            snaps[next_snaps.pop(0)] = grab()
    if t_end not in snaps:
        snaps[t_end] = grab()
    # vertical enrichment profile of the final state (lateral mean per y)
    gfinal = grab()
    yprofile = gfinal.mean(axis=0)           # mean over x, per y row
    return dict(snaps=snaps, side=side, ys=ys, yprofile=yprofile,
                t=np.array(tlog), wall_phi=np.array(wall_phi),
                bulk_phi=np.array(bulk_phi), wall_g=wall_g,
                wall_h=wall_h, chi=chi, phi0=phi0,
                mass_final=float(gfinal.mean()), mass0=phi0)
