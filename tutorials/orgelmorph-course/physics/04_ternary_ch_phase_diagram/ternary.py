"""OrgElMorph course - Physics P4: ternary Cahn-Hilliard and the Gibbs
phase diagram.

Importable core.  Real cast blends are (at least) THREE components:
two solutes -- a polymer/small-molecule donor and a fullerene acceptor
-- plus a solvent.  With two independent conserved compositions the
free-energy landscape is a SURFACE over the Gibbs triangle, and demixing
follows TIE-LINES on that triangle rather than the single axis of the
binary case.  This concept runs the coupled ternary Cahn-Hilliard system
and visualizes the composition trajectory on the ternary phase diagram
-- the natural way to read multi-component morphology.

We use the (M, K)-generic brick with M=2 solutes and K=0 (no crystal):
diffsim.physics.multiphase.MultiPhaseStepper.  Fields (phi_1, mu_1,
phi_2, mu_2) with the solvent eliminated (phi_s = 1 - phi_1 - phi_2),
Flory-Huggins exchange chemical potentials, a symmetric Onsager mobility
coupling the two solute fluxes, and gradient penalties kap_i.  (This is
the same physics as diffsim.physics.ternary_ch.TernaryCHStepper; we use
the multiphase brick so the fast cuDSS solver and the reject ladder are
available.)  The interaction matrix chi_aa[i,j] holds the pairwise
Flory chi (index 2 = the eliminated solvent): chi_aa[0,1] = solute-
solute, chi_aa[0,2]/chi_aa[1,2] = the two solute-solvent chis.

THE GIBBS TRIANGLE.  Each mesh node has a composition (phi1, phi2, phis)
with phi1+phi2+phis = 1 -- a point in the 2-simplex, drawn in an
equilateral triangle (barycentric coordinates).  The BULK average is
conserved (it never moves), but the CLOUD of local compositions starts
as a tight blob at the initial point and, as the blend demixes, spreads
along a tie-line toward the two coexisting phases.  Watching that cloud
open up IS watching ternary phase separation.
"""
import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.multiphase import MultiPhaseStepper


def build_mesh_dm(level=6, p=1, device="cuda:0"):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), device)
    return dm, mesh, cons


def _chi_aa(c12, c1s, c2s):
    """3x3 FULL-species chi (0, 1 solutes; 2 = eliminated solvent)."""
    m = np.zeros((3, 3))
    m[0, 1] = m[1, 0] = c12
    m[0, 2] = m[2, 0] = c1s
    m[1, 2] = m[2, 1] = c2s
    return m


def bary_to_xy(phi1, phi2):
    """Map (phi1, phi2, phis=1-phi1-phi2) to 2-D triangle coordinates.
    Vertices: solvent=(0,0), phi1=(1,0), phi2=(0.5, sqrt3/2)."""
    phi1 = np.asarray(phi1); phi2 = np.asarray(phi2)
    x = phi1 * 1.0 + phi2 * 0.5
    y = phi2 * (np.sqrt(3.0) / 2.0)
    return x, y


def run(dm, mesh, cons, chi=(3.5, 1.0, 0.6), mobility=0.2,
        kappa=(6e-4, 6e-4), phi0=(0.35, 0.35), amp=0.02, dt=5e-4,
        t_end=0.6, dt_max=0.02, seed=6, device="cuda:0",
        snap_times=None):
    # spinodal check: at phi0 the 2x2 exchange Hessian
    # [[1/p1+1/ps-2c1s, 1/ps+c12-c1s-c2s], [., 1/p2+1/ps-2c2s]] must be
    # indefinite (det<0) for the blend to demix -- chi_12=3.5 at
    # (0.35,0.35) gives det<0 (see the chapter).
    """March a ternary spinodal quench.  Returns per-node composition
    clouds at snapshot times plus the (conserved) mean trajectory."""
    if snap_times is None:
        snap_times = [0.0, t_end / 8, t_end / 3, t_end]
    ons = mobility * np.eye(2)
    st = MultiPhaseStepper(
        dm, M=2, K=0, chi_aa=_chi_aa(*chi), N=[1.0, 1.0, 1.0],
        onsager=ons.tolist(), kappa=list(kappa), dt=dt, bulk="p1",
        newton_tol=1e-8, newton_max=40, linsolver="cudss")
    rng = np.random.default_rng(seed)
    ic1 = phi0[0] + amp * rng.standard_normal(st.nfree)
    ic2 = phi0[1] + amp * rng.standard_normal(st.nfree)
    st.set_initial([lambda x: ic1, lambda x: ic2])

    side = int(round(np.sqrt(len(mesh.node_coords))))

    def clouds():
        p1 = np.asarray(cons.T @ st.phi(0))
        p2 = np.asarray(cons.T @ st.phi(1))
        return p1, p2

    snaps = {}
    tlog, m1, m2 = [], [], []

    def record():
        p1, p2 = clouds()
        tlog.append(st.t)
        m1.append(float(p1.mean())); m2.append(float(p2.mean()))

    record()
    if snap_times[0] <= 1e-12:
        p1, p2 = clouds()
        snaps[0.0] = (p1.copy(), p2.copy())
    pending = [s for s in snap_times if s > 1e-12]
    checks = np.linspace(t_end / 40, t_end, 40)
    for tc in checks:
        st.march(t_end=tc, dt_max=dt_max, max_steps=2000, dt_min=1e-8,
                 grow_iters=20)
        record()
        while pending and st.t >= pending[0] - 1e-9:
            p1, p2 = clouds()
            snaps[pending.pop(0)] = (p1.copy(), p2.copy())
    if t_end not in snaps:
        p1, p2 = clouds()
        snaps[t_end] = (p1.copy(), p2.copy())

    p1f, p2f = clouds()
    spread0 = float(np.std(snaps[0.0][0]))
    spreadf = float(np.std(p1f))
    # tie-line readout: split the final field at the median phi1
    med = np.median(p1f)
    lo = p1f < med
    phaseA = (float(p1f[lo].mean()), float(p2f[lo].mean()))
    phaseB = (float(p1f[~lo].mean()), float(p2f[~lo].mean()))
    return dict(snaps=snaps, side=side, chi=chi, phi0=phi0,
                t=np.array(tlog), mean1=np.array(m1),
                mean2=np.array(m2), spread0=spread0, spreadf=spreadf,
                phaseA=phaseA, phaseB=phaseB,
                mass1_drift=abs(m1[-1] - m1[0]),
                mass2_drift=abs(m2[-1] - m2[0]))
