"""OrgElMorph course - Physics P5: ternary Cahn-Hilliard with SOLVENT
EVAPORATION (the drying film).

Importable core.  An organic film is cast from solution and DRIES: the
solvent leaves through the top surface, the film thins, and the solutes
concentrate until they demix.  Evaporation is therefore the CLOCK of
morphology formation -- how fast the solvent leaves sets how much time
the blend has to phase-separate before it vitrifies.  This concept adds
that moving, evaporating top surface and shows how the drying RATE
reshapes the final morphology.

We use the (M, K)-generic brick in FILM MODE
(diffsim.physics.multiphase.MultiPhaseStepper, M=2 solutes, K=0 no
crystal, film=dict(...)).  The Landau-mapped moving frame follows the
wodo_film formulation: the computational box [0,1]x[0,1] represents a
physical strip of shrinking height h(t), the solvent flux at the top

    K = SUM_i k_e_i * avg(phi_i^top)   (>= 0, frozen per step)

drives h(t) DOWN (dh/dt = -K), and each conserved solute gains a frame-
advection term plus a surface-enrichment flux so the solute CONTENT
h * INT phi_i is conserved exactly per step (nothing evaporates but the
solvent).

THE EVAPORATION RATE (Biot number).  k_e is the nondimensional
evaporation rate of the solvent (the Biot number Bi = k_e h0 / D_s in
the wodo convention; larger k_e = more volatile solvent / faster
drying).  The physics we demonstrate:
  * DRYING CURVE: the mean solvent fraction phi_s(t) falls; a larger Bi
    dries faster (steeper curve), a smaller Bi lingers.
  * MORPHOLOGY vs RATE: fast drying quenches the blend before it can
    coarsen (finer, more kinetically-trapped domains); slow drying
    gives phase separation time to coarsen (larger domains).  This is
    the central processing knob of solution-cast organic electronics.
"""
import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.multiphase import MultiPhaseStepper


def build_mesh_dm(level=6, p=1, device="cuda:0"):
    """Laterally periodic (x), non-periodic vertical (y = the drying
    direction; the top face recedes)."""
    tree = build_uniform(level, dim=2, periodic=(True, False))
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), device)
    return dm, mesh, cons


def _chi_aa(c_pf, c_ps, c_fs):
    """3x3 FULL-species chi (0=polymer, 1=fullerene, 2=solvent)."""
    m = np.zeros((3, 3))
    m[0, 1] = m[1, 0] = c_pf
    m[0, 2] = m[2, 0] = c_ps
    m[1, 2] = m[2, 1] = c_fs
    return m


def run(dm, mesh, cons, k_e=0.3, chi=(1.5, 0.3, 0.3), N=(5.0, 5.0, 1.0),
        onsager=(0.2, 0.0, 0.2), kappa=(3e-4, 3e-4), phi0=(0.22, 0.22),
        amp=0.01, dt=1e-3, dt_max=0.015, t_end=6.0, seed=7,
        phis_stop=0.1, h_min=0.15, device="cuda:0", nchecks=50):
    """March a drying ternary film at evaporation rate k_e.  Blend
    starts dilute (mostly solvent) and dries.  Returns drying curve
    phi_s(t), film height h(t), snapshots, and a domain-scale readout of
    the final morphology."""
    ons = np.array([[onsager[0], onsager[1]],
                    [onsager[1], onsager[2]]])
    st = MultiPhaseStepper(
        dm, M=2, K=0, chi_aa=_chi_aa(*chi), N=list(N), onsager=ons,
        kappa=list(kappa), dt=dt, mob="const",
        film=dict(k_e=k_e, h0=1.0), b_reg=1e-4,
        newton_tol=1e-8, newton_max=40, linsolver="cudss")
    rng = np.random.default_rng(seed)
    icp = phi0[0] + amp * rng.standard_normal(st.nfree)
    icf = phi0[1] + amp * rng.standard_normal(st.nfree)
    st.set_initial([lambda x: icp, lambda x: icf])

    # node -> (ix, iy) grid map (robust; iy=0 substrate, iy=-1 top)
    fc = mesh.node_coords
    xs = np.unique(np.round(fc[:, 0], 12))
    ys = np.unique(np.round(fc[:, 1], 12))
    nx, ny = len(xs), len(ys)
    ix = np.searchsorted(xs, np.round(fc[:, 0], 12))
    iy = np.searchsorted(ys, np.round(fc[:, 1], 12))

    def grids():
        p = np.asarray(cons.T @ st.phi(0))
        f = np.asarray(cons.T @ st.phi(1))
        gp = np.zeros((nx, ny)); gp[ix, iy] = p
        gf = np.zeros((nx, ny)); gf[ix, iy] = f
        return gp, gf

    def phis_mean():
        gp, gf = grids()
        return float((1.0 - gp - gf).mean())

    tlog, phis_log, h_log = [], [], []

    def record():
        tlog.append(st.t)
        phis_log.append(phis_mean())
        h_log.append(st.h_curr)

    record()
    gp0, gf0 = grids()
    snaps = {0.0: (gp0.copy(), gf0.copy(), st.h_curr)}
    checks = np.linspace(t_end / nchecks, t_end, nchecks)
    reason = "t_end"
    for tc in checks:
        r = st.march(t_end=tc, dt_max=dt_max, max_steps=4000,
                     dt_min=1e-9, grow_iters=25, h_min=h_min,
                     phis_stop=phis_stop)
        record()
        if r in ("phis_stop", "h_min", "dt_underflow"):
            reason = r
            break
    gpf, gff = grids()
    snaps["final"] = (gpf.copy(), gff.copy(), st.h_curr)

    # domain-scale readout: mean lateral spectral wavelength of the
    # polymer field (a coarse "how fine is the morphology" number).
    dom = _domain_scale(gpf)
    return dict(snaps=snaps, nx=nx, ny=ny, k_e=k_e,
                t=np.array(tlog), phis=np.array(phis_log),
                h=np.array(h_log), reason=reason, t_dry=st.t,
                h_final=st.h_curr, phis_final=phis_log[-1],
                domain_scale=dom,
                contrast=float(gpf.std()))


def _domain_scale(g):
    """Characteristic lateral domain size from the first spectral
    moment of the demeaned field (periodic x).  Smaller number = finer
    morphology.  Returned in units of box width."""
    a = g - g.mean()
    F = np.abs(np.fft.rfft(a, axis=0)) ** 2       # along periodic x
    P = F.mean(axis=1)                            # average over y
    k = np.arange(P.shape[0])
    P[0] = 0.0
    if P.sum() <= 0:
        return float("nan")
    kbar = (k * P).sum() / P.sum()
    return float(1.0 / max(kbar, 1e-9))
