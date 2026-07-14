"""OrgElMorph course - Physics P9: evaporation-INDUCED crystallization.

Importable core -- the hero concept, at tutorial scale.  It assembles the
whole story: a WET ternary film (P4/P5) DRIES (evaporation, P5), the
concentrating blend PHASE-SEPARATES (Cahn-Hilliard, P1), and once the
crystallizable species is concentrated enough its crystals NUCLEATE and
GROW (Allen-Cahn, P6) coupled back to composition (P7).  The full arc

    wet film  ->  phase separation  ->  nucleation  ->  crystalline film

is exactly how a real solution-cast organic solar cell forms.

We use MultiPhaseStepper at (M, K) = (2, 1) in FILM MODE: species 0 is a
crystallizable small molecule (fullerene-class), species 1 a polymer,
and the eliminated solvent evaporates.  The r14 bulk with a crystal-
contact penalty chi_ca gives a SOLUBILITY: crystallization is forbidden
below a local small-molecule fraction phi* (set by the undercooling vs
chi_ca), so seeds implanted in the WET film DISSOLVE, while the same
seeds implanted MID-DRYING -- where drying has concentrated the small
molecule above phi* and the continuing solvent loss keeps deepening the
quench -- GROW to a crystalline film.  That ordering (crystallization
strictly AFTER significant solvent loss) is the physics.

SEEDING IS PHYSICAL AND DELICATE (the S3b campaign lessons, ledger
2026-07-12): the implanted embryo radius r0 must clear the Gibbs-Thomson
critical radius r* with margin (subcritical embryos redissolve), and the
embryo amplitude must be near 1 (half-amplitude embryos halve the bulk
driving and double r*).  We use r0 with margin and psi ~ 0.95, and read
the TERMINAL crystalline state (not a mid-growth snapshot).

TUTORIAL SIMPLIFICATION (recorded honestly).  The production S3b config
uses the Vignes composition-singular mobility (mob="fastmode_n") with a
3-decade liquid->solid drop -- physically faithful but stiff (its deep-
quench Jacobian collapses the dt ladder).  For a tutorial we use a
CONSTANT Onsager mobility (keeping the same cuDSS direct solver as the
production run); the qualitative arc (dissolve-when-wet vs
grow-when-dry) is unchanged, but the quantitative drying-front sharpness
is softened.  See test_multiphase_s3.py for the production run.
"""
import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.multiphase import MultiPhaseStepper

# fullerene-class crystal energetics (2310.11844, T = 333 K nondim),
# as in the S3b production config.
DSIG_F, DH_F, TM_F = 2.6355, 1.3072, 558.0
N_F, N_P = 5.03, 87.0


def build_mesh_dm(level=5, p=1, device="cuda:0"):
    """Laterally periodic (x), non-periodic vertical (drying direction).
    Level 5 (32x32) matches the S3b production mesh and keeps the two-leg
    drying march to a few minutes with the cuDSS direct solver; level 6
    is available (about 4x slower)."""
    tree = build_uniform(level, dim=2, periodic=(True, False))
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), device)
    return dm, mesh, cons


def make_stepper(dm, T=333.0, k_e=0.1, chi_ca_val=1.6, dt=1e-3,
                 eps2=4e-3, seed=1011):
    """S3b-class ternary drying film, tutorial mobility (const) + splu.
    chi: fullerene-polymer 1.0 (Negi), fullerene-solvent 0.7248 (2310),
    polymer-solvent 0.3 (Wodo).  chi_ca on both fullerene contacts sets
    the crystal SOLUBILITY.  Blend starts dilute (85% solvent)."""
    chi_aa = np.zeros((3, 3))
    chi_aa[0, 1] = chi_aa[1, 0] = 1.0
    chi_aa[0, 2] = chi_aa[2, 0] = 0.7248
    chi_aa[1, 2] = chi_aa[2, 1] = 0.3
    chi_ca = np.zeros((3, 3))
    chi_ca[0, 1] = chi_ca[0, 2] = chi_ca_val
    st = MultiPhaseStepper(
        dm, M=2, K=1, chi_aa=chi_aa, chi_ac=chi_ca.T.copy(),
        chi_ca=chi_ca, N=[N_F, N_P, 1.0], mob="const",
        onsager=[[0.05, 0.0], [0.0, 0.05]], kappa=[2e-4, 2e-4],
        dsig=[DSIG_F], dh=[DH_F], Tm=[TM_F], eps2=[eps2], L_psi=[N_F],
        T=T, dt=dt, bulk="r14", b_reg=1e-3, newton_tol=1e-7,
        newton_max=50, linsolver="cudss", film=dict(k_e=k_e),
        clip_psi=False, noise_seed=seed)
    # missing-splat guard (S3b lesson): verify the crystal params
    # actually reached the stepper (a dropped kw silently defaults Tm=1
    # and turns dh(1-T/Tm) into a huge MELTING drive).
    assert abs(st.Tm[0] - TM_F) < 1e-9 and abs(st.dh[0] - DH_F) < 1e-9, \
        "crystal energetics did not reach the stepper"
    return st


def set_blend(st, phi_f0=0.10, phi_p0=0.05, amp=0.01, ic_seed=1011):
    rng = np.random.default_rng(ic_seed)
    icf = phi_f0 + amp * rng.standard_normal(st.nfree)
    icp = phi_p0 + amp * rng.standard_normal(st.nfree)
    st.set_initial([lambda x: icf, lambda x: icp],
                   [lambda x: np.zeros(len(x))],
                   [lambda x: np.zeros(len(x))])


def implant(st, ctrs, r0=0.2, w=0.03, psi_amp=0.95):
    """psi discs + theta markers, committed into x AND hist (the new
    state).  r0 = 0.2 clears the Gibbs-Thomson critical radius with
    margin; psi_amp ~ 1 keeps the embryo supercritical (S3b lessons)."""
    nd = st.ndof
    coords = st.free_coords
    psi = st.x[2 * st.M::nd]
    th = st.x[2 * st.M + 1::nd]
    for k, c in enumerate(ctrs):
        r = np.hypot(coords[:, 0] - c[0], coords[:, 1] - c[1])
        disc = psi_amp * 0.5 * (1.0 - np.tanh((r - r0) / w))
        m = disc > psi
        psi[m] = disc[m]
        th[r < r0 + 3 * w] = 0.3 + 0.4 * k
    st.hist = st.x.copy()
    st.hist2 = None


def sites(st, n_seeds=3, min_sep=0.35):
    """Small-molecule-richest, well-separated sites off the moving top."""
    coords = st.free_coords
    order = np.argsort(st.phi(0))[::-1]
    ctrs = []
    for i in order:
        c = coords[i]
        if c[1] > 0.85:
            continue
        if all(np.hypot(c[0] - q[0], c[1] - q[1]) >= min_sep
               for q in ctrs):
            ctrs.append((float(c[0]), float(c[1])))
        if len(ctrs) >= n_seeds:
            break
    return ctrs


def _grid_map(mesh):
    fc = mesh.node_coords
    xs = np.unique(np.round(fc[:, 0], 12))
    ys = np.unique(np.round(fc[:, 1], 12))
    ix = np.searchsorted(xs, np.round(fc[:, 0], 12))
    iy = np.searchsorted(ys, np.round(fc[:, 1], 12))
    return len(xs), len(ys), ix, iy


def phis_mean(st, cons):
    return float(np.mean(1.0 - sum(np.asarray(cons.T @ st.phi(i))
                                   for i in range(2))))


def crys_area(st, cons):
    return float(np.mean(np.asarray(cons.T @ st.psi(0)) > 0.5))


def run_arc(dm, mesh, cons, wet=False, t_implant=8.0, t_end=20.0,
            k_e=0.1, device="cuda:0", nsnaps=4):
    """The full arc.  March the drying film to t_implant, implant seeds,
    march on.  wet=True implants EARLY (t small, still solvent-rich) to
    show dissolution; wet=False implants mid-drying to show growth.
    Returns the drying/area histories, stage snapshots, terminal metrics.
    """
    # wet=True implants EARLY (solvent-rich, below solubility -> dissolve);
    # wet=False implants mid-drying (concentrated -> grow).
    t_imp = 1.0 if wet else t_implant
    st = make_stepper(dm, k_e=k_e)
    set_blend(st)
    nx, ny, ix, iy = _grid_map(mesh)

    def snap():
        gf = np.zeros((nx, ny)); gf[ix, iy] = np.asarray(cons.T @ st.phi(0))
        gpsi = np.zeros((nx, ny)); gpsi[ix, iy] = np.asarray(cons.T @ st.psi(0))
        return gf, gpsi, st.h_curr

    tlog, phislog, arealog = [], [], []

    def rec():
        tlog.append(st.t); phislog.append(phis_mean(st, cons))
        arealog.append(crys_area(st, cons))

    rec()
    snaps = {"wet": snap()}
    marchkw = dict(dt_max=0.02, max_steps=40000, dt_min=1e-10,
                   grow_iters=40, h_min=0.14, phis_stop=0.02)

    # leg 1: dry to the implant time, recording along the way
    checks = np.linspace(t_imp / 20, t_imp, 20)
    for tc in checks:
        r = st.march(t_end=tc, **marchkw)
        rec()
        if r != "t_end":
            break
    phis_imp = phis_mean(st, cons)
    ctrs = sites(st)
    implant(st, ctrs)
    a0 = crys_area(st, cons)
    snaps["implant"] = snap()

    # leg 2: continue drying/growing to t_end
    checks2 = np.linspace(st.t + (t_end - st.t) / 20, t_end, 20)
    reason = "t_end"
    for tc in checks2:
        r = st.march(t_end=tc, **marchkw)
        rec()
        if r != "t_end":
            reason = r
            break
    snaps["final"] = snap()
    psi_full = np.asarray(cons.T @ st.psi(0))
    return dict(t=np.array(tlog), phis=np.array(phislog),
                area=np.array(arealog), snaps=snaps, nx=nx, ny=ny,
                phis_implant=phis_imp, area_implant=a0,
                area_final=crys_area(st, cons),
                psi_max=float(psi_full.max()), ctrs=ctrs,
                reason=reason, t_final=st.t, h_final=st.h_curr, k_e=k_e)
