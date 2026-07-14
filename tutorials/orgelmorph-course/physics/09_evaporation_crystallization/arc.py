"""OrgElMorph course - Physics P9: evaporation-CONDITIONED embryo growth.

Importable core -- the hero concept, at tutorial scale.  It assembles the
whole story: a WET ternary film (P4/P5) DRIES (evaporation, P5), the
concentrating blend PHASE-SEPARATES (Cahn-Hilliard, P1), and once the
crystallizable species is concentrated enough an IMPLANTED crystal embryo
can GROW (Allen-Cahn, P6) coupled back to composition (P7).  The arc

    wet film  ->  phase separation  ->  embryo growth  ->  crystalline film

is how a real solution-cast organic solar cell forms.

NAMING (a P9 correction).  The default runs do NOT nucleate crystals from
thermal noise -- they IMPLANT a supercritical embryo and ask whether the
drying-conditioned local composition lets it GROW or forces it to
DISSOLVE.  So this is "evaporation-CONDITIONED embryo growth", not
spontaneous "nucleation".  Genuine noise-driven nucleation (P8) is
available as an ADVANCED mode (make_stepper(noise_psi=...) +
run_arc(..., noise_psi=...)); it is slower and stochastic, so the checked
tutorial uses the deterministic implant.

We use MultiPhaseStepper at (M, K) = (2, 1) in FILM MODE: species 0 is a
crystallizable small molecule (fullerene-class), species 1 a polymer,
and the eliminated solvent evaporates.  The r14 bulk with a crystal-
contact penalty chi_ca gives a SOLUBILITY threshold phi* in the LOCAL
small-molecule fraction (derived in solubility_threshold from the r14 free
energy): comparing the free energy at psi=1 vs psi=0 at fixed composition,
the crystal is favoured iff

    phi_0 * drive + phi_0 * chi_ca * (1 - phi_0) < 0
    =>  phi_0 > phi* = 1 - |drive| / chi_ca,   drive = dh (T/Tm - 1) < 0.

Below phi* the crystal-contact penalty beats the undercooling drive and an
embryo redissolves; above it the embryo grows.  Drying RAISES the local
phi_0, so an embryo implanted in the WET film (phi_0 < phi*) dissolves
while the same embryo implanted MID-DRYING (phi_0 > phi*) grows.
embryo_composition_sweep VALIDATES the threshold directly (implant into
uniform blends of varying phi_0, no drying, and locate the grow/dissolve
crossover).  HONEST CAVEAT: phi* is the HOMOGENEOUS solubility; a
supercritical embryo enriches phi_0 in its neighbourhood as it orders (the
P7 crystal-bulk channel), so its EFFECTIVE growth threshold sits BELOW
phi* -- the derivation is an upper bound, and the measured crossover
confirms a composition threshold exists while lying below phi*.

SEEDING IS PHYSICAL AND DELICATE (the S3b campaign lessons, ledger
2026-07-12): the implanted embryo radius r0 must clear the Gibbs-Thomson
critical radius r* with margin (subcritical embryos redissolve), and the
embryo amplitude must be near 1 (half-amplitude embryos halve the bulk
driving and double r*).  We use r0 with margin and psi ~ 0.95, and read
the TERMINAL crystalline state (not a mid-growth snapshot).

TERMINATION.  A march that stops at the time horizon t_end is a
TIME-HORIZON stop, NOT a "drying time"; a stop at the solvent target
(phis_stop) is the dryness criterion; h_min is the height floor; a dt
underflow is a stiffness (Newton) failure.  run_arc reports the honest
termination status (see TERMINATION).

TUTORIAL SIMPLIFICATION (recorded honestly).  The production S3b config
uses the Vignes composition-singular mobility (mob="fastmode_n") with a
3-decade liquid->solid drop -- physically faithful but stiff (its deep-
quench Jacobian collapses the dt ladder).  For a tutorial we use a
CONSTANT Onsager mobility with the cuDSS direct solver; the qualitative
arc (dissolve-when-wet vs grow-when-dry) is unchanged, but the
quantitative drying-front sharpness is softened.  See
tests/test_multiphase_s3.py for the production run.
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

# Honest termination-status enum: what a stopped march actually means.
# A t_end stop is a TIME-HORIZON stop -- NOT a "drying time".
TERMINATION = {
    "t_end": "time_horizon",       # reached the requested sim horizon
    "phis_stop": "solvent_target",  # dried to the target solvent fraction
    "h_min": "height_floor",        # film thinned to the height floor
    "dt_underflow": "min_dt",       # Newton/stiffness collapsed the dt ladder
    "max_steps": "step_budget",     # exhausted the step budget
}


def drive_r14(T=333.0, dh=DH_F, Tm=TM_F):
    """Turnbull driving force drive = dh (T/Tm - 1) for the r14 bulk.
    Negative below Tm (crystallization favoured)."""
    return dh * (T / Tm - 1.0)


def solubility_threshold(chi_ca_val, T=333.0, dh=DH_F, Tm=TM_F):
    """The r14 crystal SOLUBILITY phi* in the local small-molecule fraction,
    DERIVED from the free energy.  Comparing the homogeneous free energy at
    psi=1 vs psi=0 at fixed composition, the psi-dependent part changes by

        Delta f = phi_0 * drive + phi_0 * chi_ca * (1 - phi_0),

    (bulk drive lowers it; the crystal-contact chi_ca penalty against the
    (1 - phi_0) non-crystallizing surroundings raises it).  Delta f < 0
    (crystal favoured) iff  phi_0 > phi* = 1 - |drive| / chi_ca.  Returns
    phi* clamped to [0, 1] (phi* <= 0 means no solubility barrier)."""
    drive = drive_r14(T, dh, Tm)
    phi_star = 1.0 - abs(drive) / chi_ca_val
    return float(min(max(phi_star, 0.0), 1.0))


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
                 eps2=4e-3, seed=1011, noise_psi=0.0):
    """S3b-class ternary drying film, tutorial mobility (const) + cuDSS.
    chi: fullerene-polymer 1.0 (Negi), fullerene-solvent 0.7248 (2310),
    polymer-solvent 0.3 (Wodo).  chi_ca on both fullerene contacts sets
    the crystal SOLUBILITY.  Blend starts dilute (85% solvent).
    noise_psi>0 turns on the ADVANCED FDT-noise mode (genuine nucleation,
    P8) instead of the deterministic implant -- slower and stochastic."""
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
        clip_psi=False, noise_psi=noise_psi, noise_seed=seed,
        tstep="bdf1")
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
    # local small-molecule fraction the embryo sees AT IMPLANT (it is
    # implanted at the phi_0-richest sites); this is what phi* governs.
    phi_f_imp = float(np.asarray(cons.T @ st.phi(0)).max())
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
                phi_f_implant_local=phi_f_imp,
                psi_max=float(psi_full.max()), ctrs=ctrs,
                termination=TERMINATION.get(reason, reason),
                termination_raw=reason, t_final=st.t, h_final=st.h_curr,
                k_e=k_e)


# ---------------------------------------------------------------------
# controls: static (no-evaporation) embryo, composition + radius sweeps
# ---------------------------------------------------------------------
def run_static_embryo(dm, mesh, cons, phi_f, phi_p=0.05, chi_ca_val=1.6,
                      r0=0.2, t_end=6.0, amp=0.005, k_e=0.0, T=333.0,
                      device="cuda:0"):
    """Implant one supercritical embryo into a UNIFORM blend at
    small-molecule fraction phi_f (rest solvent) with NO evaporation
    (k_e=0).  Isolates the SOLUBILITY: whether the embryo grows or
    redissolves is decided by phi_f vs phi* alone, with no drying and no
    composition gradient to confound it.  Returns the terminal crystalline
    area, psi_max, and a grow/dissolve classification."""
    st = make_stepper(dm, T=T, k_e=k_e, chi_ca_val=chi_ca_val)
    rng = np.random.default_rng(2024)
    icf = phi_f + amp * rng.standard_normal(st.nfree)
    icp = phi_p + amp * rng.standard_normal(st.nfree)
    st.set_initial([lambda x: icf, lambda x: icp],
                   [lambda x: np.zeros(len(x))],
                   [lambda x: np.zeros(len(x))])
    implant(st, [(0.5, 0.5)], r0=r0)
    a0 = crys_area(st, cons)
    reason = st.march(t_end=t_end, dt_max=0.02, max_steps=40000,
                      dt_min=1e-10, grow_iters=40)
    psi_full = np.asarray(cons.T @ st.psi(0))
    a1 = crys_area(st, cons)
    return dict(phi_f=float(phi_f), r0=float(r0), chi_ca=chi_ca_val,
                area0=float(a0), area1=float(a1),
                psi_max=float(psi_full.max()),
                grew=bool(a1 > a0 * 1.05),
                termination=TERMINATION.get(reason, reason))


def embryo_composition_sweep(dm, mesh, cons, chi_ca_val=1.6,
                             phi_fs=(0.10, 0.20, 0.30, 0.45, 0.60, 0.80),
                             device="cuda:0", **kw):
    """VALIDATE the derived solubility phi*: implant the SAME supercritical
    embryo into uniform blends of increasing phi_f (no drying) and locate
    the grow/dissolve crossover.

    HONEST FINDING.  The derived phi* is the HOMOGENEOUS solubility (no
    pre-existing crystal).  A supercritical embryo ENRICHES phi_0 in its
    neighbourhood as it orders (the P7 crystal-bulk channel), so its
    effective growth threshold sits BELOW the homogeneous phi*: the
    derivation is an UPPER BOUND, and the measured crossover confirms the
    physics (a composition threshold set by drive vs chi_ca exists) while
    lying below phi*.  Returns the per-composition fates, the measured
    crossover bracket, the derived phi*, and whether the crossover is
    (correctly) below it."""
    runs = [run_static_embryo(dm, mesh, cons, pf, chi_ca_val=chi_ca_val,
                              device=device, **kw) for pf in phi_fs]
    phi = np.array([r["phi_f"] for r in runs])
    grew = np.array([r["grew"] for r in runs])
    diss = phi[~grew]
    grow = phi[grew]
    lo = float(diss.max()) if diss.size else float("nan")
    hi = float(grow.min()) if grow.size else float("nan")
    cross = 0.5 * (lo + hi) if np.isfinite(lo) and np.isfinite(hi) else \
        (hi if np.isfinite(hi) else lo)
    phi_star = solubility_threshold(chi_ca_val)
    return dict(runs=runs, phi_fs=list(phi_fs), grew=grew.tolist(),
                crossover_lo=lo, crossover_hi=hi,
                crossover_measured=float(cross),
                crossover_bracketed=bool(np.isfinite(lo)
                                         and np.isfinite(hi)),
                phi_star_derived=phi_star,
                crossover_below_derived=bool(np.isfinite(cross)
                                             and cross <= phi_star),
                chi_ca=chi_ca_val)
