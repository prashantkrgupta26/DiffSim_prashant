"""OrgElMorph course - Physics P7: coupled Cahn-Hilliard + Allen-Cahn.

Importable core.  The real morphology problem is COUPLED: composition
(Cahn-Hilliard, P1-P5) and crystallinity (Allen-Cahn, P6) evolve
together and feed back on each other.  The signature of the coupling is
CRYSTALLIZATION-DRIVEN DEMIXING: as a species crystallizes it expels the
other components from the crystal, sharpening the composition pattern
that phase separation alone would not produce.

THE FOUR-FOLD chi.  With crystallinity the interaction between two
species depends on whether each is amorphous or crystalline, so the
single Flory chi becomes FOUR matrices.  The CONVENTION differs between
the two bulk free energies (this is the P7 correction resolved here):

  p1  (the mode this chapter runs) -- MUTUALLY-WEIGHTED, all four terms:
      chi_eff(i,j) = (1-psi_i)(1-psi_j) chi_aa + (1-psi_i) psi_j chi_ac
                   +  psi_i (1-psi_j)   chi_ca +  psi_i     psi_j   chi_cc
      At the four PURE limits chi_eff equals a single matrix each, so in
      p1 the four chi are ABSOLUTE pair interactions:
        (psi_i,psi_j) = (0,0) -> chi_aa   both amorphous
                        (1,0) -> chi_ca   i crystalline, j amorphous
                        (0,1) -> chi_ac   i amorphous, j crystalline
                        (1,1) -> chi_cc   both crystalline

  r14 -- psi^2-weighted INCREMENTS on top of chi_aa:
      chi_eff(i,j) = chi_aa + psi_i^2 chi_ca + psi_j^2 chi_ac
                   + psi_i^2 psi_j^2 chi_cc
      Here the pure limits are chi_aa, chi_aa+chi_ca, chi_aa+chi_ac,
      chi_aa+chi_ca+chi_ac+chi_cc, so in r14 the non-amorphous chi are
      DELTA-chi INCREMENTS relative to chi_aa, NOT absolute values.

  (chi_ca = chi_ac.T by pair symmetry in both.)  test_chi_limits.py
  verifies all four pure limits for BOTH conventions against the
  production evaluator diffsim.physics.multiphase._np_chi_eff.

Making chi_ca LARGER than chi_aa (p1) means "a crystal of species i
dislikes amorphous species j MORE than two amorphous phases dislike each
other" -- so crystallizing species i pushes j out.  That is the
thermodynamic origin of crystallization-driven demixing.

COMMENSURATE CONTROLS (the P7 correction).  We do NOT compare a crystal
run against a differently-masked no-crystal run (that amplified a
divide-by-floor artefact).  Instead we run THREE controls with the SAME
final crystal mask and the SAME metric:
  1. ch_only        -- K = 0, ordinary Cahn-Hilliard only;
  2. coupled_nochi  -- K = 1, the crystal grows but chi_ca = chi_aa
                       (crystal present, NO expulsion coupling);
  3. full           -- K = 1, chi_ca > chi_aa (the expulsion coupling).
The demixing metric is the composition contrast of species 0
<phi_0>_inside - <phi_0>_outside evaluated on ONE shared mask (the full
run's psi>0.5 footprint).  We report ABSOLUTE contrasts and the
attributable increments (full - coupled_nochi = the chi-coupling effect
at matched geometry; coupled_nochi - ch_only = crystal presence alone);
a relative amplification is quoted only when the ch_only denominator is
resolved above a floor.  Seed sensitivity is reported as a mean +/- sd.

CAUSALITY.  frozen_geometry: pre-seed the FINAL crystal and freeze it
(L_psi = 0), then run composition with full vs no chi-coupling -- demixing
appears ONLY with the coupling, so the coupling (not the act of
crystallizing) causes the demixing GIVEN the geometry.  kinetics: at
fixed coupling, a faster L_psi crystallizes sooner and the demixing
contrast tracks the crystalline area in time (kinetics gates the timing;
coupling sets the magnitude).
"""
import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.multiphase import MultiPhaseStepper, _np_chi_eff
from diffsim.diagnostics import conservation as _cons, energy as _en


def build_mesh_dm(level=5, p=1, device="cuda:0"):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), device)
    return dm, mesh, cons


# ---------------------------------------------------------------------
# the four-fold chi: pure-limit values (used by docs + the unit test)
# ---------------------------------------------------------------------
def chi_eff_limits(chi_aa, chi_ac, chi_ca, chi_cc, bulk="p1"):
    """chi_eff at the four pure (psi_i, psi_j) limits for a single pair,
    from the production evaluator.  Returns a dict keyed by the limit."""
    pars = dict(chi_aa=np.array([[0.0, chi_aa], [chi_aa, 0.0]]),
                chi_ac=np.array([[0.0, chi_ac], [chi_ac, 0.0]]),
                chi_ca=np.array([[0.0, chi_ca], [chi_ca, 0.0]]),
                chi_cc=np.array([[0.0, chi_cc], [chi_cc, 0.0]]),
                bulk=bulk)
    out = {}
    for name, (si, sj) in {"aa": (0.0, 0.0), "ca": (1.0, 0.0),
                           "ac": (0.0, 1.0), "cc": (1.0, 1.0)}.items():
        psi = {0: si, 1: sj}
        out[name] = float(_np_chi_eff(pars, 0, 1, lambda l: psi[l]))
    return out


def _chis(n_sp, chi_aa_val, chi_ca_val, cryst=(0,)):
    """Build the FULL-species (n_sp x n_sp) chi_aa and chi_ca.  Amorphous
    pairs interact with chi_aa_val (all off-diagonals); a crystalline
    species k (in `cryst`) interacts with every OTHER species through
    chi_ca_val.  Setting chi_ca_val == chi_aa_val is the no-coupling
    control (crystal present, no expulsion)."""
    aa = np.full((n_sp, n_sp), chi_aa_val)
    np.fill_diagonal(aa, 0.0)
    ca = np.zeros((n_sp, n_sp))
    for k in cryst:
        for j in range(n_sp):
            if j != k:
                ca[k, j] = chi_ca_val
    return aa, ca


def _disc(c, r0, w=0.03, amp=0.95):
    return lambda x: amp * 0.5 * (1.0 - np.tanh(
        (np.sqrt((x[:, 0] - c[0]) ** 2
                 + (x[:, 1] - c[1]) ** 2) - r0) / w))


def _grid_map(mesh):
    fc = mesh.node_coords
    xs = np.unique(np.round(fc[:, 0], 12))
    ys = np.unique(np.round(fc[:, 1], 12))
    ix = np.searchsorted(xs, np.round(fc[:, 0], 12))
    iy = np.searchsorted(ys, np.round(fc[:, 1], 12))
    return len(xs), len(ys), ix, iy


# ---------------------------------------------------------------------
# coupled free-energy budget (exact production conventions; p1)
# ---------------------------------------------------------------------
def _quad_weights(st):
    w = {}
    for pv in sorted(st.dm.bins):
        tb = st.dm.tables_by_p[pv]
        eids = st.mesh.bins[pv]
        h = st.mesh.tree.h()[eids]
        jac = (0.5 * h) ** st.dm.dim
        w[pv] = np.outer(jac, tb.w).ravel()
    return w


def _field_gp(st, vec):
    """(values, grads) of a free-node vector at the Gauss points, flat."""
    v, g = st._gp(vec)
    keys = sorted(v)
    return (np.concatenate([v[pv] for pv in keys]),
            np.concatenate([g[pv] for pv in keys], axis=0))


def coupled_energy(st, chi_aa_mat, chi_ca_mat, kappa, eps2, dsig, drive,
                   Ninv):
    """Ginzburg-Landau free-energy budget of the coupled (M, K) p1 film,
    integrated by quadrature with the EXACT production densities:

      f = SUM_l Ninv_l phi_l ln phi_l                       (entropy)
        + SUM_{i<j} phi_i phi_j chi_eff(i,j; psi)           (chi/coupling)
        + SUM_k phi_k^2 Ninv_k [psi_k(1-psi_k) dsig_k       (crystal bulk)
                                + psi_k^2 drive_k]
        + SUM_i (kappa_i/2)|grad phi_i|^2                    (comp gradient)
        + SUM_k (eps2_k/2)|grad psi_k|^2                     (cryst gradient)

    Returns a labelled dict (entropy/chi/cryst/grad_phi/grad_psi/total),
    each an integrated energy in the same units (diagnostics.energy)."""
    M, K = st.M, st.K
    wts = np.concatenate([_quad_weights(st)[pv] for pv in sorted(st.dm.bins)])
    phi_v, phi_g = [], []
    for i in range(M):
        v, g = _field_gp(st, st.phi(i))
        phi_v.append(v); phi_g.append(g)
    ps = 1.0 - sum(phi_v)
    all_v = phi_v + [ps]
    psi_v, psi_g = [], []
    for k in range(K):
        v, g = _field_gp(st, st.psi(k))
        psi_v.append(np.clip(v, 0.0, 1.0)); psi_g.append(g)

    def rlog(x):
        return np.log(np.clip(x, 1e-12, None))

    # entropy
    ent = sum(Ninv[l] * all_v[l] * rlog(all_v[l]) for l in range(M + 1))
    # chi / coupling (psi-dependent chi_eff over all pairs)
    pars = dict(chi_aa=chi_aa_mat, chi_ca=chi_ca_mat,
                chi_ac=chi_ca_mat.T.copy(),
                chi_cc=np.zeros_like(chi_aa_mat), bulk="p1")

    def psi_of(l):
        return psi_v[l] if l < K else 0.0
    chi = np.zeros_like(ent)
    for i in range(M + 1):
        for j in range(i + 1, M + 1):
            chi = chi + all_v[i] * all_v[j] * _np_chi_eff(pars, i, j, psi_of)
    # crystal bulk (p1)
    cry = np.zeros_like(ent)
    for k in range(K):
        s = psi_v[k]
        W = s * (1 - s) * dsig[k] + s ** 2 * drive[k]
        cry = cry + phi_v[k] ** 2 * Ninv[k] * W
    # gradients
    grad_phi = sum(0.5 * kappa[i] * np.sum(phi_g[i] ** 2, axis=1)
                   for i in range(M))
    grad_psi = (sum(0.5 * eps2[k] * np.sum(psi_g[k] ** 2, axis=1)
                    for k in range(K))
                if K > 0 else np.zeros_like(ent))
    split = {
        "entropy": _en.integrate_density(ent, wts),
        "chi": _en.integrate_density(chi, wts),
        "cryst": _en.integrate_density(cry, wts),
        "grad_phi": _en.integrate_density(grad_phi, wts),
        "grad_psi": _en.integrate_density(grad_psi, wts),
    }
    split["total"] = float(sum(split.values()))
    return split


# ---------------------------------------------------------------------
# one coupled run (a single control)
# ---------------------------------------------------------------------
def run(dm, mesh, cons, mode="full", M=2, K=1, chi_aa=1.2, chi_ca=2.6,
        dh=-1.3, dsig=1.0, Tm=1.0, T=0.55, eps2=1e-3, L_psi=6.0,
        kappa=5e-4, phi0=None, seeds=None, r0=0.13, amp=0.01, dt=5e-4,
        t_end=0.4, dt_max=0.02, seed=4, mobility=0.2, freeze_psi=False,
        track_energy=False, device="cuda:0", nchecks=16):
    """One coupled (M, K) quench with a seeded crystal of species 0.

    mode:
      "ch_only"       K = 0 (no crystal terms; ordinary Cahn-Hilliard).
      "coupled_nochi" K = 1 crystal, but chi_ca := chi_aa (no expulsion).
      "full"          K = 1 crystal with chi_ca > chi_aa (the coupling).
    freeze_psi=True sets L_psi = 0 (the crystal geometry is frozen at its
    IC -- the causality control).  Returns the fields, the crystalline
    area history, the species-0 field, and (if track_energy) the coupled
    energy budget history."""
    n_sp = M + 1
    if phi0 is None:
        phi0 = [0.45, 0.30, 0.15][:M]
    Kuse = 0 if mode == "ch_only" else K
    chi_ca_use = chi_aa if mode == "coupled_nochi" else chi_ca
    aa, ca = _chis(n_sp, chi_aa, chi_ca_use, cryst=tuple(range(min(K, M))))
    Lp = 0.0 if freeze_psi else L_psi
    kw = {}
    if Kuse > 0:
        kw = dict(chi_ac=ca.T.copy(), chi_ca=ca,
                  dsig=[dsig] * Kuse, dh=[dh] * Kuse, Tm=[Tm] * Kuse,
                  eps2=[eps2] * Kuse, L_psi=[Lp] * Kuse,
                  alpha_th=[0.0] * Kuse, beta_th=[0.0] * Kuse,
                  L_th=[5.0] * Kuse)
    st = MultiPhaseStepper(
        dm, M=M, K=Kuse, chi_aa=aa, N=[1.0] * n_sp,
        onsager=(mobility * np.eye(M)).tolist(), kappa=[kappa] * M, dt=dt,
        T=T, bulk="p1", newton_tol=1e-7, newton_max=50,
        linsolver="cudss", **kw)
    rng = np.random.default_rng(seed)
    phi_fns = [(lambda v: (lambda x: np.full(len(x), v)
                           + amp * rng.standard_normal(len(x))))(p)
               for p in phi0]
    if seeds is None:
        seeds = [(0.5, 0.5)]
    if Kuse > 0:
        coords = st.free_coords
        psi_list = []
        for k in range(Kuse):
            psv = np.zeros(st.nfree)
            for c in seeds:
                psv = np.maximum(psv, _disc(c, r0)(coords))
            psi_list.append(psv)
        st.set_initial(phi_fns,
                       [(lambda a: (lambda x: a))(psv) for psv in psi_list],
                       [(lambda x: np.zeros(len(x)))] * Kuse)
    else:
        st.set_initial(phi_fns)
    nx, ny, ix, iy = _grid_map(mesh)

    # energy budget setup (exact p1 conventions)
    drive_p1 = np.array([dh * (1.0 - T / Tm)] * max(Kuse, 1))
    ebudget = None
    if track_energy and Kuse > 0:
        ebudget = [coupled_energy(st, aa, ca, [kappa] * M, [eps2] * Kuse,
                                  [dsig] * Kuse, drive_p1, [1.0] * n_sp)]

    def crys_mask():
        return (np.asarray(cons.T @ st.psi(0)) > 0.5) if Kuse else None

    def area():
        return 0.0 if Kuse == 0 else float(np.mean(crys_mask()))

    ts, ars = [0.0], [area()]
    checks = np.linspace(t_end / nchecks, t_end, nchecks)
    for tc in checks:
        st.march(t_end=tc, dt_max=dt_max, max_steps=1500, dt_min=1e-7)
        ts.append(st.t); ars.append(area())
        if ebudget is not None:
            ebudget.append(coupled_energy(st, aa, ca, [kappa] * M,
                                          [eps2] * Kuse, [dsig] * Kuse,
                                          drive_p1, [1.0] * n_sp))

    phi0_full = np.asarray(cons.T @ st.phi(0))

    def grid(vec):
        g = np.zeros((nx, ny)); g[ix, iy] = np.asarray(vec)
        return g
    out = dict(mode=mode, M=M, K=Kuse, t=np.array(ts), area=np.array(ars),
               area_end=ars[-1], phi0_full=phi0_full,
               phi0=grid(phi0_full), nx=nx, ny=ny,
               psi_full=(np.asarray(cons.T @ st.psi(0)) if Kuse else None),
               psi=grid(np.asarray(cons.T @ st.psi(0))) if Kuse else None,
               phi0_std=float(phi0_full.std()))
    if ebudget is not None:
        out["energy"] = ebudget
        out["energy_total"] = np.array([e["total"] for e in ebudget])
    return out


# ---------------------------------------------------------------------
# the commensurate metric (SAME mask, SAME statistic for every control)
# ---------------------------------------------------------------------
def contrast_on_mask(phi0_full, mask):
    """Composition contrast of species 0 on a fixed boolean node mask:
    <phi_0>_inside - <phi_0>_outside.  Same mask + statistic for every
    control -- no per-run mask, no divide-by-floor."""
    if mask.any() and (~mask).any():
        return float(phi0_full[mask].mean() - phi0_full[~mask].mean())
    return float("nan")
